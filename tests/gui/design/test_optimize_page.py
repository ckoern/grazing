"""Tests of the Optimize workflow page."""

import dataclasses
from typing import Any

import pytest
from pytestqt.qtbot import QtBot

from grazing.core.optim.events import CancellationToken
from grazing.core.optim.optimizers import LmfitOptimizer
from grazing.core.parameters import free
from grazing.core.structure import Mirror, Periodic
from grazing.core.targets import EnergyScan, TargetSet
from grazing.design.merit import (
    Flatness,
    MeanReflectivity,
    PeakReflectivity,
    SoftMinReflectivity,
    TimePenalized,
    WeightedSum,
)
from grazing.design.merit.base import CallableMerit
from grazing.design.optim.scan import GridScan
from grazing.design.optim.stages import BraggOutline
from grazing.design.project.commands import set_mirror
from grazing.design.project.templates import new_state
from grazing.gui.common.workers import Job
from grazing.gui.design.steps import OptimizePage
from grazing.gui.design.steps.optimize import MERIT_DESCRIPTIONS, OPTICAL_MERITS

pytestmark = pytest.mark.gui


def small_state() -> Any:
    state = new_state("periodic")
    assert state.mirror is not None
    coating = Periodic.create(
        state.mirror.coating.material("H"),
        state.mirror.coating.material("L"),
        N=free(20, 10, 40),
        d=free(4.0, 3.0, 5.0),
        gamma=free(0.4, 0.2, 0.6),
    )
    return set_mirror(Mirror(coating, state.mirror.substrate)).transform(state)


@pytest.fixture
def page(qtbot: QtBot) -> OptimizePage:
    widget = OptimizePage()
    qtbot.addWidget(widget)
    widget.set_state(small_state())
    return widget


def select(combo: Any, name: str) -> None:
    combo.setCurrentIndex(combo.findData(name))
    combo.activated.emit(combo.currentIndex())


def test_help_text() -> None:
    text = OptimizePage.help_text
    for word in ("lower is", "outline", "scan", "global", "refine", "run N", "Convergence"):
        assert word in text
    assert "never overwritten" in text


def test_merit_description_follows_selection(page: OptimizePage) -> None:
    assert set(MERIT_DESCRIPTIONS) == set(OPTICAL_MERITS)
    seen = set()
    for name in OPTICAL_MERITS:
        select(page.merit_combo, name)
        assert page.merit_description.text() == MERIT_DESCRIPTIONS[name]
        seen.add(page.merit_description.text())
        index = page.merit_combo.findData(name)
        assert page.merit_combo.itemData(index, 3) == MERIT_DESCRIPTIONS[name]
    assert len(seen) == len(OPTICAL_MERITS)
    assert page.merit_description.wordWrap()


def test_flatness_shows_hint_until_mixed(page: OptimizePage) -> None:
    select(page.merit_combo, "flatness")
    assert not page.flatness_hint.isHidden()
    page.mix_check.setChecked(True)
    assert page.flatness_hint.isHidden()


def test_merit_editing_and_time_penalty(page: OptimizePage) -> None:
    state = small_state()
    commands: list[Any] = []
    page.commandRequested.connect(commands.append)
    page.time_check.setChecked(True)
    merit = commands[-1].transform(state).merit
    assert isinstance(merit, TimePenalized)
    assert isinstance(merit.merit, MeanReflectivity)
    select(page.merit_combo, "soft_min_reflectivity")
    merit = commands[-1].transform(state).merit
    assert merit.merit.type_name == "soft_min_reflectivity"
    page.set_state(state.evolve(merit=merit))
    assert page.time_check.isChecked()
    assert page.current_merit() == merit


def test_mixing_builds_weighted_sum_and_round_trips(page: OptimizePage) -> None:
    state = small_state()
    commands: list[Any] = []
    page.commandRequested.connect(commands.append)
    select(page.merit_combo, "flatness")
    page.mix_check.setChecked(True)
    select(page.secondary_combo, "mean_reflectivity")
    page.fraction.setValue(0.3)
    merit = commands[-1].transform(state).merit
    assert isinstance(merit, WeightedSum)
    (first, w1), (second, w2) = merit.terms
    assert isinstance(first, Flatness)
    assert isinstance(second, MeanReflectivity)
    assert w1 == pytest.approx(0.7)
    assert w2 == pytest.approx(0.3)

    page.time_check.setChecked(True)
    timed = commands[-1].transform(state).merit
    assert isinstance(timed, TimePenalized)
    assert timed.merit == merit

    other = OptimizePage()
    other.set_state(state.evolve(merit=timed))
    assert other.mix_check.isChecked()
    assert other.time_check.isChecked()
    assert other.merit_combo.currentData() == "flatness"
    assert other.secondary_combo.currentData() == "mean_reflectivity"
    assert other.fraction.value() == pytest.approx(0.3)
    assert other.current_merit() == timed
    assert other.custom_label.isHidden()


def test_unmixed_state_clears_mixing(page: OptimizePage) -> None:
    state = small_state()
    mixed = WeightedSum(((PeakReflectivity(), 0.6), (SoftMinReflectivity(), 0.4)))
    page.set_state(state.evolve(merit=mixed))
    assert page.mix_check.isChecked()
    page.set_state(state.evolve(merit=MeanReflectivity()))
    assert not page.mix_check.isChecked()


def test_custom_merit_detection(page: OptimizePage) -> None:
    state = small_state()
    page.set_state(state.evolve(merit=WeightedSum(((MeanReflectivity(), 1.0),))))
    assert not page.custom_label.isHidden()
    three = WeightedSum(((MeanReflectivity(), 0.5), (Flatness(), 0.3), (PeakReflectivity(), 0.2)))
    page.set_state(state.evolve(merit=three))
    assert not page.custom_label.isHidden()
    page.set_state(state.evolve(merit=MeanReflectivity()))
    assert page.custom_label.isHidden()
    nested = WeightedSum(((CallableMerit(lambda ctx: 0.0, "x"), 0.5), (MeanReflectivity(), 0.5)))
    page.set_state(state.evolve(merit=nested))
    assert not page.custom_label.isHidden()


def test_stage_table_lists_default_stages(page: OptimizePage) -> None:
    names = ["outline", "scan", "global", "refine"]
    assert page.stage_table.rowCount() == len(names)
    problem = small_state().problem()
    assert [s.name for s in page.pipeline(problem).stages] == names
    assert page.stage_combo.count() == len(names)
    assert page.stage_table.showGrid() is False


def test_disabling_stage_removes_it(page: OptimizePage) -> None:
    problem = small_state().problem()
    page._rows["scan"].enabled.setChecked(False)
    assert [s.name for s in page.pipeline(problem).stages] == ["outline", "global", "refine"]
    assert "scan" not in [page.stage_combo.itemText(i) for i in range(page.stage_combo.count())]


def test_stage_settings_reach_pipeline(page: OptimizePage) -> None:
    problem = small_state().problem()
    rows = page._rows
    method = rows["refine"].method
    assert method is not None
    method.setCurrentIndex(method.findData("nelder"))
    rows["refine"].evaluations.setValue(40)
    rows["refine"].points.setValue(12)
    rows["scan"].points.setValue(7)
    rows["global"].evaluations.setValue(55)
    stages = {s.name: s for s in page.pipeline(problem).stages}
    refine = stages["refine"].optimizer
    assert isinstance(refine, LmfitOptimizer)
    assert (refine.method, refine.max_evaluations) == ("nelder", 40)
    assert stages["refine"].points == 12
    assert isinstance(stages["scan"].optimizer, GridScan)
    assert stages["scan"].points == 7
    glob = stages["global"].optimizer
    assert isinstance(glob, LmfitOptimizer)
    assert glob.method == "differential_evolution"
    assert glob.max_evaluations == 55
    assert dict(glob.options)["popsize"] == page.population.value()
    assert rows["outline"].method is None


def two_target_state() -> Any:
    state = small_state()
    first = state.targets.targets[0]
    assert isinstance(first, EnergyScan)
    second = dataclasses.replace(
        first, energy_min=2 * first.energy_min, energy_max=2 * first.energy_max
    )
    return state.evolve(targets=TargetSet((first, second), "average_merit"))


def outline_target(page: OptimizePage, state: Any) -> Any:
    stage = {s.name: s for s in page.pipeline(state.problem()).stages}["outline"]
    assert isinstance(stage.optimizer, BraggOutline)
    return stage.optimizer.target


def test_bragg_target_choice_disabled_for_single_target(page: OptimizePage) -> None:
    combo = page._rows["outline"].target
    assert combo is not None
    assert not combo.isEnabled()
    assert outline_target(page, small_state()) is None


def test_bragg_target_choice_reaches_pipeline(page: OptimizePage) -> None:
    state = two_target_state()
    page.set_state(state)
    combo = page._rows["outline"].target
    assert combo is not None
    assert combo.isEnabled()
    assert combo.itemText(0) == "Bragg law: all targets"
    assert combo.count() == 3
    assert outline_target(page, state) is None
    combo.setCurrentIndex(2)
    assert outline_target(page, state) == 1
    page.set_state(state)
    assert combo.currentData() == 1
    page.set_state(small_state())
    assert combo.count() == 2
    assert combo.currentData() is None
    assert not combo.isEnabled()
    assert outline_target(page, small_state()) is None


def test_global_settings_reach_pipeline(page: OptimizePage) -> None:
    page.seed.setText("5")
    page.population.setValue(4)
    page.coarse.setValue(11)
    stages = {s.name: s for s in page.pipeline(small_state().problem()).stages}
    glob = stages["global"].optimizer
    assert isinstance(glob, LmfitOptimizer)
    assert dict(glob.options)["seed"] == 5
    assert stages["global"].points == 11


def test_run_single_disabled_stage_reports_error(page: OptimizePage) -> None:
    errors: list[str] = []
    page.errorOccurred.connect(errors.append)
    page._rows["scan"].enabled.setChecked(False)
    page.run("scan")
    assert errors


def test_job_returns_result_and_stage_evaluations(page: OptimizePage) -> None:
    jobs: list[Job] = []
    page.jobRequested.connect(jobs.append)
    emitted: list[tuple[Any, ...]] = []
    page.pipelineFinished.connect(lambda *args: emitted.append(args))
    page.run("outline")
    outcome = jobs[0].function(lambda _: None, CancellationToken())
    result, evaluations = outcome
    assert [s.stage.name for s in result.stages] == ["outline"]
    assert len(evaluations) == 1
    jobs[0].on_finished(outcome)
    assert len(emitted) == 1
    problem, emitted_result, run_name = emitted[0]
    assert run_name is None
    assert emitted_result is result
    assert problem is not None


def test_results_table_after_finished(page: OptimizePage) -> None:
    page.seed.setText("1")
    page.population.setValue(2)
    page.iterations.setValue(1)
    page.coarse.setValue(8)
    page.scan_points.setValue(2)
    page._rows["refine"].evaluations.setValue(5)
    jobs: list[Job] = []
    page.jobRequested.connect(jobs.append)
    page.run()
    outcome = jobs[0].function(lambda _: None, CancellationToken())
    jobs[0].on_finished(outcome)
    table = page.results_table
    assert table.rowCount() == 4
    assert [table.item(r, 0).text() for r in range(4)] == ["outline", "scan", "global", "refine"]
    assert int(table.item(0, 1).text()) == outcome[0].stages[0].result.evaluations
    for row in range(4):
        float(table.item(row, 2).text())
        float(table.item(row, 3).text())
        float(table.item(row, 4).text())
