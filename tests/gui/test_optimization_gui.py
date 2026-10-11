"""Tests of background jobs, scans, pipeline runs, tolerance analysis and candidates."""

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PySide6.QtCore import Qt
from pytestqt.qtbot import QtBot

from xstack.gui.main_window import MainWindow
from xstack.gui.plots import ConvergencePlot, DesignMapPlot
from xstack.gui.steps import AnalyzePage, ScanPage
from xstack.gui.workers import Job, JobRunner
from xstack.optim.events import CancellationToken, ProgressEvent
from xstack.parameters import free
from xstack.project.commands import set_mirror
from xstack.project.templates import new_state
from xstack.structure import Mirror, Periodic

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


def d_value(window: MainWindow) -> float:
    mirror = window.controller.state.mirror
    assert mirror is not None
    return mirror.parameters["d"].value


def wait_job(qtbot: QtBot, runner: JobRunner, timeout: int = 60000) -> None:
    qtbot.waitUntil(lambda: not runner.running, timeout=timeout)


def test_job_runner_runs_one_job_and_reports_progress(qtbot: QtBot) -> None:
    runner = JobRunner()
    results: list[Any] = []
    progress: list[Any] = []
    runner.progress.connect(progress.append)

    def compute(report: Any, cancel: CancellationToken) -> int:
        report("halfway")
        return 42

    with qtbot.waitSignal(runner.finished, timeout=5000):
        assert runner.start(Job("answer", compute, results.append))
        assert not runner.start(Job("second", compute, results.append))
    assert results == [42]
    assert progress == ["halfway"]
    assert not runner.running


def test_job_runner_failure_and_cancellation(qtbot: QtBot) -> None:
    runner = JobRunner()

    def fail(report: Any, cancel: CancellationToken) -> None:
        raise ValueError("boom")

    with qtbot.waitSignal(runner.failed, timeout=5000) as blocker:
        runner.start(Job("failing", fail, lambda _: None))
    assert "boom" in blocker.args[0]

    seen: list[bool] = []

    def wait_for_cancel(report: Any, cancel: CancellationToken) -> bool:
        while not cancel.cancelled:
            pass
        return True

    with qtbot.waitSignal(runner.finished, timeout=5000):
        runner.start(Job("cancellable", wait_for_cancel, seen.append))
        runner.cancel()
    assert seen == [True]


def test_convergence_plot_collects_points(qtbot: QtBot) -> None:
    plot = ConvergencePlot()
    qtbot.addWidget(plot)
    plot.add_point("global", 1, 0.5)
    plot.add_point("global", 2, 0.4)
    plot.add_point("refine", 1, 0.3)
    assert plot.tasks() == ("global", "refine")
    plot.clear()
    assert plot.tasks() == ()


def test_design_map_emits_selected_point(qtbot: QtBot) -> None:
    plot = DesignMapPlot()
    qtbot.addWidget(plot)
    x, y = np.array([1.0, 2.0, 3.0]), np.array([10.0, 20.0])
    values = np.arange(6.0).reshape(3, 2)
    feasible = np.ones((3, 2), dtype=bool)
    feasible[0, 0] = False
    plot.set_scan(["a", "b"], [x, y], values, feasible)
    with qtbot.waitSignal(plot.pointSelected) as blocker:
        plot.select(2, 1)
    assert blocker.args[0] == {"a": 3.0, "b": 20.0}
    plot.set_scan(["a"], [x], values[:, 0], feasible[:, 0])


def test_scan_page_axes_and_job(qtbot: QtBot) -> None:
    page = ScanPage()
    qtbot.addWidget(page)
    page.set_state(small_state())
    page.x_combo.setCurrentText("N")
    page.x_points.setValue(5)
    page.y_combo.setCurrentText("d")
    page.y_points.setValue(3)
    axes = page.axes()
    assert list(axes) == ["N", "d"]
    np.testing.assert_allclose(axes["d"], [3.0, 4.0, 5.0])
    assert np.all(axes["N"] == np.round(axes["N"]))
    jobs: list[Job] = []
    page.jobRequested.connect(jobs.append)
    page.grid_points.setValue(5)
    page.run()
    assert len(jobs) == 1
    result = jobs[0].function(lambda _: None, CancellationToken())
    assert result.objective.shape == (len(axes["N"]), 3)
    commands: list[Any] = []
    page.commandRequested.connect(commands.append)
    jobs[0].on_finished(result)
    assert page.apply_button.isEnabled()
    page.apply_best()
    new_state_ = commands[-1].transform(small_state())
    best = result.best_values()
    assert new_state_.mirror.parameters["d"].value == pytest.approx(best["d"])
    assert not page.confirmation.isHidden()
    assert page.confirmation.text().startswith("Start values set: ")
    assert f"d = {best['d']:.4g}" in page.confirmation.text()


def test_scan_page_reports_missing_bounds(qtbot: QtBot) -> None:
    page = ScanPage()
    qtbot.addWidget(page)
    state = small_state()
    params = state.mirror.parameters.replace("d", max=np.inf)
    state = state.evolve(
        mirror=state.mirror.with_coating(state.mirror.coating.with_parameters(params))
    )
    page.set_state(state)
    page.x_combo.setCurrentText("d")
    errors: list[str] = []
    page.errorOccurred.connect(errors.append)
    page.run()
    assert errors
    assert "finite bounds" in errors[0]


def test_analyze_page_tolerance_and_export(qtbot: QtBot, tmp_path: Path) -> None:
    page = AnalyzePage()
    qtbot.addWidget(page)
    state = small_state()
    page.set_state(state)
    page.samples.setValue(3)
    page.seed.setText("1")
    jobs: list[Job] = []
    page.jobRequested.connect(jobs.append)
    page.run()
    result = jobs[0].function(lambda _: None, CancellationToken())
    assert len(result.merits) == 3
    jobs[0].on_finished(result)
    assert "Nominal" in page.summary.text()
    csv_path, json_path = tmp_path / "recipe.csv", tmp_path / "recipe.json"
    assert page.export(csv_path, "csv")
    assert page.export(json_path, "json")
    with csv_path.open() as f:
        assert len(list(csv.DictReader(f))) == state.mirror.stack().num_layers
    assert "layers" in json.loads(json_path.read_text())


def test_window_pipeline_run_stores_candidates(qtbot: QtBot, window: MainWindow) -> None:
    window.controller.execute(set_mirror(small_state().mirror))
    input_design = window.controller.state.mirror
    window.optimize_page.seed.setText("1")
    window.optimize_page.iterations.setValue(1)
    window.optimize_page.population.setValue(3)
    window.optimize_page.coarse.setValue(10)
    window.optimize_page.scan_points.setValue(2)
    events: list[Any] = []
    window.job_runner.progress.connect(events.append)
    window.optimize_page.run()
    assert window.stop_action.isEnabled()
    wait_job(qtbot, window.job_runner)
    assert not window.stop_action.isEnabled()
    assert any(isinstance(e, ProgressEvent) for e in events)
    assert window.convergence_plot.tasks()
    state = window.controller.state
    names = [c.name for c in state.candidates]
    stages = [s.name for s in window.optimize_page.pipeline(state.problem()).stages]
    assert names == [f"run 1 · {stage}" for stage in stages] + ["run 1"]
    assert state.selected == "run 1"
    assert state.mirror == input_design
    assert window.controller.project.journal.entries("run")
    window.controller.undo()
    assert window.controller.state.candidates == ()
    window.controller.undo_stack.setClean()


def test_window_candidates_store_use_delete_and_overlay(qtbot: QtBot, window: MainWindow) -> None:
    qtbot.waitUntil(lambda: window.preview is not None, timeout=10000)
    window.store_candidate("first")
    assert [c.name for c in window.controller.state.candidates] == ["first"]
    window.parameter_table.table_model.setData(
        window.parameter_table.table_model.index(1, 1), "4.5"
    )
    assert d_value(window) == pytest.approx(4.5)
    window.candidate_list.setCurrentRow(0)
    window.use_button.click()
    assert d_value(window) == pytest.approx(4.0)
    window.candidate_list.item(0).setCheckState(Qt.CheckState.Checked)
    assert window.overlays == ["first"]
    qtbot.waitUntil(
        lambda: window.preview is not None and len(window.preview.overlays) == 1, timeout=10000
    )
    window.candidate_list.setCurrentRow(0)
    window.delete_button.click()
    assert window.controller.state.candidates == ()
    assert window.overlays == []
    window.controller.undo_stack.setClean()


def test_window_scan_shows_design_map_and_point_selection(qtbot: QtBot, window: MainWindow) -> None:
    window.controller.execute(set_mirror(small_state().mirror))
    page = window.scan_page
    page.x_combo.setCurrentText("N")
    page.x_points.setValue(3)
    page.y_combo.setCurrentText("d")
    page.y_points.setValue(3)
    page.grid_points.setValue(5)
    page.run()
    wait_job(qtbot, window.job_runner)
    assert window.plot_tabs.currentWidget() is window.design_map
    window.design_map.select(0, 2)
    assert d_value(window) == pytest.approx(5.0)
    window.controller.undo_stack.setClean()


def test_window_rejects_second_job(qtbot: QtBot, window: MainWindow) -> None:
    def slow(report: Any, cancel: CancellationToken) -> None:
        while not cancel.cancelled:
            pass

    assert window.start_job(Job("slow", slow, lambda _: None))
    assert not window.start_job(Job("other", slow, lambda _: None))
    assert "Another calculation" in window.statusBar().currentMessage()
    window.stop_action.trigger()
    wait_job(qtbot, window.job_runner)
