"""Tests of navigation, plot ranges, candidates, live curves and plots of the main window."""

import csv
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QPushButton
from pytestqt.qtbot import QtBot

from xstack.design.optim.events import CancellationToken, ProgressEvent
from xstack.design.optim.optimizers import LmfitOptimizer
from xstack.design.optim.pipeline import Pipeline
from xstack.design.optim.stages import Stage
from xstack.design.project.templates import new_state
from xstack.gui.common.workers import Job
from xstack.gui.design.main_window import WORKFLOW_STEPS, MainWindow
from xstack.gui.design.plots import ConvergencePlot, Curve, DesignMapPlot, ReflectivityPlot
from xstack.gui.design.preview import compute_curves

pytestmark = pytest.mark.gui


def wait_for_preview(qtbot: QtBot, window: MainWindow, previous: object = None) -> None:
    qtbot.waitUntil(
        lambda: (
            window.preview is not None
            and window.preview is not previous
            and window.preview.state is window.controller.state
        ),
        timeout=10000,
    )


def set_d(window: MainWindow, value: float) -> None:
    model = window.parameter_table.table_model
    names = list(model.parameters())
    assert model.setData(model.index(names.index("d"), 1), str(value))


def d_value(window: MainWindow) -> float:
    mirror = window.controller.state.mirror
    assert mirror is not None
    return mirror.parameters["d"].value


def set_range(window: MainWindow, axis: str, low: str, high: str) -> None:
    low_edit, high_edit = window.range_edits[axis]
    low_edit.setText(low)
    high_edit.setText(high)
    high_edit.editingFinished.emit()


def item_names(window: MainWindow) -> list[str]:
    return [window.candidate_list.item(i).text() for i in range(1, window.candidate_list.count())]


# Navigation


def test_pages_with_help_text_have_help_label(window: MainWindow) -> None:
    for index, step in enumerate(WORKFLOW_STEPS):
        container = window.step_tabs.widget(index)
        label = container.findChild(QLabel, f"help_{step.lower()}")
        if getattr(window.pages[step], "help_text", ""):
            assert label is not None, step
            assert label.text().strip()
        else:
            assert label is None, step


def test_next_and_previous_step(window: MainWindow) -> None:
    window.step_tabs.setCurrentIndex(0)
    window.next_step()
    assert window.step_tabs.currentIndex() == 1
    assert window.step_tabs.currentIndex() == 1
    window.previous_step()
    assert window.step_tabs.currentIndex() == 0
    window.previous_step()
    assert window.step_tabs.currentIndex() == 0
    window.step_tabs.setCurrentIndex(len(WORKFLOW_STEPS) - 1)
    window.next_step()
    assert window.step_tabs.currentIndex() == len(WORKFLOW_STEPS) - 1
    assert window.step_tabs.currentIndex() == len(WORKFLOW_STEPS) - 1


def test_next_buttons_and_back_buttons(window: MainWindow) -> None:
    for index, step in enumerate(WORKFLOW_STEPS):
        container = window.step_tabs.widget(index)
        forward = container.findChild(QPushButton, f"next_{step.lower()}")
        if index < len(WORKFLOW_STEPS) - 1:
            assert forward is not None, step
            window.step_tabs.setCurrentIndex(index)
            forward.click()
            assert window.step_tabs.currentIndex() == index + 1
        else:
            assert forward is None
    first = window.step_tabs.widget(0)
    back = [b for b in first.findChildren(QPushButton) if b.text().endswith("Back")]
    assert len(back) == 1
    assert not back[0].isEnabled()
    second_back = [
        b for b in window.step_tabs.widget(1).findChildren(QPushButton) if "Back" in b.text()
    ]
    assert second_back[0].isEnabled()


# Plot range


def test_energy_range_is_stored_in_ev(qtbot: QtBot, window: MainWindow) -> None:
    wait_for_preview(qtbot, window)
    first = window.preview
    set_range(window, "energy", "7.5", "8.5")
    assert window.display_ranges["energy"] == pytest.approx((7500.0, 8500.0))
    wait_for_preview(qtbot, window, first)
    assert window.preview is not None
    target = window.preview.state.targets.targets[0]  # type: ignore[union-attr]
    curve = window.preview.curves[0]
    assert curve.x[0] == pytest.approx(7500.0)
    assert curve.x[-1] == pytest.approx(8500.0)
    result = window.preview.results[0]
    assert result.x[0] == pytest.approx(target.scan_range[0])
    assert result.x[-1] == pytest.approx(target.scan_range[1])


def test_empty_range_removes_display_range(qtbot: QtBot, window: MainWindow) -> None:
    wait_for_preview(qtbot, window)
    set_range(window, "energy", "7.5", "8.5")
    assert "energy" in window.display_ranges
    set_range(window, "energy", "", "")
    assert window.display_ranges == {}


def test_invalid_ranges_report_error_and_keep_ranges(qtbot: QtBot, window: MainWindow) -> None:
    wait_for_preview(qtbot, window)
    set_range(window, "energy", "7.5", "8.5")
    valid = dict(window.display_ranges)
    for low, high in (("9", "8"), ("8", "8"), ("abc", "8")):
        window.statusBar().clearMessage()
        set_range(window, "energy", low, high)
        assert "plot range" in window.statusBar().currentMessage().lower()
        assert window.display_ranges == valid
        set_range(window, "energy", "7.5", "8.5")
        assert window.display_ranges == valid


def test_invalid_range_on_fresh_window_leaves_ranges_empty(window: MainWindow) -> None:
    set_range(window, "energy", "abc", "8")
    assert window.display_ranges == {}
    assert "Plot range" in window.statusBar().currentMessage()


def test_log_check_toggles_log_scale(window: MainWindow) -> None:
    def log_scale() -> bool:
        return window.reflectivity_plot.log_scale

    assert not log_scale()
    window.log_check.setChecked(True)
    assert log_scale()
    window.log_check.setChecked(False)
    assert not log_scale()


# Candidates


def test_store_select_and_details(qtbot: QtBot, window: MainWindow) -> None:
    wait_for_preview(qtbot, window)
    window.store_candidate("mine")
    state = window.controller.state
    assert [c.name for c in state.candidates] == ["mine"]
    assert state.candidates[0].mirror == state.mirror
    assert state.selected is None
    window.candidate_list.setCurrentRow(1)
    assert window.controller.state.selected == "mine"
    assert "mine" in window.candidate_details.text()
    window.controller.undo_stack.setClean()


def test_selected_candidate_overlay_is_automatic(qtbot: QtBot, window: MainWindow) -> None:
    wait_for_preview(qtbot, window)
    window.store_candidate("mine")
    first = window.preview
    window.candidate_list.setCurrentRow(1)
    wait_for_preview(qtbot, window, first)
    assert window.preview is not None
    assert [n for n, _ in window.preview.overlays] == ["mine"]
    assert window.overlays == []
    assert window.candidate_list.item(1).checkState() == Qt.CheckState.Unchecked
    window.controller.undo_stack.setClean()


def test_rename_candidate_keeps_selection(qtbot: QtBot, window: MainWindow) -> None:
    window.store_candidate("mine")
    window.candidate_list.setCurrentRow(1)
    window.candidate_list.item(1).setText("renamed")
    state = window.controller.state
    assert [c.name for c in state.candidates] == ["renamed"]
    assert state.selected == "renamed"
    assert item_names(window) == ["renamed"]
    assert window.candidate_list.currentItem().text() == "renamed"
    assert "renamed" in window.candidate_details.text()
    window.controller.undo_stack.setClean()


def test_rename_to_existing_name_is_rejected(qtbot: QtBot, window: MainWindow) -> None:
    window.store_candidate("one")
    window.store_candidate("two")
    window.candidate_list.item(2).setText("one")
    assert [c.name for c in window.controller.state.candidates] == ["one", "two"]
    assert item_names(window) == ["one", "two"]
    window.controller.undo_stack.setClean()


def test_use_as_start_copies_candidate(qtbot: QtBot, window: MainWindow) -> None:
    window.store_candidate("mine")
    set_d(window, 4.5)
    assert d_value(window) == pytest.approx(4.5)
    window.candidate_list.setCurrentRow(1)
    window.use_button.click()
    assert d_value(window) == pytest.approx(4.0)
    assert window.controller.state.mirror == window.controller.state.candidates[0].mirror
    window.controller.undo_stack.setClean()


def test_delete_clears_selection(qtbot: QtBot, window: MainWindow) -> None:
    window.store_candidate("mine")
    window.candidate_list.setCurrentRow(1)
    assert window.controller.state.selected == "mine"
    window.delete_button.click()
    state = window.controller.state
    assert state.candidates == ()
    assert state.selected is None
    assert window.candidate_list.count() == 1
    current = window.candidate_list.currentItem()
    assert current is not None
    assert current.data(Qt.ItemDataRole.UserRole) is None
    window.controller.undo_stack.setClean()


def test_store_button_uses_dialog_name(
    qtbot: QtBot, window: MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "xstack.gui.design.main_window.QInputDialog.getText", lambda *a, **k: ("given", True)
    )
    window.store_button.click()
    assert [c.name for c in window.controller.state.candidates] == ["given"]
    monkeypatch.setattr(
        "xstack.gui.design.main_window.QInputDialog.getText", lambda *a, **k: ("other", False)
    )
    window.store_button.click()
    assert len(window.controller.state.candidates) == 1
    window.controller.undo_stack.setClean()


# Live best curves


def test_live_best_curves_during_job(qtbot: QtBot, window: MainWindow) -> None:
    wait_for_preview(qtbot, window)

    def wait_for_cancel(report: Any, cancel: CancellationToken) -> None:
        while not cancel.cancelled:
            time.sleep(0.005)

    assert window.start_job(Job("dummy", wait_for_cancel, lambda _: None))
    assert window.job_runner.running
    window._job_started("x")
    event = ProgressEvent(
        task="stage one",
        evaluations=3,
        total=10,
        best_objective=0.5,
        current_objective=0.6,
        best_values={"d": 4.2},
        elapsed=0.1,
        best_merit=0.4,
        best_quantities={"mean reflectivity": 0.6},
    )
    window._job_progress(event)
    qtbot.waitUntil(lambda: bool(window.live_curves), timeout=10000)
    assert "stage one" in window.convergence_plot.tasks()
    assert window.convergence_plot.quantities() == ("mean reflectivity",)
    items = window.reflectivity_plot.plot_items()["energy"].listDataItems()
    assert len(items) >= 2
    window.stop_action.trigger()
    qtbot.waitUntil(lambda: not window.job_runner.running, timeout=10000)
    qtbot.waitUntil(lambda: window.live_curves == (), timeout=10000)
    assert not window.progress_bar.isVisible()


# Pipeline results


def test_apply_pipeline_result_stores_candidates(qtbot: QtBot, window: MainWindow) -> None:
    wait_for_preview(qtbot, window)
    problem = window.controller.state.problem()
    mirror = window.controller.state.mirror
    pipeline = Pipeline((Stage("tiny", LmfitOptimizer("nelder", max_evaluations=10)),))
    result = pipeline.run(problem)
    window._apply_pipeline_result(problem, result)
    state = window.controller.state
    assert [c.name for c in state.candidates] == ["run 1"]
    assert state.selected == "run 1"
    assert state.mirror == mirror
    assert window.controller.project.journal.entries("run")
    window.controller.undo_stack.setClean()


# Analyze


def test_analyze_selected_candidate_export(
    qtbot: QtBot, window: MainWindow, tmp_path: Path
) -> None:
    wait_for_preview(qtbot, window)
    page = window.analyze_page
    errors: list[str] = []
    page.errorOccurred.connect(errors.append)
    page.design_combo.setCurrentIndex(1)
    assert page.design() is None
    assert not page.export(tmp_path / "none.csv", "csv")
    assert errors
    assert "(none)" in page.design_combo.itemText(1)

    window.store_candidate("mine")
    set_d(window, 4.5)
    window.candidate_list.setCurrentRow(1)
    assert "mine" in page.design_combo.itemText(1)
    candidate = window.controller.state.candidates[0]
    assert page.design() == candidate.mirror
    path = tmp_path / "selected.csv"
    assert page.export(path, "csv")
    with path.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == candidate.mirror.stack().num_layers
    window.controller.undo_stack.setClean()


# Plots


def reflectivity_plot_label(qtbot: QtBot, reflections: int) -> str:
    state = new_state("periodic")
    mirror = state.mirror
    assert mirror is not None
    assert state.targets is not None
    target = replace(state.targets.targets[0], reflections=reflections, points=20)
    state = state.evolve(targets=replace(state.targets, targets=(target,)))
    results = compute_curves(state, mirror)
    plot = ReflectivityPlot()
    qtbot.addWidget(plot)
    plot.set_curves([Curve("c", results[0])], [target])
    return str(plot.plot_items()["energy"].getAxis("left").labelText)


def test_reflectivity_label_reflects_number_of_reflections(qtbot: QtBot) -> None:
    assert "Rⁿ" in reflectivity_plot_label(qtbot, 2)
    label = reflectivity_plot_label(qtbot, 1)
    assert "R" in label
    assert "Rⁿ" not in label


def test_every_curve_has_its_own_color(qtbot: QtBot) -> None:
    state = new_state("periodic")
    assert state.mirror is not None
    result = compute_curves(state, state.mirror)[0]
    plot = ReflectivityPlot()
    qtbot.addWidget(plot)
    plot.set_curves(
        [Curve("a", result), Curve("b", result, dashed=True), Curve("c", result)],
        state.targets.targets if state.targets else (),
    )
    items = plot.plot_items()["energy"].listDataItems()
    colors = [item.opts["pen"].color().name() for item in items]
    assert len(set(colors)) == 3


def test_convergence_plot_fills_both_panels(qtbot: QtBot) -> None:
    plot = ConvergencePlot()
    qtbot.addWidget(plot)
    plot.add_point("t", 1, 0.5, 0.4)
    plot.add_point("t", 2, 0.4, float("nan"))
    plot.add_point("t", 3, 0.3, 0.2)
    objective = plot.objective_plot.listDataItems()[0]
    merit = plot.merit_plot.listDataItems()[0]
    assert len(objective.xData) == 3
    np.testing.assert_allclose(merit.xData, [1.0, 3.0])
    np.testing.assert_allclose(merit.yData, [0.4, 0.2])


def test_design_map_select_marks_point(qtbot: QtBot) -> None:
    plot = DesignMapPlot()
    qtbot.addWidget(plot)
    x, y = np.array([1.0, 2.0, 3.0]), np.array([10.0, 20.0])
    values = np.arange(6.0).reshape(3, 2)
    plot.set_scan(["a", "b"], [x, y], values, np.ones((3, 2), dtype=bool))
    plot.select(1, 1)
    marker = plot._marker
    assert marker is not None
    xs, ys = marker.getData()
    assert list(xs) == [2.0]
    assert list(ys) == [20.0]
