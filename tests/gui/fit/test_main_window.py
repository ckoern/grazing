"""Tests for `grazing.gui.fit.main_window`."""

import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
from PySide6.QtCore import QSettings, Qt
from pytestqt.qtbot import QtBot

from grazing.core.measurement import MeasuredCurve
from grazing.core.optim.optimizers import LmfitOptimizer, Optimizer, Tracker
from grazing.core.units import deg
from grazing.fit.dataset import Dataset
from grazing.fit.problem import FitProblem
from grazing.fit.project import (
    FitProject,
    FitState,
    rename_dataset,
    set_optimizers,
    set_weighting,
)
from grazing.fit.templates import new_state
from grazing.gui.fit.main_window import WORKFLOW_STEPS, MainWindow
from grazing.gui.fit.plots import RangeViewBox
from tests.gui.fit.helpers import ANGLES, make_state

pytestmark = pytest.mark.gui

MakeWindow = Callable[[], MainWindow]


@pytest.fixture
def make_fit_window(qtbot: QtBot, settings: QSettings) -> MakeWindow:
    """Factory of fit main windows sharing the test settings."""

    def make() -> MainWindow:
        w = MainWindow(settings=settings, confirm_close=False)
        qtbot.addWidget(w)
        return w

    return make


@pytest.fixture
def fit_window(make_fit_window: MakeWindow) -> MainWindow:
    return make_fit_window()


def load(window: MainWindow, state: FitState) -> None:
    window.controller.replace_project(FitProject(state))


def wait_preview(qtbot: QtBot, window: MainWindow) -> None:
    qtbot.waitUntil(
        lambda: (
            window.preview is not None
            and not window.preview_runner.is_busy()
            and bool(window.preview.curves)
        ),
        timeout=10000,
    )


def periodic_state() -> FitState:
    """Periodic template with a simulated curve of the unfitted structure."""
    state = new_state("periodic")
    blank = Dataset.create("x", MeasuredCurve("angle", ANGLES, np.ones(ANGLES.size), energy=8048.0))
    assert state.mirror is not None
    sim = FitProblem(state.mirror, (blank,), weighting="linear").simulate(blank)
    curve = MeasuredCurve("angle", ANGLES, sim, 0.05 * sim + 1e-7, energy=8048.0)
    return state.evolve(datasets=(Dataset.create("a", curve),))


def perturbed(state: FitState) -> FitState:
    """Perturb the data so that the start values are not an exact fit (objective > 0)."""
    datasets = []
    for d in state.datasets:
        x, r, u = d.curve.arrays()
        factor = 1.0 + 0.2 * np.sin(np.arange(r.size))
        curve = MeasuredCurve("angle", x, r * factor, u, energy=8048.0)
        datasets.append(replace(d, curve=curve))
    return state.evolve(datasets=tuple(datasets))


def run_fit_and_wait(qtbot: QtBot, window: MainWindow, count: int = 1) -> None:
    window.fit_page.run_local()
    qtbot.waitUntil(
        lambda: not window.job_runner.running and len(window.controller.state.results) == count,
        timeout=20000,
    )


def test_opens_with_periodic_template(fit_window: MainWindow) -> None:
    assert "grazing-fit" in fit_window.windowTitle()
    state = fit_window.controller.state
    assert state.mirror is not None
    assert type(state.mirror.coating).__name__ == "Periodic"
    assert fit_window.step_tabs.count() == len(WORKFLOW_STEPS) == 6
    assert [fit_window.step_tabs.tabText(i).split(" ", 1)[1] for i in range(6)] == list(
        WORKFLOW_STEPS
    )
    assert fit_window.plot_tabs.count() == 4
    assert [fit_window.plot_tabs.tabText(i) for i in range(4)] == [
        "Reflectivity",
        "Correlations",
        "Layer profile",
        "Convergence",
    ]
    assert fit_window.dataset_list.count() == 0
    assert fit_window.selected_dataset is None


def test_navigation(fit_window: MainWindow) -> None:
    assert fit_window.step_tabs.currentIndex() == 0
    fit_window.next_step()
    assert fit_window.step_tabs.currentIndex() == 1
    fit_window.previous_step()
    assert fit_window.step_tabs.currentIndex() == 0
    fit_window.previous_step()
    assert fit_window.step_tabs.currentIndex() == 0
    for _ in range(10):
        fit_window.next_step()
    assert fit_window.step_tabs.currentIndex() == len(WORKFLOW_STEPS) - 1


def test_dataset_list_mirrors_state(fit_window: MainWindow) -> None:
    load(fit_window, make_state(("a", "b")))
    lst = fit_window.dataset_list
    assert lst.count() == 2
    assert lst.item(0).text().startswith("a\n")
    assert "40/40 points" in lst.item(0).text()
    assert lst.item(1).text().startswith("b\n")
    assert lst.item(0).checkState() == Qt.CheckState.Checked
    assert fit_window.selected_dataset == "a"
    assert fit_window.remove_data_button.isEnabled()


def test_uncheck_disables_dataset_and_undo(fit_window: MainWindow) -> None:
    load(fit_window, make_state(("a", "b")))
    fit_window.dataset_list.item(1).setCheckState(Qt.CheckState.Unchecked)
    state = fit_window.controller.state
    assert not state.dataset("b").enabled
    assert state.dataset("a").enabled
    assert fit_window.controller.undo_stack.count() == 1
    assert fit_window.dataset_list.item(1).checkState() == Qt.CheckState.Unchecked
    fit_window.undo_action.trigger()
    assert fit_window.controller.state.dataset("b").enabled
    assert fit_window.dataset_list.item(1).checkState() == Qt.CheckState.Checked


def test_selection_changes_selected_dataset(fit_window: MainWindow) -> None:
    load(fit_window, make_state(("a", "b")))
    fit_window.dataset_list.setCurrentRow(1)
    assert fit_window.selected_dataset == "b"
    assert fit_window.dataset_list.currentRow() == 1


def test_remove_selected_dataset_keeps_position(fit_window: MainWindow) -> None:
    load(fit_window, make_state(("a", "b", "c")))
    fit_window.dataset_list.setCurrentRow(1)
    fit_window.remove_data_button.click()
    assert fit_window.selected_dataset == "c"


def test_remove_selected_dataset(fit_window: MainWindow) -> None:
    load(fit_window, make_state(("a", "b", "c")))
    fit_window.dataset_list.setCurrentRow(1)
    fit_window.remove_data_button.click()
    names = [d.name for d in fit_window.controller.state.datasets]
    assert names == ["a", "c"]
    assert fit_window.selected_dataset in names
    assert fit_window.dataset_list.count() == 2
    fit_window.remove_data_button.click()
    fit_window.remove_data_button.click()
    assert fit_window.controller.state.datasets == ()
    assert getattr(fit_window, "selected_dataset") is None  # noqa: B009
    assert not fit_window.remove_data_button.isEnabled()


def test_rename_selected_dataset_keeps_selection(fit_window: MainWindow) -> None:
    load(fit_window, make_state(("a", "b")))
    fit_window.dataset_list.setCurrentRow(1)
    fit_window.controller.execute(rename_dataset("b", "renamed"))
    assert fit_window.selected_dataset == "renamed"
    assert fit_window.dataset_list.currentRow() == 1
    assert fit_window.dataset_list.item(1).text().startswith("renamed\n")


def test_preview_summary(qtbot: QtBot, fit_window: MainWindow) -> None:
    load(fit_window, make_state(("a",)))
    wait_preview(qtbot, fit_window)
    assert fit_window.preview is not None
    assert len(fit_window.preview.curves) == 1
    assert fit_window.preview.curves[0].model is not None
    assert "points fitted" in fit_window.summary_label.text()
    assert "40 of 40" in fit_window.summary_label.text()


def test_preview_chi2_label(qtbot: QtBot, fit_window: MainWindow) -> None:
    load(fit_window, make_state(("a",)))
    fit_window.controller.execute(set_weighting("chi2"))
    qtbot.waitUntil(lambda: fit_window.summary_label.text().startswith("χ²_red"), timeout=10000)


def test_preview_warning_without_structure(qtbot: QtBot, fit_window: MainWindow) -> None:
    load(fit_window, new_state("empty"))
    qtbot.waitUntil(lambda: "⚠" in fit_window.summary_label.text(), timeout=10000)
    assert "no structure" in fit_window.summary_label.text()


def drawn_state(qtbot: QtBot, window: MainWindow) -> None:
    load(window, make_state(("a",)))
    wait_preview(qtbot, window)


def test_range_drawn_adds_fit_range(qtbot: QtBot, fit_window: MainWindow) -> None:
    drawn_state(qtbot, fit_window)
    fit_window.reflectivity_plot.rangeDrawn.emit(0.2, 0.5, False)
    dataset = fit_window.controller.state.dataset("a")
    assert len(dataset.ranges) == 1
    assert dataset.ranges[0] == pytest.approx((deg(0.2), deg(0.5)))
    assert dataset.excluded == ()
    fit_window.undo_action.trigger()
    assert fit_window.controller.state.dataset("a").ranges == ()


def test_range_drawn_excludes(qtbot: QtBot, fit_window: MainWindow) -> None:
    drawn_state(qtbot, fit_window)
    fit_window.reflectivity_plot.rangeDrawn.emit(0.5, 0.2, True)
    dataset = fit_window.controller.state.dataset("a")
    assert dataset.ranges == ()
    assert dataset.excluded[0] == pytest.approx((deg(0.2), deg(0.5)))


def test_ranges_edited_replaces_both(qtbot: QtBot, fit_window: MainWindow) -> None:
    drawn_state(qtbot, fit_window)
    fit_window.reflectivity_plot.rangeDrawn.emit(0.2, 0.5, False)
    fit_window.reflectivity_plot.rangesEdited.emit([(0.3, 0.6), (0.7, 0.9)], [(0.4, 0.45)])
    dataset = fit_window.controller.state.dataset("a")
    assert len(dataset.ranges) == 2
    assert dataset.ranges[0] == pytest.approx((deg(0.3), deg(0.6)))
    assert dataset.ranges[1] == pytest.approx((deg(0.7), deg(0.9)))
    assert dataset.excluded[0] == pytest.approx((deg(0.4), deg(0.45)))


def test_range_viewbox_signal(qtbot: QtBot) -> None:
    box = RangeViewBox()
    with qtbot.waitSignal(box.rangeDrawn) as blocker:
        box.rangeDrawn.emit(1.0, 2.0, True)
    assert blocker.args == [1.0, 2.0, True]
    assert box.drawing_enabled


def test_run_fit_saves_result(qtbot: QtBot, fit_window: MainWindow) -> None:
    load(fit_window, perturbed(make_state(("a",))))
    wait_preview(qtbot, fit_window)
    run_fit_and_wait(qtbot, fit_window)
    stored = fit_window.controller.state.results[0]
    assert stored.name == "fit 1"
    assert fit_window.results_page.fit_list.count() == 1
    assert fit_window.results_page.fit_list.item(0).text().startswith("fit 1")
    assert fit_window.results_page.selected_fit() is not None
    assert fit_window.correlation_plot.statistics is stored.result.statistics
    assert fit_window.convergence_plot._data
    assert fit_window.run_action.isEnabled()
    assert not fit_window.stop_action.isEnabled()
    assert not fit_window.progress_bar.isVisible()
    fit_window.undo_action.trigger()
    assert fit_window.controller.state.results == ()
    assert fit_window.results_page.fit_list.count() == 0
    assert fit_window.correlation_plot.statistics is None


@dataclass(frozen=True)
class UntilCancelled(Optimizer):
    """Evaluates the start values repeatedly until the run is cancelled."""

    type_name: ClassVar[str] = "until_cancelled"

    def run(self, problem, start=None, events=None, cancel=None):
        params = start if start is not None else problem.parameters
        tracker = Tracker(problem, "until cancelled", events, cancel, None)
        while not tracker.cancelled:
            tracker.record(problem.evaluate(params.value_dict()))
            time.sleep(0.005)
        return tracker.result(params, "stopped")


def test_stop_saves_cancelled_result(qtbot: QtBot, fit_window: MainWindow) -> None:
    project = FitProject(perturbed(periodic_state()))
    project.execute(set_optimizers((UntilCancelled(),)))
    fit_window.controller.replace_project(project)
    wait_preview(qtbot, fit_window)
    fit_window.fit_page.run()
    qtbot.waitUntil(lambda: fit_window.job_runner.running, timeout=5000)
    assert fit_window.stop_action.isEnabled()
    assert not fit_window.run_action.isEnabled()
    fit_window.stop_action.trigger()
    qtbot.waitUntil(
        lambda: not fit_window.job_runner.running and len(fit_window.controller.state.results) == 1,
        timeout=20000,
    )
    result = fit_window.controller.state.results[0].result
    assert result.cancelled
    assert not fit_window.stop_action.isEnabled()
    assert fit_window.run_action.isEnabled()


def test_save_and_open_restores_fits(
    qtbot: QtBot, make_fit_window: MakeWindow, tmp_path: Path, settings: QSettings
) -> None:
    window = make_fit_window()
    load(window, make_state(("a", "b")))
    wait_preview(qtbot, window)
    run_fit_and_wait(qtbot, window)
    path = tmp_path / "p.gfit"
    assert window.save_project_as(path)
    assert not window.isWindowModified()
    assert "p.gfit" in window.windowTitle()
    assert [a.text() for a in window.recent_menu.actions()] == [str(path)]

    other = make_fit_window()
    assert other.open_project(path)
    state = other.controller.state
    assert [d.name for d in state.datasets] == ["a", "b"]
    assert [f.name for f in state.results] == ["fit 1"]
    assert other.dataset_list.count() == 2
    assert other.results_page.fit_list.count() == 1
    assert str(path) in settings.value("recent_files")


def test_open_missing_file_fails(fit_window: MainWindow, tmp_path: Path) -> None:
    assert not fit_window.open_project(tmp_path / "missing.gfit")


def test_new_project_replaces_state(qtbot: QtBot, fit_window: MainWindow) -> None:
    load(fit_window, make_state(("a",)))
    fit_window.new_project("single_layer")
    state = fit_window.controller.state
    assert state.datasets == ()
    assert state.mirror is not None
    assert type(state.mirror.coating).__name__ == "SingleLayer"
    assert fit_window.dataset_list.count() == 0
    assert "unsaved" in fit_window.windowTitle()
    assert "Single-layer" in fit_window.name_edit.text()


def test_set_theme(fit_window: MainWindow, settings: QSettings) -> None:
    fit_window.set_theme("dark")
    assert settings.value("theme") == "dark"
    assert fit_window.theme_actions["dark"].isChecked()
    fit_window.set_theme("light")
    assert fit_window.theme_actions["light"].isChecked()


def test_close_with_running_job(qtbot: QtBot, fit_window: MainWindow) -> None:
    project = FitProject(perturbed(periodic_state()))
    project.execute(set_optimizers((LmfitOptimizer("nelder", max_evaluations=100000),)))
    fit_window.controller.replace_project(project)
    fit_window.fit_page.run()
    qtbot.waitUntil(lambda: fit_window.job_runner.running, timeout=5000)
    assert fit_window.close()
    qtbot.waitUntil(lambda: not fit_window.job_runner.running, timeout=20000)
