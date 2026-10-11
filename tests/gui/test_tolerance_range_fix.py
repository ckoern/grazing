"""Tests that tolerance curves follow the plot range of the main window and the Analyze page."""

import itertools

import numpy as np
import pyqtgraph as pg
import pytest
from pytestqt.qtbot import QtBot

from xstack.gui.main_window import MainWindow
from xstack.gui.preview import ToleranceCurves, tolerance_curves
from xstack.gui.steps import AnalyzePage
from xstack.gui.widgets.tolerance_view import sample_alpha
from xstack.gui.workers import Job
from xstack.manufacturing import ErrorModel, ToleranceResult, tolerance_analysis
from xstack.optim.events import CancellationToken
from xstack.project.templates import new_state

pytestmark = pytest.mark.gui


def tiny_result(window: MainWindow, seed: int = 0) -> ToleranceResult:
    problem = window.controller.state.problem().evolve(points=20)
    return tolerance_analysis(
        problem, errors=ErrorModel(thickness_random=0.02), samples=3, seed=seed
    )


def set_range(window: MainWindow, axis: str, low: str, high: str) -> None:
    low_edit, high_edit = window.range_edits[axis]
    low_edit.setText(low)
    high_edit.setText(high)
    high_edit.editingFinished.emit()


def nominal_energy_x(window: MainWindow) -> np.ndarray | None:
    """Return the x values in keV of the nominal energy curve, or None if not drawn."""
    plots = window.tolerance_view.curve_plots()
    if "energy" not in plots:
        return None
    for item in plots["energy"].items:
        if (
            isinstance(item, pg.PlotDataItem)
            and item.opts["pen"] is not None
            and item.opts["pen"].widthF() == 2.5
        ):
            return np.asarray(item.xData)
    return None


def test_sample_alpha_bounds() -> None:
    assert sample_alpha(1) == 180
    assert sample_alpha(5) == round(340 / np.sqrt(5))
    assert sample_alpha(10000) == 14
    assert sample_alpha(0) == 180


def test_sample_alpha_is_non_increasing() -> None:
    values = [sample_alpha(n) for n in range(1, 501)]
    assert all(a >= b for a, b in itertools.pairwise(values))
    assert all(14 <= v <= 180 for v in values)


def test_analyze_job_returns_curves_on_display_range(qtbot: QtBot) -> None:
    page = AnalyzePage()
    qtbot.addWidget(page)
    page.set_state(new_state("periodic"))
    page.samples.setValue(2)
    page.seed.setText("0")
    page.display_ranges = {"energy": (0.0, 12e3)}
    jobs: list[Job] = []
    page.jobRequested.connect(jobs.append)
    emitted: list[tuple[object, ...]] = []
    page.toleranceFinished.connect(lambda *args: emitted.append(args))
    page.run()
    result, curves, ranges = jobs[0].function(lambda _: None, CancellationToken())
    assert isinstance(curves, ToleranceCurves)
    assert curves.nominal[0].x[-1] == pytest.approx(12e3)
    assert ranges == {"energy": (0.0, 12e3)}
    jobs[0].on_finished((result, curves, ranges))
    assert emitted[0][2] == {"energy": (0.0, 12e3)}


def test_analyze_job_without_display_range_has_no_curves(qtbot: QtBot) -> None:
    page = AnalyzePage()
    qtbot.addWidget(page)
    page.set_state(new_state("periodic"))
    page.samples.setValue(2)
    page.seed.setText("0")
    page.display_ranges = {}
    jobs: list[Job] = []
    page.jobRequested.connect(jobs.append)
    page.run()
    _result, curves, ranges = jobs[0].function(lambda _: None, CancellationToken())
    assert curves is None
    assert ranges == {}


def test_plot_range_is_passed_to_analyze_page(window: MainWindow) -> None:
    set_range(window, "energy", "0", "12")
    assert window.analyze_page.display_ranges == {"energy": (0.0, 12e3)}
    window.controller.undo_stack.setClean()


def test_show_tolerance_with_current_range_draws_curves_immediately(
    window: MainWindow,
) -> None:
    set_range(window, "energy", "0", "12")
    result = tiny_result(window)
    curves = tolerance_curves(window.controller.state, result, window.display_ranges)
    window._show_tolerance(result, curves, window.display_ranges)
    x = nominal_energy_x(window)
    assert x is not None
    assert np.max(x) == pytest.approx(12.0)
    window.controller.undo_stack.setClean()


def test_stale_curves_of_other_result_are_ignored(window: MainWindow) -> None:
    set_range(window, "energy", "0", "12")
    old = tiny_result(window, seed=1)
    current = tiny_result(window, seed=2)
    old_curves = tolerance_curves(window.controller.state, old, window.display_ranges)
    window._show_tolerance(current)
    window._on_tolerance_curves((old, window.display_ranges, old_curves))
    assert window.tolerance_view._result is current
    window.controller.undo_stack.setClean()


def test_curves_of_other_range_are_ignored(window: MainWindow) -> None:
    set_range(window, "energy", "0", "12")
    result = tiny_result(window)
    other_range = {"energy": (0.0, 8e3)}
    other_curves = tolerance_curves(window.controller.state, result, other_range)
    window._show_tolerance(result)
    window._on_tolerance_curves((result, other_range, other_curves))
    assert window.tolerance_view._result is result
    assert window.tolerance_view._display is not other_curves
    window.controller.undo_stack.setClean()
