"""Tests of the plot-range fields of the main window and their effect on the tolerance view."""

import numpy as np
import pyqtgraph as pg
import pytest
from pytestqt.qtbot import QtBot

from xstack.gui.main_window import MainWindow
from xstack.manufacturing import ErrorModel, tolerance_analysis

pytestmark = pytest.mark.gui


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


def test_zero_minimum_is_accepted(window: MainWindow) -> None:
    set_range(window, "energy", "0", "12")
    assert window.display_ranges["energy"] == (0.0, 12e3)
    window.controller.undo_stack.setClean()


INVALID_RANGES = [("-1", "12"), ("12", "8"), ("8", "8")]


@pytest.mark.parametrize(("low", "high"), INVALID_RANGES)
def test_invalid_range_shows_error(window: MainWindow, low: str, high: str) -> None:
    set_range(window, "energy", low, high)
    assert "Invalid plot range" in window.statusBar().currentMessage()
    window.controller.undo_stack.setClean()


@pytest.mark.parametrize(("low", "high"), INVALID_RANGES)
def test_invalid_range_keeps_previous_ranges(window: MainWindow, low: str, high: str) -> None:
    set_range(window, "energy", "0", "12")
    valid = dict(window.display_ranges)
    set_range(window, "energy", low, high)
    assert window.display_ranges == valid
    window.controller.undo_stack.setClean()


def test_range_redraws_tolerance_curves(qtbot: QtBot, window: MainWindow) -> None:
    problem = window.controller.state.problem().evolve(points=20)
    result = tolerance_analysis(
        problem, errors=ErrorModel(thickness_random=0.02), samples=3, seed=0
    )
    window._show_tolerance(result)
    x = nominal_energy_x(window)
    assert x is not None
    assert x[-1] == pytest.approx(8.2)

    set_range(window, "energy", "0", "12")
    qtbot.waitUntil(
        lambda: (x := nominal_energy_x(window)) is not None and x[-1] == pytest.approx(12.0),
        timeout=10000,
    )

    set_range(window, "energy", "", "")
    x = nominal_energy_x(window)
    assert x is not None
    assert x[-1] == pytest.approx(8.2)
    window.controller.undo_stack.setClean()
