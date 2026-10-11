"""Tests for `xstack.gui.widgets.tolerance_view`."""

import numpy as np
import pyqtgraph as pg
import pytest
from pytestqt.qtbot import QtBot

from xstack.gui.preview import tolerance_curves
from xstack.gui.widgets.tolerance_view import ToleranceView
from xstack.manufacturing import ErrorModel, ToleranceResult, tolerance_analysis
from xstack.project.templates import new_state
from xstack.targets import AngleScan, EnergyScan, TargetSet

pytestmark = pytest.mark.gui


def make_result(targets: TargetSet | None = None, samples: int = 5) -> ToleranceResult:
    problem = new_state("periodic").problem().evolve(points=30)
    if targets is not None:
        problem = problem.evolve(targets=targets)
    return tolerance_analysis(
        problem, errors=ErrorModel(thickness_random=0.02), samples=samples, seed=1
    )


def lines(plot: pg.PlotItem, width: float) -> list[pg.PlotDataItem]:
    return [
        i
        for i in plot.items
        if isinstance(i, pg.PlotDataItem)
        and i.opts["pen"] is not None
        and i.opts["pen"].widthF() == width
    ]


@pytest.fixture
def view(qtbot: QtBot) -> ToleranceView:
    v = ToleranceView()
    qtbot.addWidget(v)
    return v


def test_empty(view: ToleranceView) -> None:
    assert view.histogram_plot() is None
    assert view.curve_plots() == {}


def test_histogram_and_curves(view: ToleranceView) -> None:
    view.set_result(make_result())
    hist = view.histogram_plot()
    assert hist is not None
    assert any(
        isinstance(i, pg.PlotDataItem) and i.xData is not None and len(i.xData) > 0
        for i in hist.items
    )
    assert len([i for i in hist.items if isinstance(i, pg.InfiniteLine)]) == 3
    plots = view.curve_plots()
    assert list(plots) == ["energy"]
    assert len(lines(plots["energy"], 1)) == 5
    assert len(lines(plots["energy"], 2.5)) == 1
    assert any(isinstance(i, pg.FillBetweenItem) for i in plots["energy"].items)
    assert plots["energy"].getAxis("left").labelText == "Reflectivity R"


def test_max_curves(qtbot: QtBot) -> None:
    v = ToleranceView(max_curves=3)
    qtbot.addWidget(v)
    v.set_result(make_result())
    assert len(lines(v.curve_plots()["energy"], 1)) == 3


def test_clear_and_refresh(view: ToleranceView) -> None:
    view.set_result(make_result())
    view.set_result(None)
    assert view.histogram_plot() is None
    assert view.curve_plots() == {}
    view.set_result(make_result())
    old = view.curve_plots()["energy"]
    view.refresh()
    new = view.curve_plots()["energy"]
    assert new is not old
    assert len(lines(new, 1)) == 5


def test_angle_target(view: ToleranceView) -> None:
    base = new_state("periodic").problem().targets.targets[0]
    assert isinstance(base, EnergyScan)
    target = AngleScan(energy=8000.0, angle_min=0.01, angle_max=0.04, points=30)
    view.set_result(make_result(TargetSet(target)))
    assert list(view.curve_plots()) == ["angle"]
    assert view.curve_plots()["angle"].getAxis("bottom").labelText == "Grazing angle (deg)"


def test_reflections_label(view: ToleranceView) -> None:
    base = new_state("periodic").problem().targets.targets[0]
    view.set_result(make_result(TargetSet(base.__class__(**{**vars(base), "reflections": 2}))))
    assert view.curve_plots()["energy"].getAxis("left").labelText == "Reflectivity Rⁿ"


def test_wider_curves_span_display_range_and_mark_target(view: ToleranceView) -> None:
    state = new_state("periodic")
    result = make_result()
    curves = tolerance_curves(state, result, {"energy": (6000.0, 10000.0)})
    view.set_result(result, curves)
    energy = view.curve_plots()["energy"]
    nominal = lines(energy, 2.5)
    assert len(nominal) == 1
    x = np.asarray(nominal[0].xData)
    assert x[0] == pytest.approx(6.0)
    assert x[-1] == pytest.approx(10.0)
    regions = [i for i in energy.items if isinstance(i, pg.LinearRegionItem)]
    assert len(regions) == 1
    lo, hi = regions[0].getRegion()
    assert lo == pytest.approx(7.9)
    assert hi == pytest.approx(8.2)
