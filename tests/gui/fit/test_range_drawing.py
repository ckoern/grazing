"""Drawing and moving fit ranges with the mouse in the reflectivity plot."""

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication
from pytestqt.qtbot import QtBot

from xstack.gui.fit.plots import FitReflectivityPlot

LEFT = Qt.MouseButton.LeftButton
NONE = Qt.KeyboardModifier.NoModifier
SHIFT = Qt.KeyboardModifier.ShiftModifier
CTRL = Qt.KeyboardModifier.ControlModifier


def make_plot(qtbot: QtBot) -> FitReflectivityPlot:
    plot = FitReflectivityPlot()
    qtbot.addWidget(plot)
    plot.resize(800, 600)
    plot.show()
    qtbot.waitExposed(plot)
    plot.top.setLogMode(y=False)
    plot.top.setXRange(0.0, 3.0, padding=0.0)
    plot.top.setYRange(0.0, 1.0, padding=0.0)
    plot._add_region((1.0, 2.0), excluded=False)
    QApplication.processEvents()
    return plot


def widget_pos(plot: FitReflectivityPlot, x: float) -> QPoint:
    scene = plot.view_box.mapViewToScene(QPointF(x, 0.5))
    return plot.graphics.mapFromScene(scene)


def send(
    plot: FitReflectivityPlot, kind: QEvent.Type, pos: QPoint, buttons: object, mods: object
) -> None:
    viewport = plot.graphics.viewport()
    button = LEFT if kind != QEvent.Type.MouseMove else Qt.MouseButton.NoButton
    event = QMouseEvent(
        kind,
        QPointF(pos),
        QPointF(viewport.mapToGlobal(pos)),
        button,
        buttons,
        mods,
    )
    QApplication.sendEvent(viewport, event)
    QApplication.processEvents()


def drag(plot: FitReflectivityPlot, x0: float, x1: float, mods: object) -> None:
    """Hover over the start without modifiers, then drag from x0 to x1 with modifiers."""
    start, end = widget_pos(plot, x0), widget_pos(plot, x1)
    send(plot, QEvent.Type.MouseMove, start, Qt.MouseButton.NoButton, NONE)
    send(plot, QEvent.Type.MouseButtonPress, start, LEFT, mods)
    for i in range(1, 11):
        point = start + (end - start) * (i / 10)
        send(plot, QEvent.Type.MouseMove, point, LEFT, mods)
    send(plot, QEvent.Type.MouseButtonRelease, end, Qt.MouseButton.NoButton, mods)


def region_bounds(plot: FitReflectivityPlot) -> tuple[float, float]:
    region, _ = plot._regions[0]
    lo, hi = region.getRegion()
    return float(lo), float(hi)


def test_ctrl_drag_inside_fit_range_draws_excluded_range(qtbot: QtBot) -> None:
    plot = make_plot(qtbot)
    drawn: list[tuple[float, float, bool]] = []
    plot.rangeDrawn.connect(lambda lo, hi, ex: drawn.append((lo, hi, ex)))
    drag(plot, 1.3, 1.6, CTRL)
    assert len(drawn) == 1
    lo, hi, excluded = drawn[0]
    assert excluded
    assert 1.25 < lo < 1.35
    assert 1.55 < hi < 1.65
    assert region_bounds(plot) == (1.0, 2.0)


def test_shift_drag_inside_fit_range_draws_fit_range(qtbot: QtBot) -> None:
    plot = make_plot(qtbot)
    drawn: list[tuple[float, float, bool]] = []
    plot.rangeDrawn.connect(lambda lo, hi, ex: drawn.append((lo, hi, ex)))
    drag(plot, 1.2, 1.8, SHIFT)
    assert [d[2] for d in drawn] == [False]
    assert region_bounds(plot) == (1.0, 2.0)


def test_ctrl_drag_from_range_edge_draws(qtbot: QtBot) -> None:
    plot = make_plot(qtbot)
    drawn: list[tuple[float, float, bool]] = []
    plot.rangeDrawn.connect(lambda lo, hi, ex: drawn.append((lo, hi, ex)))
    drag(plot, 2.0, 2.4, CTRL)
    assert [d[2] for d in drawn] == [True]
    assert region_bounds(plot) == (1.0, 2.0)


def test_plain_drag_moves_range(qtbot: QtBot) -> None:
    plot = make_plot(qtbot)
    drawn: list[object] = []
    edited: list[object] = []
    plot.rangeDrawn.connect(lambda *args: drawn.append(args))
    plot.rangesEdited.connect(lambda ranges, excluded: edited.append(ranges))
    drag(plot, 1.5, 1.8, NONE)
    assert not drawn
    lo, hi = region_bounds(plot)
    assert 1.25 < lo < 1.35
    assert 2.25 < hi < 2.35
    assert edited
