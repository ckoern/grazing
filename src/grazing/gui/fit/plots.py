"""Plots of the fit application: measured and fitted curves, correlations, convergence."""

import functools
import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget

from grazing.fit.results import FitStatistics
from grazing.gui.common.style import COLORS, PLOT_PALETTE
from grazing.gui.fit.preview import FitPreview, residual_label

__all__ = [
    "ConvergencePlot",
    "CorrelationPlot",
    "FitReflectivityPlot",
    "RangeRegion",
    "RangeViewBox",
]

type Bounds = tuple[float, float]
"""Range of the scanned variable in display units (deg or keV)."""

_FIT_BRUSH = (99, 102, 241, 28)
_FIT_PEN = (99, 102, 241, 110)
_EXCLUDED_BRUSH = (244, 63, 94, 40)
_EXCLUDED_PEN = (244, 63, 94, 140)
_UNFITTED = "#b8bec9"


def _x_label(axis: str) -> str:
    return "grazing angle θ (deg)" if axis == "angle" else "energy (keV)"


class RangeViewBox(pg.ViewBox):
    """View box that draws a new range on Shift-drag (fit range) or Ctrl-drag (excluded).

    Signals:
        rangeDrawn(float, float, bool): Lower and upper bound in data coordinates and whether
            the range is excluded.
    """

    rangeDrawn = Signal(float, float, bool)

    def __init__(self) -> None:
        super().__init__()
        self._drawing: pg.LinearRegionItem | None = None
        self._excluded = False
        self.drawing_enabled = True

    def mouseDragEvent(self, ev: Any, axis: Any = None) -> None:
        """Draw a range with a modifier key held; otherwise pan or zoom as usual."""
        if not (self.drawing_enabled and (_draws_range(ev) or self._drawing is not None)):
            super().mouseDragEvent(ev, axis)
            return
        ev.accept()
        start = self.mapSceneToView(ev.buttonDownScenePos()).x()
        current = self.mapSceneToView(ev.scenePos()).x()
        if ev.isStart():
            modifiers = ev.modifiers()
            self._excluded = bool(modifiers & Qt.KeyboardModifier.ControlModifier) and not bool(
                modifiers & Qt.KeyboardModifier.ShiftModifier
            )
            brush, pen = (
                (_EXCLUDED_BRUSH, _EXCLUDED_PEN) if self._excluded else (_FIT_BRUSH, _FIT_PEN)
            )
            self._drawing = pg.LinearRegionItem(
                (start, current), movable=False, brush=pg.mkBrush(*brush), pen=pg.mkPen(*pen)
            )
            self.addItem(self._drawing)
        if self._drawing is not None:
            self._drawing.setRegion((start, current))
        if ev.isFinish():
            if self._drawing is not None:
                self.removeItem(self._drawing)
            self._drawing = None
            lo, hi = sorted((start, current))
            if hi > lo:
                self.rangeDrawn.emit(lo, hi, self._excluded)


def _draws_range(ev: Any) -> bool:
    """Whether a drag event draws a new range (left button with Shift or Ctrl)."""
    modifiers = ev.modifiers()
    drawing = Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.ControlModifier
    return ev.button() == Qt.MouseButton.LeftButton and bool(modifiers & drawing)


class RangeRegion(pg.LinearRegionItem):
    """Shaded fit or excluded range whose body and edges can be dragged.

    Drags that start with Shift or Ctrl held are handed to the view box, so that a new range can
    be drawn on top of an existing one.

    Args:
        bounds: Lower and upper bound in data coordinates.
        brush: Fill of the range.
        pen: Pen of the edges.
    """

    def __init__(self, bounds: tuple[float, float], brush: Any, pen: Any) -> None:
        super().__init__(bounds, movable=True, brush=brush, pen=pen)
        self._forwarding = False
        for line in self.lines:
            line.mouseDragEvent = functools.partial(self._line_drag, line.mouseDragEvent)

    def _forward(self, ev: Any) -> bool:
        if ev.isStart():
            self._forwarding = _draws_range(ev)
        if not self._forwarding:
            return False
        view = self.getViewBox()
        if view is not None:
            view.mouseDragEvent(ev)
        if ev.isFinish():
            self._forwarding = False
        return True

    def mouseDragEvent(self, ev: Any) -> None:
        """Move the range, or hand a Shift/Ctrl drag to the view box."""
        if not self._forward(ev):
            super().mouseDragEvent(ev)

    def _line_drag(self, original: Any, ev: Any) -> None:
        if not self._forward(ev):
            original(ev)


class FitReflectivityPlot(QWidget):
    """Measured points and fitted curves with residuals and the fit ranges of one dataset.

    Datasets with the scan axis of the selected dataset are shown; points that are not fitted
    are grey. The fit ranges (blue) and excluded ranges (red) of the selected dataset are
    shaded and can be moved by dragging their edges.

    Signals:
        rangeDrawn(float, float, bool): A new range was drawn: bounds in display units (deg or
            keV) and whether it is excluded.
        rangesEdited(object, object): The shaded ranges were moved: lists of fit ranges and of
            excluded ranges, as (lower, upper) in display units.

    Args:
        parent: Parent widget.
    """

    rangeDrawn = Signal(float, float, bool)
    rangesEdited = Signal(object, object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.graphics = pg.GraphicsLayoutWidget()
        layout.addWidget(self.graphics)
        self.view_box = RangeViewBox()
        self.view_box.rangeDrawn.connect(self.rangeDrawn)
        self.top = self.graphics.addPlot(row=0, col=0, viewBox=self.view_box)
        self.bottom = self.graphics.addPlot(row=1, col=0)
        self.graphics.ci.layout.setRowStretchFactor(0, 3)
        self.graphics.ci.layout.setRowStretchFactor(1, 1)
        self.bottom.setXLink(self.top)
        self.top.setLabel("left", "reflectivity")
        self.top.addLegend(offset=(-10, 10))
        for plot in (self.top, self.bottom):
            plot.showGrid(x=True, y=True, alpha=0.15)
            plot.getAxis("left").enableAutoSIPrefix(False)
            plot.getAxis("bottom").enableAutoSIPrefix(False)
        self.log_scale = True
        self.top.setLogMode(y=True)
        self._regions: list[tuple[pg.LinearRegionItem, bool]] = []
        self._axis: str | None = None

    def set_log_scale(self, log: bool) -> None:
        """Switch between logarithmic and linear reflectivity axis.

        Args:
            log: Use a logarithmic axis.
        """
        self.log_scale = log
        self.top.setLogMode(y=log)

    def show_preview(
        self,
        preview: FitPreview | None,
        selected: str | None,
        ranges: Sequence[Bounds] = (),
        excluded: Sequence[Bounds] = (),
        start: FitPreview | None = None,
    ) -> None:
        """Draw a preview.

        Args:
            preview: The preview, or ``None`` to clear the plot.
            selected: Name of the selected dataset, whose ranges are shaded and whose scan axis
                selects the datasets shown; ``None`` for the first dataset.
            ranges: Fit ranges of the selected dataset in display units.
            excluded: Excluded ranges of the selected dataset in display units.
            start: Preview at the start values, drawn dashed, or ``None``.
        """
        self.top.clear()
        self.bottom.clear()
        if self.top.legend is not None:
            self.top.legend.clear()
        self._regions = []
        if preview is None or not preview.curves:
            return
        names = [c.name for c in preview.curves]
        current = preview.curves[names.index(selected)] if selected in names else preview.curves[0]
        self._axis = current.axis
        self.top.setLabel("bottom", "")
        self.bottom.setLabel("bottom", _x_label(current.axis))
        self.bottom.setLabel("left", residual_label(preview.weighting))
        starts = {c.name: c for c in start.curves} if start is not None else {}
        for index, curve in enumerate(preview.curves):
            if curve.axis != current.axis:
                continue
            color = PLOT_PALETTE[index % len(PLOT_PALETTE)]
            positive = curve.measured > 0 if self.log_scale else np.ones_like(curve.fitted)
            fitted, other = curve.fitted & positive, ~curve.fitted & positive
            self.top.plot(
                curve.x[other],
                curve.measured[other],
                pen=None,
                symbol="o",
                symbolSize=3,
                symbolBrush=_UNFITTED,
                symbolPen=None,
            )
            self.top.plot(
                curve.x[fitted],
                curve.measured[fitted],
                pen=None,
                symbol="o",
                symbolSize=3,
                symbolBrush=color,
                symbolPen=None,
                name=f"{curve.name} measured",
            )
            if curve.model is not None:
                self.top.plot(
                    curve.x, curve.model, pen=pg.mkPen(color, width=1.5), name=f"{curve.name} fit"
                )
            first = starts.get(curve.name)
            if first is not None and first.model is not None:
                self.top.plot(
                    first.x,
                    first.model,
                    pen=pg.mkPen(color, width=1, style=Qt.PenStyle.DashLine),
                    name=f"{curve.name} start",
                )
            if curve.residuals is not None:
                shown = curve.fitted & np.isfinite(curve.residuals)
                self.bottom.plot(
                    curve.x[shown],
                    curve.residuals[shown],
                    pen=None,
                    symbol="o",
                    symbolSize=3,
                    symbolBrush=color,
                    symbolPen=None,
                )
        self.bottom.addLine(y=0.0, pen=pg.mkPen(COLORS["text_muted"]))
        for bounds in ranges:
            self._add_region(bounds, excluded=False)
        for bounds in excluded:
            self._add_region(bounds, excluded=True)

    def _add_region(self, bounds: Bounds, excluded: bool) -> None:
        brush, pen = (_EXCLUDED_BRUSH, _EXCLUDED_PEN) if excluded else (_FIT_BRUSH, _FIT_PEN)
        region = RangeRegion(bounds, pg.mkBrush(*brush), pg.mkPen(*pen))
        region.setZValue(-10)
        region.sigRegionChangeFinished.connect(self._regions_moved)
        self.top.addItem(region)
        self._regions.append((region, excluded))

    def _regions_moved(self) -> None:
        ranges = [tuple(sorted(r.getRegion())) for r, excluded in self._regions if not excluded]
        excluded = [tuple(sorted(r.getRegion())) for r, excluded in self._regions if excluded]
        self.rangesEdited.emit(ranges, excluded)


class CorrelationPlot(QWidget):
    """Correlation matrix of the free parameters of a fit as a heat map.

    Args:
        parent: Parent widget.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.graphics = pg.GraphicsLayoutWidget()
        layout.addWidget(self.graphics)
        self.statistics: FitStatistics | None = None

    def set_statistics(self, statistics: FitStatistics | None) -> None:
        """Show the correlations of a fit.

        Args:
            statistics: Fit statistics, or ``None`` to clear the plot.
        """
        self.statistics = statistics
        self.graphics.clear()
        if statistics is None or not statistics.names:
            return
        n = len(statistics.names)
        matrix = np.nan_to_num(np.array(statistics.correlation, dtype=float))
        plot = self.graphics.addPlot()
        colormap = pg.colormap.get("CET-D1")
        image = pg.ImageItem(matrix.T)
        image.setColorMap(colormap)
        image.setLevels((-1.0, 1.0))
        plot.addItem(image)
        plot.invertY(True)
        plot.setAspectLocked(True)
        plot.getAxis("bottom").setTicks([[(i + 0.5, str(i + 1)) for i in range(n)]])
        plot.getAxis("left").setTicks(
            [[(i + 0.5, f"{name}  {i + 1}") for i, name in enumerate(statistics.names)]]
        )
        for i in range(n):
            for j in range(i):
                value = statistics.correlation[i][j]
                if math.isfinite(value) and abs(value) >= 0.5:
                    text = pg.TextItem(f"{value:+.2f}", anchor=(0.5, 0.5), color="#1d2433")
                    text.setPos(j + 0.5, i + 0.5)
                    plot.addItem(text)
        bar = pg.ColorBarItem(values=(-1.0, 1.0), colorMap=colormap, interactive=False)
        bar.setImageItem(image, insert_in=plot)


class ConvergencePlot(QWidget):
    """Best objective of a running or finished fit versus evaluations, one color per stage.

    Args:
        parent: Parent widget.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.plot = pg.PlotWidget()
        self.plot.setLogMode(y=True)
        self.plot.setLabel("left", "best objective")
        self.plot.setLabel("bottom", "evaluations")
        self.plot.addLegend(offset=(-10, 10))
        layout.addWidget(self.plot)
        self._data: dict[str, tuple[list[float], list[float]]] = {}
        self._items: dict[str, pg.PlotDataItem] = {}
        self._offset = 0
        self._last: tuple[str, int] | None = None

    def clear(self) -> None:
        """Remove all curves."""
        self.plot.clear()
        if self.plot.plotItem.legend is not None:
            self.plot.plotItem.legend.clear()
        self._data = {}
        self._items = {}
        self._offset = 0
        self._last = None

    def add_point(self, task: str, evaluations: int, best_objective: float) -> None:
        """Append a progress point.

        Stages run one after another; evaluations of later stages continue the axis.

        Args:
            task: Name of the stage.
            evaluations: Evaluations of the stage so far.
            best_objective: Best objective of the stage so far.
        """
        if not (math.isfinite(best_objective) and best_objective > 0):
            return
        if self._last is not None and self._last[0] != task:
            self._offset += self._last[1]
        self._last = (task, evaluations)
        key = task
        while key in self._data and key != task:
            key += "'"
        xs, ys = self._data.setdefault(key, ([], []))
        xs.append(self._offset + evaluations)
        ys.append(best_objective)
        item = self._items.get(key)
        if item is None:
            color = PLOT_PALETTE[len(self._items) % len(PLOT_PALETTE)]
            item = self.plot.plot(xs, ys, pen=pg.mkPen(color, width=1.5), name=key)
            self._items[key] = item
        else:
            item.setData(xs, ys)
