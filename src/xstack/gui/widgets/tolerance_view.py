"""View of the result of a Monte Carlo tolerance analysis."""

import numpy as np
import pyqtgraph as pg
from numpy.typing import NDArray
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QVBoxLayout, QWidget

from xstack.gui.plots import scan_axis
from xstack.gui.preview import ToleranceCurves
from xstack.gui.style import COLORS, PLOT_PALETTE
from xstack.manufacturing.tolerance import ToleranceResult

__all__ = ["ToleranceView"]

_SAMPLE_ALPHA = 32
_BAND_ALPHA = 55
_FILL_ALPHA = 110
_PERCENTILES = (10.0, 90.0)


def _color(value: str, alpha: int = 255) -> QColor:
    """Return a color with the given alpha (0 to 255)."""
    color = pg.mkColor(value)
    color.setAlpha(alpha)
    return color


class ToleranceView(QWidget):
    """Distribution of the figure of merit and the reflectivity of Monte Carlo samples.

    The upper panel is a histogram of the figure of merit of the samples with the nominal value
    and the 10 % and 90 % percentiles. Below it, one panel per scan axis overlays the
    reflectivity curves of the samples, the nominal curve and the 10 % to 90 % band across the
    samples; the target range is shaded. The curves may cover a wider range than the targets
    (see `set_result`). Colors are read from the current style on every redraw.

    Args:
        max_curves: Maximum number of sample curves drawn per target.
        parent: Parent widget.
    """

    def __init__(self, max_curves: int = 200, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.max_curves = max_curves
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.graphics = pg.GraphicsLayoutWidget()
        layout.addWidget(self.graphics)
        self._result: ToleranceResult | None = None
        self._display: ToleranceCurves | None = None
        self._histogram: pg.PlotItem | None = None
        self._curves: dict[str, pg.PlotItem] = {}

    def set_result(
        self, result: ToleranceResult | None, curves: ToleranceCurves | None = None
    ) -> None:
        """Show a tolerance result.

        Args:
            result: The result, or ``None`` to clear the view.
            curves: Reflectivity curves to draw, e.g. on a wider display range. Defaults to the
                curves of the analysis on the target grids.
        """
        self._result = result
        self._display = curves
        self.refresh()

    def refresh(self) -> None:
        """Redraw the current result with the current style colors."""
        self.graphics.clear()
        self._histogram = None
        self._curves = {}
        self.graphics.setBackground(COLORS["surface"])
        result = self._result
        if result is None:
            return
        self._histogram = self._draw_histogram(result, row=0)
        axes = sorted({t.axis for t in (r.target for r in result.nominal.results)})
        for row, axis in enumerate(axes, start=1):
            self._curves[axis] = self._draw_curves(result, axis, row)

    def histogram_plot(self) -> pg.PlotItem | None:
        """Return the histogram panel, or ``None`` if no result is shown."""
        return self._histogram

    def curve_plots(self) -> dict[str, pg.PlotItem]:
        """Return the reflectivity panels by scan axis (``"energy"``, ``"angle"``)."""
        return dict(self._curves)

    def _new_plot(self, row: int) -> pg.PlotItem:
        plot = self.graphics.addPlot(row=row, col=0)
        plot.showGrid(x=True, y=True, alpha=0.15)
        for name in ("bottom", "left"):
            axis = plot.getAxis(name)
            axis.enableAutoSIPrefix(False)
            axis.setPen(COLORS["text_muted"])
            axis.setTextPen(COLORS["text_muted"])
        return plot

    def _draw_histogram(self, result: ToleranceResult, row: int) -> pg.PlotItem:
        plot = self._new_plot(row)
        plot.setLabel("bottom", "Figure of merit (lower is better)")
        plot.setLabel("left", "Samples")
        plot.addLegend(offset=(-10, 10))
        merits = result.merits[np.isfinite(result.merits)]
        if merits.size:
            counts, edges = np.histogram(merits, bins=min(30, max(5, merits.size // 3)))
            plot.plot(
                edges,
                counts,
                stepMode="center",
                fillLevel=0,
                brush=_color(COLORS["accent"], _FILL_ALPHA),
                pen=pg.mkPen(COLORS["accent"], width=1),
            )
            for q in _PERCENTILES:
                pen = pg.mkPen(PLOT_PALETTE[3], width=1.5, style=Qt.PenStyle.DashLine)
                plot.addItem(pg.InfiniteLine(float(np.percentile(merits, q)), pen=pen))
        nominal_pen = pg.mkPen(PLOT_PALETTE[1], width=2)
        plot.addItem(pg.InfiniteLine(result.nominal.merit, pen=nominal_pen))
        plot.addItem(pg.PlotDataItem([], [], pen=nominal_pen, name="nominal"))
        return plot

    def _draw_curves(self, result: ToleranceResult, axis: str, row: int) -> pg.PlotItem:
        plot = self._new_plot(row)
        display = self._display
        nominal_results = display.nominal if display else result.nominal.results
        sample_results = display.samples if display else tuple(ev.results for ev in result.samples)
        for target in {id(r.target): r.target for r in result.nominal.results}.values():
            if target.axis != axis:
                continue
            _, factor = scan_axis(target)
            lo, hi = target.scan_range
            region = pg.LinearRegionItem(
                (lo * factor, hi * factor), movable=False, brush=_color(COLORS["accent"], 22)
            )
            region.setZValue(-10)
            for line in region.lines:
                line.setPen(pg.mkPen(None))
            plot.addItem(region)
        plot.setLabel("left", "Reflectivity R")
        indices = [i for i, r in enumerate(result.nominal.results) if r.target.axis == axis]
        reflections = {result.nominal.results[i].target.reflections for i in indices}
        if any(n > 1 for n in reflections):
            plot.setLabel("left", "Reflectivity Rⁿ")
        x_min, x_max = np.inf, -np.inf
        for index in indices:
            if index >= len(nominal_results):
                continue
            nominal = nominal_results[index]
            label, factor = scan_axis(nominal.target)
            plot.setLabel("bottom", label)
            base = 2 * indices.index(index)
            sample_color = PLOT_PALETTE[base % len(PLOT_PALETTE)]
            nominal_color = PLOT_PALETTE[(base + 1) % len(PLOT_PALETTE)]
            x = nominal.x * factor
            x_min, x_max = min(x_min, float(x.min())), max(x_max, float(x.max()))
            curves = [
                results[index].reflectivity
                for results in sample_results
                if len(results) > index
                and results[index].x.shape == nominal.x.shape
                and np.allclose(results[index].x, nominal.x)
            ]
            if curves:
                data = np.asarray(curves, dtype=np.float64)
                low, high = np.nanpercentile(data, _PERCENTILES, axis=0)
                band = pg.FillBetweenItem(
                    pg.PlotDataItem(x, low, pen=None),
                    pg.PlotDataItem(x, high, pen=None),
                    brush=_color(sample_color, _BAND_ALPHA),
                )
                plot.addItem(band)
            pen = pg.mkPen(_color(sample_color, _SAMPLE_ALPHA), width=1)
            for k in self._chosen(len(curves)):
                plot.addItem(
                    pg.PlotDataItem(x, curves[k], pen=pen, skipFiniteCheck=True, antialias=False)
                )
            plot.addItem(
                pg.PlotDataItem(
                    x,
                    nominal.reflectivity,
                    pen=pg.mkPen(nominal_color, width=2.5),
                    skipFiniteCheck=True,
                )
            )
        if np.isfinite(x_min) and x_max > x_min:
            # The fill item does not report a usable bounding box, so set the range explicitly.
            plot.setXRange(x_min, x_max, padding=0.02)
        return plot

    def _chosen(self, count: int) -> NDArray[np.intp]:
        """Return evenly spaced indices of at most `max_curves` of `count` curves."""
        if count <= self.max_curves:
            return np.arange(count)
        return np.unique(np.linspace(0, count - 1, self.max_curves).round().astype(np.intp))
