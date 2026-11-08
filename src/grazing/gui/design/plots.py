"""Plot widgets based on pyqtgraph."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pyqtgraph as pg
from numpy.typing import NDArray
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget

from grazing.core.physics.models import ReflectivityResult
from grazing.core.targets import Target
from grazing.design.optim.tradeoff import TradeoffPoint, TradeoffResult
from grazing.gui.common.profile import LayerProfilePlot, bilayer_profile
from grazing.gui.common.style import COLORS, PLOT_PALETTE

__all__ = [
    "MERIT_QUANTITY",
    "ConvergencePlot",
    "Curve",
    "DesignMapPlot",
    "LayerProfilePlot",
    "ReferenceDesign",
    "ReflectivityPlot",
    "TradeoffPlot",
    "bilayer_profile",
    "quantity_value",
    "scan_axis",
]

MERIT_QUANTITY = "figure of merit"
"""Name of the trade-off quantity that selects the figure of merit itself."""

PALETTE = PLOT_PALETTE
"""Colors of plot curves (the categorical palette of the design language)."""

pg.setConfigOptions(antialias=True, background=COLORS["surface"], foreground=COLORS["text_muted"])


def scan_axis(target: Target) -> tuple[str, float]:
    """Return the axis label and the factor converting internal values for display.

    Args:
        target: The target.

    Returns:
        Label and factor (eV → keV for energy scans, rad → degrees for angle scans).
    """
    if target.axis == "energy":
        return "Energy (keV)", 1e-3
    return "Grazing angle (deg)", 180.0 / np.pi


def _reflectivity_label(counts: set[int]) -> str:
    """Return the axis label for reflectivity after the given numbers of reflections."""
    if counts <= {1}:
        return "Reflectivity R"
    if len(counts) == 1:
        (n,) = counts
        return f"Reflectivity Rⁿ (n = {n} reflections)"
    return "Reflectivity Rⁿ (n per target)"


@dataclass(frozen=True)
class Curve:
    """A reflectivity curve for display.

    Attributes:
        label: Legend label.
        result: The reflectivity result.
        dashed: Draw with a dashed line (used for overlays).
    """

    label: str
    result: ReflectivityResult
    dashed: bool = False


class ReflectivityPlot(QWidget):
    """Reflectivity versus energy or angle, with shaded target ranges.

    Energy and angle scans are shown in separate panels; every curve gets its own color.

    Args:
        parent: Parent widget.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.graphics = pg.GraphicsLayoutWidget()
        layout.addWidget(self.graphics)
        self.log_scale = False
        self._plots: dict[str, pg.PlotItem] = {}

    def set_curves(self, curves: Sequence[Curve], targets: Sequence[Target] = ()) -> None:
        """Show curves and target ranges.

        Args:
            curves: The curves to draw; curve colors follow the target order.
            targets: Targets whose scan ranges are shaded.
        """
        self.graphics.clear()
        self._plots.clear()
        axes = sorted({c.result.target.axis for c in curves} | {t.axis for t in targets})
        for row, axis in enumerate(axes):
            plot = self.graphics.addPlot(row=row, col=0)
            plot.showGrid(x=True, y=True, alpha=0.15)
            plot.addLegend(offset=(-10, 10))
            counts = {
                t.reflections
                for t in (*targets, *(c.result.target for c in curves))
                if t.axis == axis
            }
            plot.setLabel("left", _reflectivity_label(counts))
            plot.getAxis("left").enableAutoSIPrefix(False)
            plot.setLogMode(y=self.log_scale)
            self._plots[axis] = plot
        for target in targets:
            label, factor = scan_axis(target)
            plot = self._plots[target.axis]
            plot.setLabel("bottom", label)
            lo, hi = target.scan_range
            region = pg.LinearRegionItem(
                (lo * factor, hi * factor),
                movable=False,
                brush=(79, 70, 229, 22),
                pen=pg.mkPen(None),
            )
            region.setZValue(-10)
            plot.addItem(region)
        for index, curve in enumerate(curves):
            label, factor = scan_axis(curve.result.target)
            color = PALETTE[index % len(PALETTE)]
            style = (
                pg.QtCore.Qt.PenStyle.DashLine if curve.dashed else pg.QtCore.Qt.PenStyle.SolidLine
            )
            pen = pg.mkPen(color, width=2, style=style)
            y = (
                np.clip(curve.result.reflectivity, 1e-12, None)
                if self.log_scale
                else curve.result.reflectivity
            )
            plot = self._plots[curve.result.target.axis]
            plot.setLabel("bottom", label)
            plot.plot(curve.result.x * factor, y, pen=pen, name=curve.label)

    def plot_items(self) -> dict[str, pg.PlotItem]:
        """Return the plot panels by scan axis (``"energy"``, ``"angle"``)."""
        return dict(self._plots)


class ConvergencePlot(QWidget):
    """Progress of an optimization versus the number of evaluations, one color per stage.

    The upper panel shows the best objective found so far (figure of merit plus constraint
    penalties; the optimizers minimize it). The lower panel shows the quantities that the
    figure of merit rates for that best design, e.g. the mean reflectivity, one line style per
    quantity. Figures of merit without quantities are shown as such.

    Args:
        parent: Parent widget.
    """

    MERIT_LABEL = "figure of merit"
    """Label of the lower-panel curve for figures of merit without quantities."""

    _STYLES = (
        pg.QtCore.Qt.PenStyle.SolidLine,
        pg.QtCore.Qt.PenStyle.DashLine,
        pg.QtCore.Qt.PenStyle.DotLine,
        pg.QtCore.Qt.PenStyle.DashDotLine,
    )

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.graphics = pg.GraphicsLayoutWidget()
        layout.addWidget(self.graphics)
        self.objective_plot = self.graphics.addPlot(row=0, col=0)
        self.objective_plot.setLabel("left", "Best objective (FOM + penalties)")
        self.objective_plot.setLogMode(y=True)
        self.merit_plot = self.graphics.addPlot(row=1, col=0)
        self.merit_plot.setXLink(self.objective_plot)
        for plot in (self.objective_plot, self.merit_plot):
            plot.setLabel("bottom", "Evaluations")
            plot.getAxis("left").enableAutoSIPrefix(False)
            plot.showGrid(x=True, y=True, alpha=0.15)
        self.objective_plot.addLegend(offset=(-10, 10))
        self.merit_plot.addLegend(offset=(-10, 10))
        self._data: dict[str, tuple[list[float], list[float]]] = {}
        self._objective_curves: dict[str, pg.PlotDataItem] = {}
        self._quantity_data: dict[tuple[str, str], tuple[list[float], list[float]]] = {}
        self._quantity_curves: dict[tuple[str, str], pg.PlotDataItem] = {}
        self._quantities: list[str] = []
        self._update_merit_label()

    def clear(self) -> None:
        """Remove all curves."""
        for curve in self._objective_curves.values():
            self.objective_plot.removeItem(curve)
        for curve in self._quantity_curves.values():
            self.merit_plot.removeItem(curve)
        for plot in (self.objective_plot, self.merit_plot):
            if plot.legend is not None:
                plot.legend.clear()
        self._data.clear()
        self._objective_curves.clear()
        self._quantity_data.clear()
        self._quantity_curves.clear()
        self._quantities.clear()
        self._update_merit_label()

    def add_point(
        self,
        task: str,
        evaluations: int,
        best: float,
        merit: float = float("nan"),
        quantities: Mapping[str, float] | None = None,
    ) -> None:
        """Append a point to the curves of a task.

        Args:
            task: Task name, e.g. ``"global: differential_evolution"``.
            evaluations: Number of evaluations so far.
            best: Best objective so far (positive).
            merit: Figure of merit of the best design so far.
            quantities: Quantities rated by the figure of merit for the best design so far,
                by label. Without quantities, the lower panel shows `merit`.
        """
        xs, objectives = self._data.setdefault(task, ([], []))
        xs.append(float(evaluations))
        objectives.append(max(float(best), 1e-12))
        if task not in self._objective_curves:
            pen = pg.mkPen(self._color(task), width=2)
            self._objective_curves[task] = self.objective_plot.plot(pen=pen, name=task)
        self._objective_curves[task].setData(xs, objectives)
        values = dict(quantities) if quantities else {self.MERIT_LABEL: float(merit)}
        for label, value in values.items():
            self._add_quantity_point(task, label, float(evaluations), value)

    def _color(self, task: str) -> Any:
        return PALETTE[list(self._data).index(task) % len(PALETTE)]

    def _add_quantity_point(self, task: str, label: str, x: float, value: float) -> None:
        if label not in self._quantities:
            self._quantities.append(label)
            style = self._STYLES[(len(self._quantities) - 1) % len(self._STYLES)]
            self.merit_plot.legend.addItem(
                pg.PlotDataItem(pen=pg.mkPen(COLORS["text_muted"], width=2, style=style)), label
            )
            self._update_merit_label()
        key = (task, label)
        xs, ys = self._quantity_data.setdefault(key, ([], []))
        if not np.isfinite(value):
            return
        xs.append(x)
        ys.append(value)
        if key not in self._quantity_curves:
            style = self._STYLES[self._quantities.index(label) % len(self._STYLES)]
            pen = pg.mkPen(self._color(task), width=2, style=style)
            self._quantity_curves[key] = self.merit_plot.plot(pen=pen)
        self._quantity_curves[key].setData(xs, ys)

    def _update_merit_label(self) -> None:
        if len(self._quantities) == 1:
            label = f"{self._quantities[0][0].upper()}{self._quantities[0][1:]} of best design"
        else:
            label = "Best design"
        self.merit_plot.setLabel("left", label)
        if self.merit_plot.legend is not None:
            self.merit_plot.legend.setVisible(len(self._quantities) > 1)

    def tasks(self) -> tuple[str, ...]:
        """Return the names of the tasks shown."""
        return tuple(self._data)

    def quantities(self) -> tuple[str, ...]:
        """Return the labels of the quantities shown in the lower panel."""
        return tuple(self._quantities)


class DesignMapPlot(QWidget):
    """Heat map of a two-dimensional (or line plot of a one-dimensional) design-space scan.

    Infeasible designs are covered with a grey overlay and the colour scale spans the feasible
    designs. Clicking a cell emits its parameter values.

    Signals:
        pointSelected(dict): Parameter values of the clicked grid point.

    Args:
        parent: Parent widget.
    """

    pointSelected = pg.QtCore.Signal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.graphics = pg.GraphicsLayoutWidget()
        layout.addWidget(self.graphics)
        self._names: tuple[str, ...] = ()
        self._axes: tuple[NDArray[np.float64], ...] = ()
        self._plot: pg.PlotItem | None = None
        self._marker: pg.ScatterPlotItem | None = None

    def set_scan(
        self,
        names: Sequence[str],
        axes: Sequence[NDArray[np.float64]],
        values: NDArray[np.float64],
        feasible: NDArray[np.bool_],
        label: str = "Objective",
    ) -> None:
        """Show scan results.

        Args:
            names: Names of the scanned parameters (one or two).
            axes: Grid values per parameter.
            values: Values to show, shape (len(axes[0]),) or (len(axes[0]), len(axes[1])).
            feasible: Feasibility per grid point, same shape as `values`.
            label: Name of the shown quantity.
        """
        self.graphics.clear()
        self._marker = None
        self._names, self._axes = tuple(names), tuple(axes)
        plot = self.graphics.addPlot(row=0, col=0)
        plot.getAxis("left").enableAutoSIPrefix(False)
        plot.getAxis("bottom").enableAutoSIPrefix(False)
        self._plot = plot
        plot.setLabel("bottom", names[0])
        if len(names) == 1:
            plot.setLabel("left", label)
            x = axes[0]
            plot.plot(x, values, pen=pg.mkPen(PALETTE[0], width=2), symbol="o", symbolSize=5)
            bad = ~feasible
            if bad.any():
                plot.plot(
                    x[bad], values[bad], pen=None, symbol="x", symbolBrush="#888", symbolSize=8
                )
            return
        plot.setLabel("left", names[1])
        x, y = axes[0], axes[1]
        finite = np.where(np.isfinite(values), values, np.nan)
        image = pg.ImageItem(finite)
        dx = (x[-1] - x[0]) / max(len(x) - 1, 1) if len(x) > 1 else 1.0
        dy = (y[-1] - y[0]) / max(len(y) - 1, 1) if len(y) > 1 else 1.0
        rect = pg.QtCore.QRectF(x[0] - dx / 2, y[0] - dy / 2, dx * len(x), dy * len(y))
        image.setRect(rect)
        image.setColorMap(pg.colormap.get("viridis"))
        shown = finite[feasible & np.isfinite(finite)]
        if shown.size == 0:
            shown = finite[np.isfinite(finite)]
        if shown.size:
            lo, hi = float(shown.min()), float(shown.max())
            image.setLevels((lo, hi if hi > lo else lo + 1e-12))
        plot.addItem(image)
        overlay = np.zeros((*values.shape, 4), dtype=np.ubyte)
        overlay[~feasible] = (128, 128, 128, 170)
        mask = pg.ImageItem(overlay)
        mask.setRect(rect)
        plot.addItem(mask)
        bar = pg.ColorBarItem(values=image.levels, colorMap=pg.colormap.get("viridis"), label=label)
        bar.setImageItem(image, insert_in=plot)
        image.mouseClickEvent = self._clicked
        mask.mouseClickEvent = self._clicked
        self._marker = pg.ScatterPlotItem(
            size=16, symbol="+", pen=pg.mkPen("#d62728", width=3), brush=None
        )
        plot.addItem(self._marker)

    def _clicked(self, event: Any) -> None:
        if self._plot is None or len(self._axes) != 2:
            return
        pos = self._plot.vb.mapSceneToView(event.scenePos())
        x, y = self._axes
        i = int(np.argmin(np.abs(x - pos.x())))
        j = int(np.argmin(np.abs(y - pos.y())))
        self.select(i, j)

    def select(self, i: int, j: int) -> None:
        """Emit the parameter values of a grid point, as if it had been clicked.

        Args:
            i: Index along the first axis.
            j: Index along the second axis.
        """
        values = {self._names[0]: float(self._axes[0][i])}
        if len(self._axes) == 2:
            values[self._names[1]] = float(self._axes[1][j])
        self.mark(values)
        self.pointSelected.emit(values)

    def mark(self, values: dict[str, float]) -> None:
        """Mark a grid point of a two-dimensional map with a cross.

        Args:
            values: Parameter values of the point.
        """
        if self._marker is None or len(self._names) != 2:
            return
        x, y = values.get(self._names[0]), values.get(self._names[1])
        if x is not None and y is not None:
            self._marker.setData([x], [y])


def quantity_value(point: TradeoffPoint, quantity: str) -> float:
    """Return a quantity of a trade-off point.

    Args:
        point: The trade-off point.
        quantity: `MERIT_QUANTITY` or the label of one of the point's quantities.

    Returns:
        The figure of merit or the quantity; ``nan`` if the point has no such quantity.
    """
    if quantity == MERIT_QUANTITY:
        return point.merit
    return dict(point.quantities).get(quantity, float("nan"))


@dataclass(frozen=True)
class ReferenceDesign:
    """A design shown in the trade-off plot for comparison with the front.

    Attributes:
        label: Name shown next to the marker.
        coating_time: Coating time in s.
        merit: Figure of merit.
    """

    label: str
    coating_time: float
    merit: float


type _Group = list[tuple[float, float, int]]


class _Decoration(pg.ScatterPlotItem):
    """Markers that only decorate other points; clicks pass through to the points below."""

    def mouseClickEvent(self, ev: Any) -> None:
        """Ignore the click, so that the scene offers it to the items below."""
        ev.ignore()


class TradeoffPlot(QWidget):
    """Quantity versus coating time of a trade-off, with the marginal gain below.

    The upper panel shows the feasible designs as a line with markers, infeasible designs as
    hollow grey markers, designs taken over from a shorter limit as diamonds and the knee with
    a ring and a label. Reference designs (the input design and stored candidates) are drawn
    as separate labelled markers. The lower panel shows the improvement of the figure of merit
    per additional hour between neighbouring feasible designs.

    Clicking a marker selects its point; clicking the selected point again clears the selection.
    The knee ring, the selection mark and the reference designs don't take clicks. A point taken
    over from a shorter limit lies on the point it was taken from; a click there selects the
    point of the shortest limit.

    Signals:
        pointSelected(object): A marker was clicked; the index of the point in
            `TradeoffResult.points`, or ``None`` if the click cleared the selection.

    Attributes:
        feasible_xy: Coating times in h and plotted values of the feasible points.
        infeasible_xy: The same for infeasible points.
        replaced_xy: The same for points taken over from a shorter limit.
        reference_xy: The same for the reference designs.
        knee_xy: Coating time in h and value of the knee, or ``None``.
        gains: Marginal gains in figure of merit per h shown in the lower panel.
        labels: Texts of the annotations in the upper panel.
        selected: Index of the marked point, or ``None``.

    Args:
        parent: Parent widget.
    """

    pointSelected = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.graphics = pg.GraphicsLayoutWidget()
        layout.addWidget(self.graphics)
        self.main_plot = self.graphics.addPlot(row=0, col=0)
        self.gain_plot = self.graphics.addPlot(row=1, col=0)
        self.gain_plot.setXLink(self.main_plot)
        self.graphics.ci.layout.setRowStretchFactor(0, 3)
        self.graphics.ci.layout.setRowStretchFactor(1, 1)
        for plot in (self.main_plot, self.gain_plot):
            plot.showGrid(x=True, y=True, alpha=0.15)
            plot.getAxis("left").enableAutoSIPrefix(False)
        self.gain_plot.setLabel("bottom", "Coating time (h)")
        self.gain_plot.setLabel("left", "Gain (FOM / h)")
        self.main_plot.setLabel("left", MERIT_QUANTITY)
        self._result: TradeoffResult | None = None
        self._quantity = MERIT_QUANTITY
        self._items: list[tuple[pg.PlotItem, Any]] = []
        self._marker: tuple[pg.PlotItem, Any] | None = None
        self._positions: dict[int, tuple[float, float]] = {}
        self.feasible_xy: tuple[list[float], list[float]] = ([], [])
        self.infeasible_xy: tuple[list[float], list[float]] = ([], [])
        self.replaced_xy: tuple[list[float], list[float]] = ([], [])
        self.reference_xy: tuple[list[float], list[float]] = ([], [])
        self.knee_xy: tuple[float, float] | None = None
        self.gains: tuple[float, ...] = ()
        self.labels: list[str] = []
        self.selected: int | None = None

    def clear(self) -> None:
        """Remove all data."""
        for plot, item in self._items:
            plot.removeItem(item)
        self._items.clear()
        self._remove_marker()
        self._positions = {}
        self.feasible_xy = ([], [])
        self.infeasible_xy = ([], [])
        self.replaced_xy = ([], [])
        self.reference_xy = ([], [])
        self.knee_xy = None
        self.gains = ()
        self.labels = []
        self._result = None

    def _add(self, plot: pg.PlotItem, item: Any) -> None:
        plot.addItem(item)
        self._items.append((plot, item))

    def _annotate(
        self, text: str, x: float, y: float, color: str, anchor: tuple[float, float]
    ) -> None:
        item = pg.TextItem(text, color=color, anchor=anchor)
        item.setPos(x, y)
        self._add(self.main_plot, item)
        self.labels.append(text)

    def _scatter(self, group: _Group, **style: Any) -> None:
        scatter = pg.ScatterPlotItem(
            x=[p[0] for p in group], y=[p[1] for p in group], data=[p[2] for p in group], **style
        )
        scatter.sigClicked.connect(self._clicked)
        self._add(self.main_plot, scatter)

    def set_result(
        self,
        result: TradeoffResult | None,
        quantity: str = MERIT_QUANTITY,
        references: Sequence[ReferenceDesign] = (),
    ) -> None:
        """Show a trade-off.

        Args:
            result: The trade-off, or ``None`` to clear the plot.
            quantity: `MERIT_QUANTITY` or the label of a quantity to plot.
            references: Designs to show for comparison; they are only shown for the figure of
                merit.
        """
        selected = self.selected
        self.clear()
        self._result, self._quantity = result, quantity
        self.main_plot.setLabel(
            "left", "Figure of merit (lower is better)" if quantity == MERIT_QUANTITY else quantity
        )
        if result is None:
            return
        text, muted = COLORS["text"], COLORS["text_muted"]
        feasible: _Group = []
        infeasible: _Group = []
        replaced: _Group = []
        for index, point in enumerate(result.points):
            x, y = point.coating_time / 3600.0, quantity_value(point, quantity)
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            group = infeasible if not point.feasible else replaced if point.replaced else feasible
            group.append((x, y, index))
            self._positions[index] = (x, y)
            if np.isinf(point.limit):
                self._annotate("no limit", x, y, text, (1.0, 1.2))
        front = sorted([*feasible, *replaced])
        if front:
            line = pg.PlotDataItem(
                [p[0] for p in front], [p[1] for p in front], pen=pg.mkPen(PALETTE[0], width=2)
            )
            self._add(self.main_plot, line)
        solid = {"symbol": "o", "size": 9, "pen": pg.mkPen(PALETTE[0]), "brush": PALETTE[0]}
        hollow = {"symbol": "o", "size": 9, "pen": pg.mkPen(muted, width=1.5), "brush": None}
        diamond = {"symbol": "d", "size": 13, "pen": pg.mkPen(PALETTE[3]), "brush": PALETTE[3]}
        for group, style in ((feasible, solid), (infeasible, hollow), (replaced, diamond)):
            if group:
                self._scatter(group, **style)
        self.feasible_xy = ([p[0] for p in feasible], [p[1] for p in feasible])
        self.infeasible_xy = ([p[0] for p in infeasible], [p[1] for p in infeasible])
        self.replaced_xy = ([p[0] for p in replaced], [p[1] for p in replaced])
        self._draw_knee(result, quantity)
        if quantity == MERIT_QUANTITY:
            self._draw_references(references)
        self._draw_gains(result)
        self.select(selected)

    def _draw_knee(self, result: TradeoffResult, quantity: str) -> None:
        knee = result.knee()
        if knee is None or not np.isfinite(quantity_value(knee, quantity)):
            return
        x, y = knee.coating_time / 3600.0, quantity_value(knee, quantity)
        self.knee_xy = (x, y)
        ring = _Decoration(
            x=[x], y=[y], symbol="o", size=18, brush=None, pen=pg.mkPen(PALETTE[1], width=2)
        )
        self._add(self.main_plot, ring)
        self._annotate("knee", x, y, PALETTE[1], (-0.3, 1.4))

    def _draw_references(self, references: Sequence[ReferenceDesign]) -> None:
        shown = [r for r in references if np.isfinite(r.coating_time) and np.isfinite(r.merit)]
        xs, ys = [r.coating_time / 3600.0 for r in shown], [r.merit for r in shown]
        self.reference_xy = (xs, ys)
        if not shown:
            return
        muted = COLORS["text_muted"]
        marker = _Decoration(x=xs, y=ys, symbol="t", size=10, pen=pg.mkPen(muted), brush=PALETTE[6])
        self._add(self.main_plot, marker)
        for design, x, y in zip(shown, xs, ys, strict=True):
            self._annotate(design.label, x, y, muted, (0.0, -0.3))

    def _draw_gains(self, result: TradeoffResult) -> None:
        points = result.feasible_points()
        self.gains = result.marginal_gains()
        x0, x1, heights = [], [], []
        for shorter, longer, gain in zip(points, points[1:], self.gains, strict=False):
            if np.isfinite(gain):
                x0.append(shorter.coating_time / 3600.0)
                x1.append(longer.coating_time / 3600.0)
                heights.append(gain)
        if heights:
            bars = pg.BarGraphItem(
                x0=x0, x1=x1, height=heights, brush=PALETTE[2], pen=pg.mkPen(COLORS["surface"])
            )
            self._add(self.gain_plot, bars)

    def _remove_marker(self) -> None:
        if self._marker is not None:
            plot, item = self._marker
            plot.removeItem(item)
            self._marker = None

    def select(self, index: int | None) -> None:
        """Mark a point of the trade-off with a square.

        Args:
            index: Index in `TradeoffResult.points`, or ``None`` to remove the mark.
        """
        self._remove_marker()
        self.selected = index
        result = self._result
        if result is None or index is None or not 0 <= index < len(result.points):
            return
        point = result.points[index]
        x, y = point.coating_time / 3600.0, quantity_value(point, self._quantity)
        if np.isfinite(x) and np.isfinite(y):
            marker = _Decoration(
                x=[x], y=[y], symbol="s", size=17, brush=None, pen=pg.mkPen(COLORS["text"], width=2)
            )
            self.main_plot.addItem(marker)
            self._marker = (self.main_plot, marker)

    def _clicked(self, _item: Any, points: Sequence[Any]) -> None:
        if not points:
            return
        clicked = int(points[0].data())
        position = self._positions.get(clicked)
        clicked = min(i for i, xy in self._positions.items() if xy == position)
        index = None if clicked == self.selected else clicked
        self.select(index)
        self.pointSelected.emit(index)
