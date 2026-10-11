"""Plot widgets based on pyqtgraph."""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pyqtgraph as pg
from numpy.typing import NDArray
from PySide6.QtWidgets import QVBoxLayout, QWidget

from xstack.gui.style import COLORS, PLOT_PALETTE
from xstack.physics.models import ReflectivityResult
from xstack.structure.stack import Stack
from xstack.targets import Target

__all__ = [
    "ConvergencePlot",
    "Curve",
    "DesignMapPlot",
    "LayerProfilePlot",
    "ReflectivityPlot",
    "bilayer_profile",
    "scan_axis",
]

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


def bilayer_profile(
    stack: Stack,
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    """Return period and Γ of each bilayer of a stack, from the layer tags.

    Layers tagged ``"<prefix>.H"`` and ``"<prefix>.L"`` (and interlayers) with a common prefix
    form a bilayer, as generated by periodic and depth-graded coatings.

    Args:
        stack: The stack.

    Returns:
        Periods in nm and Γ values, from the top, or ``None`` if the stack has no bilayers.
    """
    groups: dict[str, dict[str, float]] = {}
    order: list[str] = []
    for layer in stack.layers:
        match = re.fullmatch(r"(.*)\.(H|L|interlayer_H_on_L|interlayer_L_on_H)", layer.tag)
        if match is None:
            continue
        prefix, role = match.groups()
        if prefix not in groups:
            groups[prefix] = {}
            order.append(prefix)
        groups[prefix][role] = groups[prefix].get(role, 0.0) + layer.thickness
    periods, gammas = [], []
    for prefix in order:
        parts = groups[prefix]
        if "H" not in parts or "L" not in parts:
            continue
        d = sum(parts.values())
        periods.append(d)
        gammas.append(parts["H"] / d if d > 0 else 0.0)
    if not periods:
        return None
    return np.array(periods), np.array(gammas)


class LayerProfilePlot(QWidget):
    """Layer thicknesses and, for bilayer coatings, period and Γ versus depth.

    Args:
        parent: Parent widget.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.graphics = pg.GraphicsLayoutWidget()
        layout.addWidget(self.graphics)

    def _add_plot(self, row: int) -> pg.PlotItem:
        plot = self.graphics.addPlot(row=row, col=0)
        plot.getAxis("left").enableAutoSIPrefix(False)
        return plot

    def set_stack(self, stack: Stack | None) -> None:
        """Show a stack.

        Args:
            stack: The stack, or ``None`` to clear the plot.
        """
        self.graphics.clear()
        if stack is None or not stack.layers:
            return
        thickness = stack.thicknesses()
        names = [layer.material.name for layer in stack.layers]
        materials = list(dict.fromkeys(names))
        plot = self._add_plot(0)
        plot.setLabel("bottom", "Layer index (0 = top)")
        plot.setLabel("left", "Thickness (nm)")
        plot.addLegend(offset=(-10, 10))
        index = np.arange(len(thickness))
        for i, material in enumerate(materials):
            mask = np.array([n == material for n in names])
            bars = pg.BarGraphItem(
                x=index[mask],
                height=thickness[mask],
                width=0.8,
                brush=PALETTE[i % len(PALETTE)],
                pen=pg.mkPen(None),
            )
            plot.addItem(bars)
            plot.legend.addItem(bars, material)
        profile = bilayer_profile(stack)
        if profile is None:
            return
        periods, gammas = profile
        bilayer = np.arange(1, len(periods) + 1)
        p_plot = self._add_plot(1)
        p_plot.setLabel("bottom", "Bilayer (1 = top)")
        p_plot.setLabel("left", "Period d (nm)")
        p_plot.plot(bilayer, periods, pen=pg.mkPen(PALETTE[0], width=2), symbol="o", symbolSize=4)
        g_plot = self._add_plot(2)
        g_plot.setXLink(p_plot)
        g_plot.setLabel("bottom", "Bilayer (1 = top)")
        g_plot.setLabel("left", "Γ")
        g_plot.plot(bilayer, gammas, pen=pg.mkPen(PALETTE[1], width=2), symbol="o", symbolSize=4)
        g_plot.setYRange(max(0.0, gammas.min() - 0.05), min(1.0, gammas.max() + 0.05))


class ConvergencePlot(QWidget):
    """Progress of an optimization versus the number of evaluations, one curve per stage.

    The upper panel shows the best objective found so far (figure of merit plus constraint
    penalties; the optimizers minimize it), the lower panel the figure of merit of that best
    design without penalties.

    Args:
        parent: Parent widget.
    """

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
        self.merit_plot.setLabel("left", "Figure of merit of best design")
        self.merit_plot.setXLink(self.objective_plot)
        for plot in (self.objective_plot, self.merit_plot):
            plot.setLabel("bottom", "Evaluations")
            plot.getAxis("left").enableAutoSIPrefix(False)
            plot.showGrid(x=True, y=True, alpha=0.15)
        self.objective_plot.addLegend(offset=(-10, 10))
        self._data: dict[str, tuple[list[float], list[float], list[float]]] = {}
        self._curves: dict[str, tuple[pg.PlotDataItem, pg.PlotDataItem]] = {}

    def clear(self) -> None:
        """Remove all curves."""
        for objective, merit in self._curves.values():
            self.objective_plot.removeItem(objective)
            self.merit_plot.removeItem(merit)
        if self.objective_plot.legend is not None:
            self.objective_plot.legend.clear()
        self._data.clear()
        self._curves.clear()

    def add_point(
        self, task: str, evaluations: int, best: float, merit: float = float("nan")
    ) -> None:
        """Append a point to the curves of a task.

        Args:
            task: Task name, e.g. ``"global: differential_evolution"``.
            evaluations: Number of evaluations so far.
            best: Best objective so far (positive).
            merit: Figure of merit of the best design so far.
        """
        xs, objectives, merits = self._data.setdefault(task, ([], [], []))
        xs.append(float(evaluations))
        objectives.append(max(float(best), 1e-12))
        merits.append(float(merit))
        if task not in self._curves:
            color = PALETTE[len(self._curves) % len(PALETTE)]
            pen = pg.mkPen(color, width=2)
            self._curves[task] = (
                self.objective_plot.plot(pen=pen, name=task),
                self.merit_plot.plot(pen=pen),
            )
        objective_curve, merit_curve = self._curves[task]
        objective_curve.setData(xs, objectives)
        merit_x = [x for x, m in zip(xs, merits, strict=True) if np.isfinite(m)]
        merit_y = [m for m in merits if np.isfinite(m)]
        merit_curve.setData(merit_x, merit_y)

    def tasks(self) -> tuple[str, ...]:
        """Return the names of the tasks shown."""
        return tuple(self._data)


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
