"""Workflow page for design-space scans."""

from collections.abc import Callable
from typing import Any

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from xstack.core.optim.events import CancellationToken
from xstack.design.optim.scan import ScanResult, scan
from xstack.design.project import ProjectState
from xstack.design.project.commands import set_parameters
from xstack.gui.common.workers import Job
from xstack.gui.design.steps.base import StepPage

__all__ = ["ScanPage"]

_NONE = "(none)"


class ScanPage(StepPage):
    """Scan one or two free parameters over their bounds and show the design map.

    Signals:
        scanFinished(ScanResult): A scan finished.

    Args:
        parent: Parent widget.
    """

    title = "Scan"
    help_text = (
        "<b>Design-space scan.</b> The scan evaluates the figure of merit on a grid of one or "
        "two free parameters over their bounds, keeping all other parameters at the values of "
        "the current input design. The resulting map shows where designs are good and where "
        "they are infeasible (grey). Clicking a cell of the map, or pressing <i>Use best grid "
        "point</i>, sets those parameter values as <b>start values</b> of the input design; "
        "the Optimize step starts from them. The scan itself does not optimize anything."
    )
    scanFinished = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state: ProjectState | None = None
        self.result: ScanResult | None = None
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<b>Design-space scan</b> over the bounds of free parameters"))
        form = QFormLayout()
        self.x_combo = QComboBox()
        self.y_combo = QComboBox()
        self.x_points = QSpinBox()
        self.y_points = QSpinBox()
        for box, value in ((self.x_points, 15), (self.y_points, 11)):
            box.setRange(2, 200)
            box.setValue(value)
        self.grid_points = QSpinBox()
        self.grid_points.setRange(5, 2000)
        self.grid_points.setValue(60)
        self.x_combo.setToolTip("Free parameter varied along the horizontal axis of the map.")
        self.y_combo.setToolTip("Optional second free parameter, varied along the vertical axis.")
        self.x_points.setToolTip("Number of grid values of the first parameter.")
        self.y_points.setToolTip("Number of grid values of the second parameter.")
        self.grid_points.setToolTip(
            "Number of points of the evaluation grid (energy or angle) used for each design."
        )
        form.addRow("First parameter", self.x_combo)
        form.addRow("Points", self.x_points)
        form.addRow("Second parameter", self.y_combo)
        form.addRow("Points", self.y_points)
        form.addRow("Target grid points", self.grid_points)
        layout.addLayout(form)
        self.run_button = QPushButton("Run scan")
        self.run_button.setToolTip("Evaluate the figure of merit on the grid.")
        self.run_button.clicked.connect(self.run)
        self.apply_button = QPushButton("Use best grid point")
        self.apply_button.setEnabled(False)
        self.apply_button.setToolTip(
            "Set the parameters of the best grid point as start values of the input design."
        )
        self.apply_button.clicked.connect(self.apply_best)
        layout.addWidget(self.run_button)
        layout.addWidget(self.apply_button)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.confirmation = QLabel()
        self.confirmation.setWordWrap(True)
        self.confirmation.setObjectName("confirmation")
        self.confirmation.setVisible(False)
        layout.addWidget(self.confirmation)
        layout.addWidget(QLabel("Click a cell of the design map to use its parameter values."))
        layout.addStretch()

    def set_state(self, state: ProjectState) -> None:
        """Offer the free parameters of the current design.

        Args:
            state: The current project state.
        """
        self._state = state
        names = list(state.mirror.parameters.free_names()) if state.mirror is not None else []
        for combo, extra in ((self.x_combo, []), (self.y_combo, [_NONE])):
            current = combo.currentText()
            items = extra + names
            if [combo.itemText(i) for i in range(combo.count())] != items:
                combo.clear()
                combo.addItems(items)
                if current in items:
                    combo.setCurrentText(current)
                elif combo is self.y_combo and len(names) > 1:
                    combo.setCurrentIndex(2)
        self.run_button.setEnabled(bool(names))

    def axes(self) -> dict[str, np.ndarray]:
        """Return the scan grid chosen in the page.

        Returns:
            Grid values by parameter name.

        Raises:
            ValueError: If no design is defined or a parameter has infinite bounds.
        """
        if self._state is None or self._state.mirror is None:
            raise ValueError("no design defined")
        params = self._state.mirror.parameters
        choices = [(self.x_combo.currentText(), self.x_points.value())]
        y = self.y_combo.currentText()
        if y not in ("", _NONE) and y != choices[0][0]:
            choices.append((y, self.y_points.value()))
        axes: dict[str, np.ndarray] = {}
        for name, points in choices:
            p = params[name]
            if not (np.isfinite(p.min) and np.isfinite(p.max)):
                raise ValueError(f"{name} needs finite bounds for a scan")
            values = np.linspace(p.min, p.max, points)
            axes[name] = np.unique(np.round(values)) if p.integer else values
        return axes

    def run(self) -> None:
        """Request a scan job with the current settings."""
        if self._state is None:
            return
        try:
            axes = self.axes()
            problem = self._state.problem().evolve(points=self.grid_points.value())
        except (ValueError, KeyError) as exc:
            self.errorOccurred.emit(f"Cannot scan: {exc}")
            return

        def compute(progress: Callable[[Any], None], cancel: CancellationToken) -> ScanResult:
            return scan(problem, axes, events=progress, cancel=cancel)

        self.jobRequested.emit(Job("Design-space scan", compute, self._finished))

    def _finished(self, result: ScanResult) -> None:
        self.result = result
        self.apply_button.setEnabled(True)
        best = ", ".join(f"{k} = {v:.4g}" for k, v in result.best_values().items())
        feasible = int(result.feasible.sum())
        self.summary.setText(
            f"Best grid point: {best} (objective {np.nanmin(result.objective):.4g}); "
            f"{feasible} of {result.feasible.size} points feasible."
        )
        self.scanFinished.emit(result)

    def apply_best(self) -> None:
        """Use the parameter values of the best grid point."""
        if self.result is not None:
            self.apply_values(self.result.best_values())

    def apply_values(self, values: dict[str, float]) -> None:
        """Set parameters to given values (bounds are widened if needed).

        Args:
            values: Parameter values by name.
        """
        if self._state is None or self._state.mirror is None:
            return
        params = self._state.mirror.parameters
        for name, value in values.items():
            p = params[name]
            params = params.replace(
                name,
                value=round(value) if p.integer else value,
                min=min(p.min, value),
                max=max(p.max, value),
            )
        text = ", ".join(f"{k} = {v:.4g}" for k, v in values.items())
        self.confirmation.setText(f"Start values set: {text}")
        self.confirmation.setVisible(True)
        self.commandRequested.emit(set_parameters(params, f"Use scan point {text}"))
