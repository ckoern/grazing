"""Workflow page for tolerance analysis and recipe export."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from xstack.gui.steps.base import StepPage
from xstack.gui.workers import Job
from xstack.io.recipe import write_recipe_csv, write_recipe_json
from xstack.manufacturing.tolerance import ToleranceResult, thickness_tolerance
from xstack.optim.events import CancellationToken
from xstack.project import ProjectState
from xstack.structure.coating import CoatingBuildError, Mirror

__all__ = ["AnalyzePage"]


class AnalyzePage(StepPage):
    """Monte Carlo tolerance analysis of the current design and recipe export.

    Signals:
        toleranceFinished(ToleranceResult): A tolerance analysis finished.

    Args:
        parent: Parent widget.
    """

    title = "Analyze"
    help_text = (
        "<b>Analyze.</b> Real deposition is never perfect, so this step tests how robust the "
        "current design is. The Monte Carlo analysis perturbs every layer thickness randomly "
        "and each material's deposition rate systematically, then re-evaluates the figure of "
        "merit for many samples. The spread of the results shows how sensitive the design is "
        "to manufacturing errors. When you are satisfied, export the layer table or the full "
        "recipe for your deposition system."
    )
    toleranceFinished = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state: ProjectState | None = None
        self.result: ToleranceResult | None = None
        layout = QVBoxLayout(self)
        design_row = QHBoxLayout()
        design_row.addWidget(QLabel("Design to analyze:"))
        self.design_combo = QComboBox()
        self.design_combo.setToolTip(
            "Analyze and export the input design (Coating step) or the candidate selected in "
            "the candidates list, e.g. an optimization result"
        )
        self.design_combo.addItem("Input design", "input")
        self.design_combo.addItem("Selected candidate", "selected")
        design_row.addWidget(self.design_combo)
        design_row.addStretch()
        layout.addLayout(design_row)
        box = QGroupBox("Thickness tolerance (Monte Carlo)")
        form = QFormLayout(box)
        self.random_error = QDoubleSpinBox()
        self.random_error.setRange(0.0, 50.0)
        self.random_error.setValue(2.0)
        self.random_error.setSuffix(" %")
        self.systematic_error = QDoubleSpinBox()
        self.systematic_error.setRange(0.0, 50.0)
        self.systematic_error.setSuffix(" %")
        self.samples = QSpinBox()
        self.samples.setRange(1, 10000)
        self.samples.setValue(50)
        self.seed = QLineEdit()
        self.seed.setPlaceholderText("random")
        self.random_error.setToolTip("Relative random thickness error of each layer (1 sigma).")
        self.systematic_error.setToolTip(
            "Relative error of the deposition rate of each material, common to all its layers."
        )
        self.samples.setToolTip("Number of randomly perturbed designs to evaluate.")
        self.seed.setToolTip("Integer seed for reproducible results; empty for a random seed.")
        form.addRow("Random error per layer", self.random_error)
        form.addRow("Rate error per material", self.systematic_error)
        form.addRow("Samples", self.samples)
        form.addRow("Random seed", self.seed)
        self.run_button = QPushButton("Run tolerance analysis")
        self.run_button.setToolTip("Evaluate the perturbed designs in the background.")
        self.run_button.clicked.connect(self.run)
        form.addRow(self.run_button)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        form.addRow(self.summary)
        layout.addWidget(box)
        export = QGroupBox("Deposition recipe")
        export_layout = QVBoxLayout(export)
        self.csv_button = QPushButton("Export layer table (CSV)…")
        self.json_button = QPushButton("Export recipe (JSON)…")
        self.csv_button.setToolTip("Write the layer table of the design as a CSV file.")
        self.json_button.setToolTip("Write the layer table and process as a JSON recipe.")
        self.csv_button.clicked.connect(lambda: self.export(None, "csv"))
        self.json_button.clicked.connect(lambda: self.export(None, "json"))
        export_layout.addWidget(self.csv_button)
        export_layout.addWidget(self.json_button)
        layout.addWidget(export)
        layout.addStretch()

    def set_state(self, state: ProjectState) -> None:
        """Remember the current state.

        Args:
            state: The current project state.
        """
        self._state = state
        selected = state.selected_candidate()
        label = f"Selected candidate ({selected.name})" if selected else "Selected candidate (none)"
        self.design_combo.setItemText(1, label)

    def design(self) -> Mirror | None:
        """Return the design chosen for analysis and export.

        Returns:
            The input design or the selected candidate's design; ``None`` if there is none.
        """
        if self._state is None:
            return None
        if self.design_combo.currentData() == "selected":
            candidate = self._state.selected_candidate()
            return candidate.mirror if candidate is not None else None
        return self._state.mirror

    def run(self) -> None:
        """Request a tolerance analysis job."""
        if self._state is None:
            return
        mirror = self.design()
        try:
            if mirror is None:
                raise ValueError("no design selected")
            problem = self._state.problem().evolve(mirror=mirror)
        except ValueError as exc:
            self.errorOccurred.emit(f"Cannot analyze: {exc}")
            return
        seed_text = self.seed.text().strip()
        settings = {
            "random_error": self.random_error.value() / 100.0,
            "systematic_error": self.systematic_error.value() / 100.0,
            "samples": self.samples.value(),
            "seed": int(seed_text) if seed_text else None,
        }

        def compute(progress: Callable[[Any], None], cancel: CancellationToken) -> ToleranceResult:
            return thickness_tolerance(problem, cancel=cancel, **settings)

        self.jobRequested.emit(Job("Tolerance analysis", compute, self._finished))

    def _finished(self, result: ToleranceResult) -> None:
        self.result = result
        m = result.merits
        self.summary.setText(
            f"Nominal figure of merit {result.nominal.merit:.4g}; samples: median "
            f"{np.median(m):.4g}, 10–90 % {np.percentile(m, 10):.4g}–{np.percentile(m, 90):.4g}; "
            f"{int(result.feasible.sum())} of {len(m)} feasible."
        )
        self.toleranceFinished.emit(result)

    def export(self, path: Path | str | None, kind: str) -> bool:
        """Write the layer table of the current design.

        Args:
            path: Output file; asked for if ``None``.
            kind: ``"csv"`` or ``"json"``.

        Returns:
            Whether a file was written.
        """
        mirror = self.design()
        if self._state is None or mirror is None:
            self.errorOccurred.emit("No design to export")
            return False
        try:
            stack = mirror.stack()
        except (CoatingBuildError, ValueError) as exc:
            self.errorOccurred.emit(f"Cannot export: {exc}")
            return False
        if path is None:
            pattern = "CSV files (*.csv)" if kind == "csv" else "JSON files (*.json)"
            chosen, _ = QFileDialog.getSaveFileName(self, "Export recipe", "", pattern)
            if not chosen:
                return False
            path = chosen
        try:
            if kind == "csv":
                write_recipe_csv(path, stack, self._state.process)
            else:
                write_recipe_json(path, stack, self._state.process)
        except (OSError, KeyError) as exc:
            self.errorOccurred.emit(f"Cannot export: {exc}")
            return False
        return True
