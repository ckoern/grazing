"""Workflow page for tolerance analysis and recipe export."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
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

from xstack.gui.files import ask_save_path
from xstack.gui.preview import ToleranceCurves, tolerance_curves
from xstack.gui.steps.base import StepPage
from xstack.gui.workers import Job
from xstack.io.recipe import write_recipe_csv, write_recipe_json
from xstack.manufacturing.tolerance import ErrorModel, ToleranceResult, tolerance_analysis
from xstack.optim.events import CancellationToken
from xstack.project import ProjectState
from xstack.structure.coating import CoatingBuildError, Mirror

__all__ = ["AnalyzePage"]

CUSTOM = "Custom"

# Preset name -> (thickness %, rate %, drift %, roughness nm, common roughness nm,
# mixing nm, mixing variation %); None marks a disabled section.
PRESETS: dict[str, tuple[tuple[float, ...] | None, ...]] = {
    "Thickness errors only": ((2.0, 0.0, 0.0), None, None),
    "Roughness only": (None, (0.1, 0.05), None),
    "Interlayer mixing only": (None, None, (0.5, 20.0)),
    "All errors": ((2.0, 1.0, 0.5), (0.1, 0.05), (0.5, 20.0)),
}


def describe_errors(errors: ErrorModel) -> str:
    """Describe an error model in one line.

    Args:
        errors: The error model.

    Returns:
        Comma-separated list of the active errors with their values.
    """
    items = [
        ("random thickness", errors.thickness_random * 100, "%"),
        ("rate error", errors.thickness_systematic * 100, "%"),
        ("rate drift", errors.thickness_drift * 100, "%"),
        ("random roughness", errors.roughness_random, "nm"),
        ("common roughness", errors.roughness_systematic, "nm"),
        ("mixed layer", errors.interlayer_thickness, "nm"),
        ("mixed-layer variation", errors.interlayer_random * 100, "%"),
    ]
    return ", ".join(f"{name} {value:.3g} {unit}" for name, value, unit in items if value > 0)


class AnalyzePage(StepPage):
    """Monte Carlo tolerance analysis of the current design and recipe export.

    Signals:
        toleranceFinished(ToleranceResult, object, object): A tolerance analysis finished;
            the arguments are the result, its reflectivity curves on the display range
            (`ToleranceCurves`, or ``None`` without display range) and that display range.

    Attributes:
        display_ranges: Plot range by scan axis; the analysis also calculates the
            reflectivity of all samples on this range for display.

    Args:
        parent: Parent widget.
    """

    title = "Analyze"
    help_text = (
        "<b>Analyze.</b> Real deposition is never perfect, so this step tests how robust the "
        "current design is. The Monte Carlo analysis draws many randomly perturbed copies of "
        "the design and re-evaluates the figure of merit for each. Three kinds of errors can "
        "be switched on separately: <i>thickness</i> errors (random per layer, a rate "
        "calibration error per material, and a linear rate drift over the run), "
        "<i>roughness</i> errors (random per interface and a common change of all "
        "interfaces) and <i>interlayer mixing</i> (a mixed layer of averaged optical "
        "constants at every interface between different materials). Pick a preset or edit "
        "the values. Two plots show the result: the distribution (histogram) of the figure "
        "of merit over all samples compared with the nominal value, and the reflectivity "
        "curves of all samples overlaid on the nominal curve. A narrow distribution and "
        "curves close to the nominal one mean a robust design. When you are satisfied, "
        "export the layer table or the full recipe for your deposition system."
    )
    toleranceFinished = Signal(object, object, object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.display_ranges: dict[str, tuple[float, float]] = {}
        self._state: ProjectState | None = None
        self.result: ToleranceResult | None = None
        self._applying = False
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
        box = QGroupBox("Error model (Monte Carlo)")
        form = QFormLayout(box)
        form.setVerticalSpacing(3)
        self.preset_combo = QComboBox()
        self.preset_combo.setToolTip(
            "Fill the controls with a typical set of manufacturing errors."
        )
        for name in PRESETS:
            self.preset_combo.addItem(name)
        self.preset_combo.addItem(CUSTOM)
        form.addRow("Preset", self.preset_combo)

        self.thickness_box = self._section(
            form,
            "Thickness",
            "Include errors of the layer thicknesses.",
            True,
        )
        self.random_error = self._spin(
            50.0, 2.0, " %", "Relative random thickness error of each layer (1 sigma)."
        )
        self.systematic_error = self._spin(
            50.0,
            0.0,
            " %",
            "Relative error of the deposition rate of each material, common to all its layers "
            "(1 sigma).",
        )
        self.drift_error = self._spin(
            50.0,
            0.0,
            " %",
            "Relative linear drift of the deposition rate over the run (1 sigma): the relative "
            "thickness change of the last layer; the first layer is unchanged.",
        )
        form.addRow("Random per layer (%)", self.random_error)
        form.addRow("Rate error per material (%)", self.systematic_error)
        form.addRow("Linear drift over run (%)", self.drift_error)

        self.roughness_box = self._section(
            form,
            "Roughness",
            "Include errors of the interface roughness.",
            False,
        )
        self.roughness_random = self._spin(
            5.0, 0.1, " nm", "Random change of the roughness of each interface (1 sigma)."
        )
        self.roughness_systematic = self._spin(
            5.0,
            0.05,
            " nm",
            "Change of the roughness common to all interfaces of a run (1 sigma).",
        )
        form.addRow("Random per interface (nm)", self.roughness_random)
        form.addRow("Common change (nm)", self.roughness_systematic)

        self.mixing_box = self._section(
            form,
            "Interlayer mixing",
            "Include a mixed layer at every interface between different materials.",
            False,
        )
        self.interlayer_thickness = self._spin(
            20.0,
            0.5,
            " nm",
            "Mean thickness of the mixed layer. A mixed layer with the averaged optical "
            "constants of both neighbours forms at every interface between different "
            "materials; half of its thickness is taken from each neighbour.",
        )
        self.interlayer_random = self._spin(
            100.0,
            20.0,
            " %",
            "Relative random variation of the mixed-layer thickness per interface (1 sigma).",
        )
        form.addRow("Mean mixed-layer thickness (nm)", self.interlayer_thickness)
        form.addRow("Random variation (%)", self.interlayer_random)

        self._sections: tuple[tuple[QCheckBox, tuple[QDoubleSpinBox, ...]], ...] = (
            (self.thickness_box, (self.random_error, self.systematic_error, self.drift_error)),
            (self.roughness_box, (self.roughness_random, self.roughness_systematic)),
            (self.mixing_box, (self.interlayer_thickness, self.interlayer_random)),
        )
        for check, spins in self._sections:
            check.toggled.connect(self._edited)
            for spin in spins:
                spin.valueChanged.connect(self._edited)
        self.preset_combo.currentTextChanged.connect(self._apply_preset)

        self.samples = QSpinBox()
        self.samples.setRange(1, 10000)
        self.samples.setValue(50)
        self.seed = QLineEdit()
        self.seed.setPlaceholderText("random")
        self.samples.setToolTip("Number of randomly perturbed designs to evaluate.")
        self.seed.setToolTip("Integer seed for reproducible results; empty for a random seed.")
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

    def _section(self, form: QFormLayout, title: str, tip: str, checked: bool) -> QCheckBox:
        check = QCheckBox(title)
        check.setChecked(checked)
        check.setToolTip(tip)
        form.addRow(check)
        return check

    @staticmethod
    def _spin(maximum: float, value: float, suffix: str, tip: str) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(0.0, maximum)
        spin.setDecimals(3)
        spin.setSingleStep(0.1)
        spin.setValue(value)
        spin.setSuffix(suffix)
        spin.setToolTip(tip)
        return spin

    def _edited(self) -> None:
        if not self._applying:
            self._set_preset(CUSTOM)

    def _set_preset(self, name: str) -> None:
        self.preset_combo.blockSignals(True)
        self.preset_combo.setCurrentText(name)
        self.preset_combo.blockSignals(False)

    def _apply_preset(self, name: str) -> None:
        preset = PRESETS.get(name)
        if preset is None:
            return
        self._applying = True
        try:
            for (check, spins), values in zip(self._sections, preset, strict=True):
                check.setChecked(values is not None)
                if values is not None:
                    for spin, value in zip(spins, values, strict=True):
                        spin.setValue(value)
        finally:
            self._applying = False

    def error_model(self) -> ErrorModel:
        """Return the error model described by the controls.

        Percent entries are converted to fractions; values of disabled sections are zero.

        Returns:
            The error model.
        """
        t = self.thickness_box.isChecked()
        r = self.roughness_box.isChecked()
        x = self.mixing_box.isChecked()
        return ErrorModel(
            thickness_random=self.random_error.value() / 100.0 if t else 0.0,
            thickness_systematic=self.systematic_error.value() / 100.0 if t else 0.0,
            thickness_drift=self.drift_error.value() / 100.0 if t else 0.0,
            roughness_random=self.roughness_random.value() if r else 0.0,
            roughness_systematic=self.roughness_systematic.value() if r else 0.0,
            interlayer_thickness=self.interlayer_thickness.value() if x else 0.0,
            interlayer_random=self.interlayer_random.value() / 100.0 if x else 0.0,
        )

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
        try:
            errors = self.error_model()
        except ValueError as exc:
            self.errorOccurred.emit(str(exc))
            return
        if errors.is_zero:
            self.errorOccurred.emit("No errors selected")
            return
        seed_text = self.seed.text().strip()
        try:
            seed = int(seed_text) if seed_text else None
        except ValueError:
            self.errorOccurred.emit("The random seed must be an integer")
            return
        samples = self.samples.value()
        state = self._state
        ranges = dict(self.display_ranges)

        def compute(
            progress: Callable[[Any], None], cancel: CancellationToken
        ) -> tuple[ToleranceResult, ToleranceCurves | None, dict[str, tuple[float, float]]]:
            result = tolerance_analysis(
                problem, errors=errors, samples=samples, seed=seed, cancel=cancel, events=progress
            )
            curves = tolerance_curves(state, result, ranges) if ranges else None
            return result, curves, ranges

        self.jobRequested.emit(Job("Tolerance analysis", compute, self._finished))

    def _finished(
        self,
        outcome: tuple[ToleranceResult, ToleranceCurves | None, dict[str, tuple[float, float]]],
    ) -> None:
        result, curves, ranges = outcome
        self.result = result
        m = result.merits
        self.summary.setText(
            f"Nominal figure of merit {result.nominal.merit:.4g}; samples: median "
            f"{np.median(m):.4g}, 10\u201390 % {np.percentile(m, 10):.4g}\u2013"
            f"{np.percentile(m, 90):.4g}, worst {np.max(m):.4g}; "
            f"{int(result.feasible.sum())} of {len(m)} feasible.\n"
            f"Errors: {describe_errors(result.errors)}."
        )
        self.toleranceFinished.emit(result, curves, ranges)

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
            chosen = ask_save_path(self, "Export recipe", pattern, f".{kind}")
            if chosen is None:
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
