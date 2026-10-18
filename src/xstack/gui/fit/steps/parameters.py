"""Workflow page for the fit parameters of the structure and of the datasets."""

from typing import Any, ClassVar

from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from xstack.core.parameters import ParameterSet
from xstack.fit.instrument import INSTRUMENT_PARAMETERS
from xstack.fit.project import Command, FitState, set_parameter
from xstack.gui.common.parameter_table import ParameterTable
from xstack.gui.fit.steps.base import FIT_COMMANDS, StepPage, display_unit

__all__ = ["PARAMETER_LABELS", "ParametersPage", "share_command", "unshare_command"]

PARAMETER_LABELS = {
    "scale": "scale",
    "background": "background",
    "offset": "offset",
    "resolution": "resolution",
    "beam_width": "beam width",
    "sample_length": "sample length",
}
"""Readable names of the instrument parameters."""


def _sequence(description: str, commands: list[Command]) -> Command:
    def transform(state: FitState) -> FitState:
        for command in commands:
            state = command.transform(state)
        return state

    return Command(description, transform)


def share_command(parameter: str, reference: str, others: list[str]) -> Command:
    """Create the command that makes an instrument parameter the same in several datasets.

    Args:
        parameter: Instrument parameter name, e.g. ``"offset"``.
        reference: Dataset whose parameter is used.
        others: Datasets that follow the reference; their parameters become expressions and
            are no longer varied.

    Returns:
        One command (a single undo step).
    """
    commands = [
        set_parameter(f"{name}.{parameter}", expr=f"{reference}.{parameter}", vary=False)
        for name in others
    ]
    return _sequence(f"Share {parameter} of {reference!r} between datasets", commands)


def unshare_command(parameter: str, values: dict[str, float]) -> Command:
    """Create the command that makes an instrument parameter independent in several datasets.

    Args:
        parameter: Instrument parameter name, e.g. ``"offset"``.
        values: Current value of the parameter by dataset name, for the datasets whose
            expression is removed; the value is kept.

    Returns:
        One command (a single undo step).
    """
    commands = [
        set_parameter(f"{name}.{parameter}", expr=None, value=value)
        for name, value in values.items()
    ]
    return _sequence(f"Make {parameter} independent", commands)


class ParametersPage(StepPage):
    """Table of all fit parameters and sharing of instrument parameters between datasets.

    Attributes:
        parameter_table: The editable parameter table.
        parameter_combo: Instrument parameter to share.
        reference_combo: Dataset whose parameter the others follow.
        share_button: Applies the sharing.
        independent_button: Removes the sharing.
        free_label: Shows the number of free parameters.
    """

    title = "Parameters"
    help_text: ClassVar[str] = (
        "Choose which parameters the fit varies (<b>Vary</b>) and within which bounds. Structure "
        "parameters come from the model, <b>instrument</b> parameters (scale, background, offset, "
        "resolution, footprint) from each dataset. Share a parameter between datasets with an "
        "<b>expression</b>, e.g. <code>cu.offset</code>. Angles are shown in degrees, beam "
        "width and sample length in mm."
    )

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state: FitState | None = None
        layout = QVBoxLayout(self)
        self.free_label = QLabel()
        self.free_label.setWordWrap(True)
        layout.addWidget(self.free_label)
        self.parameter_table = ParameterTable(FIT_COMMANDS.set_parameter, self, display_unit)
        self.parameter_table.setMinimumHeight(320)
        self.parameter_table.commandRequested.connect(self.commandRequested)
        layout.addWidget(self.parameter_table)
        layout.addWidget(self._build_share_box())

    def _build_share_box(self) -> QGroupBox:
        box = QGroupBox("Share between datasets")
        self.share_box = box
        layout = QHBoxLayout(box)
        self.parameter_combo = QComboBox()
        for name in INSTRUMENT_PARAMETERS:
            self.parameter_combo.addItem(PARAMETER_LABELS[name], name)
        self.reference_combo = QComboBox()
        self.share_button = QPushButton("Share")
        self.share_button.setToolTip(
            "All other datasets use the value of the chosen dataset (expression, not varied)."
        )
        self.independent_button = QPushButton("Make independent")
        self.independent_button.setToolTip(
            "Remove the expressions of this parameter; the datasets keep their current values."
        )
        self.share_button.clicked.connect(self.share)
        self.independent_button.clicked.connect(self.make_independent)
        for widget in (
            QLabel("Use the"),
            self.parameter_combo,
            QLabel("of"),
            self.reference_combo,
            QLabel("for all datasets"),
            self.share_button,
            self.independent_button,
        ):
            layout.addWidget(widget)
        layout.addStretch()
        box.setEnabled(False)
        return box

    def set_state(self, state: Any) -> None:
        """Show the parameters of a fit state.

        Args:
            state: The current `FitState`.
        """
        self._state = state
        try:
            parameters = state.problem().parameters
        except ValueError:
            parameters = state.mirror.parameters if state.mirror is not None else ParameterSet()
        self.parameter_table.set_parameters(parameters)
        self.free_label.setText(f"{len(parameters.free_names())} free parameters")
        names = [d.name for d in state.datasets]
        if names != [self.reference_combo.itemText(i) for i in range(self.reference_combo.count())]:
            current = self.reference_combo.currentText()
            self.reference_combo.clear()
            self.reference_combo.addItems(names)
            if current in names:
                self.reference_combo.setCurrentText(current)
        self.share_box.setEnabled(len(names) >= 2)

    def set_preview(self, preview: Any) -> None:
        """Show the number of free parameters of the latest preview.

        Args:
            preview: The latest `FitPreview`.
        """
        if preview is not None and preview.free:
            self.free_label.setText(f"{preview.free} free parameters")

    def share(self) -> None:
        """Make the selected instrument parameter follow the reference dataset."""
        if self._state is None or len(self._state.datasets) < 2:
            return
        reference = self.reference_combo.currentText()
        others = [d.name for d in self._state.datasets if d.name != reference]
        self.commandRequested.emit(
            share_command(self.parameter_combo.currentData(), reference, others)
        )

    def make_independent(self) -> None:
        """Remove the expressions of the selected instrument parameter, keeping the values."""
        if self._state is None:
            return
        parameter = self.parameter_combo.currentData()
        try:
            resolved = self._state.problem().resolve()
        except (ValueError, KeyError) as exc:
            self.errorOccurred.emit(str(exc))
            return
        values = {
            d.name: resolved[f"{d.name}.{parameter}"]
            for d in self._state.datasets
            if d.parameters[f"{d.name}.{parameter}"].expr is not None
        }
        if values:
            self.commandRequested.emit(unshare_command(parameter, values))
