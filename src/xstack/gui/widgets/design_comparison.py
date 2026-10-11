"""Comparison of the coating parameters of the selected candidate with the input design."""

import math

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from xstack.parameters import Parameter
from xstack.project.commands import promote_candidate, set_parameter
from xstack.project.project import Candidate, ProjectState
from xstack.structure.coating import Mirror

__all__ = ["DesignComparisonView"]

_COLUMNS = ("Parameter", "Unit", "Input", "Selected", "Change")
_NAME, _UNIT, _INPUT, _SELECTED, _CHANGE = range(len(_COLUMNS))
_NO_SELECTION_HINT = (
    "Select a candidate in the candidates list to compare it with the input design."
)
_FREE_MARK = "●"
_FREE_TOOLTIP = "optimized (free) parameter"
_DASH = "—"


def _format(value: float, integer: bool) -> str:
    if integer:
        return str(round(value))
    return f"{value:.5g}"


def _change(candidate: float, base: float | None, integer: bool) -> str:
    if base is None:
        return "n/a"
    if candidate == base:
        return _DASH
    if base == 0:
        return f"n/a (Δ = {_format(candidate - base, integer)})"
    return f"{(candidate - base) / abs(base) * 100:+.3g} %"


def _stack_summary(mirror: Mirror) -> str:
    try:
        stack = mirror.stack()
    except ValueError as exc:
        return f"not available ({exc})"
    return f"{stack.num_layers} layers, total thickness {stack.total_thickness:.5g} nm"


class DesignComparisonView(QWidget):
    """Table comparing the parameters of the selected candidate with the input design.

    Shows one row per coating parameter of the selected candidate, a summary of both designs and
    buttons to use the candidate as start design or to copy single values to the input design.

    Signals:
        commandRequested(Command): A change of the project that the controller should execute.
    """

    commandRequested = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = ProjectState()
        self._candidate: Candidate | None = None
        self._copyable: dict[str, Parameter] = {}

        self._header = QLabel()
        self._header.setWordWrap(True)
        font = self._header.font()
        font.setBold(True)
        self._header.setFont(font)
        self._info = QLabel()
        self._info.setWordWrap(True)

        self._table = QTableWidget(0, len(_COLUMNS))
        self._table.setHorizontalHeaderLabels(list(_COLUMNS))
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._table.setAlternatingRowColors(False)
        self._table.setShowGrid(False)
        self._table.setFrameShape(QFrame.Shape.NoFrame)
        self._table.verticalHeader().setVisible(False)
        self._table.verticalHeader().setDefaultSectionSize(28)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(True)
        header.setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self._table.itemSelectionChanged.connect(self._update_buttons)

        self._summary = QLabel()
        self._summary.setWordWrap(True)
        self._summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self._promote_button = QPushButton("Use as start design")
        self._promote_button.setToolTip(
            "Replace the input design by the selected candidate. "
            "The current input design is kept in the undo history."
        )
        self._promote_button.clicked.connect(self._on_promote)
        self._copy_button = QPushButton("Copy selected value to input")
        self._copy_button.setToolTip(
            "Set the input design's parameter in the selected row to the candidate's value. "
            "The bounds are widened if the value lies outside them."
        )
        self._copy_button.clicked.connect(self._on_copy)

        buttons = QHBoxLayout()
        buttons.addWidget(self._promote_button)
        buttons.addWidget(self._copy_button)
        buttons.addStretch(1)

        layout = QVBoxLayout(self)
        layout.addWidget(self._header)
        layout.addWidget(self._info)
        layout.addWidget(self._table, 1)
        layout.addWidget(self._summary)
        layout.addLayout(buttons)
        self.set_state(self._state)

    def header_text(self) -> str:
        """Return the text of the header line."""
        return self._header.text()

    def info_text(self) -> str:
        """Return the text of the info line, empty if there is nothing to point out."""
        return self._info.text()

    def summary_text(self) -> str:
        """Return the text of the summary block."""
        return self._summary.text()

    def table(self) -> QTableWidget:
        """Return the comparison table."""
        return self._table

    def promote_button(self) -> QPushButton:
        """Return the button that uses the candidate as start design."""
        return self._promote_button

    def copy_button(self) -> QPushButton:
        """Return the button that copies the selected value to the input design."""
        return self._copy_button

    def set_state(self, state: ProjectState) -> None:
        """Show the selected candidate of a project state next to its input design.

        Args:
            state: The project state.
        """
        self._state = state
        self._candidate = state.selected_candidate()
        self._table.clearContents()
        self._table.setRowCount(0)
        self._copyable = {}
        candidate = self._candidate
        if candidate is None:
            self._header.setText(_NO_SELECTION_HINT)
            self._info.setText("")
            self._summary.setText("")
            self._update_buttons()
            return
        self._header.setText(f"{candidate.name} ({candidate.source}) vs. input design")
        self._fill(candidate, state.mirror)
        self._summary.setText(self._summary_text(candidate, state.mirror))
        self._update_buttons()

    def _fill(self, candidate: Candidate, base_mirror: Mirror | None) -> None:
        cand_coating = candidate.mirror.coating
        base_params = None
        same_type = False
        if base_mirror is not None:
            base_params = base_mirror.coating.parameters
            same_type = type(base_mirror.coating) is type(cand_coating)
        if base_mirror is None:
            self._info.setText("The project has no input design.")
        elif not same_type:
            self._info.setText(
                f"The coating type differs ({type(cand_coating).__name__} vs. "
                f"{type(base_mirror.coating).__name__} in the input design); "
                "values cannot be compared."
            )
        else:
            self._info.setText("")
        comparable = base_params if same_type else None

        params = list(cand_coating.parameters.values())
        self._table.setRowCount(len(params))
        for row, p in enumerate(params):
            base = comparable.get(p.name) if comparable is not None else None
            if base is not None:
                self._copyable[p.name] = base
            self._set_row(row, p, base, comparable is not None)

    def _set_row(self, row: int, p: Parameter, base: Parameter | None, comparable: bool) -> None:
        free_in_input = base is not None and base.is_free
        name = QTableWidgetItem(f"{p.name} {_FREE_MARK}" if free_in_input else p.name)
        tooltip = p.description or p.name
        name.setToolTip(f"{tooltip}\n{_FREE_TOOLTIP}" if free_in_input else tooltip)
        unit = QTableWidgetItem(p.unit)
        input_text = _format(base.value, p.integer) if base is not None else _DASH
        input_item = QTableWidgetItem(input_text)
        selected = QTableWidgetItem(_format(p.value, p.integer))
        if comparable:
            change_text = _change(p.value, base.value if base is not None else None, p.integer)
        else:
            change_text = _DASH
        change = QTableWidgetItem(change_text)
        differs = base is not None and base.value != p.value
        if differs:
            font = selected.font()
            font.setBold(True)
            selected.setFont(font)
        for column, item in enumerate((name, unit, input_item, selected, change)):
            self._table.setItem(row, column, item)

    @staticmethod
    def _summary_text(candidate: Candidate, base_mirror: Mirror | None) -> str:
        lines = [f"Candidate {candidate.name}:"]
        lines.append(
            "  Figure of merit: "
            + (f"{candidate.merit:.5g}" if candidate.merit is not None else _DASH)
        )
        lines.append(
            "  Objective: "
            + (f"{candidate.objective:.5g}" if candidate.objective is not None else _DASH)
        )
        if candidate.coating_time is not None and math.isfinite(candidate.coating_time):
            lines.append(f"  Coating time: {candidate.coating_time / 3600:.4g} h")
        else:
            lines.append(f"  Coating time: {_DASH}")
        lines.append("  Structure: " + _stack_summary(candidate.mirror))
        if base_mirror is not None:
            lines.append("Input design:")
            lines.append("  Structure: " + _stack_summary(base_mirror))
        return "\n".join(lines)

    def _selected_name(self) -> str | None:
        rows = self._table.selectionModel().selectedRows()
        if not rows:
            return None
        return self._parameter_names[rows[0].row()]

    @property
    def _parameter_names(self) -> list[str]:
        if self._candidate is None:
            return []
        return list(self._candidate.mirror.coating.parameters)

    def _update_buttons(self) -> None:
        self._promote_button.setEnabled(self._candidate is not None)
        name = self._selected_name()
        self._copy_button.setEnabled(name is not None and name in self._copyable)

    def _on_promote(self) -> None:
        if self._candidate is not None:
            self.commandRequested.emit(promote_candidate(self._candidate.name))

    def _on_copy(self) -> None:
        name = self._selected_name()
        if self._candidate is None or name is None or name not in self._copyable:
            return
        value = self._candidate.mirror.coating.parameters[name].value
        base = self._copyable[name]
        changes: dict[str, float] = {"value": value}
        if value < base.min:
            changes["min"] = value
        if value > base.max:
            changes["max"] = value
        self.commandRequested.emit(set_parameter(name, **changes))
