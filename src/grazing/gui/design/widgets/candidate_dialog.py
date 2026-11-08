"""Dialog showing the coating parameters of one stored candidate."""

import math

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
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

from grazing.core.parameters import Parameter
from grazing.core.structure.coating import Mirror
from grazing.design.project.project import Candidate
from grazing.gui.design.materials_summary import materials_text

__all__ = ["CandidateParametersDialog"]

_COLUMNS = ("Parameter", "Value", "Unit", "Range", "Description")
_DASH = "—"
_INFINITY = "∞"


def _number(value: float, integer: bool = False) -> str:
    if math.isinf(value):
        return _INFINITY if value > 0 else f"-{_INFINITY}"
    if integer:
        return str(round(value))
    return f"{value:.5g}"


def _range(p: Parameter) -> str:
    if p.expr is not None:
        return f"= {p.expr}"
    if not p.vary:
        return "fixed"
    return f"{_number(p.min, p.integer)} – {_number(p.max, p.integer)}"


def _row(p: Parameter) -> tuple[str, str, str, str, str]:
    return (p.name, _number(p.value, p.integer), p.unit, _range(p), p.description)


def _stack_summary(mirror: Mirror) -> str:
    try:
        stack = mirror.stack()
    except ValueError as exc:
        return f"not available ({exc})"
    return f"{stack.num_layers} layers, total thickness {stack.total_thickness:.5g} nm"


class CandidateParametersDialog(QDialog):
    """Non-modal dialog listing the coating parameters of a candidate.

    Shows the candidate's name, origin, coating type, materials and substrate, a read-only table
    of all coating parameters with value, unit, range and description, and a summary of the
    figure of merit, objective, coating time and layer structure. The table can be copied to the
    clipboard as tab-separated text.

    Args:
        candidate: The candidate to show.
        parent: Parent widget.
    """

    def __init__(self, candidate: Candidate, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._candidate = candidate
        self.setWindowTitle(f"Parameters – {candidate.name}")
        self.resize(560, 520)

        self._name_label = QLabel(candidate.name)
        font = self._name_label.font()
        font.setBold(True)
        self._name_label.setFont(font)
        self._header = QLabel(self._header_text(candidate))
        self._header.setWordWrap(True)
        self._header.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self._table = QTableWidget(0, len(_COLUMNS))
        self._table.setHorizontalHeaderLabels(list(_COLUMNS))
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setAlternatingRowColors(False)
        self._table.setShowGrid(False)
        self._table.setFrameShape(QFrame.Shape.NoFrame)
        self._table.verticalHeader().setVisible(False)
        self._table.verticalHeader().setDefaultSectionSize(26)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(True)
        header.setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self._rows = [_row(p) for p in candidate.mirror.parameters.values()]
        self._fill()

        self._summary = QLabel(self._summary_text(candidate))
        self._summary.setWordWrap(True)
        self._summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self._copy_button = QPushButton("Copy as text")
        self._copy_button.setToolTip("Copy the parameter table as tab-separated text.")
        self._copy_button.clicked.connect(self._copy)
        self._close_button = QPushButton("Close")
        self._close_button.clicked.connect(self.close)

        buttons = QHBoxLayout()
        buttons.addWidget(self._copy_button)
        buttons.addStretch(1)
        buttons.addWidget(self._close_button)

        layout = QVBoxLayout(self)
        layout.addWidget(self._name_label)
        layout.addWidget(self._header)
        layout.addWidget(self._table, 1)
        layout.addWidget(self._summary)
        layout.addLayout(buttons)

    def table(self) -> QTableWidget:
        """Return the parameter table."""
        return self._table

    def summary_text(self) -> str:
        """Return the text of the summary block."""
        return self._summary.text()

    def header_text(self) -> str:
        """Return the text of the header: name, source, creation time, coating and materials."""
        return f"{self._name_label.text()}\n{self._header.text()}"

    def copy_button(self) -> QPushButton:
        """Return the button that copies the table to the clipboard."""
        return self._copy_button

    def close_button(self) -> QPushButton:
        """Return the button that closes the dialog."""
        return self._close_button

    def table_as_text(self) -> str:
        """Return the table as tab-separated text with a header line."""
        lines = ["\t".join(_COLUMNS)]
        lines.extend("\t".join(row) for row in self._rows)
        return "\n".join(lines)

    def _fill(self) -> None:
        self._table.setRowCount(len(self._rows))
        for r, row in enumerate(self._rows):
            for c, text in enumerate(row):
                self._table.setItem(r, c, QTableWidgetItem(text))

    def _copy(self) -> None:
        QGuiApplication.clipboard().setText(self.table_as_text())

    @staticmethod
    def _header_text(candidate: Candidate) -> str:
        mirror = candidate.mirror
        created = candidate.created.replace("T", " ")[:19] if candidate.created else _DASH
        materials = materials_text(mirror.coating)
        substrate = f"{mirror.substrate.name} (roughness {mirror.substrate_roughness:.5g} nm)"
        return (
            f"Source: {candidate.source}, created: {created}\n"
            f"Coating: {type(mirror.coating).type_name}, materials {materials}, "
            f"substrate {substrate}"
        )

    @staticmethod
    def _summary_text(candidate: Candidate) -> str:
        merit = f"{candidate.merit:.5g}" if candidate.merit is not None else _DASH
        objective = f"{candidate.objective:.5g}" if candidate.objective is not None else _DASH
        time = candidate.coating_time
        coating_time = f"{time / 3600:.4g} h" if time is not None and math.isfinite(time) else _DASH
        return "\n".join(
            [
                f"Figure of merit: {merit}",
                f"Objective: {objective}",
                f"Coating time: {coating_time}",
                f"Structure: {_stack_summary(candidate.mirror)}",
            ]
        )
