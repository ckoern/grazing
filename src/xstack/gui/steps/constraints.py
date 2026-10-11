"""Workflow page for manufacturing constraints."""

from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from xstack.gui.forms import DataclassForm
from xstack.gui.preview import Preview
from xstack.gui.steps.base import StepPage
from xstack.manufacturing.constraints import Constraint, ConstraintResult, constraints
from xstack.project import ProjectState
from xstack.project.commands import add_constraint, remove_constraint, replace_constraint

__all__ = ["ConstraintsPage"]


class ConstraintsPage(StepPage):
    """List of constraints with live status, an editor for the selected one, add/remove.

    Args:
        parent: Parent widget.
    """

    title = "Constraints"

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._constraints: tuple[Constraint, ...] = ()
        self._results: list[ConstraintResult | None] = []
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<b>Manufacturing constraints</b>"))
        self.list = QListWidget()
        self.list.currentRowChanged.connect(self._select)
        layout.addWidget(self.list)
        buttons = QHBoxLayout()
        self.type_combo = QComboBox()
        for name in constraints:
            self.type_combo.addItem(name.replace("_", " "), name)
        self.add_button = QPushButton("Add")
        self.remove_button = QPushButton("Remove")
        buttons.addWidget(self.type_combo)
        buttons.addWidget(self.add_button)
        buttons.addWidget(self.remove_button)
        layout.addLayout(buttons)
        self.add_button.clicked.connect(self._add)
        self.remove_button.clicked.connect(self._remove)
        self.form = DataclassForm()
        self.form.valueChanged.connect(self._edited)
        self.form.invalidInput.connect(self.errorOccurred)
        layout.addWidget(self.form)
        layout.addStretch()

    def set_state(self, state: ProjectState) -> None:
        """Show the constraints of a project state.

        Args:
            state: The current project state.
        """
        if state.constraints == self._constraints:
            return
        row = self.list.currentRow()
        self._constraints = state.constraints
        self._results = [None] * len(self._constraints)
        self._refresh()
        if self._constraints:
            self.list.setCurrentRow(min(max(row, 0), len(self._constraints) - 1))
        else:
            self.form.set_value(None)

    def set_preview(self, preview: Preview) -> None:
        """Show the constraint status of the latest evaluation.

        Args:
            preview: The latest preview.
        """
        if preview.evaluation is None or preview.state.constraints != self._constraints:
            return
        self._results = [r for _, r in preview.evaluation.constraints]
        self._refresh()

    def _refresh(self) -> None:
        current = self.list.currentRow()
        self.list.blockSignals(True)
        self.list.clear()
        for c, r in zip(self._constraints, self._results, strict=True):
            mark = "…" if r is None else ("✓" if r.satisfied else "✗")
            message = "" if r is None else f" — {r.message}"
            item = QListWidgetItem(
                f"{mark} {c.type_name.replace('_', ' ')} [{c.severity}]{message}"
            )
            self.list.addItem(item)
        self.list.setCurrentRow(current)
        self.list.blockSignals(False)

    def _select(self, row: int) -> None:
        self.form.set_value(self._constraints[row] if 0 <= row < len(self._constraints) else None)

    def _edited(self, constraint: Constraint) -> None:
        row = self.list.currentRow()
        if 0 <= row < len(self._constraints):
            self.commandRequested.emit(replace_constraint(row, constraint))

    def _add(self) -> None:
        cls = constraints.get(self.type_combo.currentData())
        try:
            constraint = cls()
        except (TypeError, ValueError) as exc:
            self.errorOccurred.emit(f"Cannot create constraint: {exc}")
            return
        self.commandRequested.emit(add_constraint(constraint))

    def _remove(self) -> None:
        row = self.list.currentRow()
        if 0 <= row < len(self._constraints):
            self.commandRequested.emit(remove_constraint(row))
