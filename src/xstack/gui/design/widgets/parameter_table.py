"""Editable table of coating parameters."""

import math
from typing import Any

from PySide6.QtCore import QAbstractTableModel, QModelIndex, QPersistentModelIndex, Qt, Signal
from PySide6.QtWidgets import QFrame, QHeaderView, QTableView, QWidget

from xstack.core.parameters import ParameterSet
from xstack.design.project.commands import set_parameter

__all__ = ["ParameterTable", "ParameterTableModel"]

type _Index = QModelIndex | QPersistentModelIndex

_COLUMNS = ("Parameter", "Value", "Unit", "Vary", "Min", "Max", "Expression", "Description")
_NAME, _VALUE, _UNIT, _VARY, _MIN, _MAX, _EXPR, _DESC = range(len(_COLUMNS))


def _format(value: float, integer: bool) -> str:
    if math.isinf(value):
        return "∞" if value > 0 else "-∞"
    if integer:
        return str(round(value))
    return f"{value:.6g}"


def _parse(text: str) -> float:
    text = text.strip().replace("∞", "inf")
    if text in ("", "none"):
        raise ValueError("a number is required")
    return float(text)


class ParameterTableModel(QAbstractTableModel):
    """Qt model presenting a `ParameterSet`; edits are emitted as core commands.

    Signals:
        commandRequested(Command): The user edited a cell; the command applies the edit.
    """

    commandRequested = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._params = ParameterSet()

    def set_parameters(self, params: ParameterSet) -> None:
        """Show a parameter set.

        Args:
            params: The parameters.
        """
        if params == self._params:
            return
        same_names = list(params) == list(self._params)
        self._params = params
        if same_names and len(params):
            self.dataChanged.emit(self.index(0, 0), self.index(len(params) - 1, len(_COLUMNS) - 1))
        else:
            self.beginResetModel()
            self.endResetModel()

    def parameters(self) -> ParameterSet:
        """Return the shown parameter set."""
        return self._params

    def rowCount(self, parent: _Index = QModelIndex()) -> int:  # noqa: B008
        """Return the number of parameters."""
        return 0 if parent.isValid() else len(self._params)

    def columnCount(self, parent: _Index = QModelIndex()) -> int:  # noqa: B008
        """Return the number of columns."""
        return 0 if parent.isValid() else len(_COLUMNS)

    def headerData(
        self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole
    ) -> Any:
        """Return the column titles."""
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return _COLUMNS[section]
        return None

    def _parameter(self, index: _Index) -> Any:
        return list(self._params.values())[index.row()]

    @staticmethod
    def _tooltip(p: Any, column: int) -> str:
        if column == _VARY:
            return "Let the optimizer vary this parameter between Min and Max."
        if column in (_MIN, _MAX):
            return "Bound of the range explored by the optimizer."
        if column == _EXPR:
            return "Optional expression in terms of other parameters; overrides the value."
        return p.description or p.name

    def data(self, index: _Index, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        """Return cell contents."""
        if not index.isValid():
            return None
        p = self._parameter(index)
        column = index.column()
        if role == Qt.ItemDataRole.CheckStateRole and column == _VARY:
            return Qt.CheckState.Checked if p.vary else Qt.CheckState.Unchecked
        if role == Qt.ItemDataRole.ToolTipRole:
            return self._tooltip(p, column)
        if role not in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole):
            return None
        if column == _NAME:
            return p.name
        if column == _VALUE:
            return _format(p.value, p.integer)
        if column == _UNIT:
            return p.unit
        if column == _MIN:
            return _format(p.min, p.integer)
        if column == _MAX:
            return _format(p.max, p.integer)
        if column == _EXPR:
            return p.expr or ""
        if column == _DESC:
            return p.description
        return None

    def flags(self, index: _Index) -> Qt.ItemFlag:
        """Return which cells are editable."""
        base = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        column = index.column()
        if column == _VARY:
            return base | Qt.ItemFlag.ItemIsUserCheckable
        if column in (_VALUE, _MIN, _MAX, _EXPR):
            return base | Qt.ItemFlag.ItemIsEditable
        return base

    def setData(self, index: _Index, value: Any, role: int = Qt.ItemDataRole.EditRole) -> bool:
        """Translate an edit into a command and emit `commandRequested`."""
        if not index.isValid():
            return False
        p = self._parameter(index)
        column = index.column()
        try:
            if column == _VARY and role == Qt.ItemDataRole.CheckStateRole:
                checked = Qt.CheckState(value) == Qt.CheckState.Checked
                command = set_parameter(p.name, vary=checked)
            elif role != Qt.ItemDataRole.EditRole:
                return False
            elif column == _VALUE:
                command = set_parameter(p.name, value=_parse(str(value)))
            elif column == _MIN:
                command = set_parameter(p.name, min=_parse(str(value)))
            elif column == _MAX:
                command = set_parameter(p.name, max=_parse(str(value)))
            elif column == _EXPR:
                text = str(value).strip()
                command = set_parameter(p.name, expr=text or None)
            else:
                return False
        except ValueError:
            return False
        self.commandRequested.emit(command)
        return True


class ParameterTable(QTableView):
    """Table view for editing a parameter set.

    Signals:
        commandRequested(Command): An edit that should be executed by the controller.
    """

    commandRequested = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.table_model = ParameterTableModel(self)
        self.setModel(self.table_model)
        self.table_model.commandRequested.connect(self.commandRequested)
        self.verticalHeader().setVisible(False)
        header = self.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(True)
        header.setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.setAlternatingRowColors(True)
        self.setShowGrid(False)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.verticalHeader().setDefaultSectionSize(28)

    def set_parameters(self, params: ParameterSet) -> None:
        """Show a parameter set.

        Args:
            params: The parameters.
        """
        self.table_model.set_parameters(params)
