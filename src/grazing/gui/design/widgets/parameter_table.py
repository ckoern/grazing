"""Editable table of the coating parameters of the design."""

from PySide6.QtWidgets import QWidget

from grazing.design.project.commands import set_parameter
from grazing.gui.common import parameter_table as common

__all__ = ["ParameterTable", "ParameterTableModel"]


class ParameterTableModel(common.ParameterTableModel):
    """Parameter table model emitting design commands.

    Args:
        parent: Parent widget.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(set_parameter, parent)


class ParameterTable(common.ParameterTable):
    """Parameter table emitting design commands.

    Args:
        parent: Parent widget.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(set_parameter, parent)
