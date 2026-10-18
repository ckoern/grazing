"""Workflow page for the materials of the design."""

from PySide6.QtWidgets import QWidget

from xstack.core.materials import MaterialLibrary
from xstack.gui.common import materials as common
from xstack.gui.common.materials import MaterialDialog
from xstack.gui.design.steps.base import DESIGN_COMMANDS

__all__ = ["MaterialDialog", "MaterialsPage"]


class MaterialsPage(common.MaterialsPage):
    """Materials page of the design workflow.

    Args:
        library: Material library offering the selectable materials.
        parent: Parent widget.
    """

    def __init__(
        self, library: MaterialLibrary | None = None, parent: QWidget | None = None
    ) -> None:
        super().__init__(DESIGN_COMMANDS, library, parent)
