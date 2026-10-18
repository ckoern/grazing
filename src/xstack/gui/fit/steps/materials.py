"""Workflow page for the materials of the structure model."""

from PySide6.QtWidgets import QWidget

from xstack.core.materials import MaterialLibrary
from xstack.gui.common.materials import MaterialsPage as CommonMaterialsPage
from xstack.gui.fit.steps.base import FIT_COMMANDS

__all__ = ["MaterialsPage"]


class MaterialsPage(CommonMaterialsPage):
    """Materials of the coating roles and the substrate of the structure model.

    Args:
        library: Material library offering the selectable materials.
        parent: Parent widget.
    """

    help_text = (
        "<b>Materials.</b> Each role of the coating (for example the absorber and the spacer "
        "of a multilayer) and the substrate is assigned a material. Choose a material from the "
        "library or define your own from a chemical formula and density. The density of "
        "deposited films often differs from the bulk value; it can be edited here, but it is "
        "not a fit parameter."
    )

    def __init__(
        self, library: MaterialLibrary | None = None, parent: QWidget | None = None
    ) -> None:
        super().__init__(FIT_COMMANDS, library, parent)
