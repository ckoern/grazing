"""Workflow page for the coating type, its options and parameters of the input design."""

from typing import ClassVar

from PySide6.QtWidgets import QWidget

from xstack.core.structure.coatings.graded import DepthGraded
from xstack.design.project import Command
from xstack.design.project.commands import set_parameter
from xstack.gui.common import coating as common
from xstack.gui.common.coating import (
    COATING_TYPES,
    _restack,
    convert_coating,
    has_constant_gamma,
)
from xstack.gui.design.steps.base import DESIGN_COMMANDS

__all__ = [
    "COATING_TYPES",
    "CoatingPage",
    "_restack",
    "convert_coating",
    "has_constant_gamma",
    "set_constant_gamma",
]


class CoatingPage(common.CoatingPage):
    """Coating page of the design workflow; edits the input design.

    Args:
        parent: Parent widget.
    """

    help_text: ClassVar[str] = (
        "Choose the <b>coating type</b>: a periodic multilayer, a depth-graded multilayer "
        "whose period changes with depth, a single layer, a stack of N layers of different "
        "materials, or a <b>combined coating</b> of several such blocks stacked from the top; "
        "the parameters of a block carry its name as prefix (<tt>graded.d_min</tt>). This page "
        "defines the <b>input (start) design</b>; results of the "
        "optimization are stored as candidates and never overwrite it. In the parameter table, "
        "a parameter with <b>Vary</b> checked is explored by the optimizer between its "
        "<b>Min</b> and <b>Max</b>; otherwise it stays fixed at its value. An "
        "<b>expression</b> computes a parameter from other parameters, for example "
        "<tt>2*d_min</tt>, and then it is neither fixed nor varied."
    )

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(DESIGN_COMMANDS, parent)


def set_constant_gamma(coating: DepthGraded, constant: bool) -> Command:
    """Return a design command that switches between one Γ and a linear Γ gradient.

    Args:
        coating: The current depth-graded coating.
        constant: Use one Γ for all graded bilayers.

    Returns:
        The command.
    """
    command: Command = common.set_constant_gamma(coating, constant, set_parameter)
    return command
