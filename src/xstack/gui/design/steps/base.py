"""Common base of the design workflow pages and the design's structure commands."""

from xstack.design.project.commands import set_coating, set_mirror, set_parameter
from xstack.gui.common.pages import StepPage, StructureCommands

__all__ = ["DESIGN_COMMANDS", "StepPage"]

DESIGN_COMMANDS = StructureCommands(set_mirror, set_coating, set_parameter)
"""Commands of the design project used by the shared structure pages."""
