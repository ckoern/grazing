"""Common base of the design workflow pages and the design's structure commands."""

from grazing.design.project.commands import (
    set_coating,
    set_mirror,
    set_optical_constants,
    set_parameter,
)
from grazing.gui.common.pages import StepPage, StructureCommands

__all__ = ["DESIGN_COMMANDS", "StepPage"]

DESIGN_COMMANDS = StructureCommands(set_mirror, set_coating, set_parameter, set_optical_constants)
"""Commands of the design project used by the shared structure pages."""
