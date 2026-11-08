"""Common base of the fit workflow pages and the fit's structure commands."""

from grazing.core.parameters import Parameter
from grazing.fit.project import set_coating, set_mirror, set_optical_constants, set_parameter
from grazing.fit.report import display_value
from grazing.gui.common.pages import StepPage, StructureCommands

__all__ = ["FIT_COMMANDS", "StepPage", "display_unit"]

FIT_COMMANDS = StructureCommands(set_mirror, set_coating, set_parameter, set_optical_constants)
"""Commands of the fit project used by the shared structure pages."""


def display_unit(parameter: Parameter) -> tuple[float, str]:
    """Return the factor from the internal to the shown value and the shown unit.

    Angles are shown in degrees and beam and sample dimensions in mm.

    Args:
        parameter: The parameter.

    Returns:
        Factor and unit label.
    """
    factor, unit = display_value(parameter, 1.0)
    return factor, unit
