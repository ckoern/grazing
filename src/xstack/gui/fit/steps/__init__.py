"""Pages of the fit workflow."""

from xstack.gui.fit.steps.base import FIT_COMMANDS, StepPage, display_unit
from xstack.gui.fit.steps.data import DataPage
from xstack.gui.fit.steps.fit import FitPage
from xstack.gui.fit.steps.materials import MaterialsPage
from xstack.gui.fit.steps.parameters import ParametersPage
from xstack.gui.fit.steps.results import ResultsPage
from xstack.gui.fit.steps.structure import StructurePage

__all__ = [
    "FIT_COMMANDS",
    "DataPage",
    "FitPage",
    "MaterialsPage",
    "ParametersPage",
    "ResultsPage",
    "StepPage",
    "StructurePage",
    "display_unit",
]
