"""Pages of the fit workflow."""

from grazing.gui.fit.steps.base import FIT_COMMANDS, StepPage, display_unit
from grazing.gui.fit.steps.data import DataPage
from grazing.gui.fit.steps.fit import FitPage
from grazing.gui.fit.steps.materials import MaterialsPage
from grazing.gui.fit.steps.parameters import ParametersPage
from grazing.gui.fit.steps.results import ResultsPage
from grazing.gui.fit.steps.structure import StructurePage

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
