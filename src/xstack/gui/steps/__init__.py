"""Pages of the design workflow."""

from xstack.gui.steps.analyze import AnalyzePage
from xstack.gui.steps.base import StepPage
from xstack.gui.steps.coating import CoatingPage, convert_coating
from xstack.gui.steps.constraints import ConstraintsPage
from xstack.gui.steps.estimate import EstimatePage
from xstack.gui.steps.goal import GoalPage
from xstack.gui.steps.materials import MaterialDialog, MaterialsPage
from xstack.gui.steps.optimize import OptimizePage
from xstack.gui.steps.process import ProcessPage
from xstack.gui.steps.scan import ScanPage
from xstack.gui.steps.tradeoff import TradeoffPage

__all__ = [
    "AnalyzePage",
    "CoatingPage",
    "ConstraintsPage",
    "EstimatePage",
    "GoalPage",
    "MaterialDialog",
    "MaterialsPage",
    "OptimizePage",
    "ProcessPage",
    "ScanPage",
    "StepPage",
    "TradeoffPage",
    "convert_coating",
]
