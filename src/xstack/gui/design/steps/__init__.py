"""Pages of the design workflow."""

from xstack.gui.design.steps.analyze import AnalyzePage
from xstack.gui.design.steps.base import StepPage
from xstack.gui.design.steps.coating import CoatingPage
from xstack.gui.design.steps.constraints import ConstraintsPage
from xstack.gui.design.steps.estimate import EstimatePage
from xstack.gui.design.steps.goal import GoalPage
from xstack.gui.design.steps.materials import MaterialDialog, MaterialsPage
from xstack.gui.design.steps.optimize import OptimizePage
from xstack.gui.design.steps.process import ProcessPage
from xstack.gui.design.steps.scan import ScanPage
from xstack.gui.design.steps.tradeoff import TradeoffPage

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
]
