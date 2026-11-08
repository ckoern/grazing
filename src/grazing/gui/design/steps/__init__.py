"""Pages of the design workflow."""

from grazing.gui.design.steps.analyze import AnalyzePage
from grazing.gui.design.steps.base import StepPage
from grazing.gui.design.steps.coating import CoatingPage
from grazing.gui.design.steps.constraints import ConstraintsPage
from grazing.gui.design.steps.estimate import EstimatePage
from grazing.gui.design.steps.goal import GoalPage
from grazing.gui.design.steps.materials import MaterialDialog, MaterialsPage
from grazing.gui.design.steps.optimize import OptimizePage
from grazing.gui.design.steps.process import ProcessPage
from grazing.gui.design.steps.scan import ScanPage
from grazing.gui.design.steps.tradeoff import TradeoffPage

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
