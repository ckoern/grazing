"""Design problems, optimizers and staged optimization."""

from xstack.design.optim.events import CancellationToken, EventSink, ProgressEvent
from xstack.design.optim.optimizers import (
    GLOBAL_METHODS,
    LmfitOptimizer,
    OptimizationResult,
    Optimizer,
    optimizers,
    register_optimizer,
)
from xstack.design.optim.pipeline import Pipeline, PipelineResult, StageResult
from xstack.design.optim.problem import DesignProblem, Evaluation
from xstack.design.optim.scan import GridScan, ScanResult, scan
from xstack.design.optim.stages import BraggOutline, Stage
from xstack.design.optim.tradeoff import TradeoffPoint, TradeoffResult, time_tradeoff

__all__ = [
    "GLOBAL_METHODS",
    "BraggOutline",
    "CancellationToken",
    "DesignProblem",
    "Evaluation",
    "EventSink",
    "GridScan",
    "LmfitOptimizer",
    "OptimizationResult",
    "Optimizer",
    "Pipeline",
    "PipelineResult",
    "ProgressEvent",
    "ScanResult",
    "Stage",
    "StageResult",
    "TradeoffPoint",
    "TradeoffResult",
    "optimizers",
    "register_optimizer",
    "scan",
    "time_tradeoff",
]
