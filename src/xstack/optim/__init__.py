"""Design problems, optimizers and staged optimization."""

from xstack.optim.events import CancellationToken, EventSink, ProgressEvent
from xstack.optim.optimizers import (
    GLOBAL_METHODS,
    LmfitOptimizer,
    OptimizationResult,
    Optimizer,
    optimizers,
    register_optimizer,
)
from xstack.optim.pipeline import Pipeline, PipelineResult, StageResult
from xstack.optim.problem import DesignProblem, Evaluation
from xstack.optim.scan import GridScan, ScanResult, scan
from xstack.optim.stages import BraggOutline, Stage

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
    "optimizers",
    "register_optimizer",
    "scan",
]
