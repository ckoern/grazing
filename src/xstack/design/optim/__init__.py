"""Design problems, design-specific optimizers, staged optimization, scans and trade-offs."""

from xstack.design.optim.pipeline import Pipeline, PipelineResult, StageResult
from xstack.design.optim.problem import DesignProblem, Evaluation
from xstack.design.optim.scan import GridScan, ScanResult, scan
from xstack.design.optim.stages import BraggOutline, Stage
from xstack.design.optim.tradeoff import TradeoffPoint, TradeoffResult, time_tradeoff

__all__ = [
    "BraggOutline",
    "DesignProblem",
    "Evaluation",
    "GridScan",
    "Pipeline",
    "PipelineResult",
    "ScanResult",
    "Stage",
    "StageResult",
    "TradeoffPoint",
    "TradeoffResult",
    "scan",
    "time_tradeoff",
]
