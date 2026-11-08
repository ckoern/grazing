"""Design problems, design-specific optimizers, staged optimization, scans and trade-offs."""

from grazing.design.optim.pipeline import Pipeline, PipelineResult, StageResult
from grazing.design.optim.problem import DesignProblem, Evaluation
from grazing.design.optim.scan import GridScan, ScanResult, scan
from grazing.design.optim.stages import BraggOutline, Stage
from grazing.design.optim.tradeoff import TradeoffPoint, TradeoffResult, time_tradeoff

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
