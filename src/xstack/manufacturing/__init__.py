"""Deposition process models, constraints and tolerance analysis."""

from xstack.manufacturing.constraints import (
    Constraint,
    ConstraintResult,
    MaxCoatingTime,
    MaxLayers,
    MaxTotalThickness,
    MinLayerThickness,
    constraints,
    register_constraint,
)
from xstack.manufacturing.process import DepositionProcess, DepositionStep

__all__ = [
    "Constraint",
    "ConstraintResult",
    "DepositionProcess",
    "DepositionStep",
    "MaxCoatingTime",
    "MaxLayers",
    "MaxTotalThickness",
    "MinLayerThickness",
    "constraints",
    "register_constraint",
]
