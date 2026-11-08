"""Deposition process models, constraints and tolerance analysis."""

from grazing.design.manufacturing.constraints import (
    Constraint,
    ConstraintResult,
    MaxCoatingTime,
    MaxLayers,
    MaxTotalThickness,
    MinLayerThickness,
    constraints,
    register_constraint,
)
from grazing.design.manufacturing.process import DepositionProcess, DepositionStep
from grazing.design.manufacturing.tolerance import (
    ErrorModel,
    ToleranceResult,
    mixed_material,
    perturb_stack,
    perturb_thicknesses,
    thickness_tolerance,
    tolerance_analysis,
)

__all__ = [
    "Constraint",
    "ConstraintResult",
    "DepositionProcess",
    "DepositionStep",
    "ErrorModel",
    "MaxCoatingTime",
    "MaxLayers",
    "MaxTotalThickness",
    "MinLayerThickness",
    "ToleranceResult",
    "constraints",
    "mixed_material",
    "perturb_stack",
    "perturb_thicknesses",
    "register_constraint",
    "thickness_tolerance",
    "tolerance_analysis",
]
