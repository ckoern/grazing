"""Figures of merit for evaluating reflectivity against targets."""

from grazing.design.merit.base import (
    CallableMerit,
    Curve,
    CurveMerit,
    EvaluationContext,
    FigureOfMerit,
    merits,
    register_merit,
)
from grazing.design.merit.composite import TimePenalized, WeightedSum
from grazing.design.merit.optical import (
    Flatness,
    MeanReflectivity,
    PeakReflectivity,
    ProfileDeviation,
    SoftMinReflectivity,
)

__all__ = [
    "CallableMerit",
    "Curve",
    "CurveMerit",
    "EvaluationContext",
    "FigureOfMerit",
    "Flatness",
    "MeanReflectivity",
    "PeakReflectivity",
    "ProfileDeviation",
    "SoftMinReflectivity",
    "TimePenalized",
    "WeightedSum",
    "merits",
    "register_merit",
]
