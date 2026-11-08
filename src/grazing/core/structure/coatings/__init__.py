"""Coating implementations such as periodic and depth-graded multilayers."""

from grazing.core.structure.coatings.composite import Composite
from grazing.core.structure.coatings.graded import (
    DepthGraded,
    GradingLaw,
    LinearLaw,
    PowerLaw,
    grading_laws,
)
from grazing.core.structure.coatings.periodic import Periodic
from grazing.core.structure.coatings.simple import NLayer, SingleLayer

__all__ = [
    "Composite",
    "DepthGraded",
    "GradingLaw",
    "LinearLaw",
    "NLayer",
    "Periodic",
    "PowerLaw",
    "SingleLayer",
    "grading_laws",
]
