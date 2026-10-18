"""Coating implementations such as periodic and depth-graded multilayers."""

from xstack.core.structure.coatings.composite import Composite
from xstack.core.structure.coatings.graded import (
    DepthGraded,
    GradingLaw,
    LinearLaw,
    PowerLaw,
    grading_laws,
)
from xstack.core.structure.coatings.periodic import Periodic
from xstack.core.structure.coatings.simple import NLayer, SingleLayer

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
