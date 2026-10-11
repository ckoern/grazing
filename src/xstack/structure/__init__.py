"""Layer stacks and coating descriptions."""

from xstack.structure.coating import Coating, CoatingBuildError, Mirror, coatings, register_coating
from xstack.structure.coatings import (
    Composite,
    DepthGraded,
    NLayer,
    Periodic,
    SingleLayer,
    grading_laws,
)
from xstack.structure.stack import Layer, Stack

__all__ = [
    "Coating",
    "CoatingBuildError",
    "Composite",
    "DepthGraded",
    "Layer",
    "Mirror",
    "NLayer",
    "Periodic",
    "SingleLayer",
    "Stack",
    "coatings",
    "grading_laws",
    "register_coating",
]
