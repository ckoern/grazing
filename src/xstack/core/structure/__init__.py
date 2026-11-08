"""Layer stacks and coating descriptions."""

from xstack.core.structure.coating import (
    Coating,
    CoatingBuildError,
    Mirror,
    coatings,
    register_coating,
)
from xstack.core.structure.coatings import (
    Composite,
    DepthGraded,
    NLayer,
    Periodic,
    SingleLayer,
    grading_laws,
)
from xstack.core.structure.conversion import (
    constant_gamma_changes,
    convert_coating,
    has_constant_gamma,
    restack_layers,
    unique_block_name,
)
from xstack.core.structure.stack import Layer, Stack

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
    "constant_gamma_changes",
    "convert_coating",
    "grading_laws",
    "has_constant_gamma",
    "register_coating",
    "restack_layers",
    "unique_block_name",
]
