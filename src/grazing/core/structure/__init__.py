"""Layer stacks and coating descriptions."""

from grazing.core.structure.coating import (
    Coating,
    CoatingBuildError,
    Mirror,
    coatings,
    register_coating,
)
from grazing.core.structure.coatings import (
    Composite,
    DepthGraded,
    NLayer,
    Periodic,
    SingleLayer,
    grading_laws,
)
from grazing.core.structure.conversion import (
    constant_gamma_changes,
    convert_coating,
    has_constant_gamma,
    restack_layers,
    unique_block_name,
)
from grazing.core.structure.stack import Layer, Stack

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
