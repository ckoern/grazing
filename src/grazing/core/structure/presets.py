"""Starting structures from a few choices: coating kind, materials, top layer and substrate.

Used to set up new projects (e.g. by the project wizards of the applications). The structures
have typical start values; thicknesses, periods, Γ and the number of bilayers are free within
generous bounds, roughnesses are fixed at zero.

Examples:
    >>> mirror = build_mirror(StructureChoice("periodic", MATERIAL_PAIRS["Pt/C"], top="L"))
    >>> mirror.coating.top, mirror.coating.materials["H"].name, mirror.substrate.name
    ('L', 'Pt', 'Si')
    >>> layers = build_mirror(StructureChoice("layers", ("Ru", "B4C", "Cr")))
    >>> layers.coating.parameters["t_Ru"].value
    10.0
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from grazing.core.materials import MaterialLibrary, default_library
from grazing.core.materials.material import Material
from grazing.core.parameters import free
from grazing.core.structure.coating import Coating, Mirror
from grazing.core.structure.coatings.graded import DepthGraded
from grazing.core.structure.coatings.periodic import Periodic
from grazing.core.structure.coatings.simple import NLayer, SingleLayer

__all__ = [
    "COATING_KINDS",
    "MATERIAL_PAIRS",
    "CoatingKind",
    "StructureChoice",
    "build_coating",
    "build_mirror",
]

type CoatingKind = Literal[
    "single_layer", "layers", "periodic", "graded_linear", "graded_power_law"
]
"""Kinds of starting structures."""

COATING_KINDS: dict[CoatingKind, str] = {
    "single_layer": "Single layer",
    "layers": "Individual layers",
    "periodic": "Periodic multilayer",
    "graded_linear": "Depth-graded multilayer, linear",
    "graded_power_law": "Depth-graded multilayer, power law",
}
"""Labels of the coating kinds."""

MATERIAL_PAIRS: dict[str, tuple[str, str]] = {
    "W/Si": ("W", "Si"),
    "Pt/C": ("Pt", "C"),
    "Ir/B4C": ("Ir", "B4C"),
    "Ir/C": ("Ir", "C"),
    "W/B4C": ("W", "B4C"),
    "Mo/Si": ("Mo", "Si"),
    "Pt/SiC": ("Pt", "SiC"),
    "Ni/C": ("Ni", "C"),
    "Ru/B4C": ("Ru", "B4C"),
    "Cr/Sc": ("Cr", "Sc"),
    "La/B4C": ("La", "B4C"),
}
"""Common multilayer material combinations: absorber (H) and spacer (L) by label."""

_MULTILAYER: tuple[CoatingKind, ...] = ("periodic", "graded_linear", "graded_power_law")


@dataclass(frozen=True)
class StructureChoice:
    """The choices that define a starting structure.

    Attributes:
        kind: Coating kind.
        materials: Material names: one for a single layer, the layers from the top for
            individual layers, absorber (H) and spacer (L) for multilayers.
        top: Top layer of each bilayer of a multilayer, ``"H"`` or ``"L"``.
        substrate: Name of the substrate material.

    Raises:
        ValueError: If the kind is unknown, the number of materials does not fit the kind or
            ``top`` is invalid.
    """

    kind: CoatingKind = "periodic"
    materials: tuple[str, ...] = MATERIAL_PAIRS["W/Si"]
    top: Literal["H", "L"] = "H"
    substrate: str = "Si"

    def __post_init__(self) -> None:
        object.__setattr__(self, "materials", tuple(self.materials))
        if self.kind not in COATING_KINDS:
            raise ValueError(f"unknown coating kind {self.kind!r}")
        count = len(self.materials)
        if self.kind == "single_layer" and count != 1:
            raise ValueError("a single layer needs exactly one material")
        if self.kind == "layers" and count < 1:
            raise ValueError("individual layers need at least one material")
        if self.kind in _MULTILAYER and count != 2:
            raise ValueError("a multilayer needs two materials (absorber and spacer)")
        if self.top not in ("H", "L"):
            raise ValueError("the top layer must be 'H' or 'L'")

    @property
    def is_multilayer(self) -> bool:
        """Whether the coating consists of bilayers."""
        return self.kind in _MULTILAYER


def _materials(names: Sequence[str], library: MaterialLibrary) -> list[Material]:
    return [library.get(name) for name in names]


def build_coating(choice: StructureChoice, library: MaterialLibrary | None = None) -> Coating:
    """Create the coating of a starting structure.

    Args:
        choice: The choices.
        library: Library providing the materials. Defaults to the default library.

    Returns:
        The coating with start values and free parameters.

    Raises:
        KeyError: If a material is not in the library.
    """
    library = library or default_library()
    materials = _materials(choice.materials, library)
    if choice.kind == "single_layer":
        return SingleLayer.create(materials[0], t=free(30.0, 1.0, 200.0))
    if choice.kind == "layers":
        thicknesses = [free(10.0, 0.5, 200.0) for _ in materials]
        return NLayer.create(materials, thicknesses)
    h, l_ = materials
    if choice.kind == "periodic":
        return Periodic.create(
            h,
            l_,
            N=free(40, 5, 200),
            d=free(4.0, 1.5, 15.0),
            gamma=free(0.4, 0.15, 0.7),
            top=choice.top,
        )
    law = "linear" if choice.kind == "graded_linear" else "power_law"
    coating = DepthGraded.create(
        h,
        l_,
        N=free(150, 20, 400),
        d_min=free(3.0, 1.5, 6.0),
        d_max=free(10.0, 5.0, 25.0),
        gamma_top=free(0.4, 0.15, 0.7),
        gamma_bottom=free(0.4, 0.15, 0.7),
        law=law,
        top=choice.top,
    )
    return coating


def build_mirror(choice: StructureChoice, library: MaterialLibrary | None = None) -> Mirror:
    """Create a starting structure: the coating on its substrate.

    Args:
        choice: The choices.
        library: Library providing the materials. Defaults to the default library.

    Returns:
        The mirror.

    Raises:
        KeyError: If a material is not in the library.
    """
    library = library or default_library()
    return Mirror(build_coating(choice, library), library.get(choice.substrate))
