"""Physical description of a planar layer stack."""

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from grazing.core.materials import VACUUM, Material, MaterialLibrary, default_library
from grazing.core.serialization import register_serializable

__all__ = ["Layer", "Stack"]


@dataclass(frozen=True)
class Layer:
    """A homogeneous layer.

    Args:
        material: Material of the layer.
        thickness: Thickness in nm, not negative.
        roughness: Width σ in nm of the interface at the top of this layer, i.e. between this
            layer and the medium above it.
        tag: Free-form label describing the origin of the layer, e.g. ``"graded[17].H"``.

    Raises:
        ValueError: If thickness or roughness is negative.
    """

    material: Material
    thickness: float
    roughness: float = 0.0
    tag: str = ""

    def __post_init__(self) -> None:
        if self.thickness < 0:
            raise ValueError(f"layer thickness must not be negative, got {self.thickness}")
        if self.roughness < 0:
            raise ValueError(f"layer roughness must not be negative, got {self.roughness}")


@dataclass(frozen=True)
class Stack:
    """A planar stack of layers on a substrate.

    This is the fully resolved physical description used for reflectivity calculations,
    coating time estimates and constraint checks.

    Args:
        layers: Layers ordered from the top (next to the ambient medium) to the bottom (next to
            the substrate).
        substrate: Substrate material (semi-infinite).
        substrate_roughness: Width σ in nm of the interface at the top of the substrate.
        ambient: Ambient material (semi-infinite), vacuum by default.

    Raises:
        ValueError: If the substrate roughness is negative.

    Examples:
        >>> from grazing.core.materials import get
        >>> stack = Stack([Layer(get("W"), 2.0), Layer(get("Si"), 3.0)] * 10, get("Si"))
        >>> stack.num_layers, stack.total_thickness
        (20, 50.0)
    """

    layers: tuple[Layer, ...]
    substrate: Material
    substrate_roughness: float = 0.0
    ambient: Material = VACUUM

    def __init__(
        self,
        layers: Iterable[Layer],
        substrate: Material,
        substrate_roughness: float = 0.0,
        ambient: Material = VACUUM,
    ) -> None:
        if substrate_roughness < 0:
            raise ValueError("substrate roughness must not be negative")
        object.__setattr__(self, "layers", tuple(layers))
        object.__setattr__(self, "substrate", substrate)
        object.__setattr__(self, "substrate_roughness", float(substrate_roughness))
        object.__setattr__(self, "ambient", ambient)

    @property
    def num_layers(self) -> int:
        """Number of layers (excluding ambient and substrate)."""
        return len(self.layers)

    @property
    def total_thickness(self) -> float:
        """Sum of all layer thicknesses in nm."""
        return float(sum(layer.thickness for layer in self.layers))

    @property
    def media(self) -> tuple[Material, ...]:
        """All media from the ambient to the substrate."""
        return (self.ambient, *(layer.material for layer in self.layers), self.substrate)

    def thicknesses(self) -> NDArray[np.float64]:
        """Return the layer thicknesses in nm, shape (num_layers,)."""
        return np.array([layer.thickness for layer in self.layers], dtype=np.float64)

    def roughnesses(self) -> NDArray[np.float64]:
        """Return the interface widths in nm from top to bottom, shape (num_layers + 1,)."""
        values = [layer.roughness for layer in self.layers] + [self.substrate_roughness]
        return np.array(values, dtype=np.float64)

    def distinct_refractive_indices(
        self, energy: ArrayLike, library: MaterialLibrary | None = None
    ) -> tuple[NDArray[np.complex128], NDArray[np.intp]]:
        """Return the refractive indices of the distinct materials and the medium mapping.

        Args:
            energy: Photon energies in eV, shape (P,).
            library: Library used to compute (and cache) optical constants. Defaults to the
                shared default library.

        Returns:
            Complex refractive indices of the U distinct materials, shape (P, U), and for each
            medium from the ambient to the substrate the index of its column, shape
            (num_layers + 2,).

        Raises:
            ValueError: If `energy` is not one-dimensional.
        """
        lib = library or default_library()
        e = np.atleast_1d(np.asarray(energy, dtype=np.float64))
        if e.ndim != 1:
            raise ValueError("energy must be one-dimensional")
        columns: dict[Material, int] = {}
        index = np.array([columns.setdefault(m, len(columns)) for m in self.media], dtype=np.intp)
        n = np.stack([lib.refractive_index(m, e) for m in columns], axis=1)
        return n, index

    def refractive_indices(
        self, energy: ArrayLike, library: MaterialLibrary | None = None
    ) -> NDArray[np.complex128]:
        """Return the refractive indices of all media.

        Args:
            energy: Photon energies in eV, shape (P,).
            library: Library used to compute (and cache) optical constants. Defaults to the
                shared default library.

        Returns:
            Complex refractive indices, shape (P, num_layers + 2), ambient first.
        """
        n, index = self.distinct_refractive_indices(energy, library)
        return n[:, index]


register_serializable("Layer", Layer)
register_serializable("Stack", Stack)
