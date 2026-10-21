"""Collections of named materials with cached optical constants."""

from collections import OrderedDict
from collections.abc import Iterable, Iterator, Mapping
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from xstack.core.materials.material import VACUUM, Material
from xstack.core.materials.providers import DEFAULT_PROVIDER, delta_beta, providers

__all__ = ["PRESETS", "MaterialLibrary", "default_library", "provider_library"]

PRESETS: tuple[Material, ...] = (
    VACUUM,
    Material("Si", "Si", 2.33),
    Material("W", "W", 19.25),
    Material("Mo", "Mo", 10.22),
    Material("Pt", "Pt", 21.45),
    Material("Au", "Au", 19.32),
    Material("Ir", "Ir", 22.56),
    Material("Ru", "Ru", 12.37),
    Material("Rh", "Rh", 12.41),
    Material("Pd", "Pd", 12.02),
    Material("Ni", "Ni", 8.91),
    Material("Cr", "Cr", 7.19),
    Material("Ti", "Ti", 4.51),
    Material("Cu", "Cu", 8.96),
    Material("Ta", "Ta", 16.65),
    Material("Sc", "Sc", 2.99),
    Material("La", "La", 6.15),
    Material("Al", "Al", 2.70),
    Material("C", "C", 2.20),
    Material("B4C", "B4C", 2.52),
    Material("SiC", "SiC", 3.21),
    Material("Si3N4", "Si3N4", 3.17),
    Material("SiO2", "SiO2", 2.20),
    Material("Al2O3", "Al2O3", 3.95),
    Material("MoSi2", "MoSi2", 6.26),
    Material("WSi2", "WSi2", 9.30),
)
"""Preset materials with bulk densities in g/cm³. Thin films are often less dense."""

_CACHE_SIZE = 256


class MaterialLibrary:
    """A collection of materials, addressed by unique name.

    The library also selects optical-constants providers and caches computed optical
    constants per material, provider and energy grid.

    Args:
        materials: Initial materials. Defaults to `PRESETS`.
        default_provider: Provider used for materials that don't specify their own.

    Raises:
        KeyError: If `default_provider` is not registered.

    Examples:
        >>> library = MaterialLibrary()
        >>> library.get("W").density
        19.25
    """

    def __init__(
        self,
        materials: Iterable[Material] | None = None,
        default_provider: str = DEFAULT_PROVIDER,
    ) -> None:
        providers.get(default_provider)
        self._materials: dict[str, Material] = {}
        self._default_provider = default_provider
        self._cache: OrderedDict[tuple[Material, str, bytes], NDArray[np.complex128]] = (
            OrderedDict()
        )
        for material in PRESETS if materials is None else materials:
            self.add(material)

    @property
    def default_provider(self) -> str:
        """Name of the provider used for materials that don't specify their own."""
        return self._default_provider

    @default_provider.setter
    def default_provider(self, name: str) -> None:
        providers.get(name)
        self._default_provider = name

    def add(self, material: Material, *, replace: bool = False) -> None:
        """Add a material.

        Args:
            material: The material to add.
            replace: Allow replacing an existing material with the same name.

        Raises:
            ValueError: If a material with the same name exists and `replace` is False.
        """
        if material.name in self._materials and not replace:
            raise ValueError(f"material {material.name!r} already exists")
        self._materials[material.name] = material

    def remove(self, name: str) -> None:
        """Remove a material.

        Args:
            name: Name of the material.

        Raises:
            KeyError: If no material with this name exists.
        """
        self.get(name)
        del self._materials[name]

    def get(self, name: str) -> Material:
        """Return a material by name.

        Args:
            name: Name of the material.

        Returns:
            The material.

        Raises:
            KeyError: If no material with this name exists.
        """
        try:
            return self._materials[name]
        except KeyError:
            raise KeyError(f"unknown material {name!r}") from None

    def names(self) -> tuple[str, ...]:
        """Return the names of all materials in insertion order."""
        return tuple(self._materials)

    def __contains__(self, name: object) -> bool:
        return name in self._materials

    def __iter__(self) -> Iterator[Material]:
        return iter(self._materials.values())

    def __len__(self) -> int:
        return len(self._materials)

    def provider_for(self, material: Material) -> str:
        """Return the name of the provider used for a material.

        Args:
            material: The material.

        Returns:
            The material's own provider, or the library default.
        """
        return material.provider or self._default_provider

    def delta_beta(self, material: Material | str, energy: ArrayLike) -> NDArray[np.complex128]:
        """Return δ + iβ of a material, using the cache.

        Args:
            material: The material, or the name of a material in this library.
            energy: Photon energies in eV, scalar or array.

        Returns:
            Complex array δ + iβ with the shape of `energy`. The array is read-only.

        Raises:
            KeyError: If the material or its provider is unknown.
            ValueError: If an energy is outside the supported range.
        """
        mat = self.get(material) if isinstance(material, str) else material
        e = np.asarray(energy, dtype=np.float64)
        provider = self.provider_for(mat)
        key = (mat, provider, repr(e.shape).encode() + e.tobytes())
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        result = delta_beta(mat, e, provider)
        result.flags.writeable = False
        self._cache[key] = result
        if len(self._cache) > _CACHE_SIZE:
            self._cache.popitem(last=False)
        return result

    def refractive_index(
        self, material: Material | str, energy: ArrayLike
    ) -> NDArray[np.complex128]:
        """Return the complex refractive index n = 1 − δ + iβ of a material.

        Args:
            material: The material, or the name of a material in this library.
            energy: Photon energies in eV, scalar or array.

        Returns:
            Complex array n with the shape of `energy`.

        Raises:
            KeyError: If the material or its provider is unknown.
            ValueError: If an energy is outside the supported range.
        """
        db = self.delta_beta(material, energy)
        return 1.0 - db.real + 1j * db.imag

    def clear_cache(self) -> None:
        """Discard all cached optical constants."""
        self._cache.clear()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the library.

        Returns:
            Dictionary with the default provider and all materials.
        """
        return {
            "default_provider": self._default_provider,
            "materials": [
                {
                    "name": m.name,
                    "formula": m.formula,
                    "density": m.density,
                    "provider": m.provider,
                }
                for m in self
            ],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MaterialLibrary":
        """Create a library from the representation returned by `to_dict`.

        Args:
            data: Dictionary as produced by `to_dict`.

        Returns:
            The library.
        """
        materials = [
            Material(m["name"], m["formula"], float(m["density"]), m.get("provider"))
            for m in data["materials"]
        ]
        return cls(materials, data.get("default_provider", DEFAULT_PROVIDER))


_default: MaterialLibrary | None = None
_by_provider: dict[str, MaterialLibrary] = {}


def default_library() -> MaterialLibrary:
    """Return the shared default library, created with `PRESETS` on first use.

    Returns:
        The default material library.
    """
    global _default
    if _default is None:
        _default = MaterialLibrary()
    return _default


def provider_library(provider: str) -> MaterialLibrary:
    """Return a shared library that computes optical constants with a given provider.

    Materials that specify their own provider keep it. Libraries are created once per provider,
    so their caches of optical constants are shared by all callers.

    Args:
        provider: Name of a registered optical-constants provider.

    Returns:
        The default library if its default provider is `provider`, otherwise a library with
        `PRESETS` and `provider` as default provider.

    Raises:
        KeyError: If the provider is not registered.

    Examples:
        >>> provider_library("chantler").default_provider
        'chantler'
    """
    default = default_library()
    if provider == default.default_provider:
        return default
    if provider not in _by_provider:
        _by_provider[provider] = MaterialLibrary(default_provider=provider)
    return _by_provider[provider]
