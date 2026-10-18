"""Material definitions."""

from dataclasses import dataclass, field

import xraylib

from xstack.core.serialization import register_serializable
from xstack.core.units import AVOGADRO

__all__ = ["VACUUM", "Material"]

_CM3_TO_NM3 = 1e-21


@dataclass(frozen=True)
class Material:
    """A homogeneous material defined by its chemical formula and mass density.

    A material with an empty formula is vacuum and has a refractive index of exactly 1.

    Args:
        name: Unique, human-readable name, e.g. ``"W"`` or ``"B4C (sputtered)"``.
        formula: Chemical formula, e.g. ``"B4C"`` or ``"Ca(OH)2"``. Empty for vacuum.
        density: Mass density in g/cm³. Must be 0 for vacuum and positive otherwise.
        provider: Name of the optical-constants provider for this material. ``None`` uses the
            default of the material library or of the calling function.

    Raises:
        ValueError: If the formula cannot be parsed or the density is invalid.

    Examples:
        >>> b4c = Material("B4C", "B4C", 2.52)
        >>> b4c.composition
        ((5, 4.0), (6, 1.0))
        >>> round(b4c.molar_mass, 2)
        55.25
    """

    name: str
    formula: str
    density: float
    provider: str | None = None
    composition: tuple[tuple[int, float], ...] = field(init=False, repr=False, compare=False)
    """Atomic numbers and number of atoms per formula unit, sorted by atomic number."""
    molar_mass: float = field(init=False, repr=False, compare=False)
    """Molar mass of one formula unit in g/mol (0 for vacuum)."""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("material name must not be empty")
        if not self.formula:
            if self.density != 0.0:
                raise ValueError(f"vacuum material {self.name!r} must have density 0")
            object.__setattr__(self, "composition", ())
            object.__setattr__(self, "molar_mass", 0.0)
            return
        if not self.density > 0.0:
            raise ValueError(f"material {self.name!r} must have a positive density")
        try:
            parsed = xraylib.CompoundParser(self.formula)
        except ValueError as exc:
            raise ValueError(f"material {self.name!r}: {exc}") from None
        composition = tuple(
            (int(z), float(n)) for z, n in zip(parsed["Elements"], parsed["nAtoms"], strict=True)
        )
        molar_mass = sum(n * xraylib.AtomicWeight(z) for z, n in composition)
        object.__setattr__(self, "composition", composition)
        object.__setattr__(self, "molar_mass", float(molar_mass))

    @property
    def is_vacuum(self) -> bool:
        """Whether the material is vacuum."""
        return not self.composition

    def atom_densities(self) -> dict[int, float]:
        """Return the number density of each element.

        Returns:
            Mapping from atomic number to atoms per nm³. Empty for vacuum.
        """
        if self.is_vacuum:
            return {}
        units_per_nm3 = self.density * _CM3_TO_NM3 * AVOGADRO / self.molar_mass
        return {z: n * units_per_nm3 for z, n in self.composition}

    def with_density(self, density: float) -> "Material":
        """Return a copy of this material with a different density.

        Args:
            density: New mass density in g/cm³.

        Returns:
            The modified material, with the same name.
        """
        return Material(self.name, self.formula, density, self.provider)


VACUUM = Material("vacuum", "", 0.0)
"""The vacuum material (refractive index 1)."""

register_serializable("Material", Material)
