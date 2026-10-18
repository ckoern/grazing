"""Providers of optical constants.

An optical-constants provider returns δ + iβ for a material, where the complex refractive
index is n = 1 − δ + iβ. Element-based providers compute it from the atomic scattering factors
f1 + i·f2 of the constituent elements:

    δ + iβ = r_e λ² / (2π) · Σ_a n_a (f1_a + i·f2_a)

with the classical electron radius r_e, the wavelength λ and the atom number densities n_a.

By default, f2 includes Compton (incoherent) scattering in addition to photoabsorption, because
Compton-scattered photons are removed from the specularly reflected beam. This matches the
CXRO index-of-refraction calculator and matters mainly for light elements above ~10 keV.

Available providers (names in `providers`):

* ``"henke"``: CXRO/Henke tables, roughly 10 eV to 30 keV.
* ``"xraylib"``: xraylib anomalous scattering factors, all energies.
* ``"henke_xraylib"``: Henke up to 30 keV and xraylib above. This is `DEFAULT_PROVIDER`.
* ``"henke_photo"``, ``"xraylib_photo"``, ``"henke_xraylib_photo"``: the same without the
  Compton contribution (photoabsorption only).
"""

from abc import ABC, abstractmethod
from functools import cache
from importlib import resources
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import xraylib
from numpy.typing import ArrayLike, NDArray

from xstack.core.materials.material import Material
from xstack.core.plugins import Registry
from xstack.core.units import AVOGADRO, CLASSICAL_ELECTRON_RADIUS, energy_to_wavelength, to_kev

__all__ = [
    "DEFAULT_PROVIDER",
    "CombinedProvider",
    "HenkeProvider",
    "OpticalConstantsProvider",
    "ScatteringFactorProvider",
    "TabulatedProvider",
    "XraylibProvider",
    "compton_f2",
    "delta_beta",
    "providers",
    "refractive_index",
]

DEFAULT_PROVIDER = "henke_xraylib"
"""Name of the provider used when neither the caller nor the material specifies one."""

_MISSING_F1 = -9000.0
_CM2_TO_NM2 = 1e14
_COMPTON_MIN_KEV = 0.1


@runtime_checkable
class OpticalConstantsProvider(Protocol):
    """Interface of all optical-constants providers."""

    @property
    def name(self) -> str:
        """Short name of the provider."""
        ...

    @property
    def energy_range(self) -> tuple[float, float]:
        """Nominal range of supported photon energies in eV (min, max)."""
        ...

    def delta_beta(self, material: Material, energy: NDArray[np.float64]) -> NDArray[np.complex128]:
        """Return δ + iβ of a material.

        Args:
            material: The material.
            energy: Photon energies in eV, any shape.

        Returns:
            Complex array δ + iβ with the same shape as `energy`.

        Raises:
            ValueError: If an energy is outside the supported range for this material.
        """
        ...


class ScatteringFactorProvider(ABC):
    """Base class for providers based on per-element atomic scattering factors.

    Subclasses implement `scattering_factors`. The Compton contribution and the conversion to
    δ + iβ are added here.

    Args:
        include_compton: Add Compton scattering to f2.
    """

    def __init__(self, *, include_compton: bool = True) -> None:
        self._include_compton = include_compton

    @property
    def include_compton(self) -> bool:
        """Whether f2 includes Compton scattering."""
        return self._include_compton

    @property
    @abstractmethod
    def name(self) -> str:
        """Short name of the provider."""

    @property
    @abstractmethod
    def energy_range(self) -> tuple[float, float]:
        """Nominal range of supported photon energies in eV (min, max)."""

    @abstractmethod
    def scattering_factors(
        self, atomic_number: int, energy: NDArray[np.float64]
    ) -> NDArray[np.complex128]:
        """Return the forward atomic scattering factor f1 + i·f2 of an element.

        f2 here contains photoabsorption only.

        Args:
            atomic_number: Atomic number Z of the element.
            energy: Photon energies in eV, one-dimensional.

        Returns:
            Complex array f1 + i·f2 with the same shape as `energy`.

        Raises:
            ValueError: If an energy is outside the supported range for this element.
        """

    def delta_beta(self, material: Material, energy: NDArray[np.float64]) -> NDArray[np.complex128]:
        """Return δ + iβ of a material.

        Args:
            material: The material.
            energy: Photon energies in eV, any shape.

        Returns:
            Complex array δ + iβ with the same shape as `energy`.

        Raises:
            ValueError: If an energy is outside the supported range for this material.
        """
        energy = np.asarray(energy, dtype=np.float64)
        flat = energy.ravel()
        total = np.zeros(flat.shape, dtype=np.complex128)
        for z, density in material.atom_densities().items():
            f = self.scattering_factors(z, flat)
            if self._include_compton:
                f = f + 1j * compton_f2(z, flat)
            total += density * f
        wavelength = energy_to_wavelength(flat)
        result = CLASSICAL_ELECTRON_RADIUS * wavelength**2 / (2.0 * np.pi) * total
        return result.reshape(energy.shape)


class HenkeProvider(ScatteringFactorProvider):
    """Atomic scattering factors from the CXRO/Henke tables.

    The tables cover Z = 1 to 92 and roughly 10 eV to 30 keV (some elements extend to lower
    energies). f1 is interpolated linearly in log(E), f2 linearly in log(E)–log(f2).

    Args:
        data_dir: Directory with ``<element>.nff`` files. Defaults to the tables bundled with
            xstack.
        include_compton: Add Compton scattering to f2.
    """

    def __init__(self, data_dir: Path | None = None, *, include_compton: bool = True) -> None:
        super().__init__(include_compton=include_compton)
        self._data_dir = data_dir

    @property
    def name(self) -> str:
        """Short name of the provider."""
        return "henke" if self._include_compton else "henke_photo"

    @property
    def energy_range(self) -> tuple[float, float]:
        """Nominal range of supported photon energies in eV (min, max)."""
        return (10.0, 30000.0)

    def element_range(self, atomic_number: int) -> tuple[float, float]:
        """Return the energy range in which both f1 and f2 are tabulated for an element.

        Args:
            atomic_number: Atomic number Z of the element.

        Returns:
            (min, max) photon energy in eV.
        """
        log_e, _, _ = self._table(atomic_number)
        return (float(np.exp(log_e[0])), float(np.exp(log_e[-1])))

    def scattering_factors(
        self, atomic_number: int, energy: NDArray[np.float64]
    ) -> NDArray[np.complex128]:
        """Return the forward atomic scattering factor f1 + i·f2 of an element.

        Args:
            atomic_number: Atomic number Z of the element.
            energy: Photon energies in eV, one-dimensional.

        Returns:
            Complex array f1 + i·f2 with the same shape as `energy`.

        Raises:
            ValueError: If an energy is outside the tabulated range for this element.
        """
        log_e, f1, log_f2 = self._table(atomic_number)
        e_min, e_max = np.exp(log_e[0]), np.exp(log_e[-1])
        energy = np.asarray(energy, dtype=np.float64)
        # Small tolerance so that the table end points themselves are accepted.
        if energy.size and (
            energy.min() < e_min * (1 - 1e-12) or energy.max() > e_max * (1 + 1e-12)
        ):
            symbol = xraylib.AtomicNumberToSymbol(atomic_number)
            raise ValueError(
                f"Henke data for {symbol} cover {e_min:g}-{e_max:g} eV, "
                f"requested {energy.min():g}-{energy.max():g} eV"
            )
        x = np.log(energy)
        return np.interp(x, log_e, f1) + 1j * np.exp(np.interp(x, log_e, log_f2))

    def _table(
        self, atomic_number: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        if self._data_dir is None:
            return _bundled_henke_table(atomic_number)
        return _load_henke_table(self._data_dir / _henke_filename(atomic_number))


def _henke_filename(atomic_number: int) -> str:
    if not 1 <= atomic_number <= 92:
        raise ValueError(f"Henke data are available for Z = 1-92, not Z = {atomic_number}")
    return f"{xraylib.AtomicNumberToSymbol(atomic_number).lower()}.nff"


@cache
def _bundled_henke_table(
    atomic_number: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    resource = resources.files("xstack.core.materials").joinpath("data", "henke")
    with resources.as_file(resource / _henke_filename(atomic_number)) as path:
        return _load_henke_table(path)


def _load_henke_table(
    path: Path,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Load an nff file and return log(E), f1 and log(f2) on unique, tabulated energies."""
    data = np.loadtxt(path, skiprows=1, ndmin=2)
    energy, f1, f2 = data[:, 0], data[:, 1], data[:, 2]
    valid = f1 > _MISSING_F1
    energy, f1, f2 = energy[valid], f1[valid], f2[valid]
    unique, inverse, counts = np.unique(energy, return_inverse=True, return_counts=True)
    f1 = np.bincount(inverse, weights=f1) / counts
    f2 = np.bincount(inverse, weights=f2) / counts
    return np.log(unique), f1, np.log(f2)


class XraylibProvider(ScatteringFactorProvider):
    """Atomic scattering factors from xraylib.

    Uses f1 = Z + f'(E) and f2 = −f''(E) with the anomalous scattering factors ``Fi`` and
    ``Fii`` of xraylib, which cover all elements and the whole X-ray range.

    Args:
        include_compton: Add Compton scattering to f2.
    """

    @property
    def name(self) -> str:
        """Short name of the provider."""
        return "xraylib" if self._include_compton else "xraylib_photo"

    @property
    def energy_range(self) -> tuple[float, float]:
        """Nominal range of supported photon energies in eV (min, max)."""
        return (100.0, 1e6)

    def scattering_factors(
        self, atomic_number: int, energy: NDArray[np.float64]
    ) -> NDArray[np.complex128]:
        """Return the forward atomic scattering factor f1 + i·f2 of an element.

        Args:
            atomic_number: Atomic number Z of the element.
            energy: Photon energies in eV, one-dimensional.

        Returns:
            Complex array f1 + i·f2 with the same shape as `energy`.

        Raises:
            ValueError: If xraylib has no data for the element or energy.
        """
        energy = np.asarray(energy, dtype=np.float64)
        out = np.empty(energy.shape, dtype=np.complex128)
        for i, e_kev in enumerate(to_kev(energy).tolist()):
            out[i] = _xraylib_factor(atomic_number, e_kev)
        return out


@cache
def _xraylib_factor(atomic_number: int, energy_kev: float) -> complex:
    f1 = atomic_number + xraylib.Fi(atomic_number, energy_kev)
    f2 = -xraylib.Fii(atomic_number, energy_kev)
    return complex(f1, f2)


def compton_f2(atomic_number: int, energy: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the Compton scattering contribution to f2 of an element.

    The contribution is f2_C = σ_C / (2 r_e λ) with the atomic Compton cross section σ_C
    from xraylib. Below 100 eV, where it is negligible, it is zero.

    Args:
        atomic_number: Atomic number Z of the element.
        energy: Photon energies in eV, one-dimensional.

    Returns:
        f2_C with the same shape as `energy`.
    """
    energy = np.asarray(energy, dtype=np.float64)
    sigma = np.array([_compton_cross_section(atomic_number, e) for e in to_kev(energy).tolist()])
    return sigma / (2.0 * CLASSICAL_ELECTRON_RADIUS * energy_to_wavelength(energy))


@cache
def _compton_cross_section(atomic_number: int, energy_kev: float) -> float:
    """Atomic Compton cross section in nm² (zero below the xraylib data range)."""
    if energy_kev < _COMPTON_MIN_KEV:
        return 0.0
    mass_cs = xraylib.CS_Compt(atomic_number, energy_kev)
    return float(mass_cs * xraylib.AtomicWeight(atomic_number) / AVOGADRO * _CM2_TO_NM2)


class CombinedProvider:
    """Uses one provider below a switching energy and another one above.

    Args:
        low: Provider for energies up to and including `switch_energy`.
        high: Provider for energies above `switch_energy`.
        switch_energy: Switching energy in eV.
        name: Short name of the combined provider.
    """

    def __init__(
        self,
        low: OpticalConstantsProvider,
        high: OpticalConstantsProvider,
        switch_energy: float,
        name: str,
    ) -> None:
        self._low = low
        self._high = high
        self._switch = switch_energy
        self._name = name

    @property
    def name(self) -> str:
        """Short name of the provider."""
        return self._name

    @property
    def switch_energy(self) -> float:
        """Switching energy in eV."""
        return self._switch

    @property
    def energy_range(self) -> tuple[float, float]:
        """Nominal range of supported photon energies in eV (min, max)."""
        return (self._low.energy_range[0], self._high.energy_range[1])

    def delta_beta(self, material: Material, energy: NDArray[np.float64]) -> NDArray[np.complex128]:
        """Return δ + iβ of a material.

        Args:
            material: The material.
            energy: Photon energies in eV, any shape.

        Returns:
            Complex array δ + iβ with the same shape as `energy`.

        Raises:
            ValueError: If an energy is outside the supported range for this material.
        """
        energy = np.asarray(energy, dtype=np.float64)
        out = np.empty(energy.shape, dtype=np.complex128)
        low = energy <= self._switch
        if low.any():
            out[low] = self._low.delta_beta(material, energy[low])
        if (~low).any():
            out[~low] = self._high.delta_beta(material, energy[~low])
        return out


class TabulatedProvider:
    """Optical constants of one specific material from a user-supplied table.

    δ is interpolated linearly in log(E), β linearly in log(E)–log(β).

    Args:
        name: Short name of the provider, used to refer to it from a `Material`.
        energy: Tabulated photon energies in eV, strictly increasing.
        delta: Tabulated δ values.
        beta: Tabulated β values, positive.

    Raises:
        ValueError: If the table is malformed.
    """

    def __init__(self, name: str, energy: ArrayLike, delta: ArrayLike, beta: ArrayLike) -> None:
        e = np.asarray(energy, dtype=np.float64)
        d = np.asarray(delta, dtype=np.float64)
        b = np.asarray(beta, dtype=np.float64)
        if e.ndim != 1 or e.shape != d.shape or e.shape != b.shape or e.size < 2:
            raise ValueError("energy, delta and beta must be 1-D arrays of equal length >= 2")
        if np.any(np.diff(e) <= 0) or e[0] <= 0:
            raise ValueError("energies must be positive and strictly increasing")
        if np.any(b <= 0):
            raise ValueError("beta values must be positive")
        self._name = name
        self._log_e = np.log(e)
        self._delta = d
        self._log_beta = np.log(b)

    @classmethod
    def from_file(cls, name: str, path: Path, skiprows: int = 0) -> "TabulatedProvider":
        """Create a provider from a text file with columns energy (eV), δ, β.

        Args:
            name: Short name of the provider.
            path: Path to the whitespace- or comma-separated text file.
            skiprows: Number of header lines to skip.

        Returns:
            The provider.
        """
        text = path.read_text().replace(",", " ")
        data = np.loadtxt(text.splitlines(), skiprows=skiprows, ndmin=2)
        return cls(name, data[:, 0], data[:, 1], data[:, 2])

    @property
    def name(self) -> str:
        """Short name of the provider."""
        return self._name

    @property
    def energy_range(self) -> tuple[float, float]:
        """Tabulated range of photon energies in eV (min, max)."""
        return (float(np.exp(self._log_e[0])), float(np.exp(self._log_e[-1])))

    def delta_beta(self, material: Material, energy: NDArray[np.float64]) -> NDArray[np.complex128]:
        """Return the tabulated δ + iβ, independent of the material's composition.

        Args:
            material: The material (unused apart from error messages).
            energy: Photon energies in eV, any shape.

        Returns:
            Complex array δ + iβ with the same shape as `energy`.

        Raises:
            ValueError: If an energy is outside the tabulated range.
        """
        energy = np.asarray(energy, dtype=np.float64)
        e_min, e_max = self.energy_range
        if energy.size and (
            energy.min() < e_min * (1 - 1e-12) or energy.max() > e_max * (1 + 1e-12)
        ):
            raise ValueError(
                f"table {self._name!r} for {material.name!r} covers {e_min:g}-{e_max:g} eV"
            )
        x = np.log(energy)
        delta = np.interp(x, self._log_e, self._delta)
        beta = np.exp(np.interp(x, self._log_e, self._log_beta))
        return delta + 1j * beta


providers: Registry[OpticalConstantsProvider] = Registry("optical constants provider")
"""Registry of all available optical-constants providers."""

for _suffix, _compton in (("", True), ("_photo", False)):
    _henke = HenkeProvider(include_compton=_compton)
    _xraylib = XraylibProvider(include_compton=_compton)
    providers.register(_henke.name, _henke)
    providers.register(_xraylib.name, _xraylib)
    providers.register(
        DEFAULT_PROVIDER + _suffix,
        CombinedProvider(_henke, _xraylib, 30000.0, DEFAULT_PROVIDER + _suffix),
    )


def delta_beta(
    material: Material, energy: ArrayLike, provider: str | None = None
) -> NDArray[np.complex128]:
    """Return δ + iβ of a material.

    The provider is chosen in this order: the `provider` argument, the material's own
    provider, `DEFAULT_PROVIDER`.

    Args:
        material: The material.
        energy: Photon energies in eV, scalar or array.
        provider: Name of a registered provider.

    Returns:
        Complex array δ + iβ with the shape of `energy`. Zero for vacuum.

    Raises:
        KeyError: If the provider is not registered.
        ValueError: If an energy is outside the supported range.
    """
    e = np.asarray(energy, dtype=np.float64)
    if np.any(e <= 0):
        raise ValueError("photon energies must be positive")
    if material.is_vacuum:
        return np.zeros(e.shape, dtype=np.complex128)
    name = provider or material.provider or DEFAULT_PROVIDER
    return providers.get(name).delta_beta(material, e)


def refractive_index(
    material: Material, energy: ArrayLike, provider: str | None = None
) -> NDArray[np.complex128]:
    """Return the complex refractive index n = 1 − δ + iβ of a material.

    Args:
        material: The material.
        energy: Photon energies in eV, scalar or array.
        provider: Name of a registered provider. See `delta_beta` for the selection order.

    Returns:
        Complex array n with the shape of `energy`.

    Raises:
        KeyError: If the provider is not registered.
        ValueError: If an energy is outside the supported range.

    Examples:
        >>> from xstack.core.materials import Material
        >>> n = refractive_index(Material("Si", "Si", 2.33), 8048.0)
        >>> print(f"{1 - n.real:.3e}")
        7.580e-06
    """
    db = delta_beta(material, energy, provider)
    return 1.0 - db.real + 1j * db.imag
