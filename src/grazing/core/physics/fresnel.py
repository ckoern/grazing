"""Interface coefficients and interface roughness models.

Conventions: the stack is described by M media (ambient, layers, substrate) and has M − 1
interfaces. Interface j separates medium j (above) from medium j + 1 (below). Arrays carry the
evaluation points (energy/angle pairs) along the first axis.

For s polarization the coefficients refer to the electric field, for p polarization to the
magnetic field. In both cases the field component used is continuous across a sharp interface,
so that t = 1 + r. The reflectance ``|r|²`` is the same in either convention.
"""

from dataclasses import dataclass
from typing import ClassVar, Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from grazing.core.plugins import Registry

__all__ = [
    "DebyeWaller",
    "InterfaceModel",
    "NevotCroce",
    "Polarization",
    "SharpInterface",
    "fresnel_coefficients",
    "interface_models",
    "interface_reflection",
    "wavevector_z",
]

type Polarization = Literal["s", "p"]
"""Polarization of the incident wave."""

type ComplexArray = NDArray[np.complex128]


def wavevector_z(
    n: ComplexArray, theta: NDArray[np.float64], wavelength: NDArray[np.float64]
) -> ComplexArray:
    """Return the wavevector component perpendicular to the surface in every medium.

    k_z,j = k·√(n_j² − n_0²·cos²θ), with the branch chosen such that Im(k_z,j) ≥ 0, which
    corresponds to waves that decay in the direction of propagation.

    Args:
        n: Complex refractive indices, shape (P, M). Column 0 is the ambient medium.
        theta: Grazing angle of incidence in the ambient medium in rad, shape (P,).
        wavelength: Vacuum wavelength in nm, shape (P,).

    Returns:
        k_z in 1/nm, shape (P, M).
    """
    return _kz(n, n[:, :1], theta, wavelength)


def _kz(
    n: ComplexArray, n0: ComplexArray, theta: NDArray[np.float64], wavelength: NDArray[np.float64]
) -> ComplexArray:
    k = 2.0 * np.pi / wavelength
    # n_j² − n_0²cos²θ written without the cancellation in 1 − cos²θ at grazing incidence.
    radicand = (n0 * np.sin(theta)[:, None]) ** 2 + (n - n0) * (n + n0)
    kz = k[:, None] * np.sqrt(radicand)
    return np.where(kz.imag < 0.0, -kz, kz)


def interface_reflection(
    kz_above: ComplexArray,
    kz_below: ComplexArray,
    n_above: ComplexArray,
    n_below: ComplexArray,
    polarization: Polarization,
) -> ComplexArray:
    """Return the reflection coefficients of sharp interfaces for waves incident from above.

    Args:
        kz_above: k_z in the medium above each interface.
        kz_below: k_z in the medium below each interface.
        n_above: Refractive index of the medium above each interface.
        n_below: Refractive index of the medium below each interface.
        polarization: ``"s"`` or ``"p"``.

    Returns:
        Reflection coefficients, broadcast shape of the inputs.

    Raises:
        ValueError: If the polarization is unknown.
    """
    if polarization == "s":
        return (kz_above - kz_below) / (kz_above + kz_below)
    if polarization == "p":
        a = n_below**2 * kz_above
        b = n_above**2 * kz_below
        return (a - b) / (a + b)
    raise ValueError(f"unknown polarization {polarization!r}, expected 's' or 'p'")


def fresnel_coefficients(
    kz: ComplexArray, n: ComplexArray, polarization: Polarization
) -> ComplexArray:
    """Return the reflection coefficients of all sharp interfaces of a stack.

    Args:
        kz: Perpendicular wavevector components, shape (P, M), from `wavevector_z`.
        n: Complex refractive indices, shape (P, M).
        polarization: ``"s"`` or ``"p"``.

    Returns:
        Reflection coefficients r_j for waves incident from above, shape (P, M − 1).

    Raises:
        ValueError: If the polarization is unknown.
    """
    return interface_reflection(kz[:, :-1], kz[:, 1:], n[:, :-1], n[:, 1:], polarization)


class InterfaceModel(Protocol):
    """Model of imperfect interfaces of width σ.

    It provides the factors by which the reflection and transmission coefficients of a sharp
    interface are multiplied. All arrays broadcast against each other; ``kz_above`` and
    ``kz_below`` are in 1/nm and ``sigma`` in nm.
    """

    @property
    def name(self) -> str:
        """Registry name of the model."""
        ...

    def reflection_factor(
        self, kz_above: ComplexArray, kz_below: ComplexArray, sigma: NDArray[np.float64]
    ) -> ComplexArray:
        """Return the factor applied to the reflection coefficients.

        Args:
            kz_above: k_z in the medium above each interface.
            kz_below: k_z in the medium below each interface.
            sigma: Interface widths in nm.

        Returns:
            The factors, broadcast shape of the inputs.
        """
        ...

    def transmission_factor(
        self, kz_above: ComplexArray, kz_below: ComplexArray, sigma: NDArray[np.float64]
    ) -> ComplexArray:
        """Return the factor applied to the transmission coefficients.

        Args:
            kz_above: k_z in the medium above each interface.
            kz_below: k_z in the medium below each interface.
            sigma: Interface widths in nm.

        Returns:
            The factors, broadcast shape of the inputs.
        """
        ...


def _gaussian_transmission(
    kz_above: ComplexArray, kz_below: ComplexArray, sigma: NDArray[np.float64]
) -> ComplexArray:
    return np.exp(0.5 * (kz_above - kz_below) ** 2 * np.asarray(sigma, dtype=np.float64) ** 2)


@dataclass(frozen=True)
class SharpInterface:
    """Ideal sharp interfaces: the roughness is ignored."""

    name: ClassVar[str] = "sharp"

    def reflection_factor(
        self, kz_above: ComplexArray, kz_below: ComplexArray, sigma: NDArray[np.float64]
    ) -> ComplexArray:
        """Return ones.

        Args:
            kz_above: k_z in the medium above each interface.
            kz_below: k_z in the medium below each interface.
            sigma: Interface widths in nm (ignored).

        Returns:
            Ones with the broadcast shape of the inputs.
        """
        shape = np.broadcast_shapes(np.shape(kz_above), np.shape(kz_below), np.shape(sigma))
        return np.ones(shape, dtype=np.complex128)

    def transmission_factor(
        self, kz_above: ComplexArray, kz_below: ComplexArray, sigma: NDArray[np.float64]
    ) -> ComplexArray:
        """Return ones.

        Args:
            kz_above: k_z in the medium above each interface.
            kz_below: k_z in the medium below each interface.
            sigma: Interface widths in nm (ignored).

        Returns:
            Ones with the broadcast shape of the inputs.
        """
        return self.reflection_factor(kz_above, kz_below, sigma)


@dataclass(frozen=True)
class NevotCroce:
    """Névot–Croce factors for interfaces with a Gaussian height or composition profile.

    r → r·exp(−2·k_z,j·k_z,j+1·σ²) and t → t·exp((k_z,j − k_z,j+1)²·σ²/2)
    (L. Névot, P. Croce, Rev. Phys. Appl. 15, 761 (1980)).
    """

    name: ClassVar[str] = "nevot_croce"

    def reflection_factor(
        self, kz_above: ComplexArray, kz_below: ComplexArray, sigma: NDArray[np.float64]
    ) -> ComplexArray:
        """Return exp(−2·k_z,j·k_z,j+1·σ²).

        Args:
            kz_above: k_z in the medium above each interface.
            kz_below: k_z in the medium below each interface.
            sigma: Interface widths in nm.

        Returns:
            The factors, broadcast shape of the inputs.
        """
        return np.exp(-2.0 * kz_above * kz_below * np.asarray(sigma, dtype=np.float64) ** 2)

    def transmission_factor(
        self, kz_above: ComplexArray, kz_below: ComplexArray, sigma: NDArray[np.float64]
    ) -> ComplexArray:
        """Return exp((k_z,j − k_z,j+1)²·σ²/2).

        Args:
            kz_above: k_z in the medium above each interface.
            kz_below: k_z in the medium below each interface.
            sigma: Interface widths in nm.

        Returns:
            The factors, broadcast shape of the inputs.
        """
        return _gaussian_transmission(kz_above, kz_below, sigma)


@dataclass(frozen=True)
class DebyeWaller:
    """Debye–Waller factors for interfaces with a Gaussian height or composition profile.

    r → r·exp(−2·k_z,j²·σ²), with k_z,j taken in the medium above the interface. The
    transmission factor is the same as for `NevotCroce`.
    """

    name: ClassVar[str] = "debye_waller"

    def reflection_factor(
        self, kz_above: ComplexArray, kz_below: ComplexArray, sigma: NDArray[np.float64]
    ) -> ComplexArray:
        """Return exp(−2·k_z,j²·σ²).

        Args:
            kz_above: k_z in the medium above each interface.
            kz_below: k_z in the medium below each interface.
            sigma: Interface widths in nm.

        Returns:
            The factors, broadcast shape of the inputs.
        """
        factor = np.exp(-2.0 * kz_above**2 * np.asarray(sigma, dtype=np.float64) ** 2)
        return np.broadcast_to(factor, np.broadcast_shapes(factor.shape, np.shape(kz_below)))

    def transmission_factor(
        self, kz_above: ComplexArray, kz_below: ComplexArray, sigma: NDArray[np.float64]
    ) -> ComplexArray:
        """Return exp((k_z,j − k_z,j+1)²·σ²/2).

        Args:
            kz_above: k_z in the medium above each interface.
            kz_below: k_z in the medium below each interface.
            sigma: Interface widths in nm.

        Returns:
            The factors, broadcast shape of the inputs.
        """
        return _gaussian_transmission(kz_above, kz_below, sigma)


interface_models: Registry[InterfaceModel] = Registry("interface model")
"""Registry of interface roughness models."""

for _model in (SharpInterface(), NevotCroce(), DebyeWaller()):
    interface_models.register(_model.name, _model)
