"""Interface coefficients and interface roughness models.

Conventions: the stack is described by M media (ambient, layers, substrate) and has M − 1
interfaces. Interface j separates medium j (above) from medium j + 1 (below). Arrays carry the
evaluation points (energy/angle pairs) along the first axis.

For s polarization the coefficients refer to the electric field, for p polarization to the
magnetic field. In both cases the field component used is continuous across a sharp interface,
so that t = 1 + r. The reflectance |r|² is the same in either convention.
"""

from collections.abc import Callable
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from xstack.plugins import Registry

__all__ = [
    "InterfaceModel",
    "Polarization",
    "debye_waller",
    "fresnel_coefficients",
    "interface_models",
    "nevot_croce",
    "sharp",
    "wavevector_z",
]

type Polarization = Literal["s", "p"]
"""Polarization of the incident wave."""

type ComplexArray = NDArray[np.complex128]

type InterfaceModel = Callable[
    [ComplexArray, ComplexArray, NDArray[np.float64]], tuple[ComplexArray, ComplexArray]
]
"""Callable ``(kz_above, kz_below, sigma) -> (r_factor, t_factor)``.

Returns the factors by which the sharp-interface reflection and transmission coefficients are
multiplied to account for an interface of width ``sigma`` (nm). ``kz_above`` and ``kz_below``
have shape (P, M − 1) and are given in 1/nm, ``sigma`` has shape (M − 1,).
"""


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
    k = 2.0 * np.pi / wavelength
    n0 = n[:, :1]
    # n_j² − n_0²cos²θ written without the cancellation in 1 − cos²θ at grazing incidence.
    radicand = (n0 * np.sin(theta)[:, None]) ** 2 + (n - n0) * (n + n0)
    kz = k[:, None] * np.sqrt(radicand)
    return np.where(kz.imag < 0.0, -kz, kz)


def fresnel_coefficients(
    kz: ComplexArray, n: ComplexArray, polarization: Polarization
) -> ComplexArray:
    """Return the reflection coefficients of all sharp interfaces.

    Args:
        kz: Perpendicular wavevector components, shape (P, M), from `wavevector_z`.
        n: Complex refractive indices, shape (P, M).
        polarization: ``"s"`` or ``"p"``.

    Returns:
        Reflection coefficients r_j for waves incident from above, shape (P, M − 1).

    Raises:
        ValueError: If the polarization is unknown.
    """
    upper, lower = kz[:, :-1], kz[:, 1:]
    if polarization == "s":
        return (upper - lower) / (upper + lower)
    if polarization == "p":
        eps = n**2
        a = eps[:, 1:] * upper
        b = eps[:, :-1] * lower
        return (a - b) / (a + b)
    raise ValueError(f"unknown polarization {polarization!r}, expected 's' or 'p'")


def sharp(
    kz_above: ComplexArray, kz_below: ComplexArray, sigma: NDArray[np.float64]
) -> tuple[ComplexArray, ComplexArray]:
    """Ideal sharp interfaces: the roughness is ignored.

    Args:
        kz_above: k_z in the medium above each interface, shape (P, M − 1).
        kz_below: k_z in the medium below each interface, shape (P, M − 1).
        sigma: Interface widths in nm, shape (M − 1,).

    Returns:
        Factors (1, 1) for r and t, shape (P, M − 1) each.
    """
    ones = np.ones(np.broadcast_shapes(kz_above.shape, kz_below.shape), dtype=np.complex128)
    return ones, ones


def nevot_croce(
    kz_above: ComplexArray, kz_below: ComplexArray, sigma: NDArray[np.float64]
) -> tuple[ComplexArray, ComplexArray]:
    """Névot–Croce factors for interfaces with a Gaussian height or composition profile.

    r → r·exp(−2·k_z,j·k_z,j+1·σ²) and t → t·exp((k_z,j − k_z,j+1)²·σ²/2)
    (L. Névot, P. Croce, Rev. Phys. Appl. 15, 761 (1980)).

    Args:
        kz_above: k_z in the medium above each interface, shape (P, M − 1).
        kz_below: k_z in the medium below each interface, shape (P, M − 1).
        sigma: Interface widths in nm, shape (M − 1,).

    Returns:
        Factors for r and t, shape (P, M − 1) each.
    """
    s2 = np.asarray(sigma, dtype=np.float64) ** 2
    r_factor = np.exp(-2.0 * kz_above * kz_below * s2)
    t_factor = np.exp(0.5 * (kz_above - kz_below) ** 2 * s2)
    return r_factor, t_factor


def debye_waller(
    kz_above: ComplexArray, kz_below: ComplexArray, sigma: NDArray[np.float64]
) -> tuple[ComplexArray, ComplexArray]:
    """Debye–Waller factors for interfaces with a Gaussian height or composition profile.

    r → r·exp(−2·k_z,j²·σ²), with k_z,j taken in the medium above the interface. The
    transmission factor is the same as for `nevot_croce`.

    Args:
        kz_above: k_z in the medium above each interface, shape (P, M − 1).
        kz_below: k_z in the medium below each interface, shape (P, M − 1).
        sigma: Interface widths in nm, shape (M − 1,).

    Returns:
        Factors for r and t, shape (P, M − 1) each.
    """
    s2 = np.asarray(sigma, dtype=np.float64) ** 2
    r_factor = np.exp(-2.0 * kz_above**2 * s2)
    t_factor = np.exp(0.5 * (kz_above - kz_below) ** 2 * s2)
    return r_factor, t_factor


interface_models: Registry[InterfaceModel] = Registry("interface model")
"""Registry of interface roughness models."""

interface_models.register("sharp", sharp)
interface_models.register("nevot_croce", nevot_croce)
interface_models.register("debye_waller", debye_waller)
