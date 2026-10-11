"""Reflectivity kernel based on the Parratt recursion.

The kernel works on plain arrays and knows nothing about materials or coatings. A stack of M
media (ambient, M − 2 layers, substrate) is described by

* ``n``: complex refractive indices, shape (P, M), for P evaluation points,
* ``thickness``: layer thicknesses in nm, shape (M − 2,),
* ``roughness``: interface widths in nm, shape (M − 1,), interface j lying between media
  j and j + 1,
* ``theta`` and ``wavelength``: grazing angle (rad) and vacuum wavelength (nm), shape (P,).

The recursion runs from the substrate to the surface. It uses only decaying exponentials and is
numerically stable for thick and strongly absorbing stacks.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from xstack.physics.fresnel import (
    Polarization,
    fresnel_coefficients,
    interface_models,
    wavevector_z,
)

__all__ = ["KernelResult", "parratt", "reflectance"]


@dataclass(frozen=True)
class KernelResult:
    """Complex amplitudes and intensities of a stack for one polarization.

    Attributes:
        r: Complex reflection amplitude, shape (P,).
        t: Complex transmission amplitude into the substrate, referenced to the top of the
            substrate, shape (P,). For p polarization it refers to the magnetic field.
        reflectance: Reflected intensity fraction |r|², shape (P,).
        transmittance: Fraction of the incident energy flux transmitted into the substrate,
            shape (P,).
    """

    r: NDArray[np.complex128]
    t: NDArray[np.complex128]
    reflectance: NDArray[np.float64]
    transmittance: NDArray[np.float64]


def parratt(
    n: ArrayLike,
    thickness: ArrayLike,
    roughness: ArrayLike,
    theta: ArrayLike,
    wavelength: ArrayLike,
    polarization: Polarization = "s",
    interface_model: str = "nevot_croce",
) -> KernelResult:
    """Compute reflection and transmission of a planar multilayer stack.

    Args:
        n: Complex refractive indices n = 1 − δ + iβ, shape (P, M) with M ≥ 2. Column 0 is
            the ambient medium, column M − 1 the substrate.
        thickness: Layer thicknesses in nm, shape (M − 2,).
        roughness: Interface widths σ in nm, shape (M − 1,). Entry j belongs to the interface
            between media j and j + 1.
        theta: Grazing angles of incidence in rad, shape (P,) or scalar. Must be in (0, π/2].
        wavelength: Vacuum wavelengths in nm, shape (P,) or scalar.
        polarization: ``"s"`` or ``"p"``.
        interface_model: Name of a model registered in
            `xstack.physics.fresnel.interface_models`.

    Returns:
        Amplitudes and intensities for all P points.

    Raises:
        ValueError: If array shapes are inconsistent or values are out of range.
        KeyError: If the interface model is unknown.
    """
    n_arr = np.asarray(n, dtype=np.complex128)
    if n_arr.ndim != 2 or n_arr.shape[1] < 2:
        raise ValueError("n must have shape (P, M) with M >= 2")
    points, media = n_arr.shape
    t_arr = np.asarray(thickness, dtype=np.float64)
    sigma = np.asarray(roughness, dtype=np.float64)
    if t_arr.shape != (media - 2,):
        raise ValueError(f"thickness must have shape ({media - 2},), got {t_arr.shape}")
    if sigma.shape != (media - 1,):
        raise ValueError(f"roughness must have shape ({media - 1},), got {sigma.shape}")
    if np.any(t_arr < 0) or np.any(sigma < 0):
        raise ValueError("thicknesses and roughnesses must not be negative")
    theta_arr = np.broadcast_to(np.asarray(theta, dtype=np.float64), (points,))
    lam = np.broadcast_to(np.asarray(wavelength, dtype=np.float64), (points,))
    if np.any(theta_arr <= 0) or np.any(theta_arr > np.pi / 2):
        raise ValueError("grazing angles must be in (0, pi/2]")
    if np.any(lam <= 0):
        raise ValueError("wavelengths must be positive")
    model = interface_models.get(interface_model)

    kz = wavevector_z(n_arr, theta_arr, lam)
    r_sharp = fresnel_coefficients(kz, n_arr, polarization)
    r_factor, t_factor = model(kz[:, :-1], kz[:, 1:], sigma)
    # Interface- and layer-major copies, so that the loop below works on contiguous rows.
    r_int = np.ascontiguousarray((r_sharp * r_factor).T)
    t_int = np.ascontiguousarray(((1.0 + r_sharp) * t_factor).T)
    # Phase factors exp(i·kz·d) of the layers (media 1 … M − 2).
    propagation = np.ascontiguousarray(np.exp(1j * kz[:, 1:-1] * t_arr).T)

    # x: ratio of up- to down-going amplitude at the bottom of the current medium.
    x = np.zeros(points, dtype=np.complex128)
    y = np.empty_like(x)
    denominator = np.empty_like(x)
    t_total = np.ones(points, dtype=np.complex128)
    for j in range(media - 2, -1, -1):
        if j < media - 2:
            phase = propagation[j]
            np.multiply(x, phase * phase, out=y)
            t_total *= phase
        else:
            y[:] = x
        np.multiply(r_int[j], y, out=denominator)
        denominator += 1.0
        x = (r_int[j] + y) / denominator
        t_total *= t_int[j] / denominator

    reflectance = np.abs(x) ** 2
    if polarization == "s":
        flux_ratio = kz[:, -1].real / kz[:, 0].real
    else:
        eps = n_arr**2
        flux_ratio = (kz[:, -1] / eps[:, -1]).real / (kz[:, 0] / eps[:, 0]).real
    transmittance = flux_ratio * np.abs(t_total) ** 2
    return KernelResult(r=x, t=t_total, reflectance=reflectance, transmittance=transmittance)


def reflectance(
    n: ArrayLike,
    thickness: ArrayLike,
    roughness: ArrayLike,
    theta: ArrayLike,
    wavelength: ArrayLike,
    polarization: Polarization | Literal["unpolarized"] = "unpolarized",
    interface_model: str = "nevot_croce",
) -> NDArray[np.float64]:
    """Compute the reflectance of a planar multilayer stack.

    Args:
        n: Complex refractive indices, shape (P, M). See `parratt`.
        thickness: Layer thicknesses in nm, shape (M − 2,).
        roughness: Interface widths σ in nm, shape (M − 1,).
        theta: Grazing angles of incidence in rad, shape (P,) or scalar.
        wavelength: Vacuum wavelengths in nm, shape (P,) or scalar.
        polarization: ``"s"``, ``"p"`` or ``"unpolarized"`` (mean of s and p).
        interface_model: Name of a registered interface model.

    Returns:
        Reflectance, shape (P,).

    Raises:
        ValueError: If array shapes are inconsistent or values are out of range.
        KeyError: If the interface model is unknown.
    """
    if polarization == "unpolarized":
        r_s = parratt(n, thickness, roughness, theta, wavelength, "s", interface_model)
        r_p = parratt(n, thickness, roughness, theta, wavelength, "p", interface_model)
        return 0.5 * (r_s.reflectance + r_p.reflectance)
    return parratt(
        n, thickness, roughness, theta, wavelength, polarization, interface_model
    ).reflectance
