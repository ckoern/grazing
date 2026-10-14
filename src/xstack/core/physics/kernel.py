"""Reflectivity kernel based on the Parratt recursion.

The kernel works on plain arrays and knows nothing about materials or coatings. A stack of M
media (ambient, M − 2 layers, substrate) is described by

* ``n``: complex refractive indices, shape (P, M), for P evaluation points, or shape (P, U)
  for U distinct media together with ``media``, an index array of shape (M,) that assigns a
  column of ``n`` to every medium,
* ``thickness``: layer thicknesses in nm, shape (M − 2,),
* ``roughness``: interface widths in nm, shape (M − 1,), interface j lying between media
  j and j + 1,
* ``theta`` and ``wavelength``: grazing angle (rad) and vacuum wavelength (nm), shape (P,).

Quantities that depend only on the media (wavevectors, interface coefficients, layer phase
factors) are computed once per distinct medium, interface or layer type, which makes stacks
with many repeated layers fast. The recursion runs from the substrate to the surface. It uses
only decaying exponentials and is numerically stable for thick and strongly absorbing stacks.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from xstack.core.physics.fresnel import (
    InterfaceModel,
    Polarization,
    _kz,
    interface_models,
    interface_reflection,
)

__all__ = ["KernelResult", "parratt", "reflectance"]

type _Complex = NDArray[np.complex128]


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


@dataclass(frozen=True)
class _Prepared:
    """Validated inputs with the media reduced to distinct columns."""

    n: _Complex
    media: NDArray[np.intp]
    kz: _Complex
    thickness: NDArray[np.float64]
    roughness: NDArray[np.float64]

    @property
    def points(self) -> int:
        return int(self.n.shape[0])


def _distinct_columns(n: _Complex) -> tuple[_Complex, NDArray[np.intp]]:
    rows = np.ascontiguousarray(n.T)
    labels: dict[bytes, int] = {}
    index = np.empty(rows.shape[0], dtype=np.intp)
    first: list[int] = []
    for j, row in enumerate(rows):
        key = row.tobytes()
        if key not in labels:
            labels[key] = len(first)
            first.append(j)
        index[j] = labels[key]
    return n[:, first], index


def _prepare(
    n: ArrayLike,
    thickness: ArrayLike,
    roughness: ArrayLike,
    theta: ArrayLike,
    wavelength: ArrayLike,
    media: ArrayLike | None,
) -> _Prepared:
    n_arr = np.asarray(n, dtype=np.complex128)
    if n_arr.ndim != 2:
        raise ValueError("n must be a two-dimensional array")
    points = n_arr.shape[0]
    if media is None:
        if n_arr.shape[1] < 2:
            raise ValueError("n must have shape (P, M) with M >= 2")
        n_arr, index = _distinct_columns(n_arr)
    else:
        index = np.asarray(media, dtype=np.intp)
        if index.ndim != 1 or index.size < 2:
            raise ValueError("media must be a one-dimensional index array with at least 2 entries")
        if index.min() < 0 or index.max() >= n_arr.shape[1]:
            raise ValueError("media indices must refer to columns of n")
    count = index.size
    t_arr = np.asarray(thickness, dtype=np.float64)
    sigma = np.asarray(roughness, dtype=np.float64)
    if t_arr.shape != (count - 2,):
        raise ValueError(f"thickness must have shape ({count - 2},), got {t_arr.shape}")
    if sigma.shape != (count - 1,):
        raise ValueError(f"roughness must have shape ({count - 1},), got {sigma.shape}")
    if np.any(t_arr < 0) or np.any(sigma < 0):
        raise ValueError("thicknesses and roughnesses must not be negative")
    theta_arr = np.broadcast_to(np.asarray(theta, dtype=np.float64), (points,))
    lam = np.broadcast_to(np.asarray(wavelength, dtype=np.float64), (points,))
    if np.any(theta_arr <= 0) or np.any(theta_arr > np.pi / 2):
        raise ValueError("grazing angles must be in (0, pi/2]")
    if np.any(lam <= 0):
        raise ValueError("wavelengths must be positive")
    kz = _kz(n_arr, n_arr[:, index[:1]], theta_arr, lam)
    return _Prepared(n_arr, index, kz, t_arr, sigma)


def _combinations(*keys: NDArray[np.generic]) -> tuple[list[tuple[object, ...]], NDArray[np.intp]]:
    """Return the distinct key tuples and the index of each element's tuple."""
    table: dict[tuple[object, ...], int] = {}
    index = np.empty(len(keys[0]), dtype=np.intp)
    for i, key in enumerate(zip(*(k.tolist() for k in keys), strict=True)):
        index[i] = table.setdefault(key, len(table))
    return list(table), index


def _interface_rows(
    prep: _Prepared, polarization: Polarization, model: InterfaceModel, transmission: bool
) -> tuple[_Complex, _Complex | None]:
    """Return interface reflection (and transmission) coefficients, shape (M − 1, P)."""
    above, below = prep.media[:-1], prep.media[1:]
    combos, index = _combinations(above, below, prep.roughness)
    a = np.array([c[0] for c in combos], dtype=np.intp)
    b = np.array([c[1] for c in combos], dtype=np.intp)
    s = np.array([c[2] for c in combos], dtype=np.float64)
    kz_a, kz_b = prep.kz[:, a], prep.kz[:, b]
    r_sharp = interface_reflection(kz_a, kz_b, prep.n[:, a], prep.n[:, b], polarization)
    r = r_sharp * model.reflection_factor(kz_a, kz_b, s)
    r_rows = np.ascontiguousarray(r.T[index])
    if not transmission:
        return r_rows, None
    t = (1.0 + r_sharp) * model.transmission_factor(kz_a, kz_b, s)
    return r_rows, np.ascontiguousarray(t.T[index])


def _propagation_rows(prep: _Prepared) -> _Complex:
    """Return the layer phase factors exp(i·kz·d), shape (M − 2, P)."""
    if prep.thickness.size == 0:
        return np.empty((0, prep.points), dtype=np.complex128)
    combos, index = _combinations(prep.media[1:-1], prep.thickness)
    m = np.array([c[0] for c in combos], dtype=np.intp)
    d = np.array([c[1] for c in combos], dtype=np.float64)
    phase = np.exp(1j * prep.kz[:, m] * d)
    return np.ascontiguousarray(phase.T[index])


def _recursion(
    r_rows: _Complex, t_rows: _Complex | None, propagation: _Complex
) -> tuple[_Complex, _Complex]:
    """Run the Parratt recursion from the substrate to the surface.

    Args:
        r_rows: Interface reflection coefficients, shape (M − 1, P).
        t_rows: Interface transmission coefficients, shape (M − 1, P), or ``None`` to skip
            the transmission amplitude.
        propagation: Layer phase factors exp(i·kz·d), shape (M − 2, P).

    Returns:
        Reflection amplitudes and transmission amplitudes (ones if `t_rows` is ``None``).
    """
    interfaces, points = r_rows.shape
    phase2 = propagation * propagation
    # x: ratio of up- to down-going amplitude at the bottom of the current medium.
    x = np.zeros(points, dtype=np.complex128)
    y = np.empty_like(x)
    denominator = np.empty_like(x)
    t_total = np.ones(points, dtype=np.complex128)
    for j in range(interfaces - 1, -1, -1):
        if j < interfaces - 1:
            np.multiply(x, phase2[j], out=y)
            if t_rows is not None:
                t_total *= propagation[j]
        else:
            y[:] = x
        np.multiply(r_rows[j], y, out=denominator)
        denominator += 1.0
        x = (r_rows[j] + y) / denominator
        if t_rows is not None:
            t_total *= t_rows[j] / denominator
    return x, t_total


def parratt(
    n: ArrayLike,
    thickness: ArrayLike,
    roughness: ArrayLike,
    theta: ArrayLike,
    wavelength: ArrayLike,
    polarization: Polarization = "s",
    interface_model: str = "nevot_croce",
    media: ArrayLike | None = None,
) -> KernelResult:
    """Compute reflection and transmission of a planar multilayer stack.

    Args:
        n: Complex refractive indices n = 1 − δ + iβ, shape (P, M) with M ≥ 2, column 0 being
            the ambient medium and column M − 1 the substrate. With `media`, shape (P, U) for
            U distinct media.
        thickness: Layer thicknesses in nm, shape (M − 2,).
        roughness: Interface widths σ in nm, shape (M − 1,). Entry j belongs to the interface
            between media j and j + 1.
        theta: Grazing angles of incidence in rad, shape (P,) or scalar. Must be in (0, π/2].
        wavelength: Vacuum wavelengths in nm, shape (P,) or scalar.
        polarization: ``"s"`` or ``"p"``.
        interface_model: Name of a model registered in
            `xstack.core.physics.fresnel.interface_models`.
        media: Optional column index into `n` for each of the M media, ambient first.

    Returns:
        Amplitudes and intensities for all P points.

    Raises:
        ValueError: If array shapes are inconsistent or values are out of range.
        KeyError: If the interface model is unknown.
    """
    prep = _prepare(n, thickness, roughness, theta, wavelength, media)
    model = interface_models.get(interface_model)
    r_rows, t_rows = _interface_rows(prep, polarization, model, transmission=True)
    x, t_total = _recursion(r_rows, t_rows, _propagation_rows(prep))

    kz0, kz_sub = prep.kz[:, prep.media[0]], prep.kz[:, prep.media[-1]]
    if polarization == "s":
        flux_ratio = kz_sub.real / kz0.real
    else:
        eps0, eps_sub = prep.n[:, prep.media[0]] ** 2, prep.n[:, prep.media[-1]] ** 2
        flux_ratio = (kz_sub / eps_sub).real / (kz0 / eps0).real
    return KernelResult(
        r=x, t=t_total, reflectance=np.abs(x) ** 2, transmittance=flux_ratio * np.abs(t_total) ** 2
    )


def reflectance(
    n: ArrayLike,
    thickness: ArrayLike,
    roughness: ArrayLike,
    theta: ArrayLike,
    wavelength: ArrayLike,
    polarization: Polarization | Literal["unpolarized"] = "unpolarized",
    interface_model: str = "nevot_croce",
    media: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Compute the reflectance of a planar multilayer stack.

    Args:
        n: Complex refractive indices, shape (P, M), or (P, U) with `media`. See `parratt`.
        thickness: Layer thicknesses in nm, shape (M − 2,).
        roughness: Interface widths σ in nm, shape (M − 1,).
        theta: Grazing angles of incidence in rad, shape (P,) or scalar.
        wavelength: Vacuum wavelengths in nm, shape (P,) or scalar.
        polarization: ``"s"``, ``"p"`` or ``"unpolarized"`` (mean of s and p).
        interface_model: Name of a registered interface model.
        media: Optional column index into `n` for each of the M media, ambient first.

    Returns:
        Reflectance, shape (P,).

    Raises:
        ValueError: If array shapes are inconsistent or values are out of range.
        KeyError: If the interface model is unknown.
    """
    prep = _prepare(n, thickness, roughness, theta, wavelength, media)
    model = interface_models.get(interface_model)
    propagation = _propagation_rows(prep)
    if polarization == "unpolarized":
        # Both polarizations in one pass of the recursion.
        r_s, _ = _interface_rows(prep, "s", model, transmission=False)
        r_p, _ = _interface_rows(prep, "p", model, transmission=False)
        x, _ = _recursion(
            np.concatenate([r_s, r_p], axis=1), None, np.concatenate([propagation] * 2, axis=1)
        )
        r = np.abs(x) ** 2
        return np.asarray(0.5 * (r[: prep.points] + r[prep.points :]), dtype=np.float64)
    r_rows, _ = _interface_rows(prep, polarization, model, transmission=False)
    x, _ = _recursion(r_rows, None, propagation)
    return np.asarray(np.abs(x) ** 2, dtype=np.float64)
