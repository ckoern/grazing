"""Analytic estimates for periodic and depth-graded multilayers.

These closed-form relations are fast enough to scan large parameter spaces and provide start
values and bounds for the full calculation. Optical constants enter through the decrement δ
and the absorption index β of the absorber (H) and spacer (L).

Refraction is included through the period-averaged decrement δ̄ = Γ·δ_H + (1 − Γ)·δ_L:

    m·λ = 2·d·√(sin²θ − 2δ̄)

Setting δ̄ = 0 gives the plain Bragg law m·λ = 2·d·sinθ.
"""

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq

__all__ = [
    "absorption_bilayers",
    "bragg_angle",
    "bragg_period",
    "bragg_wavelength",
    "extinction_bilayers",
    "mean_delta",
    "optimal_gamma",
    "relative_bandwidth",
]


def mean_delta(delta_h: ArrayLike, delta_l: ArrayLike, gamma: ArrayLike) -> NDArray[np.float64]:
    """Return the period-averaged decrement δ̄ = Γ·δ_H + (1 − Γ)·δ_L.

    Args:
        delta_h: Decrement of the absorber.
        delta_l: Decrement of the spacer.
        gamma: Absorber fraction Γ.

    Returns:
        δ̄, broadcast over the inputs.
    """
    g = np.asarray(gamma, dtype=np.float64)
    return np.asarray(g * np.asarray(delta_h) + (1.0 - g) * np.asarray(delta_l), dtype=np.float64)


def bragg_angle(
    period: ArrayLike, wavelength: ArrayLike, order: int = 1, delta: ArrayLike = 0.0
) -> NDArray[np.float64]:
    """Return the grazing angle of a Bragg peak.

    Args:
        period: Multilayer period d in nm.
        wavelength: Vacuum wavelength in nm.
        order: Diffraction order m ≥ 1.
        delta: Period-averaged decrement δ̄ (0 for the plain Bragg law).

    Returns:
        Grazing angle in rad. NaN where no Bragg reflection exists.
    """
    s2 = (order * np.asarray(wavelength) / (2.0 * np.asarray(period))) ** 2 + 2.0 * np.asarray(
        delta
    )
    with np.errstate(invalid="ignore"):
        return np.asarray(np.arcsin(np.where(s2 <= 1.0, np.sqrt(s2), np.nan)), dtype=np.float64)


def bragg_period(
    theta: ArrayLike, wavelength: ArrayLike, order: int = 1, delta: ArrayLike = 0.0
) -> NDArray[np.float64]:
    """Return the period that reflects a wavelength at a given angle.

    Args:
        theta: Grazing angle in rad.
        wavelength: Vacuum wavelength in nm.
        order: Diffraction order m ≥ 1.
        delta: Period-averaged decrement δ̄ (0 for the plain Bragg law).

    Returns:
        Period d in nm. NaN where θ is below the critical angle √(2δ̄).
    """
    s2 = np.sin(np.asarray(theta)) ** 2 - 2.0 * np.asarray(delta)
    with np.errstate(invalid="ignore", divide="ignore"):
        root = np.where(s2 > 0.0, np.sqrt(np.abs(s2)), np.nan)
        return np.asarray(order * np.asarray(wavelength) / (2.0 * root), dtype=np.float64)


def bragg_wavelength(
    period: ArrayLike, theta: ArrayLike, order: int = 1, delta_coefficient: ArrayLike = 0.0
) -> NDArray[np.float64]:
    """Return the wavelength of a Bragg peak at a given angle.

    The decrement scales with the wavelength squared away from absorption edges, δ̄ = c·λ².
    With this, mλ = 2d√(sin²θ − 2cλ²) is solved for λ.

    Args:
        period: Multilayer period d in nm.
        theta: Grazing angle in rad.
        order: Diffraction order m ≥ 1.
        delta_coefficient: c = δ̄/λ² in 1/nm² (0 for the plain Bragg law).

    Returns:
        Vacuum wavelength in nm.
    """
    d = np.asarray(period, dtype=np.float64)
    c = np.asarray(delta_coefficient, dtype=np.float64)
    return np.asarray(np.sin(np.asarray(theta)) / np.sqrt((order / (2.0 * d)) ** 2 + 2.0 * c))


def optimal_gamma(beta_h: float, beta_l: float) -> float:
    """Return Γ that maximizes the first-order peak reflectivity of a periodic multilayer.

    Solves tan(πΓ) = π·[Γ + β_L / (β_H − β_L)] (A. V. Vinogradov, B. Ya. Zeldovich,
    Appl. Opt. 16, 89 (1977)).

    Args:
        beta_h: Absorption index of the absorber.
        beta_l: Absorption index of the spacer. Must be smaller than `beta_h`.

    Returns:
        Optimal Γ in (0, 0.5].

    Raises:
        ValueError: If `beta_h` is not larger than `beta_l`.
    """
    if not beta_h > beta_l >= 0.0:
        raise ValueError("optimal_gamma requires beta_h > beta_l >= 0")
    ratio = beta_l / (beta_h - beta_l)

    def f(g: float) -> float:
        return math.tan(math.pi * g) - math.pi * (g + ratio)

    upper = 0.5 - 1e-12
    if f(upper) < 0.0:
        return 0.5
    return float(brentq(f, 1e-12, upper, xtol=1e-14))


def extinction_bilayers(
    delta_h: float,
    beta_h: float,
    delta_l: float,
    beta_l: float,
    gamma: float,
    theta: float,
    order: int = 1,
) -> float:
    """Estimate the number of bilayers after which a periodic multilayer is extinction-limited.

    Each interface reflects with amplitude ≈ Δn / (2·sin²θ), with the index contrast
    Δn = (δ_H − δ_L) − i(β_H − β_L). The two interfaces of one period add up at the m-th
    Bragg condition to ``|r_1| ≈ |Δn|·|sin(πmΓ)| / sin²θ``. Its inverse estimates the number of
    periods that interact with the beam.

    Args:
        delta_h: Decrement of the absorber.
        beta_h: Absorption index of the absorber.
        delta_l: Decrement of the spacer.
        beta_l: Absorption index of the spacer.
        gamma: Absorber fraction Γ.
        theta: Grazing angle in rad.
        order: Diffraction order m.

    Returns:
        Estimated number of bilayers (not rounded).
    """
    contrast = abs(complex(delta_h - delta_l, beta_h - beta_l))
    r1 = contrast * abs(math.sin(math.pi * order * gamma)) / math.sin(theta) ** 2
    return math.inf if r1 == 0.0 else 1.0 / r1


def absorption_bilayers(
    beta_h: float, beta_l: float, gamma: float, period: float, theta: float, wavelength: float
) -> float:
    """Estimate the number of bilayers after which absorption limits a periodic multilayer.

    It is the number of periods at which the intensity on the way in and out has dropped by
    1/e: N_abs = λ·sinθ / (8π·β̄·d), with β̄ = Γ·β_H + (1 − Γ)·β_L.

    Args:
        beta_h: Absorption index of the absorber.
        beta_l: Absorption index of the spacer.
        gamma: Absorber fraction Γ.
        period: Period d in nm.
        theta: Grazing angle in rad.
        wavelength: Vacuum wavelength in nm.

    Returns:
        Estimated number of bilayers (not rounded).
    """
    beta = gamma * beta_h + (1.0 - gamma) * beta_l
    if beta <= 0.0:
        return math.inf
    return wavelength * math.sin(theta) / (8.0 * math.pi * beta * period)


def relative_bandwidth(bilayers: float, order: int = 1) -> float:
    """Estimate the relative bandwidth Δλ/λ = Δθ/tanθ of a Bragg peak.

    Valid for unsaturated multilayers. For saturated ones, pass the effective number of
    bilayers, e.g. the smaller of `extinction_bilayers` and `absorption_bilayers`.

    Args:
        bilayers: (Effective) number of bilayers.
        order: Diffraction order m.

    Returns:
        Δλ/λ (full width).
    """
    return 1.0 / (order * bilayers)
