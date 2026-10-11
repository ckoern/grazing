"""Tier 4: semi-analytic physics checks of multilayer reflectivity."""

import numpy as np
import pytest

from xstack.materials import Material, delta_beta, get
from xstack.physics import bragg, reflectivity
from xstack.structure import DepthGraded, Mirror, Periodic
from xstack.units import energy_to_wavelength

W, SI, C = get("W"), get("Si"), get("C")
ENERGY = 8048.0
LAMBDA = energy_to_wavelength(ENERGY)


def peak(theta: np.ndarray, r: np.ndarray) -> tuple[float, float, float]:
    """Return centre of the half-maximum region, its full width and the maximum."""
    i = int(np.argmax(r))
    above = theta[r >= r[i] / 2]
    # Restrict to the contiguous region around the maximum.
    lo = i
    while lo > 0 and r[lo - 1] >= r[i] / 2:
        lo -= 1
    hi = i
    while hi < len(r) - 1 and r[hi + 1] >= r[i] / 2:
        hi += 1
    assert above.size > 0
    return 0.5 * (theta[lo] + theta[hi]), theta[hi] - theta[lo], float(r[i])


def periodic_mirror(h: Material, low: Material, n: int, d: float, gamma: float) -> Mirror:
    return Mirror(Periodic.create(h, low, N=n, d=d, gamma=gamma), SI)


@pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.parametrize(("h", "low"), [(W, SI), (W, C)], ids=["W/Si", "W/C"])
def test_peak_positions_follow_corrected_bragg_law(h: Material, low: Material, order: int) -> None:
    d, gamma = 5.0, 0.3
    mirror = periodic_mirror(h, low, 60, d, gamma)
    delta = bragg.mean_delta(delta_beta(h, ENERGY).real, delta_beta(low, ENERGY).real, gamma)
    theta_b = float(bragg.bragg_angle(d, LAMBDA, order, delta))
    theta = np.linspace(0.85 * theta_b, 1.15 * theta_b, 6001)
    centre, width, _ = peak(theta, reflectivity(mirror.stack(), ENERGY, theta, "s"))
    assert abs(centre - theta_b) < 0.3 * width
    assert centre > float(bragg.bragg_angle(d, LAMBDA, order))


def test_gamma_one_half_suppresses_second_order() -> None:
    def order_peak(gamma: float, order: int) -> float:
        mirror = periodic_mirror(W, C, 40, 5.0, gamma)
        theta_b = float(bragg.bragg_angle(5.0, LAMBDA, order, 2e-5))
        theta = np.linspace(0.9 * theta_b, 1.1 * theta_b, 2001)
        return float(reflectivity(mirror.stack(), ENERGY, theta, "s").max())

    # Refraction makes the optical thickness fraction differ slightly from the geometric Γ, so
    # the suppression is strong but not perfect.
    assert order_peak(0.5, 2) < 1e-2 * order_peak(0.5, 1)
    assert order_peak(0.5, 2) < 0.05 * order_peak(0.3, 2)
    assert order_peak(0.3, 2) > 0.1 * order_peak(0.3, 1)


def test_bandwidth_scales_inversely_with_bilayers() -> None:
    # Weak contrast (Si/C) keeps the multilayer far from saturation.
    widths = []
    for n in (20, 40, 80):
        mirror = periodic_mirror(SI, C, n, 4.0, 0.4)
        theta_b = float(bragg.bragg_angle(4.0, LAMBDA, 1, 7e-6))
        theta = np.linspace(0.85 * theta_b, 1.15 * theta_b, 8001)
        _, width, _ = peak(theta, reflectivity(mirror.stack(), ENERGY, theta, "s"))
        widths.append(width)
    assert widths[0] / widths[1] == pytest.approx(2.0, rel=0.05)
    assert widths[1] / widths[2] == pytest.approx(2.0, rel=0.05)


def test_reflectivity_saturates_with_bilayers() -> None:
    peaks = []
    for n in (10, 40, 160, 320):
        mirror = periodic_mirror(W, SI, n, 4.0, 0.4)
        theta = np.linspace(0.018, 0.022, 2001)
        peaks.append(reflectivity(mirror.stack(), ENERGY, theta, "s").max())
    assert peaks[0] < peaks[1] < peaks[2]
    assert peaks[3] == pytest.approx(peaks[2], rel=1e-3)


def test_optimal_gamma_is_near_maximum_reflectivity() -> None:
    db_w, db_si = delta_beta(W, ENERGY), delta_beta(SI, ENERGY)
    g_opt = bragg.optimal_gamma(float(db_w.imag), float(db_si.imag))
    gammas = np.linspace(0.1, 0.6, 26)
    peaks = []
    for g in gammas:
        mirror = periodic_mirror(W, SI, 200, 4.0, float(g))
        theta = np.linspace(0.018, 0.0225, 1501)
        peaks.append(reflectivity(mirror.stack(), ENERGY, theta, "s").max())
    best = gammas[int(np.argmax(peaks))]
    assert abs(best - g_opt) < 0.1


def test_depth_graded_covers_predicted_energy_band() -> None:
    theta = 4e-3
    coating = DepthGraded.create(W, SI, N=150, d_min=2.5, d_max=8.0, c=0.25, gamma_top=0.4)
    mirror = Mirror(coating, SI)
    delta_coeff = 1.6e-5 / LAMBDA**2 * 0.4 + 7.6e-6 / LAMBDA**2 * 0.6
    e_low = 1239.84198 / float(bragg.bragg_wavelength(8.0, theta, 1, delta_coeff))
    e_high = 1239.84198 / float(bragg.bragg_wavelength(2.5, theta, 1, delta_coeff))
    energies = np.linspace(e_low, e_high, 200)[20:-20]
    r = reflectivity(mirror.stack(), energies, theta, "s")
    assert r.min() > 0.05
    outside = reflectivity(mirror.stack(), np.array([1.6 * e_high, 1.8 * e_high]), theta, "s")
    assert outside.max() < 0.2 * r.mean()
