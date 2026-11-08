import math

import numpy as np
import pytest

from grazing.core.physics import bragg


def test_plain_bragg_law_round_trip() -> None:
    theta = bragg.bragg_angle(4.0, 0.154, order=1)
    assert math.sin(theta) == pytest.approx(0.154 / 8.0)
    assert bragg.bragg_period(theta, 0.154) == pytest.approx(4.0)
    assert bragg.bragg_wavelength(4.0, theta) == pytest.approx(0.154)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_refraction_corrected_round_trip(order: int) -> None:
    lam, d, delta = 0.154, 4.0, 2.3e-5
    theta = bragg.bragg_angle(d, lam, order, delta)
    assert order * lam == pytest.approx(2 * d * math.sqrt(math.sin(theta) ** 2 - 2 * delta))
    assert bragg.bragg_period(theta, lam, order, delta) == pytest.approx(d)
    c = delta / lam**2
    assert bragg.bragg_wavelength(d, theta, order, c) == pytest.approx(lam)


def test_refraction_shifts_peak_to_larger_angle() -> None:
    assert bragg.bragg_angle(4.0, 0.154, delta=2e-5) > bragg.bragg_angle(4.0, 0.154)


def test_no_reflection_and_below_critical_angle_give_nan() -> None:
    assert np.isnan(bragg.bragg_angle(0.05, 0.154))
    assert np.isnan(bragg.bragg_period(1e-3, 0.154, delta=1e-5))


def test_vectorized() -> None:
    d = np.array([3.0, 4.0, 5.0])
    theta = bragg.bragg_angle(d, 0.154)
    assert theta.shape == (3,)
    np.testing.assert_allclose(bragg.bragg_period(theta, 0.154), d)


def test_mean_delta() -> None:
    assert bragg.mean_delta(4e-5, 1e-5, 0.25) == pytest.approx(1.75e-5)


def test_optimal_gamma_satisfies_equation() -> None:
    beta_h, beta_l = 3.3e-6, 1.7e-7
    g = bragg.optimal_gamma(beta_h, beta_l)
    assert 0 < g < 0.5
    assert math.tan(math.pi * g) == pytest.approx(math.pi * (g + beta_l / (beta_h - beta_l)))


def test_optimal_gamma_limits() -> None:
    # Non-absorbing spacer: tan(πΓ) = πΓ has the solution Γ → 0 only; the optimum is small.
    assert bragg.optimal_gamma(1e-6, 0.0) < 0.05
    # Spacer almost as absorbing as the absorber: Γ approaches 0.5.
    assert bragg.optimal_gamma(1e-6, 0.99e-6) == pytest.approx(0.5, abs=0.01)
    with pytest.raises(ValueError, match="beta_h > beta_l"):
        bragg.optimal_gamma(1e-7, 1e-6)


def test_extinction_and_absorption_estimates() -> None:
    n_ext = bragg.extinction_bilayers(4.6e-5, 3.3e-6, 7.6e-6, 1.7e-7, 0.4, 0.0204)
    n_abs = bragg.absorption_bilayers(3.3e-6, 1.7e-7, 0.4, 4.0, 0.0204, 0.154)
    assert 5 < n_ext < 30
    assert 5 < n_abs < 50
    assert bragg.extinction_bilayers(1e-5, 0, 1e-5, 0, 0.4, 0.02) == math.inf
    assert bragg.absorption_bilayers(0, 0, 0.4, 4.0, 0.02, 0.154) == math.inf
    # Γ = 1/2 suppresses the second order.
    assert bragg.extinction_bilayers(4e-5, 1e-6, 1e-5, 0, 0.5, 0.04, order=2) > 1e10


def test_relative_bandwidth() -> None:
    assert bragg.relative_bandwidth(100) == pytest.approx(0.01)
    assert bragg.relative_bandwidth(100, order=2) == pytest.approx(0.005)
