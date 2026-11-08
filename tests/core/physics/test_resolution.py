"""Tests for the resolution helpers `extended_grid` and `gaussian_smear`."""

import numpy as np
import pytest

from grazing.core.physics.resolution import extended_grid, gaussian_smear


class TestExtendedGrid:
    @pytest.mark.parametrize(
        ("lo", "hi", "points", "sigma"),
        [
            (0.0, 1.0, 11, 0.05),
            (10.0, 20.0, 5, 0.3),
            (1e4, 7e4, 200, 50.0),
            (0.01, 0.03, 50, 1e-3),
            (5.0, 6.0, 1000, 0.5),
        ],
    )
    def test_covers_range(self, lo: float, hi: float, points: int, sigma: float) -> None:
        grid, sl = extended_grid(lo, hi, points, sigma)
        inner = grid[sl]
        step = grid[1] - grid[0]
        assert inner[0] == pytest.approx(lo, abs=step * 1e-6 + 1e-12)
        assert inner[0] <= lo + 1e-9 * abs(lo)
        assert inner[-1] >= hi - step * 1e-6
        assert inner[-1] - hi < step * (1 + 1e-6)

    @pytest.mark.parametrize(
        ("lo", "hi", "points", "sigma"),
        [(0.0, 1.0, 11, 0.05), (10.0, 20.0, 5, 0.3), (0.01, 0.03, 50, 1e-3), (5.0, 6.0, 1000, 0.5)],
    )
    def test_spacing_and_padding(self, lo: float, hi: float, points: int, sigma: float) -> None:
        grid, sl = extended_grid(lo, hi, points, sigma)
        steps = np.diff(grid)
        step = steps[0]
        np.testing.assert_allclose(steps, step, rtol=1e-6)
        assert step <= sigma / 3.0 * (1 + 1e-12)
        assert step <= (hi - lo) / (points - 1) * (1 + 1e-12)
        assert grid[sl][0] - grid[0] >= 4.0 * sigma - 1e-9 * abs(sigma)
        assert grid[-1] - grid[sl][-1] >= 4.0 * sigma - 1e-9 * abs(sigma)

    def test_uses_original_step_when_finer(self) -> None:
        grid, sl = extended_grid(0.0, 1.0, 101, 10.0)
        assert grid[1] - grid[0] == pytest.approx(0.01)
        assert grid[sl][0] == pytest.approx(0.0)
        assert grid[sl][-1] == pytest.approx(1.0)


class TestGaussianSmear:
    def test_preserves_constant(self) -> None:
        y = np.full(200, 0.37)
        np.testing.assert_allclose(gaussian_smear(y, 0.1, 0.5), 0.37, atol=1e-12)

    def test_preserves_shape_of_output(self) -> None:
        y = np.random.default_rng(0).random(77)
        assert gaussian_smear(y, 0.1, 0.35).shape == y.shape

    def test_preserves_peak_area(self) -> None:
        step, sigma = 0.05, 0.4
        x = np.arange(-20, 20 + step / 2, step)
        y = np.exp(-0.5 * ((x - 1.0) / 0.3) ** 2)
        out = gaussian_smear(y, step, sigma)
        assert out.sum() == pytest.approx(y.sum(), rel=1e-6)

    def test_zero_sigma_returns_copy(self) -> None:
        y = np.array([1.0, 2.0, 3.0])
        out = gaussian_smear(y, 0.1, 0.0)
        np.testing.assert_array_equal(out, y)
        assert out is not y
        out[0] = 99.0
        assert y[0] == 1.0

    def test_broadens_with_expected_variance(self) -> None:
        step, sigma_in, sigma = 0.02, 0.3, 0.4
        x = np.arange(-15, 15 + step / 2, step)
        y = np.exp(-0.5 * (x / sigma_in) ** 2)
        out = gaussian_smear(y, step, sigma)
        assert out.max() < y.max()

        def variance(v: np.ndarray) -> float:
            w = v / v.sum()
            mean = float(np.dot(w, x))
            return float(np.dot(w, (x - mean) ** 2))

        assert variance(y) == pytest.approx(sigma_in**2, rel=1e-3)
        assert variance(out) == pytest.approx(sigma_in**2 + sigma**2, rel=1e-3)

    def test_delta_gives_kernel(self) -> None:
        step, sigma = 0.1, 0.5
        y = np.zeros(101)
        y[50] = 1.0
        out = gaussian_smear(y, step, sigma)
        x = (np.arange(101) - 50) * step
        expected = np.where(np.abs(x) <= 4.0 * sigma, np.exp(-0.5 * (x / sigma) ** 2), 0.0)
        expected /= expected.sum()
        np.testing.assert_allclose(out, expected, atol=1e-12)
