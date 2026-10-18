"""Tests for `xstack.fit.instrument`."""

import math

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.special import erf

from xstack.core import materials
from xstack.core.parameters import Parameter, ParameterSet, free
from xstack.core.physics.models import ParrattModel
from xstack.core.structure import Mirror, Periodic
from xstack.core.units import deg, kev
from xstack.fit.instrument import INSTRUMENT_PARAMETERS, Instrument, footprint_factor, smear

FWHM = 2.0 * math.sqrt(2.0 * math.log(2.0))


class TestCreate:
    def test_names_and_order(self) -> None:
        instrument = Instrument.create("angle")
        assert tuple(instrument.parameters) == INSTRUMENT_PARAMETERS
        assert INSTRUMENT_PARAMETERS == (
            "scale",
            "background",
            "offset",
            "resolution",
            "beam_width",
            "sample_length",
        )

    def test_defaults(self) -> None:
        values = Instrument.create("angle").parameters.value_dict()
        assert values == {
            "scale": 1.0,
            "background": 0.0,
            "offset": 0.0,
            "resolution": 0.0,
            "beam_width": 0.0,
            "sample_length": 0.0,
        }
        assert Instrument.create("angle").parameters.free_names() == ()

    @pytest.mark.parametrize(("axis", "unit"), [("angle", "rad"), ("energy", "eV")])
    def test_units_per_axis(self, axis: str, unit: str) -> None:
        params = Instrument.create(axis).parameters  # type: ignore[arg-type]
        assert params["offset"].unit == unit
        assert params["resolution"].unit == unit
        assert params["beam_width"].unit == "nm"
        assert params["sample_length"].unit == "nm"
        assert params["scale"].unit == ""

    def test_free_specs(self) -> None:
        instrument = Instrument.create(
            "energy", scale=free(1.0, 0.5, 2.0), offset=free(0.0, -5.0, 5.0), background=1e-6
        )
        assert instrument.parameters.free_names() == ("scale", "offset")
        assert instrument.parameters["scale"].min == 0.5
        assert instrument.parameters["background"].value == 1e-6
        assert not instrument.parameters["background"].is_free

    def test_expression_spec(self) -> None:
        instrument = Instrument.create("angle", background="scale * 2")
        assert instrument.parameters["background"].expr == "scale * 2"

    def test_bad_axis(self) -> None:
        with pytest.raises(ValueError, match="unknown axis"):
            Instrument("wavelength", Instrument.create("angle").parameters)  # type: ignore[arg-type]

    def test_wrong_parameter_names(self) -> None:
        with pytest.raises(ValueError, match="instrument parameters must be"):
            Instrument("angle", ParameterSet([Parameter("scale", 1.0)]))

    def test_with_parameters(self) -> None:
        instrument = Instrument.create("angle")
        params = instrument.parameters.replace("scale", value=2.0)
        assert instrument.with_parameters(params).parameters["scale"].value == 2.0
        assert instrument.parameters["scale"].value == 1.0


def affine(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return 0.1 + 2.0 * x


class TestApply:
    def test_identity(self) -> None:
        x = np.array([0.01, 0.02, 0.03])
        np.testing.assert_allclose(Instrument.create("angle").apply(x, affine), affine(x))

    def test_scale_and_background(self) -> None:
        x = np.array([0.01, 0.02, 0.03])
        instrument = Instrument.create("angle", scale=3.0, background=0.5)
        np.testing.assert_allclose(instrument.apply(x, affine), 3.0 * affine(x) + 0.5)

    def test_offset_shifts_evaluation_point(self) -> None:
        x = np.array([0.01, 0.02, 0.03])
        instrument = Instrument.create("angle", offset=0.004)
        np.testing.assert_allclose(instrument.apply(x, affine), affine(x + 0.004))

    def test_energy_offset(self) -> None:
        x = np.array([8000.0, 8100.0])
        instrument = Instrument.create("energy", offset=-25.0)
        np.testing.assert_allclose(instrument.apply(x, affine), affine(x - 25.0))

    def test_values_override(self) -> None:
        x = np.array([0.01, 0.02])
        instrument = Instrument.create("angle", scale=3.0)
        out = instrument.apply(x, affine, values={"scale": 2.0, "offset": 0.001})
        np.testing.assert_allclose(out, 2.0 * affine(x + 0.001))

    def test_footprint_applied_for_angle_scan(self) -> None:
        x = np.array([0.001, 0.01])
        instrument = Instrument.create("angle", beam_width=1e5, sample_length=1e7)
        expected = affine(x) * footprint_factor(x, 1e5, 1e7)
        np.testing.assert_allclose(instrument.apply(x, affine), expected)
        assert expected[0] < affine(x)[0]

    def test_footprint_energy_scan_uses_fixed_angle(self) -> None:
        x = np.array([8000.0, 8100.0])
        instrument = Instrument.create("energy", beam_width=1e5, sample_length=1e7)
        out = instrument.apply(x, lambda e: np.ones_like(e), fixed_angle=0.002)
        np.testing.assert_allclose(out, footprint_factor(np.full(2, 0.002), 1e5, 1e7))
        assert out[0] == out[1]

    def test_resolution_smears(self) -> None:
        x = np.linspace(0.01, 0.02, 50)
        instrument = Instrument.create("angle", resolution=1e-3)
        out = instrument.apply(x, lambda t: t**2)
        np.testing.assert_allclose(out, x**2 + 1e-6, rtol=1e-4)


class TestFootprint:
    theta = np.array([0.0005, 0.002, 0.01, 0.05])

    def test_disabled(self) -> None:
        for bw, sl in [(0.0, 1e6), (1e5, 0.0), (0.0, 0.0)]:
            np.testing.assert_array_equal(footprint_factor(self.theta, bw, sl), np.ones(4))

    def test_matches_erf_formula(self) -> None:
        bw, sl = 2e5, 3e7
        sigma = bw / FWHM
        expected = erf(sl * np.sin(self.theta) / (2 * math.sqrt(2) * sigma))
        np.testing.assert_allclose(footprint_factor(self.theta, bw, sl), expected, rtol=1e-12)

    def test_fraction_between_zero_and_one(self) -> None:
        f = footprint_factor(self.theta, 2e5, 3e7)
        assert np.all((f > 0) & (f <= 1))
        assert np.all(np.diff(f) > 0)

    def test_tends_to_one_for_large_footprint(self) -> None:
        f = footprint_factor(np.array([0.1]), 1e4, 1e9)
        assert f[0] == pytest.approx(1.0, abs=1e-12)

    def test_half_beam_at_fwhm(self) -> None:
        # The sample covers ±FWHM/2 of the beam profile when L sin θ = FWHM.
        theta = np.array([math.asin(0.5)])
        f = footprint_factor(theta, 1e5, 2e5)
        assert f[0] == pytest.approx(erf(1e5 / (2 * math.sqrt(2) * 1e5 / FWHM)))
        assert f[0] == pytest.approx(0.7615, abs=1e-3)


class Recorder:
    """Records the sizes and range of the arrays a function was evaluated at."""

    def __init__(self, func=None) -> None:
        self.calls: list[NDArray[np.float64]] = []
        self.func = func or (lambda x: 1.0 + 2.0 * x + 3.0 * x**2)

    def __call__(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        self.calls.append(np.array(x))
        return np.asarray(self.func(x))


def quadratic(a: float = 1.0, b: float = 2.0, c: float = 300.0):
    def f(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return a + b * x + c * x**2

    return f


class TestSmear:
    def test_zero_sigma_is_direct_evaluation(self) -> None:
        x = np.linspace(0.01, 0.02, 7)
        rec = Recorder()
        out = smear(rec, x, 0.0)
        np.testing.assert_array_equal(out, 1.0 + 2.0 * x + 3.0 * x**2)
        assert len(rec.calls) == 1
        np.testing.assert_array_equal(rec.calls[0], x)

    def test_empty_input(self) -> None:
        assert smear(quadratic(), np.zeros(0), 1e-3).size == 0

    def test_dense_points_use_grid(self) -> None:
        sigma = 1e-3
        x = np.linspace(0.01, 0.03, 200)
        rec = Recorder(quadratic())
        out = smear(rec, x, sigma)
        # One evaluation on a uniform grid with spacing sigma/10, not 15 nodes per point.
        assert len(rec.calls) == 1
        assert rec.calls[0].size < 15 * x.size
        np.testing.assert_allclose(np.diff(rec.calls[0]), 0.1 * sigma, rtol=1e-6)
        exact = quadratic()(x) + 300.0 * sigma**2
        np.testing.assert_allclose(out, exact, rtol=1e-6)

    def test_sparse_points_use_gauss_hermite(self) -> None:
        sigma = 1e-3
        x = np.asarray(
            0.01 + 8 * sigma * np.arange(5), dtype=np.float64
        )  # gaps below the cluster separation of 10 sigma
        rec = Recorder(quadratic())
        out = smear(rec, x, sigma)
        assert len(rec.calls) == 1
        assert rec.calls[0].size == 15 * x.size
        exact = quadratic()(x) + 300.0 * sigma**2
        np.testing.assert_allclose(out, exact, rtol=1e-6)

    def test_both_paths_agree_on_smooth_function(self) -> None:
        sigma = 2e-3
        x = np.array([0.02, 0.03, 0.04])
        func = lambda t: np.exp(-t / 0.05)  # noqa: E731
        hermite = smear(func, x, sigma)
        dense = np.linspace(0.02, 0.04, 400)
        grid = smear(func, dense, sigma)
        np.testing.assert_allclose(np.interp(x, dense, grid), hermite, rtol=1e-6)

    def test_clusters_separated_by_large_gaps(self) -> None:
        sigma = 1e-3
        first = np.linspace(0.01, 0.012, 30)
        second = np.linspace(0.5, 0.502, 30)
        x = np.concatenate([first, second])
        rec = Recorder(quadratic())
        out = smear(rec, x, sigma)
        assert len(rec.calls) == 2
        # No evaluation in the gap between the clusters.
        evaluated = np.concatenate(rec.calls)
        assert not np.any((evaluated > 0.012 + 6 * sigma) & (evaluated < 0.5 - 6 * sigma))
        np.testing.assert_allclose(out, quadratic()(x) + 300.0 * sigma**2, rtol=1e-6)

    def test_negative_angles_are_clipped(self) -> None:
        def strict(x: NDArray[np.float64]) -> NDArray[np.float64]:
            if np.any(x <= 0.0):
                raise AssertionError("evaluated at non-positive angle")
            return 1.0 + 0.0 * x

        x = np.linspace(1e-4, 5e-3, 40)
        out = smear(strict, x, 2e-3)
        assert np.all(np.isfinite(out))
        # Sparse path too.
        out = smear(strict, np.array([1e-4, 0.02]), 2e-3)
        assert np.all(np.isfinite(out))

    def test_angles_clipped_to_right_angle(self) -> None:
        seen: list[float] = []

        def func(x: NDArray[np.float64]) -> NDArray[np.float64]:
            seen.append(float(np.max(x)))
            return 0.0 * x

        smear(func, np.array([math.pi / 2 - 1e-4]), 1e-2, "angle")
        assert max(seen) <= math.pi / 2
        seen.clear()
        smear(func, np.array([8000.0]), 100.0, "energy")
        assert max(seen) > 8100.0

    def test_energy_axis_has_no_upper_limit(self) -> None:
        x = np.linspace(8000.0, 8100.0, 50)
        out = smear(quadratic(1.0, 0.0, 1e-6), x, 5.0, "energy")
        np.testing.assert_allclose(out, quadratic(1.0, 0.0, 1e-6)(x) + 1e-6 * 25.0, rtol=1e-6)


def test_smear_matches_brute_force_convolution_of_parratt_multilayer() -> None:
    w, si = materials.get("W"), materials.get("Si")
    stack = Mirror(Periodic.create(w, si, N=20, d=4.0), si).stack()
    energy = kev(8.048)
    model = ParrattModel()

    def reflectivity(theta: NDArray[np.float64]) -> NDArray[np.float64]:
        return model.reflectance(stack, np.full_like(theta, energy), theta, "unpolarized", None)

    sigma = deg(0.01)
    x = np.linspace(deg(0.9), deg(1.4), 60)
    smeared = smear(reflectivity, x, sigma)

    step = sigma / 40
    fine = np.arange(x[0] - 7 * sigma, x[-1] + 7 * sigma, step)
    values = reflectivity(fine)
    half = int(7 * sigma / step)
    kernel = np.exp(-0.5 * (step * np.arange(-half, half + 1) / sigma) ** 2)
    brute = np.interp(x, fine, np.convolve(values, kernel / kernel.sum(), mode="same"))

    assert brute.max() > 0.1  # the range covers the Bragg peak
    assert np.max(np.abs(smeared / brute - 1.0)) < 1e-2
