"""Tests for `grazing.core.measurement`."""

import math
from typing import Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from grazing.core.measurement import MeasuredCurve
from grazing.core.serialization import from_data, to_data
from grazing.core.units import deg, kev


def angle_curve(**kwargs: Any) -> MeasuredCurve:
    args: dict[str, Any] = {
        "axis": "angle",
        "x": deg(np.array([0.3, 0.1, 0.2])),
        "reflectivity": [0.2, 0.9, 0.5],
        "energy": kev(8.0),
    }
    args.update(kwargs)
    return MeasuredCurve(**args)


def energy_curve(**kwargs: Any) -> MeasuredCurve:
    args: dict[str, Any] = {
        "axis": "energy",
        "x": [3000.0, 1000.0, 2000.0],
        "reflectivity": [0.1, 0.7, 0.4],
        "angle": deg(0.5),
    }
    args.update(kwargs)
    return MeasuredCurve(**args)


def test_construction_sorts_and_aligns() -> None:
    curve = angle_curve(uncertainty=[0.03, 0.01, 0.02])
    assert curve.x == pytest.approx(tuple(deg(np.array([0.1, 0.2, 0.3]))))
    assert curve.reflectivity == (0.9, 0.5, 0.2)
    assert curve.uncertainty == (0.01, 0.02, 0.03)


def test_scalar_input_is_one_point() -> None:
    curve = MeasuredCurve("energy", 1000.0, 0.5)
    assert curve.size == 1


def test_defaults() -> None:
    curve = energy_curve()
    assert curve.polarization == "unpolarized"
    assert curve.name == ""
    assert curve.source == ""
    assert curve.uncertainty is None
    assert curve.energy is None


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"axis": "time"}, "unknown axis"),
        ({"polarization": "circular"}, "unknown polarization"),
        ({"x": [], "reflectivity": []}, "non-zero length"),
        ({"reflectivity": [0.1, 0.2]}, "non-zero length"),
        ({"uncertainty": [0.1, 0.2]}, "non-zero length"),
        ({"reflectivity": [0.1, math.nan, 0.2]}, "finite"),
        ({"x": [0.1, math.inf, 0.2]}, "finite"),
        ({"uncertainty": [0.1, math.nan, 0.2]}, "finite"),
        ({"x": [0.1, 0.0, 0.2]}, "grazing angles"),
        ({"x": [0.1, -0.1, 0.2]}, "grazing angles"),
        ({"x": [0.1, math.pi / 2 + 1e-6, 0.2]}, "grazing angles"),
        ({"uncertainty": [0.1, -0.1, 0.2]}, "negative"),
        ({"energy": 0.0}, "energy must be positive"),
        ({"energy": -5.0}, "energy must be positive"),
        ({"energy": math.nan}, "energy must be positive"),
        ({"angle": 0.0}, "angle must be"),
        ({"angle": 2.0}, "angle must be"),
    ],
)
def test_angle_validation(kwargs: dict[str, Any], match: str) -> None:
    base: dict[str, Any] = {
        "axis": "angle",
        "x": [0.1, 0.2, 0.3],
        "reflectivity": [0.5, 0.4, 0.3],
        "energy": 8000.0,
    }
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        MeasuredCurve(**base)


def test_angle_of_pi_half_allowed() -> None:
    assert MeasuredCurve("angle", [math.pi / 2], [0.1]).size == 1


@pytest.mark.parametrize("x", [[0.0, 1000.0], [-1.0, 1000.0]])
def test_energy_must_be_positive(x: list[float]) -> None:
    with pytest.raises(ValueError, match="energies must be positive"):
        MeasuredCurve("energy", x, [0.1, 0.2])


def test_energy_scan_accepts_large_values() -> None:
    assert MeasuredCurve("energy", [10.0, 1e5], [0.1, 0.2]).size == 2


def test_grid_angle_scan() -> None:
    energy, theta = angle_curve().grid()
    np.testing.assert_allclose(energy, 8000.0)
    np.testing.assert_allclose(theta, deg(np.array([0.1, 0.2, 0.3])))


def test_grid_energy_scan() -> None:
    energy, theta = energy_curve().grid()
    np.testing.assert_allclose(energy, [1000.0, 2000.0, 3000.0])
    np.testing.assert_allclose(theta, deg(0.5))
    assert theta.shape == energy.shape


@pytest.mark.parametrize(
    ("curve", "missing"),
    [(angle_curve(energy=None), "energy"), (energy_curve(angle=None, name="e"), "angle")],
)
def test_grid_without_fixed_value(curve: MeasuredCurve, missing: str) -> None:
    with pytest.raises(ValueError, match=f"the {missing} of curve"):
        curve.grid()


def test_grid_error_names_unnamed_curve() -> None:
    with pytest.raises(ValueError, match=r"\(unnamed\)"):
        angle_curve(energy=None).grid()


def test_crop_inclusive_bounds() -> None:
    curve = energy_curve(uncertainty=[0.3, 0.1, 0.2], name="c")
    cropped = curve.crop(1000.0, 2000.0)
    assert cropped.x == (1000.0, 2000.0)
    assert cropped.reflectivity == (0.7, 0.4)
    assert cropped.uncertainty == (0.1, 0.2)
    assert cropped.name == "c"
    assert cropped.angle == curve.angle


def test_crop_without_uncertainty() -> None:
    assert energy_curve().crop(1500.0, 5000.0).uncertainty is None


def test_crop_empty_raises() -> None:
    with pytest.raises(ValueError, match="no measured point"):
        energy_curve().crop(1100.0, 1900.0)


def test_with_fixed_angle_scan_sets_energy() -> None:
    curve = angle_curve(energy=None).with_fixed(9000.0)
    assert curve.energy == 9000.0
    assert curve.fixed == 9000.0
    assert curve.angle is None


def test_with_fixed_energy_scan_sets_angle() -> None:
    curve = energy_curve(angle=None).with_fixed(0.01)
    assert curve.angle == 0.01
    assert curve.fixed == 0.01


def test_with_fixed_keeps_original() -> None:
    curve = angle_curve()
    curve.with_fixed(1000.0)
    assert curve.energy == 8000.0


def test_with_fixed_out_of_range_raises() -> None:
    with pytest.raises(ValueError, match="energy must be positive"):
        angle_curve().with_fixed(-1.0)
    with pytest.raises(ValueError, match="angle must be"):
        energy_curve().with_fixed(4.0)


def test_fixed_scan_range_size() -> None:
    a, e = angle_curve(), energy_curve()
    assert a.fixed == 8000.0
    assert e.fixed == deg(0.5)
    assert a.size == e.size == 3
    assert e.scan_range == (1000.0, 3000.0)
    assert a.scan_range == pytest.approx((deg(0.1), deg(0.3)))


def test_arrays() -> None:
    x, r, u = energy_curve().arrays()
    assert u is None
    np.testing.assert_array_equal(x, [1000.0, 2000.0, 3000.0])
    np.testing.assert_array_equal(r, [0.7, 0.4, 0.1])
    _, _, u = energy_curve(uncertainty=[3, 2, 1]).arrays()
    assert u is not None
    np.testing.assert_array_equal(u, [2.0, 1.0, 3.0])


@pytest.mark.parametrize("with_uncertainty", [False, True])
@pytest.mark.parametrize("make", [angle_curve, energy_curve])
def test_serialization_round_trip(make: Any, with_uncertainty: bool) -> None:
    kwargs: dict[str, Any] = {"name": "n", "source": "f.csv", "polarization": "s"}
    if with_uncertainty:
        kwargs["uncertainty"] = [0.01, 0.02, 0.03]
    curve = make(**kwargs)
    assert from_data(to_data(curve)) == curve


def test_equality() -> None:
    assert angle_curve() == angle_curve()
    assert angle_curve() != angle_curve(name="other")
    assert angle_curve(
        x=deg(np.array([0.1, 0.2, 0.3])), reflectivity=[0.9, 0.5, 0.2]
    ) == angle_curve(x=deg(np.array([0.3, 0.2, 0.1])), reflectivity=[0.2, 0.5, 0.9])


@given(
    points=st.lists(
        st.tuples(
            st.floats(min_value=1.0, max_value=1e5),
            st.floats(min_value=0.0, max_value=1.0),
            st.floats(min_value=0.0, max_value=1.0),
        ),
        min_size=1,
        max_size=20,
        unique_by=lambda p: p[0],
    ),
    seed=st.integers(0, 2**32 - 1),
)
def test_sorting_invariant_under_permutation(
    points: list[tuple[float, float, float]], seed: int
) -> None:
    order = np.random.default_rng(seed).permutation(len(points))
    shuffled = [points[i] for i in order]

    def build(pts: list[tuple[float, float, float]]) -> MeasuredCurve:
        x, r, u = zip(*pts, strict=True)
        return MeasuredCurve("energy", x, r, u)

    curve = build(shuffled)
    assert curve == build(points)
    assert list(curve.x) == sorted(curve.x)
    expected = {p[0]: (p[1], p[2]) for p in points}
    for x, r, u in zip(curve.x, curve.reflectivity, curve.uncertainty or (), strict=True):
        assert expected[x] == (r, u)
