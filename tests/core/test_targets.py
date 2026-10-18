"""Tests for `Profile`, `EnergyScan`, `AngleScan` and `TargetSet`."""

import dataclasses
import math

import numpy as np
import pytest

from xstack.core import materials
from xstack.core.physics.models import ParrattModel
from xstack.core.structure import Stack
from xstack.core.targets import AngleScan, EnergyScan, Profile, TargetSet
from xstack.core.units import deg, kev


def energy_scan(**kwargs: object) -> EnergyScan:
    params: dict[str, object] = {
        "angle": deg(0.2),
        "energy_min": kev(15),
        "energy_max": kev(70),
        "points": 11,
    }
    params.update(kwargs)
    return EnergyScan(**params)  # type: ignore[arg-type]


def angle_scan(**kwargs: object) -> AngleScan:
    params: dict[str, object] = {
        "energy": 8048.0,
        "angle_min": 0.01,
        "angle_max": 0.03,
        "points": 11,
    }
    params.update(kwargs)
    return AngleScan(**params)  # type: ignore[arg-type]


class TestProfile:
    def test_interpolation(self) -> None:
        p = Profile([0.0, 1.0, 3.0], [0.0, 2.0, 0.0])
        assert p(0.5) == pytest.approx(1.0)
        assert p(2.0) == pytest.approx(1.0)
        np.testing.assert_allclose(p([0.0, 1.0, 3.0]), [0.0, 2.0, 0.0])

    def test_clamped_outside_range(self) -> None:
        p = Profile([1.0, 2.0], [5.0, 7.0])
        np.testing.assert_allclose(p([-10.0, 100.0]), [5.0, 7.0])

    def test_constant(self) -> None:
        p = Profile.constant(0.3)
        np.testing.assert_allclose(p(np.linspace(-5, 1e6, 7)), 0.3)

    def test_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="same"):
            Profile([0.0, 1.0], [1.0])

    def test_empty(self) -> None:
        with pytest.raises(ValueError, match="non-zero"):
            Profile([], [])

    @pytest.mark.parametrize("x", [[0.0, 0.0, 1.0], [2.0, 1.0]])
    def test_not_strictly_increasing(self, x: list[float]) -> None:
        with pytest.raises(ValueError, match="increasing"):
            Profile(x, [1.0] * len(x))

    def test_frozen_and_hashable(self) -> None:
        p = Profile([0.0, 1.0], [1.0, 2.0])
        with pytest.raises(dataclasses.FrozenInstanceError):
            p.x = (1.0,)  # type: ignore[misc]
        assert hash(p) == hash(Profile([0.0, 1.0], [1.0, 2.0]))
        assert p == Profile([0.0, 1.0], [1.0, 2.0])


class TestValidation:
    @pytest.mark.parametrize("make", [energy_scan, angle_scan])
    @pytest.mark.parametrize("weight", [0.0, -1.0, math.nan])
    def test_weight(self, make: object, weight: float) -> None:
        with pytest.raises(ValueError, match="weight"):
            make(weight=weight)  # type: ignore[operator]

    @pytest.mark.parametrize("make", [energy_scan, angle_scan])
    def test_points(self, make: object) -> None:
        with pytest.raises(ValueError, match="points"):
            make(points=1)  # type: ignore[operator]
        make(points=2)  # type: ignore[operator]

    @pytest.mark.parametrize("make", [energy_scan, angle_scan])
    def test_polarization(self, make: object) -> None:
        with pytest.raises(ValueError, match="polarization"):
            make(polarization="x")  # type: ignore[operator]
        for pol in ("s", "p", "unpolarized"):
            assert make(polarization=pol).polarization == pol  # type: ignore[operator]

    @pytest.mark.parametrize("make", [energy_scan, angle_scan])
    def test_resolution(self, make: object) -> None:
        with pytest.raises(ValueError, match="resolution"):
            make(resolution=-1e-3)  # type: ignore[operator]
        make(resolution=0.0)  # type: ignore[operator]

    @pytest.mark.parametrize("angle", [0.0, -0.1, math.pi / 2 + 0.01])
    def test_energy_scan_angle(self, angle: float) -> None:
        with pytest.raises(ValueError, match="angle"):
            energy_scan(angle=angle)

    def test_energy_scan_angle_upper_limit_allowed(self) -> None:
        energy_scan(angle=math.pi / 2)

    @pytest.mark.parametrize(("lo", "hi"), [(0.0, 10.0), (-5.0, 10.0), (10.0, 10.0), (20.0, 10.0)])
    def test_energy_range(self, lo: float, hi: float) -> None:
        with pytest.raises(ValueError, match="energy"):
            energy_scan(energy_min=lo, energy_max=hi)

    @pytest.mark.parametrize(
        ("lo", "hi"), [(0.0, 0.1), (-0.1, 0.1), (0.1, 0.1), (0.2, 0.1), (0.1, 2.0)]
    )
    def test_angle_range(self, lo: float, hi: float) -> None:
        with pytest.raises(ValueError, match="angle"):
            angle_scan(angle_min=lo, angle_max=hi)

    def test_angle_scan_energy(self) -> None:
        with pytest.raises(ValueError, match="energy"):
            angle_scan(energy=0.0)

    def test_frozen_and_hashable(self) -> None:
        t = energy_scan()
        with pytest.raises(dataclasses.FrozenInstanceError):
            t.weight = 2.0  # type: ignore[misc]
        assert hash(t) == hash(energy_scan())
        assert t == energy_scan()
        assert t != energy_scan(weight=2.0)


class TestGrids:
    def test_energy_scan_axis_and_range(self) -> None:
        t = energy_scan()
        assert t.axis == "energy"
        assert t.scan_range == (kev(15), kev(70))

    def test_angle_scan_axis_and_range(self) -> None:
        t = angle_scan()
        assert t.axis == "angle"
        assert t.scan_range == (0.01, 0.03)

    def test_energy_grid(self) -> None:
        t = energy_scan(points=5)
        e, th = t.grid()
        assert e.shape == th.shape == (5,)
        np.testing.assert_allclose(e, np.linspace(kev(15), kev(70), 5))
        np.testing.assert_allclose(th, deg(0.2))

    def test_angle_grid(self) -> None:
        t = angle_scan(points=5)
        e, th = t.grid()
        assert e.shape == th.shape == (5,)
        np.testing.assert_allclose(e, 8048.0)
        np.testing.assert_allclose(th, np.linspace(0.01, 0.03, 5))

    def test_grid_points_override(self) -> None:
        e, th = energy_scan(points=5).grid(9)
        assert e.shape == th.shape == (9,)

    def test_grid_at_energy(self) -> None:
        values = np.array([20e3, 30e3, 40e3])
        e, th = energy_scan().grid_at(values)
        np.testing.assert_allclose(e, values)
        np.testing.assert_allclose(th, deg(0.2))
        assert th.shape == values.shape

    def test_grid_at_angle(self) -> None:
        values = np.array([0.012, 0.02])
        e, th = angle_scan().grid_at(values)
        np.testing.assert_allclose(e, 8048.0)
        np.testing.assert_allclose(th, values)

    def test_scan_values(self) -> None:
        np.testing.assert_allclose(angle_scan(points=3).scan_values(), [0.01, 0.02, 0.03])
        assert energy_scan().scan_values(7).shape == (7,)

    def test_with_points(self) -> None:
        t = energy_scan(points=5, weight=2.0)
        t2 = t.with_points(50)
        assert t2.points == 50
        assert t.points == 5
        assert t2.weight == 2.0
        assert isinstance(t2, EnergyScan)
        assert t2.scan_values().shape == (50,)


class TestWeights:
    def test_trapezoid_shape_and_sum(self) -> None:
        w = energy_scan(points=6).weights()
        assert w.shape == (6,)
        assert w.sum() == pytest.approx(1.0)
        np.testing.assert_allclose(w, np.array([0.5, 1, 1, 1, 1, 0.5]) / 5.0)

    def test_points_override(self) -> None:
        w = angle_scan(points=6).weights(4)
        np.testing.assert_allclose(w, np.array([0.5, 1, 1, 0.5]) / 3.0)

    def test_spectrum(self) -> None:
        spectrum = Profile([0.0, 1.0], [0.0, 1.0])
        t = angle_scan(angle_min=0.0 + 1e-9, angle_max=1.0, points=5, spectrum=spectrum)
        x = t.scan_values()
        expected = np.array([0.5, 1, 1, 1, 0.5]) * x
        np.testing.assert_allclose(t.weights(), expected / expected.sum())
        assert t.weights().sum() == pytest.approx(1.0)

    def test_zero_spectrum(self) -> None:
        t = energy_scan(spectrum=Profile.constant(0.0))
        with pytest.raises(ValueError, match="spectrum"):
            t.weights()

    def test_mean_of_constant(self) -> None:
        t = energy_scan(points=13, spectrum=Profile([1e4, 8e4], [1.0, 3.0]))
        assert float(np.dot(t.weights(), np.full(13, 0.7))) == pytest.approx(0.7)


class TestTargetSet:
    def test_single_target_not_in_tuple(self) -> None:
        t = energy_scan()
        ts = TargetSet(t)
        assert ts.targets == (t,)
        assert len(ts) == 1

    def test_list_accepted(self) -> None:
        ts = TargetSet([energy_scan(), energy_scan(weight=2.0)])
        assert isinstance(ts.targets, tuple)
        assert len(ts) == 2

    def test_empty(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            TargetSet(())

    def test_unknown_combine(self) -> None:
        with pytest.raises(ValueError, match="combine"):
            TargetSet(energy_scan(), combine="max")  # type: ignore[arg-type]

    def test_default_combine(self) -> None:
        assert TargetSet(energy_scan()).combine == "average_reflectivity"

    def test_average_reflectivity_same_axis_different_angles_ok(self) -> None:
        TargetSet([energy_scan(angle=deg(0.15)), energy_scan(angle=deg(0.25))])

    @pytest.mark.parametrize(
        "other",
        [
            lambda: angle_scan(),
            lambda: energy_scan(energy_max=kev(60)),
            lambda: energy_scan(energy_min=kev(16)),
            lambda: energy_scan(points=12),
        ],
        ids=["axis", "max", "min", "points"],
    )
    def test_average_reflectivity_mismatch(self, other: object) -> None:
        with pytest.raises(ValueError, match="average_reflectivity"):
            TargetSet([energy_scan(), other()])  # type: ignore[operator]

    def test_average_merit_allows_mismatch(self) -> None:
        ts = TargetSet(
            [energy_scan(), angle_scan(), energy_scan(energy_max=kev(60), points=30)],
            combine="average_merit",
        )
        assert len(ts) == 3

    def test_normalized_weights(self) -> None:
        ts = TargetSet([energy_scan(weight=1.0), energy_scan(weight=3.0)])
        np.testing.assert_allclose(ts.normalized_weights(), [0.25, 0.75])

    def test_with_points(self) -> None:
        ts = TargetSet([energy_scan(weight=1.0), energy_scan(weight=3.0)], combine="average_merit")
        ts2 = ts.with_points(40)
        assert [t.points for t in ts2.targets] == [40, 40]
        assert [t.weight for t in ts2.targets] == [1.0, 3.0]
        assert ts2.combine == "average_merit"
        assert [t.points for t in ts.targets] == [11, 11]

    def test_frozen_and_hashable(self) -> None:
        ts = TargetSet([energy_scan(), energy_scan(weight=2.0)])
        with pytest.raises(dataclasses.FrozenInstanceError):
            ts.combine = "average_merit"  # type: ignore[misc]
        same = TargetSet([energy_scan(), energy_scan(weight=2.0)])
        assert hash(ts) == hash(same)
        assert ts == same


class TestReflections:
    def test_default_is_one(self) -> None:
        assert energy_scan().reflections == 1
        assert angle_scan().reflections == 1

    @pytest.mark.parametrize("make", [energy_scan, angle_scan])
    def test_zero_raises(self, make: object) -> None:
        with pytest.raises(ValueError):
            make(reflections=0)  # type: ignore[operator]

    def test_two_reflections_squares_reflectivity(self) -> None:
        si = materials.get("Si")
        stack = Stack([], si)
        model = ParrattModel()
        one = model.evaluate(stack, angle_scan(points=7, reflections=1))
        two = model.evaluate(stack, angle_scan(points=7, reflections=2))
        np.testing.assert_allclose(two.reflectivity, one.reflectivity**2, rtol=1e-12)


class TestLabel:
    def test_named_target_returns_name(self) -> None:
        assert energy_scan(name="Cu K").label(3) == "Cu K"

    def test_unnamed_target_is_numbered_from_one(self) -> None:
        assert energy_scan().label(0) == "target 1"
        assert angle_scan().label(4) == "target 5"
