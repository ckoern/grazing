"""Tests for the figures of merit, evaluated on synthetic reflectivity results."""

import dataclasses
import itertools

import numpy as np
import pytest

from xstack.materials import get
from xstack.merit import (
    CallableMerit,
    EvaluationContext,
    Flatness,
    MeanReflectivity,
    PeakReflectivity,
    ProfileDeviation,
    SoftMinReflectivity,
    TimePenalized,
    WeightedSum,
    merits,
)
from xstack.physics.models import ReflectivityResult
from xstack.structure import Stack
from xstack.targets import EnergyScan, Profile, Target, TargetSet
from xstack.units import deg, kev

POINTS = 5
STACK = Stack([], get("Si"))


def scan(weight: float = 1.0, **kwargs: object) -> EnergyScan:
    params: dict[str, object] = {
        "angle": deg(0.2),
        "energy_min": kev(10),
        "energy_max": kev(20),
        "points": POINTS,
        "weight": weight,
    }
    params.update(kwargs)
    return EnergyScan(**params)  # type: ignore[arg-type]


def result(target: Target, r: object) -> ReflectivityResult:
    x = target.scan_values()
    energy, theta = target.grid_at(x)
    return ReflectivityResult(target, x, energy, theta, np.asarray(r, dtype=np.float64))


def context(
    targets: list[Target],
    curves: list[object],
    combine: str = "average_reflectivity",
    coating_time: float | None = None,
) -> EvaluationContext:
    ts = TargetSet(targets, combine=combine)  # type: ignore[arg-type]
    results = tuple(result(t, r) for t, r in zip(targets, curves, strict=True))
    return EvaluationContext(STACK, ts, results, coating_time)


R1 = [0.2, 0.4, 0.6, 0.8, 1.0]
R2 = [0.0, 0.0, 0.5, 0.5, 1.0]
W_TRAPEZOID = np.array([0.5, 1, 1, 1, 0.5]) / 4.0


def single(r: object = R1, weight: float = 1.0, **kwargs: object) -> EvaluationContext:
    return context([scan(weight, **kwargs)], [r])


class TestEvaluationContext:
    def test_result_count_validated(self) -> None:
        t = scan()
        with pytest.raises(ValueError, match="one reflectivity result per target"):
            EvaluationContext(STACK, TargetSet([t, scan()]), (result(t, R1),))
        with pytest.raises(ValueError, match="one reflectivity result per target"):
            EvaluationContext(STACK, TargetSet(t), ())

    def test_total_thickness(self) -> None:
        assert single().total_thickness == 0.0

    def test_single_target_curve(self) -> None:
        (curve,) = single(R1).curves()
        np.testing.assert_allclose(curve.reflectivity, R1)
        np.testing.assert_allclose(curve.weights, W_TRAPEZOID)
        np.testing.assert_allclose(curve.x, np.linspace(kev(10), kev(20), POINTS))
        assert curve.profile is None
        assert curve.weight == 1.0
        assert curve.mean() == pytest.approx(float(np.dot(W_TRAPEZOID, R1)))

    def test_single_target_profile(self) -> None:
        profile = Profile([kev(10), kev(20)], [0.0, 1.0])
        (curve,) = single(R1, profile=profile).curves()
        assert curve.profile is not None
        np.testing.assert_allclose(curve.profile, np.linspace(0, 1, POINTS))

    def test_average_reflectivity_curve(self) -> None:
        ctx = context([scan(1.0), scan(3.0)], [R1, R2])
        (curve,) = ctx.curves()
        np.testing.assert_allclose(curve.reflectivity, 0.25 * np.array(R1) + 0.75 * np.array(R2))
        np.testing.assert_allclose(curve.weights, W_TRAPEZOID)
        assert curve.weight == 1.0

    def test_average_reflectivity_spectrum_weights_are_averaged(self) -> None:
        s1 = Profile([kev(10), kev(20)], [1.0, 3.0])
        s2 = Profile([kev(10), kev(20)], [3.0, 1.0])
        t1, t2 = scan(1.0, spectrum=s1), scan(3.0, spectrum=s2)
        (curve,) = context([t1, t2], [R1, R2]).curves()
        expected = 0.25 * t1.weights() + 0.75 * t2.weights()
        np.testing.assert_allclose(curve.weights, expected)

    def test_average_merit_curves(self) -> None:
        ctx = context([scan(1.0), scan(3.0)], [R1, R2], combine="average_merit")
        c1, c2 = ctx.curves()
        np.testing.assert_allclose(c1.reflectivity, R1)
        np.testing.assert_allclose(c2.reflectivity, R2)
        assert c1.weight == pytest.approx(0.25)
        assert c2.weight == pytest.approx(0.75)
        np.testing.assert_allclose(c1.weights, W_TRAPEZOID)

    def test_average_merit_single_target(self) -> None:
        ctx = context([scan(5.0)], [R1], combine="average_merit")
        (curve,) = ctx.curves()
        assert curve.weight == 1.0


class TestMeanReflectivity:
    def test_constant_reflectivity(self) -> None:
        ctx = single(np.full(POINTS, 0.37))
        assert MeanReflectivity().scalar(ctx) == pytest.approx(1 - 0.37)

    def test_hand_computed_value(self) -> None:
        expected = 1.0 - float(np.dot(W_TRAPEZOID, R1))
        assert MeanReflectivity().scalar(single()) == pytest.approx(expected)
        # by hand: <R> = 0.25 * (0.1 + 0.4 + 0.6 + 0.8 + 0.5) = 0.6
        assert expected == pytest.approx(1.0 - 0.6)

    def test_residuals(self) -> None:
        ctx = single()
        res = MeanReflectivity().residuals(ctx)
        np.testing.assert_allclose(res, np.sqrt(W_TRAPEZOID) * (1 - np.array(R1)))
        assert np.sum(res**2) == pytest.approx(float(np.dot(W_TRAPEZOID, (1 - np.array(R1)) ** 2)))

    def test_average_reflectivity_scalar(self) -> None:
        ctx = context([scan(1.0), scan(3.0)], [R1, R2])
        mixed = 0.25 * np.array(R1) + 0.75 * np.array(R2)
        assert MeanReflectivity().scalar(ctx) == pytest.approx(1 - np.dot(W_TRAPEZOID, mixed))

    def test_average_merit_scalar(self) -> None:
        ctx = context([scan(1.0), scan(3.0)], [R1, R2], combine="average_merit")
        m1, m2 = np.dot(W_TRAPEZOID, R1), np.dot(W_TRAPEZOID, R2)
        assert MeanReflectivity().scalar(ctx) == pytest.approx(0.25 * (1 - m1) + 0.75 * (1 - m2))

    def test_average_merit_vs_average_reflectivity_residual_lengths(self) -> None:
        targets: list[Target] = [scan(1.0), scan(3.0)]
        merit = MeanReflectivity()
        averaged = merit.residuals(context(targets, [R1, R2], "average_reflectivity"))
        per_target = merit.residuals(context(targets, [R1, R2], "average_merit"))
        assert averaged.shape == (POINTS,)
        assert per_target.shape == (2 * POINTS,)

    def test_residual_sum_of_squares_matches_per_target_weighting(self) -> None:
        ctx = context([scan(1.0), scan(3.0)], [R1, R2], combine="average_merit")
        res = MeanReflectivity().residuals(ctx)
        expected = 0.25 * np.dot(W_TRAPEZOID, (1 - np.array(R1)) ** 2) + 0.75 * np.dot(
            W_TRAPEZOID, (1 - np.array(R2)) ** 2
        )
        assert np.sum(res**2) == pytest.approx(float(expected))

    def test_spectrum_weighting(self) -> None:
        t = scan(spectrum=Profile([kev(10), kev(20)], [0.0, 1.0]))
        ctx = context([t], [R1])
        assert MeanReflectivity().scalar(ctx) == pytest.approx(1 - np.dot(t.weights(), R1))


class TestSoftMinReflectivity:
    def test_approaches_hard_minimum(self) -> None:
        value = SoftMinReflectivity(softness=1e-4).scalar(single())
        assert value == pytest.approx(1 - min(R1), abs=1e-3)

    def test_constant_curve(self) -> None:
        ctx = single(np.full(POINTS, 0.4))
        for softness in (1e-3, 0.05, 1.0):
            assert SoftMinReflectivity(softness=softness).scalar(ctx) == pytest.approx(0.6)

    def test_monotone_in_softness(self) -> None:
        values = [SoftMinReflectivity(softness=s).scalar(single()) for s in (1e-3, 1e-2, 0.1, 1.0)]
        # softer minima lie closer to the mean, so the merit decreases towards 1 - <R>
        assert all(a > b for a, b in itertools.pairwise(values))
        assert values[-1] > 1 - float(np.dot(W_TRAPEZOID, R1))

    def test_between_mean_and_hard_value(self) -> None:
        value = SoftMinReflectivity(softness=0.05).scalar(single())
        assert 1 - float(np.dot(W_TRAPEZOID, R1)) < value < 1 - min(R1) + 0.05

    def test_residuals(self) -> None:
        merit = SoftMinReflectivity(softness=0.05)
        res = merit.residuals(single())
        assert res.shape == (POINTS,)
        # the weights p_i sum to 1, so the residuals are dominated by the lowest R
        assert np.argmax(np.abs(res)) == 0

    @pytest.mark.parametrize("softness", [0.0, -0.1])
    def test_invalid_softness(self, softness: float) -> None:
        with pytest.raises(ValueError, match="softness"):
            SoftMinReflectivity(softness=softness)


class TestProfileDeviation:
    PROFILE = Profile.constant(0.5)

    def ctx(self) -> EvaluationContext:
        return single(R1, profile=self.PROFILE)

    def test_two_sided(self) -> None:
        diff = np.array(R1) - 0.5
        expected = np.sqrt(np.dot(W_TRAPEZOID, diff**2))
        merit = ProfileDeviation()
        assert merit.scalar(self.ctx()) == pytest.approx(float(expected))
        np.testing.assert_allclose(merit.residuals(self.ctx()), np.sqrt(W_TRAPEZOID) * diff)

    def test_below(self) -> None:
        diff = np.minimum(np.array(R1) - 0.5, 0.0)
        merit = ProfileDeviation(mode="below")
        assert merit.scalar(self.ctx()) == pytest.approx(
            float(np.sqrt(np.dot(W_TRAPEZOID, diff**2)))
        )
        np.testing.assert_allclose(merit.residuals(self.ctx()), np.sqrt(W_TRAPEZOID) * diff)

    def test_below_ignores_excess(self) -> None:
        ctx = single(np.full(POINTS, 0.9), profile=self.PROFILE)
        assert ProfileDeviation(mode="below").scalar(ctx) == 0.0
        assert ProfileDeviation().scalar(ctx) == pytest.approx(0.4)

    def test_exact_match(self) -> None:
        ctx = single(np.full(POINTS, 0.5), profile=self.PROFILE)
        assert ProfileDeviation().scalar(ctx) == pytest.approx(0.0, abs=1e-15)

    def test_missing_profile(self) -> None:
        merit = ProfileDeviation()
        with pytest.raises(ValueError, match="profile"):
            merit.scalar(single())
        with pytest.raises(ValueError, match="profile"):
            merit.residuals(single())

    def test_unknown_mode(self) -> None:
        with pytest.raises(ValueError, match="mode"):
            ProfileDeviation(mode="above")  # type: ignore[arg-type]


class TestFlatness:
    def test_constant_is_zero(self) -> None:
        ctx = single(np.full(POINTS, 0.3))
        assert Flatness().scalar(ctx) == pytest.approx(0.0, abs=1e-12)
        np.testing.assert_allclose(Flatness().residuals(ctx), 0.0, atol=1e-12)

    def test_scale_invariant(self) -> None:
        a = Flatness().scalar(single(R1))
        b = Flatness().scalar(single(0.5 * np.array(R1)))
        assert a > 0.1
        assert a == pytest.approx(b)

    def test_hand_computed(self) -> None:
        r = np.array(R1)
        mean = np.dot(W_TRAPEZOID, r)
        expected = np.sqrt(np.dot(W_TRAPEZOID, (r - mean) ** 2)) / mean
        assert Flatness().scalar(single()) == pytest.approx(float(expected))

    def test_zero_reflectivity_is_finite(self) -> None:
        assert np.isfinite(Flatness().scalar(single(np.zeros(POINTS))))


class TestPeakReflectivity:
    def test_value_and_residual(self) -> None:
        ctx = single([0.1, 0.9, 0.3, 0.2, 0.0])
        assert PeakReflectivity().scalar(ctx) == pytest.approx(0.1)
        np.testing.assert_allclose(PeakReflectivity().residuals(ctx), [0.1])

    def test_average_merit_residual_count(self) -> None:
        ctx = context([scan(), scan()], [R1, R2], combine="average_merit")
        assert PeakReflectivity().residuals(ctx).shape == (2,)
        assert PeakReflectivity().scalar(ctx) == pytest.approx(0.0)


class TestWeightedSum:
    def test_value(self) -> None:
        ctx = single()
        ws = WeightedSum(((MeanReflectivity(), 2.0), (PeakReflectivity(), 0.5)))
        expected = 2.0 * MeanReflectivity().scalar(ctx) + 0.5 * PeakReflectivity().scalar(ctx)
        assert ws.scalar(ctx) == pytest.approx(expected)

    def test_residuals_concatenated_and_scaled(self) -> None:
        ctx = single()
        ws = WeightedSum(((MeanReflectivity(), 4.0), (PeakReflectivity(), 9.0)))
        res = ws.residuals(ctx)
        np.testing.assert_allclose(
            res,
            np.concatenate(
                [2.0 * MeanReflectivity().residuals(ctx), 3.0 * PeakReflectivity().residuals(ctx)]
            ),
        )
        assert res.shape == (POINTS + 1,)

    def test_zero_weight_allowed(self) -> None:
        ws = WeightedSum(((MeanReflectivity(), 0.0), (PeakReflectivity(), 1.0)))
        assert ws.scalar(single()) == pytest.approx(PeakReflectivity().scalar(single()))

    def test_empty(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            WeightedSum(())

    def test_negative_weight(self) -> None:
        with pytest.raises(ValueError, match="negative"):
            WeightedSum(((MeanReflectivity(), -1.0),))


class TestTimePenalized:
    def test_zero_weight_is_optical(self) -> None:
        ctx = single()
        merit = TimePenalized(MeanReflectivity(), weight=0.0, reference_time=100.0)
        ctx = dataclasses.replace(ctx, coating_time=500.0)
        assert merit.scalar(ctx) == pytest.approx(MeanReflectivity().scalar(ctx))

    def test_full_weight_is_time(self) -> None:
        ctx = dataclasses.replace(single(), coating_time=50.0)
        merit = TimePenalized(MeanReflectivity(), weight=1.0, reference_time=200.0)
        assert merit.scalar(ctx) == pytest.approx(0.25)

    def test_linear_in_weight(self) -> None:
        ctx = dataclasses.replace(single(), coating_time=50.0)
        optical = MeanReflectivity().scalar(ctx)
        for w in (0.1, 0.3, 0.5, 0.9):
            merit = TimePenalized(MeanReflectivity(), weight=w, reference_time=200.0)
            assert merit.scalar(ctx) == pytest.approx((1 - w) * optical + w * 0.25)

    def test_residuals(self) -> None:
        ctx = dataclasses.replace(single(), coating_time=50.0)
        w = 0.4
        merit = TimePenalized(MeanReflectivity(), weight=w, reference_time=200.0)
        res = merit.residuals(ctx)
        assert res.shape == (POINTS + 1,)
        np.testing.assert_allclose(res[:-1], np.sqrt(1 - w) * MeanReflectivity().residuals(ctx))
        assert res[-1] ** 2 == pytest.approx(w * 0.25)
        total = np.sum(res**2)
        assert total == pytest.approx(
            (1 - w) * float(np.sum(MeanReflectivity().residuals(ctx) ** 2)) + w * 0.25
        )

    def test_missing_coating_time(self) -> None:
        merit = TimePenalized(MeanReflectivity())
        with pytest.raises(ValueError, match="coating time"):
            merit.scalar(single())
        with pytest.raises(ValueError, match="coating time"):
            merit.residuals(single())

    @pytest.mark.parametrize("weight", [-0.1, 1.1])
    def test_invalid_weight(self, weight: float) -> None:
        with pytest.raises(ValueError, match="weight"):
            TimePenalized(MeanReflectivity(), weight=weight)

    @pytest.mark.parametrize("reference", [0.0, -5.0])
    def test_invalid_reference(self, reference: float) -> None:
        with pytest.raises(ValueError, match="reference_time"):
            TimePenalized(MeanReflectivity(), reference_time=reference)


class TestCallableMerit:
    def test_scalar(self) -> None:
        merit = CallableMerit(lambda ctx: ctx.curves()[0].reflectivity.max() * 2, label="twice")
        assert merit.scalar(single()) == pytest.approx(2.0)
        assert merit.label == "twice"

    def test_default_residuals(self) -> None:
        merit = CallableMerit(lambda ctx: 4.0)
        np.testing.assert_allclose(merit.residuals(single()), [2.0])

    def test_negative_value_residual_is_zero(self) -> None:
        merit = CallableMerit(lambda ctx: -1.0)
        np.testing.assert_allclose(merit.residuals(single()), [0.0])

    def test_uses_context(self) -> None:
        merit = CallableMerit(lambda ctx: ctx.total_thickness + len(ctx.results))
        assert merit.scalar(single()) == 1.0


class TestRegistry:
    @pytest.mark.parametrize(
        ("name", "cls"),
        [
            ("mean_reflectivity", MeanReflectivity),
            ("soft_min_reflectivity", SoftMinReflectivity),
            ("profile_deviation", ProfileDeviation),
            ("flatness", Flatness),
            ("peak_reflectivity", PeakReflectivity),
            ("weighted_sum", WeightedSum),
            ("time_penalized", TimePenalized),
            ("callable", CallableMerit),
        ],
    )
    def test_registered(self, name: str, cls: type) -> None:
        assert name in merits
        assert merits.get(name) is cls
        assert cls.type_name == name

    def test_frozen(self) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            MeanReflectivity().type_name = "x"  # type: ignore[misc]
        with pytest.raises(dataclasses.FrozenInstanceError):
            SoftMinReflectivity().softness = 1.0  # type: ignore[misc]
