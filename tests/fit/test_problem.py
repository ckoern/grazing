"""Tests for `xstack.fit.problem`."""

import math
from dataclasses import replace

import numpy as np
import pytest

from xstack.core import materials
from xstack.core.measurement import MeasuredCurve
from xstack.core.parameters import Parameter, ParameterSet, free
from xstack.core.physics.models import KinematicModel
from xstack.core.structure import Mirror, SingleLayer
from xstack.fit.dataset import Dataset
from xstack.fit.problem import FAILED_OBJECTIVE, WEIGHTINGS, FitProblem, Weighting

PT, SI = materials.get("Pt"), materials.get("Si")
ANGLES = np.linspace(0.002, 0.012, 40)
ENERGIES = np.linspace(7000.0, 9000.0, 40)


def make_mirror(t: float = 30.0) -> Mirror:
    return Mirror(SingleLayer.create(PT, t=free(t, 10.0, 60.0)), SI)


def angle_curve(r: np.ndarray | None = None, sigma: float | None = 0.05) -> MeasuredCurve:
    values = np.ones_like(ANGLES) if r is None else r
    u = None if sigma is None else sigma * np.maximum(values, 1e-3)
    return MeasuredCurve("angle", ANGLES, values, u, energy=8048.0)


def energy_curve(r: np.ndarray | None = None) -> MeasuredCurve:
    values = np.ones_like(ENERGIES) if r is None else r
    return MeasuredCurve("energy", ENERGIES, values, 0.05 * values, angle=0.006)


def synthetic(curve: MeasuredCurve, mirror: Mirror | None = None) -> MeasuredCurve:
    """Return `curve` with the reflectivity simulated for scale 1 and unit instrument."""
    probe = Dataset.create("probe", curve)
    problem = FitProblem(mirror or make_mirror(), (probe,), weighting="linear")
    sim = problem.simulate(probe)
    u = 0.05 * np.maximum(sim, 1e-6)
    return replace(curve, reflectivity=tuple(sim), uncertainty=tuple(u))


def make_problem(
    weighting: Weighting = "log", scale: float = 1.0, weight: float = 1.0
) -> tuple[FitProblem, Dataset]:
    data = synthetic(angle_curve())
    dataset = Dataset.create("a", data, scale=scale)
    dataset = replace(dataset, weight=weight)
    return FitProblem(make_mirror(), (dataset,), weighting=weighting), dataset


class TestConstruction:
    def test_parameters_for_two_datasets(self) -> None:
        a = Dataset.create("a", angle_curve(), scale=free(1.0, 0.5, 2.0))
        b = Dataset.create("b", energy_curve(), offset=free(0.0, -5.0, 5.0))
        problem = FitProblem(make_mirror(), (a, b), weighting="linear")
        names = list(problem.parameters)
        assert names[:2] == ["t", "sigma"]
        assert names[2:8] == [f"a.{n}" for n in a.instrument.parameters]
        assert names[8:] == [f"b.{n}" for n in b.instrument.parameters]
        assert problem.parameters.free_names() == ("t", "a.scale", "b.offset")

    def test_duplicate_dataset_names(self) -> None:
        a = Dataset.create("a", angle_curve())
        with pytest.raises(ValueError, match="unique"):
            FitProblem(make_mirror(), (a, a))

    def test_parameter_name_collision(self) -> None:
        base = SingleLayer.create(PT)
        params = ParameterSet([*base.parameters.values(), Parameter("a.scale", 1.0)])
        coating = replace(base, parameters=params)
        a = Dataset.create("a", angle_curve())
        with pytest.raises(ValueError, match=r"\['a.scale'\] collide with coating parameters"):
            FitProblem(Mirror(coating, SI), (a,))

    def test_unknown_weighting(self) -> None:
        with pytest.raises(ValueError, match="unknown weighting"):
            FitProblem(
                make_mirror(),
                (Dataset.create("a", angle_curve()),),
                weighting="bad",  # type: ignore[arg-type]
            )

    def test_chi2_without_uncertainties(self) -> None:
        a = Dataset.create("a", angle_curve(sigma=None))
        with pytest.raises(ValueError, match="uncertainties"):
            FitProblem(make_mirror(), (a,), weighting="chi2")
        FitProblem(make_mirror(), (a,), weighting="log")

    def test_chi2_ignores_disabled_dataset_without_uncertainties(self) -> None:
        a = Dataset.create("a", angle_curve())
        b = replace(Dataset.create("b", angle_curve(sigma=None)), enabled=False)
        FitProblem(make_mirror(), (a, b), weighting="chi2")

    def test_no_selected_points(self) -> None:
        a = Dataset.create("a", angle_curve()).with_ranges([(1.0, 2.0)])
        with pytest.raises(ValueError, match="no measured point"):
            FitProblem(make_mirror(), (a,))
        with pytest.raises(ValueError, match="no measured point"):
            FitProblem(make_mirror(), (replace(Dataset.create("a", angle_curve()), enabled=False),))

    def test_datasets_stored_as_tuple(self) -> None:
        problem = FitProblem(make_mirror(), [Dataset.create("a", angle_curve())])  # type: ignore[arg-type]
        assert isinstance(problem.datasets, tuple)

    def test_bounds_hints_empty(self) -> None:
        assert make_problem()[0].bounds_hints() == {}

    def test_evolve(self) -> None:
        problem, _ = make_problem("log")
        other = problem.evolve(weighting="linear")
        assert other.weighting == "linear"
        assert problem.weighting == "log"


class TestFitMask:
    def test_log_drops_non_positive_reflectivity(self) -> None:
        r = np.ones_like(ANGLES)
        r[[3, 7]] = 0.0
        r[11] = -1.0
        d = Dataset.create("a", angle_curve(r))
        problem = FitProblem(make_mirror(), (d,), weighting="log")
        mask = problem.fit_mask(d)
        assert not mask[[3, 7, 11]].any()
        assert mask.sum() == ANGLES.size - 3

    def test_linear_keeps_non_positive_reflectivity(self) -> None:
        r = np.ones_like(ANGLES)
        r[3] = 0.0
        d = Dataset.create("a", angle_curve(r))
        assert FitProblem(make_mirror(), (d,), weighting="linear").fit_mask(d).all()

    def test_chi2_drops_zero_uncertainty(self) -> None:
        u = np.full_like(ANGLES, 0.1)
        u[[2, 5]] = 0.0
        curve = MeasuredCurve("angle", ANGLES, np.ones_like(ANGLES), u, energy=8048.0)
        d = Dataset.create("a", curve)
        problem = FitProblem(make_mirror(), (d,), weighting="chi2")
        mask = problem.fit_mask(d)
        assert not mask[[2, 5]].any()
        assert mask.sum() == ANGLES.size - 2

    def test_respects_ranges(self) -> None:
        d = Dataset.create("a", angle_curve()).with_ranges([(0.004, 0.008)])
        mask = FitProblem(make_mirror(), (d,)).fit_mask(d)
        assert mask.sum() == int(((ANGLES >= 0.004) & (ANGLES <= 0.008)).sum())

    def test_disabled_dataset_all_false(self) -> None:
        a = Dataset.create("a", angle_curve())
        b = replace(Dataset.create("b", angle_curve()), enabled=False)
        problem = FitProblem(make_mirror(), (a, b))
        assert not problem.fit_mask(b).any()
        assert problem.fit_mask(b).shape == (ANGLES.size,)
        assert problem.fit_mask(a).all()


class TestWithParameters:
    def test_routes_values(self) -> None:
        a = Dataset.create("a", angle_curve(), scale=free(1.0, 0.5, 2.0))
        b = Dataset.create("b", angle_curve(), offset=free(0.0, -0.01, 0.01))
        problem = FitProblem(make_mirror(), (a, b), weighting="linear")
        params = problem.parameters.with_values({"t": 40.0, "a.scale": 1.5, "b.offset": 0.002})
        new = problem.with_parameters(params)
        assert new.mirror.parameters["t"].value == 40.0
        assert new.datasets[0].instrument.parameters["scale"].value == 1.5
        assert new.datasets[1].instrument.parameters["offset"].value == 0.002
        assert new.datasets[1].instrument.parameters["scale"].value == 1.0
        assert new.parameters == params
        assert problem.mirror.parameters["t"].value == 30.0

    def test_keeps_other_settings(self) -> None:
        problem, _ = make_problem("chi2")
        new = problem.with_parameters(problem.parameters)
        assert new.weighting == "chi2"
        assert new.model == problem.model


class TestResolve:
    def make(self) -> FitProblem:
        a = Dataset.create("a", angle_curve(), offset=free(0.0, -0.01, 0.01))
        b = Dataset.create("b", angle_curve(), offset="a.offset")
        return FitProblem(make_mirror(), (a, b), weighting="linear")

    def test_shared_parameter_resolved(self) -> None:
        problem = self.make()
        assert problem.parameters["b.offset"].expr == "a.offset"
        values = problem.resolve({"a.offset": 0.003})
        assert values["b.offset"] == 0.003
        assert problem.resolve()["b.offset"] == 0.0

    def test_unknown_name(self) -> None:
        with pytest.raises(KeyError):
            self.make().resolve({"nope": 1.0})

    def test_out_of_bounds(self) -> None:
        with pytest.raises(ValueError):
            self.make().resolve({"t": -1.0})

    def test_shared_parameter_takes_effect_in_evaluation(self) -> None:
        problem = self.make()
        ev = problem.evaluate({"a.offset": 0.001})
        assert ev.values["b.offset"] == 0.001
        fixed_a = Dataset.create("a", angle_curve(), offset=0.001)
        fixed_b = Dataset.create("b", angle_curve(), offset=0.001)
        reference = FitProblem(make_mirror(), (fixed_a, fixed_b), weighting="linear").evaluate()
        np.testing.assert_allclose(ev.models[1], reference.models[1])
        np.testing.assert_allclose(ev.models[0], ev.models[1])
        unshifted = problem.evaluate({"a.offset": 0.0})
        assert not np.allclose(unshifted.models[1], ev.models[1])


class TestResiduals:
    @pytest.mark.parametrize("weighting", WEIGHTINGS)
    def test_zero_for_perfect_model(self, weighting: Weighting) -> None:
        problem, _ = make_problem(weighting)
        ev = problem.evaluate()
        assert ev.error is None
        np.testing.assert_allclose(ev.residuals, 0.0, atol=1e-9)
        assert ev.objective == pytest.approx(0.0, abs=1e-15)
        assert ev.points == ANGLES.size

    def test_log_residual_of_scaled_model(self) -> None:
        problem, _ = make_problem("log", scale=2.0)
        ev = problem.evaluate()
        np.testing.assert_allclose(ev.residuals, math.log10(2.0), rtol=1e-9)
        assert ev.objective == pytest.approx(ANGLES.size * math.log10(2.0) ** 2)

    def test_linear_residual_of_scaled_model(self) -> None:
        problem, dataset = make_problem("linear", scale=2.0)
        ev = problem.evaluate()
        data = np.array(dataset.curve.reflectivity)
        np.testing.assert_allclose(ev.residuals, ev.models[0] - data)
        np.testing.assert_allclose(ev.residuals, data, rtol=1e-9)  # 2 R - R

    def test_chi2_residual_of_scaled_model(self) -> None:
        problem, dataset = make_problem("chi2", scale=2.0)
        ev = problem.evaluate()
        data = np.array(dataset.curve.reflectivity)
        sigma = np.array(dataset.curve.uncertainty)
        np.testing.assert_allclose(ev.residuals, (ev.models[0] - data) / sigma)
        np.testing.assert_allclose(ev.objective, ev.residuals @ ev.residuals)

    @pytest.mark.parametrize("weighting", WEIGHTINGS)
    def test_dataset_weight_scales_residuals(self, weighting: Weighting) -> None:
        plain = make_problem(weighting, scale=2.0)[0].evaluate()
        heavy = make_problem(weighting, scale=2.0, weight=4.0)[0].evaluate()
        np.testing.assert_allclose(heavy.residuals, 2.0 * plain.residuals)
        assert heavy.objective == pytest.approx(4.0 * plain.objective)

    def test_background_and_offset_enter_model(self) -> None:
        data = synthetic(angle_curve())
        dataset = Dataset.create("a", data, background=0.01)
        problem = FitProblem(make_mirror(), (dataset,), weighting="linear")
        np.testing.assert_allclose(problem.evaluate().residuals, 0.01, rtol=1e-9)

    def test_only_fitted_points_in_dataset_order(self) -> None:
        a = Dataset.create("a", synthetic(angle_curve()), scale=2.0).with_ranges([(0.002, 0.005)])
        b = Dataset.create("b", synthetic(angle_curve()), scale=3.0).with_ranges(
            [(0.008, 0.012)], excluded=[(0.0095, 0.0105)]
        )
        problem = FitProblem(make_mirror(), (a, b), weighting="log")
        ev = problem.evaluate()
        na, nb = int(problem.fit_mask(a).sum()), int(problem.fit_mask(b).sum())
        assert 0 < na < ANGLES.size
        assert 0 < nb < ANGLES.size
        assert ev.residuals.size == na + nb == ev.points
        np.testing.assert_allclose(ev.residuals[:na], math.log10(2.0), rtol=1e-9)
        np.testing.assert_allclose(ev.residuals[na:], math.log10(3.0), rtol=1e-9)
        assert [m.size for m in ev.models] == [na, nb]

    def test_disabled_dataset_has_no_residuals_or_model(self) -> None:
        a = Dataset.create("a", synthetic(angle_curve()), scale=2.0)
        b = replace(Dataset.create("b", synthetic(angle_curve()), scale=3.0), enabled=False)
        ev = FitProblem(make_mirror(), (a, b), weighting="log").evaluate()
        assert ev.residuals.size == ANGLES.size
        assert [m.size for m in ev.models] == [ANGLES.size, 0]
        np.testing.assert_allclose(ev.residuals, math.log10(2.0), rtol=1e-9)

    def test_log_floor_for_vanishing_model(self) -> None:
        # A model reflectivity of exactly zero is floored instead of giving -inf.
        problem, _ = make_problem("log")
        problem = problem.evolve(model=KinematicModel())
        ev = problem.evaluate()
        assert ev.error is None
        assert np.all(np.isfinite(ev.residuals))

    def test_energy_scan(self) -> None:
        mirror = make_mirror()
        data = synthetic(energy_curve(), mirror)
        dataset = Dataset.create("e", data, scale=2.0)
        problem = FitProblem(mirror, (dataset,), weighting="log")
        ev = problem.evaluate()
        assert ev.error is None
        np.testing.assert_allclose(ev.residuals, math.log10(2.0), rtol=1e-9)

    def test_energy_scan_simulate_uses_fixed_angle(self) -> None:
        mirror = make_mirror()
        dataset = Dataset.create("e", energy_curve())
        problem = FitProblem(mirror, (dataset,), weighting="linear")
        stack = mirror.stack()
        expected = problem.model.reflectance(
            stack, ENERGIES, np.full_like(ENERGIES, 0.006), "unpolarized", None
        )
        np.testing.assert_allclose(problem.simulate(dataset), expected)

    def test_energy_scan_footprint_uses_curve_angle(self) -> None:
        mirror = make_mirror()
        curve = energy_curve()
        plain = FitProblem(mirror, (Dataset.create("e", curve),), weighting="linear")
        foot = FitProblem(
            mirror,
            (Dataset.create("e", curve, beam_width=1e5, sample_length=1e7),),
            weighting="linear",
        )
        ratio = foot.simulate(foot.datasets[0]) / plain.simulate(plain.datasets[0])
        assert np.all(ratio < 1.0)
        assert np.ptp(ratio) < 1e-12

    def test_mixed_axes(self) -> None:
        mirror = make_mirror()
        a = Dataset.create("a", synthetic(angle_curve(), mirror), scale=2.0)
        e = Dataset.create("e", synthetic(energy_curve(), mirror), scale=2.0)
        ev = FitProblem(mirror, (a, e), weighting="log").evaluate()
        assert ev.points == ANGLES.size + ENERGIES.size
        np.testing.assert_allclose(ev.residuals, math.log10(2.0), rtol=1e-9)


class TestSimulate:
    def test_problem_rejects_missing_fixed_value(self) -> None:
        curve = MeasuredCurve("angle", ANGLES, np.ones_like(ANGLES))
        with pytest.raises(ValueError, match=r"fixed energy or angle is not set.*\['a'\]"):
            FitProblem(make_mirror(), (Dataset.create("a", curve),), weighting="linear")

    def test_missing_fixed_energy(self) -> None:
        good = Dataset.create(
            "good", MeasuredCurve("angle", ANGLES, np.ones_like(ANGLES), energy=8048.0)
        )
        problem = FitProblem(make_mirror(), (good,), weighting="linear")
        unset = Dataset.create("a", MeasuredCurve("angle", ANGLES, np.ones_like(ANGLES)))
        with pytest.raises(ValueError, match="energy of dataset 'a' is not set"):
            problem.simulate(unset)

    def test_missing_fixed_angle(self) -> None:
        good = Dataset.create(
            "good", MeasuredCurve("angle", ANGLES, np.ones_like(ANGLES), energy=8048.0)
        )
        problem = FitProblem(make_mirror(), (good,), weighting="linear")
        unset = Dataset.create("e", MeasuredCurve("energy", ENERGIES, np.ones_like(ENERGIES)))
        with pytest.raises(ValueError, match="angle of dataset 'e' is not set"):
            problem.simulate(unset)

    def test_disabled_dataset_may_lack_fixed_value(self) -> None:
        good = Dataset.create(
            "good", MeasuredCurve("angle", ANGLES, np.ones_like(ANGLES), energy=8048.0)
        )
        unset = replace(
            Dataset.create("a", MeasuredCurve("angle", ANGLES, np.ones_like(ANGLES))), enabled=False
        )
        FitProblem(make_mirror(), (good, unset), weighting="linear")

    def test_custom_points(self) -> None:
        problem, dataset = make_problem("linear")
        x = np.array([ANGLES[3], ANGLES[10]])
        np.testing.assert_allclose(
            problem.simulate(dataset, x), problem.simulate(dataset)[[3, 10]], rtol=1e-12
        )

    def test_values_override_stack(self) -> None:
        problem, dataset = make_problem("linear")
        thick = problem.simulate(dataset, values=problem.resolve({"t": 45.0}))
        assert not np.allclose(thick, problem.simulate(dataset))

    def test_dataset_not_in_problem(self) -> None:
        problem, _ = make_problem("linear")
        other = Dataset.create("other", angle_curve(), scale=2.0)
        values = problem.resolve() | {
            f"other.{k}": v for k, v in other.instrument.parameters.value_dict().items()
        }
        np.testing.assert_allclose(
            problem.simulate(other, values=values), 2.0 * problem.simulate(problem.datasets[0])
        )


class TestFailure:
    def test_value_out_of_bounds(self) -> None:
        problem, _ = make_problem()
        ev = problem.evaluate({"t": -1.0})
        assert ev.error is not None
        assert ev.objective == FAILED_OBJECTIVE
        assert math.isnan(ev.merit)
        assert ev.residuals.size == 0
        assert ev.stack is None
        assert ev.values["t"] == -1.0

    def test_broken_expression(self) -> None:
        dataset = Dataset.create("a", angle_curve(), background="unknown_name + 1")
        problem = FitProblem(make_mirror(), (dataset,), weighting="linear")
        ev = problem.evaluate()
        assert ev.error is not None
        assert ev.objective == FAILED_OBJECTIVE

    def test_expression_outside_bounds(self) -> None:
        a = Dataset.create("a", angle_curve(), scale=free(1.0, 0.5, 2.0))
        b = Dataset.create("b", angle_curve(), scale="a.scale * 10")
        problem = FitProblem(make_mirror(), (a, b), weighting="linear")
        assert problem.evaluate({"a.scale": 0.5}).error is None
        b_bounded = replace(
            b,
            instrument=b.instrument.with_parameters(
                b.instrument.parameters.replace("scale", max=3.0)
            ),
        )
        problem = FitProblem(make_mirror(), (a, b_bounded), weighting="linear")
        assert problem.evaluate({"a.scale": 1.5}).error is not None

    def test_residual_vector_for_failure(self) -> None:
        problem, _ = make_problem()
        r = problem.residuals({"t": -1.0}, 7)
        assert r.shape == (7,)
        assert r @ r == pytest.approx(FAILED_OBJECTIVE)

    def test_residual_vector_for_success(self) -> None:
        problem, _ = make_problem("log", scale=2.0)
        r = problem.residuals(None, 999)
        assert r.shape == (ANGLES.size,)


class TestCache:
    def test_same_values_same_object(self) -> None:
        problem, _ = make_problem()
        first = problem.evaluate({"t": 31.0})
        assert problem.evaluate({"t": 31.0}) is first
        assert problem.evaluate({"t": 31.0}) is first

    def test_current_values(self) -> None:
        problem, _ = make_problem()
        assert problem.evaluate() is problem.evaluate()

    def test_different_values_differ(self) -> None:
        problem, _ = make_problem("linear")
        a = problem.evaluate({"t": 31.0})
        b = problem.evaluate({"t": 32.0})
        assert a is not b
        assert a.objective != b.objective
        assert problem.evaluate({"t": 32.0}) is b

    def test_failures_cached(self) -> None:
        problem, _ = make_problem()
        assert problem.evaluate({"t": -1.0}) is problem.evaluate({"t": -1.0})


class TestQuantities:
    def test_chi2(self) -> None:
        problem, _ = make_problem("chi2", scale=2.0)
        ev = problem.evaluate()
        q = problem.quantities(ev)
        assert list(q) == ["χ²_red"]
        # No free parameters in the dataset, but "t" is free.
        assert q["χ²_red"] == pytest.approx(ev.objective / (ev.points - 1))

    @pytest.mark.parametrize("weighting", ["log", "linear"])
    def test_rms(self, weighting: Weighting) -> None:
        problem, _ = make_problem(weighting, scale=2.0)
        ev = problem.evaluate()
        q = problem.quantities(ev)
        assert list(q) == ["rms residual"]
        assert q["rms residual"] == pytest.approx(math.sqrt(ev.objective / ev.points))

    def test_failed_evaluation(self) -> None:
        problem, _ = make_problem()
        assert problem.quantities(problem.evaluate({"t": -1.0})) == {}
