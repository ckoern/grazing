"""Tests for `xstack.fit.results`."""

import json
import math

import numpy as np
import pytest

from xstack.core import materials
from xstack.core.measurement import MeasuredCurve
from xstack.core.optim.optimizers import LmfitOptimizer, OptimizationResult
from xstack.core.parameters import free
from xstack.core.serialization import from_data, to_data
from xstack.core.structure import Mirror, SingleLayer
from xstack.fit.dataset import Dataset
from xstack.fit.problem import FitProblem, Weighting
from xstack.fit.results import STRONG_CORRELATION, FitResult, FitStatistics, analyze

PT, SI = materials.get("Pt"), materials.get("Si")
ANGLES = np.linspace(0.002, 0.012, 60)


def fixed_mirror() -> Mirror:
    return Mirror(SingleLayer.create(PT, t=30.0), SI)


def linear_problem(weighting: Weighting, **instrument: object) -> tuple[FitProblem, np.ndarray]:
    """A fit in which only instrument parameters are free; returns it and the sample curve."""
    rng = np.random.default_rng(3)
    base = Dataset.create("a", MeasuredCurve("angle", ANGLES, np.ones(60), energy=8048.0))
    reference = FitProblem(fixed_mirror(), (base,), weighting="linear")
    sample = reference.simulate(base)
    measured = 1.05 * sample + 0.002 + 0.02 * sample * rng.standard_normal(60)
    sigma = 0.02 * sample + 1e-4
    curve = MeasuredCurve("angle", ANGLES, measured, sigma, energy=8048.0)
    dataset = Dataset.create("a", curve, **instrument)  # type: ignore[arg-type]
    return FitProblem(fixed_mirror(), (dataset,), weighting=weighting), sample


def design_matrix(problem: FitProblem, sample: np.ndarray) -> np.ndarray:
    return np.column_stack([sample, np.ones_like(sample)])


class TestAnalyzeLinear:
    def make(self, weighting: Weighting) -> tuple[FitProblem, np.ndarray]:
        return linear_problem(
            weighting, scale=free(1.0, 0.5, 2.0), background=free(0.001, -0.1, 0.1)
        )

    def test_chi2_matches_weighted_least_squares(self) -> None:
        problem, sample = self.make("chi2")
        sigma = np.array(problem.datasets[0].curve.uncertainty)
        j = design_matrix(problem, sample) / sigma[:, None]
        covariance = np.linalg.inv(j.T @ j)
        stats = analyze(problem)
        assert stats.names == ("a.scale", "a.background")
        np.testing.assert_allclose(stats.errors, np.sqrt(np.diag(covariance)), rtol=1e-4)
        expected = covariance / np.sqrt(np.outer(np.diag(covariance), np.diag(covariance)))
        np.testing.assert_allclose(np.array(stats.correlation), expected, rtol=1e-4, atol=1e-6)
        assert stats.determined
        assert stats.weighting == "chi2"
        assert stats.points == 60

    def test_linear_scales_covariance_by_residual_scatter(self) -> None:
        problem, sample = self.make("linear")
        j = design_matrix(problem, sample)
        residuals = problem.evaluate().residuals
        objective = float(residuals @ residuals)
        covariance = objective / (60 - 2) * np.linalg.inv(j.T @ j)
        stats = analyze(problem)
        np.testing.assert_allclose(stats.errors, np.sqrt(np.diag(covariance)), rtol=1e-4)
        assert stats.objective == pytest.approx(objective)
        assert stats.reduced_objective == pytest.approx(objective / 58)
        assert stats.degrees_of_freedom == 58

    def test_values_and_names(self) -> None:
        problem, _ = self.make("chi2")
        stats = analyze(problem)
        assert stats.values == (1.0, 0.001)

    def test_correlation_matrix_properties(self) -> None:
        stats = analyze(self.make("chi2")[0])
        c = np.array(stats.correlation)
        np.testing.assert_allclose(c, c.T, atol=1e-12)
        np.testing.assert_allclose(np.diag(c), 1.0)
        assert np.all(np.abs(c) <= 1.0)
        assert stats.correlation_of("a.scale", "a.background") == pytest.approx(c[0, 1])
        assert stats.correlation_of("a.background", "a.scale") == pytest.approx(c[0, 1])

    def test_error_lookup(self) -> None:
        stats = analyze(self.make("chi2")[0])
        assert stats.error("a.scale") == stats.errors[0]
        assert stats.error("a.background") == stats.errors[1]
        with pytest.raises(KeyError, match="not a free parameter"):
            stats.error("t")
        with pytest.raises(KeyError):
            stats.correlation_of("a.scale", "t")
        with pytest.raises(KeyError):
            stats.correlation_of("t", "a.scale")

    def test_parameter_at_bound_uses_one_sided_difference(self) -> None:
        problem, sample = linear_problem(
            "chi2", scale=free(1.0, 1.0, 2.0), background=free(0.001, -0.1, 0.1)
        )
        sigma = np.array(problem.datasets[0].curve.uncertainty)
        j = design_matrix(problem, sample) / sigma[:, None]
        expected = np.sqrt(np.diag(np.linalg.inv(j.T @ j)))
        stats = analyze(problem)
        assert stats.determined
        np.testing.assert_allclose(stats.errors, expected, rtol=1e-4)

    def test_upper_bound(self) -> None:
        problem, sample = linear_problem(
            "chi2", scale=free(2.0, 0.5, 2.0), background=free(0.001, -0.1, 0.1)
        )
        sigma = np.array(problem.datasets[0].curve.uncertainty)
        j = design_matrix(problem, sample) / sigma[:, None]
        expected = np.sqrt(np.diag(np.linalg.inv(j.T @ j)))
        np.testing.assert_allclose(analyze(problem).errors, expected, rtol=1e-4)

    def test_free_parameter_with_equal_bounds_excluded(self) -> None:
        problem, _ = linear_problem(
            "chi2", scale=free(1.0, 1.0, 1.0), background=free(0.001, -0.1, 0.1)
        )
        assert analyze(problem).names == ("a.background",)


class TestAnalyzeErrors:
    def test_no_free_parameters(self) -> None:
        problem, _ = linear_problem("chi2")
        with pytest.raises(ValueError, match="no free parameters"):
            analyze(problem)

    def test_integer_parameters_excluded(self) -> None:
        from xstack.core.structure import Periodic

        w = materials.get("W")
        mirror = Mirror(Periodic.create(w, SI, N=free(10, 5, 20), d=4.0), SI)
        assert mirror.parameters.free_names() == ("N",)
        curve = MeasuredCurve("angle", ANGLES, np.full(60, 0.1), energy=8048.0)
        problem = FitProblem(mirror, (Dataset.create("a", curve),), weighting="linear")
        with pytest.raises(ValueError, match="no free parameters"):
            analyze(problem)

    def test_irrelevant_parameter_is_undetermined(self) -> None:
        # beam_width has no effect while sample_length is zero.
        problem, _ = linear_problem(
            "chi2", background=free(0.001, -0.1, 0.1), beam_width=free(1e5, 0.0, 1e6)
        )
        stats = analyze(problem)
        assert stats.names == ("a.background", "a.beam_width")
        assert all(math.isnan(e) for e in stats.errors)
        assert not stats.determined
        c = stats.correlation
        assert c[0][0] == c[1][1] == 1.0
        assert math.isnan(c[0][1])
        assert math.isnan(c[1][0])
        assert stats.strong_correlations() == []
        assert stats.objective > 0
        assert stats.points == 60

    def test_too_few_points(self) -> None:
        curve = MeasuredCurve("angle", ANGLES[:2], [0.5, 0.4], [0.1, 0.1], energy=8048.0)
        dataset = Dataset.create("a", curve, scale=free(1.0, 0.5, 2.0), background=free(0.0, -1, 1))
        problem = FitProblem(fixed_mirror(), (dataset,), weighting="chi2")
        stats = analyze(problem)
        assert stats.degrees_of_freedom == 0
        assert not stats.determined
        assert math.isnan(stats.reduced_objective)


def make_stats(correlation: tuple[tuple[float, ...], ...]) -> FitStatistics:
    n = len(correlation)
    names = tuple("abcdef"[:n])
    return FitStatistics(names, (1.0,) * n, (0.1,) * n, correlation, 1.0, 10, "chi2")


class TestStrongCorrelations:
    def test_ordering_and_threshold(self) -> None:
        c = (
            (1.0, 0.95, -0.99, 0.1),
            (0.95, 1.0, 0.5, -0.85),
            (-0.99, 0.5, 1.0, 0.0),
            (0.1, -0.85, 0.0, 1.0),
        )
        stats = make_stats(c)
        assert stats.strong_correlations(0.9) == [("a", "c", -0.99), ("a", "b", 0.95)]
        assert STRONG_CORRELATION == 0.8
        assert [p[:2] for p in stats.strong_correlations()] == [
            ("a", "c"),
            ("a", "b"),
            ("b", "d"),
        ]
        assert stats.strong_correlations(0.999) == []
        assert len(stats.strong_correlations(0.0)) == 6

    def test_threshold_inclusive(self) -> None:
        stats = make_stats(((1.0, 0.9), (0.9, 1.0)))
        assert stats.strong_correlations(0.9) == [("a", "b", 0.9)]

    def test_nan_ignored(self) -> None:
        nan = math.nan
        assert make_stats(((1.0, nan), (nan, 1.0))).strong_correlations(0.0) == []


class TestFitResult:
    def test_from_optimization_with_least_squares(self) -> None:
        blank = Dataset.create("a", MeasuredCurve("angle", ANGLES, np.ones(60), energy=8048.0))
        truth = FitProblem(fixed_mirror(), (blank,), weighting="linear")
        sim = truth.simulate(blank)
        curve = MeasuredCurve("angle", ANGLES, sim, 0.02 * sim + 1e-6, energy=8048.0)
        dataset = Dataset.create("a", curve)
        mirror = Mirror(SingleLayer.create(PT, t=free(33.0, 10.0, 60.0)), SI)
        problem = FitProblem(mirror, (dataset,), weighting="chi2")
        optimization = LmfitOptimizer("least_squares").run(problem, None, None, None)
        result = FitResult.from_optimization(problem, optimization)
        assert result.method == "least_squares"
        assert result.evaluations == optimization.evaluations > 0
        assert result.parameters["t"].value == pytest.approx(30.0, abs=1e-3)
        assert not result.cancelled
        assert result.note == ""
        assert result.created.endswith("+00:00")
        assert result.elapsed >= 0.0
        stats = result.statistics
        assert stats is not None
        assert stats.names == ("t",)
        assert stats.values[0] == pytest.approx(30.0, abs=1e-3)
        assert stats.determined
        assert stats.error("t") > 0.0
        assert stats.objective == pytest.approx(0.0, abs=1e-3)

    def test_failed_analysis_stores_note(self) -> None:
        problem, _ = linear_problem("chi2")  # nothing free
        evaluation = problem.evaluate()
        optimization = OptimizationResult(
            problem.parameters, evaluation, 3, (1.0,), "manual", "message", True, 0.5
        )
        result = FitResult.from_optimization(problem, optimization)
        assert result.statistics is None
        assert "no free parameters" in result.note
        assert result.method == "manual"
        assert result.message == "message"
        assert result.cancelled
        assert result.evaluations == 3
        assert result.elapsed == 0.5

    def test_serialization_round_trip(self) -> None:
        problem, _ = linear_problem(
            "chi2", scale=free(1.0, 0.5, 2.0), background=free(0.001, -0.1, 0.1)
        )
        stats = analyze(problem)
        result = FitResult(
            problem.parameters,
            stats,
            "least_squares",
            "ok",
            12,
            False,
            1.5,
            "2026-01-01T00:00:00+00:00",
        )
        restored = from_data(json.loads(json.dumps(to_data(result))))
        assert isinstance(restored, FitResult)
        assert restored == result
        assert restored.statistics == stats

    def test_serialization_without_statistics(self) -> None:
        problem, _ = linear_problem("chi2")
        result = FitResult(problem.parameters, None, note="because")
        assert from_data(json.loads(json.dumps(to_data(result)))) == result

    def test_statistics_with_nan_survive_json(self) -> None:
        nan = math.nan
        stats = FitStatistics(("a",), (1.0,), (nan,), ((1.0,),), 2.0, 5, "log")
        restored = from_data(json.loads(json.dumps(to_data(stats))))
        assert isinstance(restored, FitStatistics)
        assert math.isnan(restored.errors[0])
        assert restored.names == ("a",)
