"""Tests of the optimizers with a problem that has nothing to do with mirror design."""

import itertools
from collections.abc import Mapping
from dataclasses import dataclass, field, replace

import numpy as np
import pytest
from numpy.typing import NDArray

from xstack.core.optim.events import CancellationToken, ProgressEvent
from xstack.core.optim.optimizers import LmfitOptimizer, prepare_parameters
from xstack.core.parameters import Parameter, ParameterSet

TRUE_A = 2.0
TRUE_B = -1.0

_rng = np.random.default_rng(1234)
X = np.linspace(0.0, 10.0, 40)
Y = TRUE_A * X + TRUE_B + _rng.normal(0.0, 0.05, X.size)


def line_parameters() -> ParameterSet:
    return ParameterSet([Parameter(n, 0.5, vary=True, min=-10.0, max=10.0) for n in ("a", "b")])


@dataclass(frozen=True)
class LineEvaluation:
    values: Mapping[str, float]
    objective: float
    merit: float
    residuals: NDArray[np.float64]
    error: str | None = None


@dataclass(frozen=True)
class LineProblem:
    parameters: ParameterSet = field(default_factory=line_parameters)

    def with_parameters(self, parameters: ParameterSet) -> "LineProblem":
        return replace(self, parameters=parameters)

    def _values(self, values: Mapping[str, float] | None) -> dict[str, float]:
        merged = self.parameters.value_dict()
        if values is not None:
            merged.update(values)
        return merged

    def evaluate(self, values: Mapping[str, float] | None = None) -> LineEvaluation:
        merged = self._values(values)
        r = np.asarray(merged["a"] * X + merged["b"] - Y, dtype=np.float64)
        chi2 = float(np.sum(r**2))
        return LineEvaluation(merged, chi2, chi2, r)

    def residuals(self, evaluation: LineEvaluation, size: int) -> NDArray[np.float64]:
        return evaluation.residuals

    def quantities(self, evaluation: LineEvaluation) -> dict[str, float]:
        return {"chi2": evaluation.objective}

    def bounds_hints(self) -> dict[str, tuple[float, float]]:
        return {}


@dataclass(frozen=True)
class HintedLineProblem(LineProblem):
    hints: tuple[tuple[str, tuple[float, float]], ...] = ()

    def bounds_hints(self) -> dict[str, tuple[float, float]]:
        return dict(self.hints)


@dataclass(frozen=True)
class CountingLineProblem(LineProblem):
    calls: list[int] = field(default_factory=lambda: [0])

    def with_parameters(self, parameters: ParameterSet) -> "CountingLineProblem":
        return replace(self, parameters=parameters)

    def evaluate(self, values: Mapping[str, float] | None = None) -> LineEvaluation:
        self.calls[0] += 1
        return super().evaluate(values)


def assert_valid_result(result_history: tuple[float, ...], evaluations: int) -> None:
    assert evaluations == len(result_history)
    assert all(b <= a for a, b in itertools.pairwise(result_history))


class TestLmfitOptimizerOnLine:
    @pytest.mark.parametrize("method", ["leastsq", "nelder"])
    def test_recovers_line(self, method: str) -> None:
        problem = LineProblem()
        result = LmfitOptimizer(method).run(problem)
        assert result.parameters["a"].value == pytest.approx(TRUE_A, abs=0.05)
        assert result.parameters["b"].value == pytest.approx(TRUE_B, abs=0.1)
        assert isinstance(result.best, LineEvaluation)
        assert result.best.objective < problem.evaluate().objective
        assert result.method == method
        assert_valid_result(result.history, result.evaluations)

    def test_differential_evolution_improves(self) -> None:
        problem = LineProblem()
        optimizer = LmfitOptimizer(
            "differential_evolution", max_evaluations=300, options=(("seed", 1),)
        )
        result = optimizer.run(problem)
        assert isinstance(result.best, LineEvaluation)
        assert result.best.objective < problem.evaluate().objective
        assert_valid_result(result.history, result.evaluations)

    @pytest.mark.parametrize("method", ["leastsq", "least_squares", "nelder"])
    def test_evaluates_once_per_objective_call(self, method: str) -> None:
        problem = CountingLineProblem()
        result = LmfitOptimizer(method).run(problem)
        assert result.evaluations > 1
        assert problem.calls[0] == result.evaluations

    def test_start_parameters(self) -> None:
        start = line_parameters().with_values({"a": 1.9, "b": -0.9})
        result = LmfitOptimizer("leastsq").run(LineProblem(), start=start)
        assert result.parameters["a"].value == pytest.approx(TRUE_A, abs=0.05)

    def test_progress_events(self) -> None:
        received: list[ProgressEvent] = []
        LmfitOptimizer("nelder").run(LineProblem(), events=received.append)
        assert received
        assert all("chi2" in e.best_quantities for e in received)
        assert received[-1].task == "nelder"
        objectives = [e.best_objective for e in received]
        assert objectives == sorted(objectives, reverse=True)

    def test_cancelled_before_start(self) -> None:
        token = CancellationToken()
        token.cancel()
        problem = LineProblem()
        result = LmfitOptimizer("nelder").run(problem, cancel=token)
        assert result.cancelled
        assert isinstance(result.best, LineEvaluation)
        assert result.best.objective <= problem.evaluate().objective
        assert set(result.parameters) == {"a", "b"}

    def test_no_free_parameter(self) -> None:
        fixed = ParameterSet([Parameter("a", 1.0), Parameter("b", 0.0)])
        with pytest.raises(ValueError, match="no free"):
            LmfitOptimizer("leastsq").run(LineProblem(fixed))

    def test_global_method_requires_finite_bounds(self) -> None:
        unbounded = ParameterSet([Parameter("a", 1.0, vary=True), Parameter("b", 0.0)])
        with pytest.raises(ValueError, match="finite bounds"):
            LmfitOptimizer("differential_evolution").run(LineProblem(unbounded))


class TestPrepareParameters:
    def test_bounds_hints_intersect(self) -> None:
        problem = HintedLineProblem(hints=(("a", (1.0, 20.0)),))
        params = prepare_parameters(problem, None, vary_integers=False)
        assert (params["a"].min, params["a"].max) == (1.0, 10.0)
        assert params["a"].value == 1.0
        assert (params["b"].min, params["b"].max) == (-10.0, 10.0)

    def test_empty_intersection_is_ignored(self) -> None:
        problem = HintedLineProblem(hints=(("a", (20.0, 30.0)),))
        params = prepare_parameters(problem, None, vary_integers=False)
        assert (params["a"].min, params["a"].max) == (-10.0, 10.0)

    def test_unknown_hint_is_ignored(self) -> None:
        problem = HintedLineProblem(hints=(("zzz", (0.0, 1.0)),))
        assert prepare_parameters(problem, None, vary_integers=False) == problem.parameters

    def test_hints_applied_by_optimizer(self) -> None:
        problem = HintedLineProblem(hints=(("a", (0.0, 1.5)),))
        result = LmfitOptimizer("leastsq").run(problem)
        assert result.parameters["a"].value <= 1.5

    def test_integers_fixed_unless_requested(self) -> None:
        params = ParameterSet(
            [
                Parameter("n", 3, vary=True, min=1, max=10, integer=True),
                Parameter("a", 1.0, vary=True, min=0, max=2),
            ]
        )
        problem = LineProblem(params)
        fixed = prepare_parameters(problem, None, vary_integers=False)
        assert fixed.free_names() == ("a",)
        varied = prepare_parameters(problem, None, vary_integers=True)
        assert varied.free_names() == ("n", "a")


class TestDesignOnlyOptimizers:
    def test_reject_generic_problem(self) -> None:
        from xstack.design.optim import BraggOutline, GridScan

        with pytest.raises(TypeError, match="DesignProblem"):
            BraggOutline().run(LineProblem())
        with pytest.raises(TypeError, match="DesignProblem"):
            GridScan(points=3).run(LineProblem())
