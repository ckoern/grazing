"""Tests for the lmfit optimizer."""

import itertools
import math

import pytest

from tests.optim.test_problem import SI, W, make_problem
from xstack.manufacturing.constraints import MinLayerThickness
from xstack.merit import MeanReflectivity
from xstack.optim import (
    CancellationToken,
    DesignProblem,
    LmfitOptimizer,
    ProgressEvent,
    optimizers,
)
from xstack.optim.optimizers import prepare_parameters
from xstack.optim.problem import Evaluation
from xstack.parameters import free
from xstack.structure import Mirror, Periodic

D_OPT = 4.0


def _de_options(method: str = "differential_evolution") -> tuple[tuple[str, object], ...]:
    return (("seed", 2), ("popsize", 5)) if method == "differential_evolution" else ()


def integer_problem() -> DesignProblem:
    """Problem with a free number of bilayers."""
    coating = Periodic.create(
        W, SI, N=free(20, 10, 30), d=free(3.9, 3.5, 4.5), gamma=free(0.3, 0.1, 0.7)
    )
    return DesignProblem(Mirror(coating, SI), make_problem().targets, MeanReflectivity())


class TestRegistry:
    def test_registered(self) -> None:
        assert "lmfit" in optimizers
        assert "grid_scan" in optimizers
        assert optimizers.get("lmfit") is LmfitOptimizer

    def test_settings_are_immutable(self) -> None:
        opt = LmfitOptimizer("nelder", max_evaluations=10)
        with pytest.raises(AttributeError):
            opt.method = "leastsq"  # type: ignore[misc]


class TestErrors:
    def test_no_free_parameters(self) -> None:
        problem = make_problem(d=4.0, gamma=0.3)
        with pytest.raises(ValueError, match="no free parameters"):
            LmfitOptimizer("nelder").run(problem)

    def test_only_integer_free_with_local_method(self) -> None:
        coating = Periodic.create(W, SI, N=free(20, 10, 30), d=4.0, gamma=0.3)
        problem = DesignProblem(Mirror(coating, SI), make_problem().targets, MeanReflectivity())
        with pytest.raises(ValueError, match="no free parameters"):
            LmfitOptimizer("leastsq").run(problem)

    def test_global_method_needs_finite_bounds(self) -> None:
        problem = make_problem()
        params = problem.parameters.replace("d", min=float("-inf"), max=float("inf"))
        with pytest.raises(ValueError, match="finite bounds"):
            LmfitOptimizer("differential_evolution").run(problem, start=params)


class TestIntegerParameters:
    def test_prepare_fixes_integers_for_local_methods(self) -> None:
        params = prepare_parameters(integer_problem(), None, vary_integers=False)
        assert params.free_names() == ("d", "gamma")
        assert params["N"].vary is False

    def test_prepare_keeps_integers_for_global_methods(self) -> None:
        params = prepare_parameters(integer_problem(), None, vary_integers=True)
        assert set(params.free_names()) == {"N", "d", "gamma"}

    def test_local_method_keeps_n(self) -> None:
        problem = integer_problem()
        result = LmfitOptimizer("nelder", max_evaluations=40).run(problem)
        assert result.parameters["N"].value == 20
        assert result.best.values["N"] == 20

    def test_global_method_varies_n(self) -> None:
        problem = integer_problem()
        opt = LmfitOptimizer(
            "differential_evolution", max_evaluations=15, options=(("seed", 3), ("popsize", 6))
        )
        result = opt.run(problem)
        n = result.parameters["N"].value
        assert n == round(n)
        assert 10 <= n <= 30
        assert result.best.values["N"] == n


class TestConstraintHints:
    def test_hints_narrow_bounds(self) -> None:
        problem = make_problem(
            d=4.0,
            gamma=free(0.5, 0.0, 1.0),
            constraints=(MinLayerThickness(minimum=1.0),),
        )
        params = prepare_parameters(problem, None, vary_integers=False)
        assert params["gamma"].min == pytest.approx(0.25)
        assert params["gamma"].max == pytest.approx(0.75)

    def test_hints_never_widen_bounds(self) -> None:
        problem = make_problem(
            d=4.0,
            gamma=free(0.5, 0.4, 0.6),
            constraints=(MinLayerThickness(minimum=1.0),),
        )
        params = prepare_parameters(problem, None, vary_integers=False)
        assert (params["gamma"].min, params["gamma"].max) == pytest.approx((0.4, 0.6))

    def test_optimizer_respects_hinted_bounds(self) -> None:
        # The unconstrained optimum gamma = 0.234 lies below the hinted lower bound 0.25.
        problem = make_problem(
            d=3.958,
            gamma=free(0.5, 0.0, 1.0),
            constraints=(MinLayerThickness(minimum=0.99),),
        )
        result = LmfitOptimizer("nelder", max_evaluations=60).run(problem)
        gamma = result.parameters["gamma"]
        assert gamma.min == pytest.approx(0.99 / 3.958)
        assert gamma.value >= gamma.min - 1e-12
        assert result.best.feasible


class TestConvergence:
    def test_leastsq_and_nelder_agree(self) -> None:
        problem = make_problem(n=30, points=9, d=free(3.95, 3.0, 5.0), gamma=free(0.3, 0.1, 0.7))
        start = problem.evaluate()
        r_ls = LmfitOptimizer("leastsq").run(problem)
        r_nm = LmfitOptimizer("nelder").run(problem)
        assert r_ls.best.merit < start.merit
        assert r_nm.best.merit < start.merit
        # The merit is flat along the valley, so compare merits and loosely the parameters.
        assert r_ls.best.merit == pytest.approx(r_nm.best.merit, abs=5e-3)
        assert r_ls.parameters["d"].value == pytest.approx(r_nm.parameters["d"].value, abs=0.15)
        assert r_ls.parameters["d"].value == pytest.approx(D_OPT, abs=0.15)

    def test_differential_evolution_reproducible(self) -> None:
        problem = make_problem()
        opt = LmfitOptimizer(
            "differential_evolution", max_evaluations=15, options=(("seed", 7), ("popsize", 6))
        )
        a = opt.run(problem)
        b = opt.run(problem)
        assert a.history == b.history
        assert a.parameters.value_dict() == b.parameters.value_dict()

    def test_differential_evolution_improves(self) -> None:
        problem = make_problem()
        opt = LmfitOptimizer(
            "differential_evolution", max_evaluations=60, options=(("seed", 1), ("popsize", 8))
        )
        result = opt.run(problem)
        assert result.best.merit < problem.evaluate().merit
        assert result.best.merit < 0.4

    @pytest.mark.parametrize("method", ["differential_evolution", "nelder", "leastsq"])
    def test_never_worse_than_start(self, method: str) -> None:
        problem = make_problem()
        good = LmfitOptimizer("least_squares").run(problem).parameters
        start = problem.with_parameters(good).evaluate().objective
        opt = LmfitOptimizer(method, max_evaluations=15, options=_de_options(method))
        result = opt.run(problem, good)
        assert result.history[0] == pytest.approx(start)
        assert result.best.objective <= start

    def test_differential_evolution_population_contains_start(self) -> None:
        problem = make_problem()
        start = problem.parameters.with_values({"d": 3.95, "gamma": 0.24})
        seen: list[dict[str, float]] = []
        original = problem.evaluate

        def spy(values: dict[str, float] | None = None) -> Evaluation:
            seen.append(dict(values or {}))
            return original(values)

        object.__setattr__(problem, "evaluate", spy)
        opt = LmfitOptimizer("differential_evolution", max_evaluations=15, options=_de_options())
        opt.run(problem, start)
        hits = [v for v in seen[1:] if v.get("d") == pytest.approx(3.95)]
        assert hits
        assert hits[0]["gamma"] == pytest.approx(0.24)

    def test_differential_evolution_explicit_init_kept(self) -> None:
        problem = make_problem()
        opt = LmfitOptimizer(
            "differential_evolution",
            max_evaluations=10,
            options=(("seed", 1), ("init", "random"), ("popsize", 5)),
        )
        assert opt.run(problem).evaluations > 0

    def test_max_evaluations_limits_work(self) -> None:
        problem = make_problem()
        result = LmfitOptimizer("nelder", max_evaluations=15).run(problem)
        assert result.evaluations <= 20

    def test_start_parameters_are_used(self) -> None:
        problem = make_problem()
        start = problem.parameters.with_values({"d": 3.95, "gamma": 0.24})
        result = LmfitOptimizer("nelder", max_evaluations=5).run(problem, start=start)
        assert result.history[0] == pytest.approx(
            problem.evaluate({"d": 3.95, "gamma": 0.24}).objective
        )


class TestResultFields:
    def test_result_fields(self) -> None:
        problem = make_problem()
        result = LmfitOptimizer("nelder").run(problem)
        assert result.method == "nelder"
        assert result.evaluations == len(result.history) > 0
        assert all(b <= a for a, b in itertools.pairwise(result.history))
        assert result.history[-1] == result.best.objective
        assert result.elapsed > 0
        assert not result.cancelled
        for name in ("d", "gamma"):
            p = result.parameters[name]
            assert p.min <= p.value <= p.max
            assert p.value == pytest.approx(result.best.values[name])
        assert result.best.objective == pytest.approx(
            problem.evaluate(result.parameters.value_dict()).objective
        )

    def test_start_design_not_buildable_for_residual_methods(self) -> None:
        problem = make_problem(gamma="d * 0.5")
        with pytest.raises(ValueError, match="cannot be built"):
            LmfitOptimizer("leastsq").run(problem)


class TestEvents:
    def test_events_carry_best_quantities(self) -> None:
        events: list[ProgressEvent] = []
        LmfitOptimizer("nelder", max_evaluations=30).run(make_problem(), events=events.append)
        assert events
        for event in events:
            assert set(event.best_quantities) == {"mean reflectivity"}
            assert 0.0 <= event.best_quantities["mean reflectivity"] <= 1.0
        assert events[-1].best_quantities["mean reflectivity"] == pytest.approx(
            1.0 - events[-1].best_merit
        )

    def test_progress_events(self) -> None:
        events: list[ProgressEvent] = []
        LmfitOptimizer("nelder", max_evaluations=30).run(make_problem(), events=events.append)
        assert events
        counts = [e.evaluations for e in events]
        assert counts == sorted(counts)
        assert len(set(counts)) == len(counts)
        best = [e.best_objective for e in events]
        assert all(b <= a for a, b in itertools.pairwise(best))
        assert all(e.task == "nelder" for e in events)
        assert events[-1].best_objective <= events[0].best_objective
        assert set(events[0].best_values) >= {"d", "gamma"}

    def test_cancellation(self) -> None:
        problem = make_problem()
        opt = LmfitOptimizer("nelder")
        full = opt.run(problem)
        token = CancellationToken()

        def on_event(event: ProgressEvent) -> None:
            if event.evaluations >= 5:
                token.cancel()

        cancelled = opt.run(problem, events=on_event, cancel=token)
        assert cancelled.cancelled
        assert not full.cancelled
        assert cancelled.evaluations < full.evaluations
        assert cancelled.best.stack is not None

    def test_cancellation_of_global_method(self) -> None:
        token = CancellationToken()

        def on_event(event: ProgressEvent) -> None:
            if event.evaluations >= 10:
                token.cancel()

        opt = LmfitOptimizer("differential_evolution", options=(("seed", 1), ("popsize", 8)))
        result = opt.run(make_problem(), events=on_event, cancel=token)
        assert result.cancelled
        assert result.evaluations < 200

    def test_precancelled_token(self) -> None:
        token = CancellationToken()
        token.cancel()
        result = LmfitOptimizer("nelder").run(make_problem(), cancel=token)
        assert result.cancelled
        assert result.evaluations <= 5


class TestExpressions:
    @pytest.mark.parametrize("method", ["nelder", "leastsq"])
    def test_expression_parameter_stays_consistent(self, method: str) -> None:
        problem = make_problem(d=free(3.5, 3.0, 5.0), gamma="d * 0.06")
        assert problem.parameters.free_names() == ("d",)
        result = LmfitOptimizer(method, max_evaluations=40).run(problem)
        assert result.best.values["gamma"] == pytest.approx(0.06 * result.best.values["d"])
        assert result.parameters["gamma"].expr == "d * 0.06"
        assert result.best.error is None
        assert result.history[-1] <= problem.evaluate().objective


class TestEventMerit:
    def test_best_merit_finite_for_buildable_design(self) -> None:
        events: list[ProgressEvent] = []
        LmfitOptimizer("nelder", max_evaluations=20).run(make_problem(), events=events.append)
        assert events
        assert all(math.isfinite(e.best_merit) for e in events)
