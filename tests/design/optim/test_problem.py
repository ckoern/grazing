"""Tests for the optimization events and the design problem."""

import math
from typing import ClassVar

import numpy as np
import pytest

from xstack.core.materials import get
from xstack.core.optim import CancellationToken, ProgressEvent
from xstack.core.parameters import free
from xstack.core.structure import Mirror, Periodic
from xstack.core.targets import EnergyScan, TargetSet
from xstack.design.manufacturing.constraints import MaxTotalThickness, MinLayerThickness
from xstack.design.manufacturing.process import DepositionProcess
from xstack.design.merit import MeanReflectivity
from xstack.design.optim import DesignProblem, Evaluation

W = get("W")
SI = get("Si")
ENERGY = 8048.0
THETA = 0.02


def make_problem(
    n: int = 20,
    d: object = None,
    gamma: object = None,
    points: int = 5,
    **kwargs: object,
) -> DesignProblem:
    """Small W/Si problem at 8048 eV with a narrow band."""
    coating = Periodic.create(
        W,
        SI,
        N=n,
        d=free(3.5, 3.0, 5.0) if d is None else d,  # type: ignore[arg-type]
        gamma=free(0.4, 0.1, 0.7) if gamma is None else gamma,  # type: ignore[arg-type]
    )
    targets = TargetSet(
        EnergyScan(
            angle=THETA,
            energy_min=ENERGY - 20,
            energy_max=ENERGY + 20,
            points=points,
            polarization="s",
        )
    )
    return DesignProblem(Mirror(coating, SI), targets, MeanReflectivity(), **kwargs)  # type: ignore[arg-type]


class TestEvents:
    def test_token_starts_uncancelled(self) -> None:
        assert not CancellationToken().cancelled

    def test_token_cancel_is_sticky(self) -> None:
        token = CancellationToken()
        token.cancel()
        token.cancel()
        assert token.cancelled

    def test_tokens_are_independent(self) -> None:
        a, b = CancellationToken(), CancellationToken()
        a.cancel()
        assert not b.cancelled

    def test_progress_event_fields_and_immutability(self) -> None:
        event = ProgressEvent("t", 3, None, 0.5, 0.6, {"d": 4.0}, 0.1)
        assert event.task == "t"
        assert event.evaluations == 3
        assert event.total is None
        assert event.best_values == {"d": 4.0}
        with pytest.raises(AttributeError):
            event.task = "x"  # type: ignore[misc]


class TestEvaluate:
    def test_parameters_property(self) -> None:
        problem = make_problem()
        assert problem.parameters is problem.mirror.parameters
        assert set(problem.parameters.free_names()) == {"d", "gamma"}

    def test_evaluation_contents(self) -> None:
        ev = make_problem().evaluate()
        assert isinstance(ev, Evaluation)
        assert ev.stack is not None
        assert len(ev.results) == 1
        assert ev.error is None
        assert ev.feasible
        assert 0.0 <= ev.merit <= 1.0
        assert ev.values["N"] == 20
        assert ev.values["d"] == 3.5

    def test_objective_equals_merit_without_constraints(self) -> None:
        ev = make_problem().evaluate()
        assert ev.objective == ev.merit
        assert ev.constraints == ()

    def test_coating_time_only_with_process(self) -> None:
        assert make_problem().evaluate().coating_time is None
        process = DepositionProcess({"W": 0.1, "Si": 0.2}, overhead_per_layer=1.0)
        ev = make_problem(process=process).evaluate()
        assert ev.stack is not None
        assert ev.coating_time is not None
        assert ev.coating_time == pytest.approx(process.time(ev.stack))
        assert ev.coating_time > 0

    def test_evaluate_overrides_values(self) -> None:
        problem = make_problem()
        ev = problem.evaluate({"d": 4.0, "gamma": 0.25})
        assert ev.values["d"] == 4.0
        assert ev.values["gamma"] == 0.25
        assert ev.merit != problem.evaluate().merit
        # The problem itself is unchanged.
        assert problem.parameters["d"].value == 3.5

    def test_evaluate_rounds_integer_values(self) -> None:
        coating = Periodic.create(W, SI, N=free(20, 5, 40), d=4.0, gamma=0.3)
        problem = DesignProblem(Mirror(coating, SI), make_problem().targets, MeanReflectivity())
        assert problem.evaluate({"N": 10.4}).values["N"] == 10

    def test_better_design_has_lower_merit(self) -> None:
        problem = make_problem()
        assert problem.evaluate({"d": 3.96, "gamma": 0.234}).merit < problem.evaluate().merit


class TestConstraintPenalties:
    # d = 3.5, gamma = 0.2 gives a 0.7 nm absorber layer.
    thin: ClassVar[dict[str, float]] = {"d": 3.5, "gamma": 0.2}
    thick: ClassVar[dict[str, float]] = {"d": 3.5, "gamma": 0.5}

    def test_soft_penalty(self) -> None:
        constraint = MaxTotalThickness(maximum=50.0, severity="soft")
        problem = make_problem(constraints=(constraint,), soft_penalty=2.5)
        ev = problem.evaluate(self.thick)
        ((_, result),) = ev.constraints
        assert not result.satisfied
        assert result.violation > 0
        assert ev.objective == pytest.approx(ev.merit + 2.5 * result.violation**2)
        assert ev.feasible

    def test_satisfied_soft_constraint_adds_nothing(self) -> None:
        problem = make_problem(constraints=(MaxTotalThickness(maximum=500.0, severity="soft"),))
        ev = problem.evaluate(self.thick)
        assert ev.objective == ev.merit

    def test_hard_penalty(self) -> None:
        problem = make_problem(constraints=(MinLayerThickness(minimum=1.5),), hard_penalty=7.0)
        ev = problem.evaluate(self.thin)
        ((_, result),) = ev.constraints
        assert not result.satisfied
        assert ev.objective == pytest.approx(ev.merit + 7.0 * (1.0 + result.violation))
        assert not ev.feasible

    def test_feasible_beats_infeasible(self) -> None:
        problem = make_problem(constraints=(MinLayerThickness(minimum=1.2),))
        feasible = [problem.evaluate({"d": d, "gamma": g}) for d in (3.5, 4.0) for g in (0.4, 0.5)]
        infeasible = [
            problem.evaluate({"d": d, "gamma": g}) for d in (3.0, 3.5) for g in (0.1, 0.2)
        ]
        assert all(e.feasible for e in feasible)
        assert not any(e.feasible for e in infeasible)
        assert max(e.objective for e in feasible) < min(e.objective for e in infeasible)

    def test_infeasible_ranked_by_violation(self) -> None:
        problem = make_problem(constraints=(MinLayerThickness(minimum=1.2),))
        worse = problem.evaluate({"d": 3.0, "gamma": 0.1})
        mild = problem.evaluate({"d": 3.5, "gamma": 0.3})
        better = problem.evaluate({"d": 3.5, "gamma": 0.5})
        assert better.feasible
        assert not mild.feasible
        assert not worse.feasible
        assert worse.objective > mild.objective > better.objective

    def test_hard_and_soft_combined(self) -> None:
        problem = make_problem(
            constraints=(
                MinLayerThickness(minimum=1.5),
                MaxTotalThickness(maximum=50.0, severity="soft"),
            ),
            hard_penalty=5.0,
            soft_penalty=3.0,
        )
        ev = problem.evaluate(self.thin)
        hard = next(r for c, r in ev.constraints if c.severity == "hard")
        soft = next(r for c, r in ev.constraints if c.severity == "soft")
        expected = ev.merit + 3.0 * soft.violation**2 + 5.0 * (1.0 + hard.violation)
        assert ev.objective == pytest.approx(expected)


class TestUnbuildable:
    def test_gamma_expression_out_of_bounds(self) -> None:
        problem = make_problem(gamma="d * 0.5")
        ev = problem.evaluate({"d": 4.0})
        assert ev.objective == 2.0 * problem.hard_penalty
        assert math.isnan(ev.merit)
        assert ev.stack is None
        assert ev.error
        assert not ev.feasible
        assert ev.results == ()

    def test_expression_inside_bounds_builds(self) -> None:
        problem = make_problem(gamma="d * 0.05")
        ev = problem.evaluate({"d": 4.0})
        assert ev.error is None
        assert ev.values["gamma"] == pytest.approx(0.2)

    def test_period_too_small_for_interlayers(self) -> None:
        coating = Periodic.create(
            W,
            SI,
            N=10,
            d=free(4.0, 0.5, 5.0),
            gamma=0.4,
            interlayers={"interlayer_H_on_L": get("C"), "interlayer_L_on_H": get("C")},
            t_interlayer_H_on_L=1.0,
            t_interlayer_L_on_H=1.0,
        )
        problem = DesignProblem(
            Mirror(coating, SI), make_problem().targets, MeanReflectivity(), hard_penalty=3.0
        )
        assert problem.evaluate().error is None
        ev = problem.evaluate({"d": 0.9})
        assert ev.error
        assert ev.stack is None
        assert ev.objective == 6.0
        assert math.isnan(ev.merit)


class TestDerivedProblems:
    def test_with_parameters_returns_new_problem(self) -> None:
        problem = make_problem()
        params = problem.parameters.replace("d", min=3.8, max=4.2, value=4.0)
        other = problem.with_parameters(params)
        assert other is not problem
        assert other.parameters["d"].min == 3.8
        assert problem.parameters["d"].min == 3.0
        assert other.targets is problem.targets

    def test_evolve_returns_new_problem(self) -> None:
        problem = make_problem()
        other = problem.evolve(hard_penalty=99.0, points=7)
        assert other is not problem
        assert other.hard_penalty == 99.0
        assert other.points == 7
        assert problem.hard_penalty == 10.0
        assert problem.points is None

    def test_bounds_hints_empty_without_constraints(self) -> None:
        assert make_problem().bounds_hints() == {}

    def test_bounds_hints_from_min_layer_thickness(self) -> None:
        problem = make_problem(d=4.0, constraints=(MinLayerThickness(minimum=1.0),))
        lo, hi = problem.bounds_hints()["gamma"]
        assert lo == pytest.approx(0.25)
        assert hi == pytest.approx(0.75)

    def test_bounds_hints_intersect(self) -> None:
        problem = make_problem(
            d=4.0,
            constraints=(MinLayerThickness(minimum=1.0), MinLayerThickness(minimum=1.4)),
        )
        lo, hi = problem.bounds_hints()["gamma"]
        assert lo == pytest.approx(0.35)
        assert hi == pytest.approx(0.65)

    def test_bounds_hints_ignore_soft_constraints(self) -> None:
        problem = make_problem(
            d=4.0,
            constraints=(MinLayerThickness(minimum=1.0, severity="soft"),),
        )
        assert problem.bounds_hints() == {}

    def test_bounds_hints_empty_when_period_varies(self) -> None:
        problem = make_problem(constraints=(MinLayerThickness(minimum=1.0),))
        assert problem.bounds_hints() == {}


class TestResiduals:
    def test_length_is_merit_residuals_plus_two(self) -> None:
        problem = make_problem()
        ev = problem.evaluate()
        assert ev.stack is not None
        from xstack.design.merit.base import EvaluationContext

        ctx = EvaluationContext(ev.stack, problem.targets, ev.results, ev.coating_time)
        n_merit = len(problem.merit.residuals(ctx))
        r = problem.residuals(None, n_merit + 2)
        assert r.shape == (n_merit + 2,)
        assert np.all(r[-2:] == 0.0)

    def test_unbuildable_design_has_fixed_length(self) -> None:
        problem = make_problem(gamma="d * 0.5")
        r = problem.residuals({"d": 4.0}, 7)
        assert r.shape == (7,)
        assert np.sum(r**2) == pytest.approx(2.0 * problem.hard_penalty)

    def test_penalty_entries_squared_equal_penalty_terms(self) -> None:
        problem = make_problem(
            constraints=(
                MinLayerThickness(minimum=1.5),
                MaxTotalThickness(maximum=50.0, severity="soft"),
            ),
            hard_penalty=5.0,
            soft_penalty=3.0,
        )
        values = {"d": 3.5, "gamma": 0.2}
        ev = problem.evaluate(values)
        hard = next(r for c, r in ev.constraints if c.severity == "hard")
        soft = next(r for c, r in ev.constraints if c.severity == "soft")
        n = len(problem.residuals(values, 1))
        r = problem.residuals(values, n)
        assert r[-2] ** 2 == pytest.approx(3.0 * soft.violation**2)
        assert r[-1] ** 2 == pytest.approx(5.0 * (1.0 + hard.violation))
        merit_part = ev.objective - r[-2] ** 2 - r[-1] ** 2
        assert merit_part == pytest.approx(ev.merit)

    def test_sum_of_squares_without_penalties(self) -> None:
        # With a single target point set, sum of squares of the MeanReflectivity residuals is
        # the weighted mean of (1 - R)^2, which bounds the merit (1 - <R>)^2 from above.
        problem = make_problem()
        ev = problem.evaluate()
        n = len(problem.residuals(None, 1))
        r = problem.residuals(None, n)
        assert np.sum(r**2) >= (1.0 - 1e-12) * ev.merit**2


def test_evaluate_values_outside_bounds_is_unbuildable() -> None:
    problem = make_problem()
    name = next(n for n, p in problem.parameters.items() if np.isfinite(p.max))
    evaluation = problem.evaluate({name: problem.parameters[name].max + 1.0})
    assert evaluation.stack is None
    assert evaluation.error is not None
    assert evaluation.objective == pytest.approx(2.0 * problem.hard_penalty)


class TestQuantities:
    def test_mean_reflectivity_of_evaluation(self) -> None:
        problem = make_problem()
        evaluation = problem.evaluate()
        q = problem.quantities(evaluation)
        assert set(q) == {"mean reflectivity"}
        assert q["mean reflectivity"] == pytest.approx(1.0 - evaluation.merit)

    def test_unbuildable_design_gives_empty(self) -> None:
        problem = make_problem()
        name = next(n for n, p in problem.parameters.items() if np.isfinite(p.max))
        evaluation = problem.evaluate({name: problem.parameters[name].max + 1.0})
        assert evaluation.stack is None
        assert problem.quantities(evaluation) == {}
