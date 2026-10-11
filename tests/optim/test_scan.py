"""Tests for design-space scans and the grid-scan optimizer."""

import math

import numpy as np
import pytest

from tests.optim.test_problem import SI, W, make_problem
from xstack.manufacturing.constraints import MinLayerThickness
from xstack.merit import MeanReflectivity
from xstack.optim import CancellationToken, DesignProblem, GridScan, ProgressEvent, scan
from xstack.parameters import free
from xstack.structure import Mirror, Periodic


class TestScan:
    def test_grid_shape_and_axes(self) -> None:
        problem = make_problem()
        result = scan(problem, {"d": [4.0, 3.8, 3.9, 3.8], "gamma": np.linspace(0.2, 0.3, 3)})
        assert result.names == ("d", "gamma")
        np.testing.assert_allclose(result.axes[0], [3.8, 3.9, 4.0])
        np.testing.assert_allclose(result.axes[1], [0.2, 0.25, 0.3])
        assert result.objective.shape == (3, 3)
        assert result.merit.shape == (3, 3)
        assert result.feasible.shape == (3, 3)
        assert result.feasible.dtype == bool
        assert result.feasible.all()
        assert not np.isnan(result.objective).any()

    def test_values_match_direct_evaluation(self) -> None:
        problem = make_problem()
        result = scan(problem, {"d": [3.8, 4.0], "gamma": [0.2, 0.3]})
        for i, d in enumerate((3.8, 4.0)):
            for j, g in enumerate((0.2, 0.3)):
                ev = problem.evaluate({"d": d, "gamma": g})
                assert result.objective[i, j] == pytest.approx(ev.objective)
                assert result.merit[i, j] == pytest.approx(ev.merit)

    def test_best_values_match_argmin(self) -> None:
        result = scan(make_problem(), {"d": np.linspace(3.8, 4.1, 4), "gamma": [0.2, 0.25, 0.3]})
        i, j = np.unravel_index(np.argmin(result.objective), result.objective.shape)
        best = result.best_values()
        assert best == {"d": float(result.axes[0][i]), "gamma": float(result.axes[1][j])}
        opt = result.optimization
        assert opt.best.objective == pytest.approx(result.objective.min())
        assert opt.evaluations == 12
        assert opt.method == "scan"
        assert not opt.cancelled

    def test_unlisted_parameters_keep_values(self) -> None:
        problem = make_problem()
        result = scan(problem, {"d": [3.9, 4.0]})
        assert result.objective.shape == (2,)
        assert result.optimization.best.values["gamma"] == 0.4

    def test_integer_axis_rounded_and_deduplicated(self) -> None:
        coating = Periodic.create(W, SI, N=free(20, 5, 40), d=4.0, gamma=0.3)
        problem = DesignProblem(Mirror(coating, SI), make_problem().targets, MeanReflectivity())
        result = scan(problem, {"N": [10.2, 10.4, 15.6, 16.4, 20]})
        np.testing.assert_array_equal(result.axes[0], [10.0, 16.0, 20.0])
        assert result.objective.shape == (3,)
        # More bilayers reflect more.
        assert result.merit[0] > result.merit[-1]

    def test_values_outside_bounds_are_evaluated(self) -> None:
        problem = make_problem()
        result = scan(problem, {"d": [2.0, 4.0, 6.0]})
        assert result.objective.shape == (3,)
        assert not np.isnan(result.objective).any()
        assert result.optimization.parameters["d"].min <= 2.0
        assert result.optimization.parameters["d"].max >= 6.0
        # The problem itself is unchanged.
        assert problem.parameters["d"].max == 5.0

    def test_empty_axes_mapping(self) -> None:
        with pytest.raises(ValueError, match="axis"):
            scan(make_problem(), {})

    def test_empty_axis(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            scan(make_problem(), {"d": []})

    def test_unknown_parameter(self) -> None:
        with pytest.raises(KeyError):
            scan(make_problem(), {"nope": [1.0]})

    def test_unbuildable_points_give_nan_merit(self) -> None:
        problem = make_problem(gamma="d * 0.5", hard_penalty=4.0)
        result = scan(problem, {"d": [3.0, 5.0]})
        assert np.isnan(result.merit).all()
        np.testing.assert_allclose(result.objective, 8.0)
        assert not result.feasible.any()

    def test_cancellation_leaves_nan(self) -> None:
        token = CancellationToken()

        def on_event(event: ProgressEvent) -> None:
            if event.evaluations >= 3:
                token.cancel()

        result = scan(
            make_problem(),
            {"d": np.linspace(3.8, 4.1, 4), "gamma": [0.2, 0.25, 0.3]},
            events=on_event,
            cancel=token,
        )
        assert result.optimization.cancelled
        assert int(np.isnan(result.objective).sum()) == 12 - 3
        assert np.isnan(result.merit).sum() == 9
        assert result.feasible.sum() == 3
        assert math.isfinite(result.objective.flat[0])

    def test_feasibility_mask(self) -> None:
        problem = make_problem(constraints=(MinLayerThickness(minimum=1.2),))
        gammas = [0.1, 0.3, 0.5]
        result = scan(problem, {"gamma": gammas}, None)
        # Layers thinner than 1.2 nm for gamma = 0.1 and 0.3 at d = 3.5.
        np.testing.assert_array_equal(result.feasible, [False, False, True])
        assert result.objective[2] == pytest.approx(result.merit[2])
        assert result.objective[0] > result.merit[0]
        assert result.best_values() == {"gamma": 0.5}

    def test_cancelled_before_start(self) -> None:
        token = CancellationToken()
        token.cancel()
        result = scan(make_problem(), {"d": [3.8, 3.9]}, cancel=token)
        assert np.isnan(result.objective).all()

    def test_events_reported(self) -> None:
        events: list[ProgressEvent] = []
        scan(make_problem(), {"d": [3.8, 3.9, 4.0]}, events=events.append)
        assert events
        assert events[-1].total == 3
        assert all(e.task == "scan" for e in events)


class TestGridScan:
    def test_axes_use_linspace_within_bounds(self) -> None:
        problem = make_problem()
        axes = GridScan(points=5).axes(problem.parameters)
        assert set(axes) == {"d", "gamma"}
        np.testing.assert_allclose(axes["d"], np.linspace(3.0, 5.0, 5))
        np.testing.assert_allclose(axes["gamma"], np.linspace(0.1, 0.7, 5))

    def test_fixed_parameters_not_scanned(self) -> None:
        problem = make_problem(d=4.0)
        assert set(GridScan(points=4).axes(problem.parameters)) == {"gamma"}

    def test_integer_parameters_enumerated(self) -> None:
        coating = Periodic.create(W, SI, N=free(20, 18, 22), d=4.0, gamma=0.3)
        problem = DesignProblem(Mirror(coating, SI), make_problem().targets, MeanReflectivity())
        axes = GridScan(points=11).axes(problem.parameters)
        np.testing.assert_array_equal(axes["N"], [18, 19, 20, 21, 22])

    def test_integer_parameters_sampled_when_range_large(self) -> None:
        coating = Periodic.create(W, SI, N=free(20, 1, 100), d=4.0, gamma=0.3)
        problem = DesignProblem(Mirror(coating, SI), make_problem().targets, MeanReflectivity())
        axes = GridScan(points=5).axes(problem.parameters)
        np.testing.assert_allclose(axes["N"], np.linspace(1, 100, 5))

    def test_explicit_values_override(self) -> None:
        problem = make_problem()
        grid = GridScan(points=5, values=(("d", (3.9, 4.0)),))
        axes = grid.axes(problem.parameters)
        np.testing.assert_allclose(axes["d"], [3.9, 4.0])
        assert len(axes["gamma"]) == 5

    def test_explicit_values_for_fixed_parameter(self) -> None:
        problem = make_problem(d=4.0)
        axes = GridScan(points=3, values=(("d", (3.9, 4.0)),)).axes(problem.parameters)
        assert set(axes) == {"gamma", "d"}

    def test_infinite_bounds(self) -> None:
        problem = make_problem()
        params = problem.parameters.replace("d", min=-math.inf, max=math.inf)
        with pytest.raises(ValueError, match="finite bounds"):
            GridScan().axes(params)

    def test_infinite_bounds_with_explicit_values_ok(self) -> None:
        problem = make_problem()
        params = problem.parameters.replace("d", min=-math.inf, max=math.inf)
        axes = GridScan(values=(("d", (4.0,)),), points=3).axes(params)
        np.testing.assert_allclose(axes["d"], [4.0])

    def test_run_returns_grid_optimum(self) -> None:
        problem = make_problem()
        grid = GridScan(points=4, values=(("d", (3.9, 3.96, 4.02)), ("gamma", (0.2, 0.25, 0.3))))
        result = grid.run(problem)
        reference = scan(problem, {"d": (3.9, 3.96, 4.02), "gamma": (0.2, 0.25, 0.3)})
        best = reference.best_values()
        assert result.evaluations == 9
        assert result.parameters["d"].value == pytest.approx(best["d"])
        assert result.parameters["gamma"].value == pytest.approx(best["gamma"])
        assert result.best.objective == pytest.approx(reference.objective.min())
        assert result.best.objective <= problem.evaluate().objective

    def test_run_with_constraint_hints(self) -> None:
        problem = make_problem(
            d=4.0,
            gamma=free(0.5, 0.0, 1.0),
            constraints=(MinLayerThickness(minimum=1.0),),
        )
        result = GridScan(points=5).run(problem)
        assert result.best.feasible
        assert result.parameters["gamma"].value >= 0.25 - 1e-12

    def test_run_cancelled_before_start(self) -> None:
        token = CancellationToken()
        token.cancel()
        result = GridScan(points=3).run(make_problem(), cancel=token)
        assert result.cancelled
