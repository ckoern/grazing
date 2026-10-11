"""Tests for the staged design pipeline."""

import pytest

from tests.optim.test_problem import ENERGY, THETA, make_problem
from xstack.materials import get
from xstack.merit import MeanReflectivity
from xstack.optim import CancellationToken, DesignProblem, Pipeline
from xstack.optim.optimizers import LmfitOptimizer
from xstack.optim.pipeline import StageResult
from xstack.optim.stages import Stage
from xstack.parameters import free
from xstack.structure import Composite, Mirror, Periodic, SingleLayer
from xstack.targets import EnergyScan, TargetSet

W = get("W")
SI = get("Si")
FAST = {"global_iterations": 2, "population": 4, "coarse_points": 4, "scan_points": 3, "seed": 0}


def integer_problem() -> DesignProblem:
    return make_problem(n=free(15, 5, 40))  # type: ignore[arg-type]


def composite_problem() -> DesignProblem:
    ml = Periodic.create(W, SI, N=15, d=free(3.5, 3.0, 5.0), gamma=free(0.4, 0.1, 0.7), name="ml")
    cap = SingleLayer.create(SI, t=2.0, name="cap")
    targets = TargetSet(
        EnergyScan(angle=THETA, energy_min=ENERGY - 20, energy_max=ENERGY + 20, points=5)
    )
    return DesignProblem(Mirror(Composite.create([cap, ml]), SI), targets, MeanReflectivity())


class TestDefaultStages:
    def test_stage_names(self) -> None:
        pipeline = Pipeline.default(make_problem(), **FAST)
        assert [s.name for s in pipeline.stages] == ["outline", "scan", "global", "refine"]

    def test_scan_free_names(self) -> None:
        pipeline = Pipeline.default(integer_problem(), **FAST)
        scan = pipeline.stages[1]
        assert scan.name == "scan"
        assert scan.free is not None
        assert set(scan.free) == {"gamma", "N"}
        assert scan.points == 4

    def test_scan_free_names_in_composite(self) -> None:
        pipeline = Pipeline.default(composite_problem(), **FAST)
        scan = next(s for s in pipeline.stages if s.name == "scan")
        assert scan.free == ("ml.gamma",)

    def test_scan_skipped_without_gamma_or_integer(self) -> None:
        problem = make_problem(gamma=0.4)
        pipeline = Pipeline.default(problem, **FAST)
        assert [s.name for s in pipeline.stages] == ["outline", "global", "refine"]

    def test_scan_skipped_when_gamma_fixed_and_n_fixed(self) -> None:
        problem = make_problem(n=20, gamma=0.5)
        names = [s.name for s in Pipeline.default(problem, **FAST).stages]
        assert "scan" not in names

    def test_global_budget_scales_with_free_parameters(self) -> None:
        pipeline = Pipeline.default(make_problem(), **FAST)
        global_stage = next(s for s in pipeline.stages if s.name == "global")
        assert isinstance(global_stage.optimizer, LmfitOptimizer)
        assert global_stage.optimizer.max_evaluations == 3 * 4 * 2
        assert ("seed", 0) in global_stage.optimizer.options


class TestRun:
    def test_on_stage_called_in_order(self) -> None:
        problem = make_problem()
        pipeline = Pipeline.default(problem, **FAST)
        seen: list[str] = []
        result = pipeline.run(problem, on_stage=lambda s: seen.append(s.stage.name))
        assert seen == ["outline", "scan", "global", "refine"]
        assert [s.stage.name for s in result.stages] == seen
        assert not result.cancelled

    def test_parameters_handed_over(self) -> None:
        problem = make_problem()
        first = Stage("a", LmfitOptimizer("leastsq", 20), free=("d",))
        second = Stage("b", LmfitOptimizer("leastsq", 20), free=("gamma",))
        result = Pipeline((first, second)).run(problem)
        a, b = (s.result for s in result.stages)
        assert b.parameters["d"].value == a.parameters["d"].value
        assert a.parameters["gamma"].value == problem.parameters["gamma"].value
        original = problem.parameters
        for name in original:
            assert result.parameters[name].value == pytest.approx(b.parameters[name].value)
            assert result.parameters[name].min == original[name].min
            assert result.parameters[name].max == original[name].max
            assert result.parameters[name].vary == original[name].vary

    def test_start_of_next_stage_is_previous_result(self) -> None:
        problem = make_problem()
        starts: list[float] = []

        class Probe(LmfitOptimizer):
            def run(self, problem, start=None, events=None, cancel=None):
                assert start is not None
                starts.append(start["d"].value)
                return super().run(problem, start, events, cancel)

        stages = (
            Stage("a", Probe("leastsq", 15), free=("d",)),
            Stage("b", Probe("leastsq", 15), free=("d",)),
        )
        result = Pipeline(stages).run(problem)
        assert starts[0] == problem.parameters["d"].value
        assert starts[1] == result.stages[0].result.parameters["d"].value

    def test_final_uses_original_problem(self) -> None:
        problem = make_problem(points=9)
        stage = Stage("coarse", LmfitOptimizer("leastsq", 10), points=3)
        result = Pipeline((stage,)).run(problem)
        assert len(result.final.results[0].reflectivity) == 9
        expected = problem.with_parameters(result.parameters).evaluate()
        assert result.final.objective == pytest.approx(expected.objective)
        assert len(result.stages[0].result.best.results[0].reflectivity) == 3

    def test_final_bounds_are_those_of_start(self) -> None:
        problem = integer_problem()
        result = Pipeline.default(problem, **FAST).run(problem)
        narrowed = result.stages[0].result.parameters
        assert (narrowed["N"].min, narrowed["N"].max) != (
            problem.parameters["N"].min,
            problem.parameters["N"].max,
        )
        for name, p in problem.parameters.items():
            assert (result.parameters[name].min, result.parameters[name].max) == (p.min, p.max)

    def test_coarse_points_capped_at_target_points(self) -> None:
        pipeline = Pipeline.default(make_problem(points=5), **{**FAST, "coarse_points": 50})
        assert pipeline.stages[-2].points == 5

    def test_empty_pipeline(self) -> None:
        problem = make_problem()
        result = Pipeline().run(problem)
        assert result.stages == ()
        assert result.parameters == problem.parameters
        assert result.final.objective == pytest.approx(problem.evaluate().objective)

    def test_start_parameters(self) -> None:
        problem = make_problem()
        start = problem.parameters.replace("d", value=4.5)
        result = Pipeline((Stage("a", LmfitOptimizer("leastsq", 1), free=("gamma",)),)).run(
            problem, start
        )
        assert result.parameters["d"].value == 4.5

    def test_cancel_after_first_stage(self) -> None:
        problem = make_problem()
        token = CancellationToken()
        called: list[str] = []

        def on_stage(res: StageResult) -> None:
            called.append(res.stage.name)
            token.cancel()

        result = Pipeline.default(problem, **FAST).run(problem, cancel=token, on_stage=on_stage)
        assert called == ["outline"]
        assert len(result.stages) == 1
        assert result.cancelled

    def test_events_carry_stage_names(self) -> None:
        problem = make_problem()
        tasks: set[str] = set()
        Pipeline.default(problem, **FAST).run(problem, events=lambda e: tasks.add(e.task))
        prefixes = {t.split(":")[0] for t in tasks}
        assert prefixes == {"outline", "scan", "global", "refine"}

    def test_end_to_end_improves_merit(self) -> None:
        problem = make_problem(n=20, d=free(3.0, 2.5, 6.0), gamma=free(0.6, 0.1, 0.8))
        start = problem.evaluate()
        result = Pipeline.default(problem, **FAST).run(problem)
        assert result.final.merit < start.merit
        assert result.final.objective <= start.objective
        assert result.final.feasible
        for name in problem.parameters:
            assert result.parameters[name].vary == problem.parameters[name].vary
