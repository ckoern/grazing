"""Tests for the coating-time trade-off."""

import math

import pytest

from grazing.core.optim import CancellationToken, ProgressEvent
from grazing.core.parameters import ParameterSet
from grazing.core.serialization import from_data, to_data
from grazing.design.manufacturing.constraints import MaxCoatingTime
from grazing.design.manufacturing.process import DepositionProcess
from grazing.design.optim import DesignProblem, Pipeline
from grazing.design.optim.tradeoff import (
    TradeoffPoint,
    TradeoffResult,
    default_limits,
    time_tradeoff,
)
from grazing.design.project import Project
from grazing.design.project.commands import store_tradeoff
from tests.design.optim.test_stages import periodic_problem

PROCESS = DepositionProcess({"W": 0.05, "Si": 0.1}, overhead_per_layer=20.0)
FAST = {"global_iterations": 2, "population": 4, "coarse_points": 4, "scan_points": 3, "seed": 0}


def problem() -> DesignProblem:
    return periodic_problem().evolve(process=PROCESS)


def pipeline() -> Pipeline:
    return Pipeline.default(problem(), **FAST)


def point(limit: float, time: float, merit: float, feasible: bool = True) -> TradeoffPoint:
    objective = merit if feasible else merit + 10.0
    return TradeoffPoint(limit, ParameterSet(), objective, merit, time, feasible)


class TestDefaultLimits:
    def test_log_spaced_up_to_reference(self) -> None:
        limits = default_limits(1000.0, points=3, lowest=0.25)
        assert limits == pytest.approx((250.0, 500.0, 1000.0))

    def test_single_point(self) -> None:
        assert default_limits(100.0, points=1) == (100.0,)

    @pytest.mark.parametrize(
        ("reference", "points", "lowest"),
        [(0.0, 3, 0.2), (math.inf, 3, 0.2), (100.0, 0, 0.2), (100.0, 3, 0.0), (100.0, 3, 1.5)],
    )
    def test_invalid(self, reference: float, points: int, lowest: float) -> None:
        with pytest.raises(ValueError):
            default_limits(reference, points, lowest)


class TestResult:
    def test_marginal_gains_per_hour(self) -> None:
        result = TradeoffResult(
            (point(3600, 3600, 0.5), point(7200, 7200, 0.3), point(math.inf, 10800, 0.25))
        )
        assert result.marginal_gains() == pytest.approx((0.2, 0.05))

    def test_infeasible_points_are_skipped(self) -> None:
        result = TradeoffResult((point(1800, 3000, 0.9, feasible=False), point(7200, 7200, 0.3)))
        assert result.feasible_points() == (result.points[1],)
        assert result.marginal_gains() == ()

    def test_knee(self) -> None:
        points = (point(1, 1, 1.0), point(2, 2, 0.2), point(3, 3, 0.15), point(4, 4, 0.1))
        assert TradeoffResult(points).knee() == points[1]

    def test_no_knee_for_straight_line_or_few_points(self) -> None:
        line = (point(1, 1, 0.3), point(2, 2, 0.2), point(3, 3, 0.1))
        assert TradeoffResult(line).knee() is None
        assert TradeoffResult(line[:2]).knee() is None


class TestValidation:
    def test_requires_process(self) -> None:
        with pytest.raises(ValueError, match="deposition process"):
            time_tradeoff(periodic_problem(), [3600.0])

    def test_positive_limits(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            time_tradeoff(problem(), [0.0])

    def test_mode(self) -> None:
        with pytest.raises(ValueError, match="mode"):
            time_tradeoff(problem(), [3600.0], mode="fast")  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def sweep() -> TradeoffResult:
    return time_tradeoff(problem(), points=3, lowest=0.3, pipeline=pipeline())


class TestSweep:
    def test_unconstrained_point_and_default_limits(self, sweep: TradeoffResult) -> None:
        limits = [p.limit for p in sweep.points]
        assert len(limits) == 4
        assert limits == sorted(limits)
        assert math.isinf(limits[-1])
        free_time = sweep.points[-1].coating_time
        assert limits[0] == pytest.approx(0.3 * free_time, rel=1e-6)
        assert limits[-2] < free_time

    def test_points_respect_their_limit(self, sweep: TradeoffResult) -> None:
        for p in sweep.points:
            assert p.feasible
            assert p.coating_time <= p.limit * (1 + 1e-9)

    def test_front_is_monotone(self, sweep: TradeoffResult) -> None:
        merits = [p.merit for p in sweep.points]
        assert merits == sorted(merits, reverse=True)
        assert all(g >= 0 for g in sweep.marginal_gains())

    def test_fewer_bilayers_for_shorter_limits(self, sweep: TradeoffResult) -> None:
        n = [p.parameters["N"].value for p in sweep.points]
        assert n[0] < n[-1]

    def test_parameters_keep_start_settings(self, sweep: TradeoffResult) -> None:
        start = problem().parameters
        for p in sweep.points:
            for name, param in start.items():
                assert (p.parameters[name].min, p.parameters[name].max) == (param.min, param.max)
                assert p.parameters[name].vary == param.vary

    def test_quantities(self, sweep: TradeoffResult) -> None:
        assert all(p.quantities for p in sweep.points)


class TestRun:
    def test_explicit_limits_without_unconstrained_point(self) -> None:
        result = time_tradeoff(problem(), [3000.0, 1500.0], pipeline=pipeline())
        assert [p.limit for p in result.points] == [1500.0, 3000.0]

    def test_existing_time_limit_is_replaced(self) -> None:
        tight = problem().evolve(constraints=(MaxCoatingTime(60.0),))
        result = time_tradeoff(tight, [3000.0], pipeline=pipeline())
        assert result.points[0].feasible

    def test_thorough_mode(self) -> None:
        result = time_tradeoff(problem(), [3000.0, 1500.0], mode="thorough", pipeline=pipeline())
        assert result.mode == "thorough"
        assert all(p.feasible for p in result.points)

    def test_events_are_prefixed_with_the_limit(self) -> None:
        tasks: set[str] = set()

        def sink(event: ProgressEvent) -> None:
            tasks.add(event.task.split(":")[0])

        time_tradeoff(problem(), [3600.0, 1800.0], pipeline=pipeline(), events=sink)
        assert tasks == {"1 h", "0.5 h"}

    def test_cancel_after_first_point(self) -> None:
        token = CancellationToken()
        seen: list[TradeoffPoint] = []

        def on_point(p: TradeoffPoint) -> None:
            seen.append(p)
            token.cancel()

        result = time_tradeoff(
            problem(),
            [3000.0, 2000.0, 1000.0],
            pipeline=pipeline(),
            cancel=token,
            on_point=on_point,
        )
        assert result.cancelled
        assert [p.limit for p in result.points] == [3000.0]
        assert seen == list(result.points)


class TestDominance:
    def test_better_design_from_shorter_limit_replaces_point(self) -> None:
        from grazing.design.optim.tradeoff import _dominance_filtered

        short = point(1000, 900, 0.2)
        long = point(2000, 1800, 0.3)
        infeasible = point(3000, 3500, 0.1, feasible=False)
        result = _dominance_filtered([short, long, infeasible])
        assert result[0] == short
        assert result[1].replaced
        assert (result[1].limit, result[1].merit) == (2000, 0.2)
        assert result[2].replaced
        assert result[2].limit == 3000


class TestPersistence:
    def test_round_trip(self, sweep: TradeoffResult) -> None:
        restored = from_data(to_data(sweep))
        assert isinstance(restored, TradeoffResult)
        assert len(restored.points) == len(sweep.points)
        assert math.isinf(restored.points[-1].limit)
        for a, b in zip(restored.points, sweep.points, strict=True):
            assert a.parameters.value_dict() == b.parameters.value_dict()
            assert a.quantities == b.quantities

    def test_store_in_project(self, sweep: TradeoffResult, tmp_path: object) -> None:
        from pathlib import Path

        project = Project()
        project.execute(store_tradeoff(sweep))
        path = Path(str(tmp_path)) / "tradeoff.grazing"
        project.save(path)
        loaded = Project.load(path).state.tradeoff
        assert loaded is not None
        assert [p.merit for p in loaded.points] == [p.merit for p in sweep.points]
        project.undo()
        assert project.state.tradeoff is None
