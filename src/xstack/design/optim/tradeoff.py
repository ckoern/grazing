"""Trade-off between the figure of merit and the coating time.

`time_tradeoff` optimizes a design problem for a sequence of maximum coating times (an
ε-constraint sweep). Every point is the best design found within its time limit, so the points
approximate the Pareto front of figure of merit versus coating time.
"""

import itertools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np

from xstack.core.parameters import ParameterSet
from xstack.core.serialization import register_serializable
from xstack.design.manufacturing.constraints import MaxCoatingTime
from xstack.design.optim.events import CancellationToken, EventSink, ProgressEvent
from xstack.design.optim.optimizers import LmfitOptimizer, Optimizer
from xstack.design.optim.pipeline import Pipeline
from xstack.design.optim.problem import DesignProblem, Evaluation
from xstack.design.optim.scan import GridScan
from xstack.design.optim.stages import Stage

__all__ = [
    "TradeoffMode",
    "TradeoffPoint",
    "TradeoffResult",
    "default_limits",
    "time_tradeoff",
]

TradeoffMode = Literal["continuation", "thorough"]

_INTEGER_GRID_SIZE = 200
"""Maximum number of grid points of the integer scan in a continuation step."""

_KNEE_DISTANCE = 0.01
"""Minimum scaled distance of a knee from the line between the ends of the front."""

_REFINE_EVALUATIONS = 500
"""Evaluation budget of the default local optimizer of a continuation step."""


@dataclass(frozen=True)
class TradeoffPoint:
    """The best design found for one maximum coating time.

    Attributes:
        limit: Maximum coating time in s the design was optimized for; ``inf`` for the design
            optimized without a time limit.
        parameters: The design parameters, with the bounds and free/fixed settings of the start
            parameters.
        objective: Objective of the design under its time limit (figure of merit plus
            constraint penalties).
        merit: Figure of merit of the design, ``nan`` if it could not be built.
        coating_time: Coating time of the design in s, ``nan`` if it could not be built.
        feasible: Whether the design satisfies all hard constraints, including the time limit.
        quantities: Quantities rated by the figure of merit as (label, value) pairs.
        replaced: Whether the design was taken over from a smaller time limit because it was
            better than the design optimized for this limit.
    """

    limit: float
    parameters: ParameterSet
    objective: float
    merit: float
    coating_time: float
    feasible: bool
    quantities: tuple[tuple[str, float], ...] = ()
    replaced: bool = False


@dataclass(frozen=True)
class TradeoffResult:
    """Result of a coating-time trade-off.

    Attributes:
        points: The points in order of increasing time limit.
        mode: How the points were optimized (see `time_tradeoff`).
        cancelled: Whether the sweep was cancelled before all limits were done.
    """

    points: tuple[TradeoffPoint, ...] = field(default=())
    mode: TradeoffMode = "continuation"
    cancelled: bool = False

    def feasible_points(self) -> tuple[TradeoffPoint, ...]:
        """Return the feasible points in order of increasing coating time.

        Returns:
            The points.
        """
        feasible = [p for p in self.points if p.feasible and math.isfinite(p.coating_time)]
        return tuple(sorted(feasible, key=lambda p: p.coating_time))

    def marginal_gains(self) -> tuple[float, ...]:
        """Return the improvement of the figure of merit per additional hour of coating time.

        The gain between neighbouring feasible points (see `feasible_points`) is the decrease of
        the figure of merit divided by the increase of the coating time in h. Positive values
        mean that the longer coating is better.

        Returns:
            One value per pair of neighbouring feasible points (figure of merit per h); ``nan``
            where the coating times are equal.
        """
        points = self.feasible_points()
        gains = []
        for shorter, longer in itertools.pairwise(points):
            hours = (longer.coating_time - shorter.coating_time) / 3600.0
            gains.append((shorter.merit - longer.merit) / hours if hours > 0 else math.nan)
        return tuple(gains)

    def knee(self) -> TradeoffPoint | None:
        """Return the point beyond which more coating time improves the design least.

        The knee is the feasible point farthest from the straight line between the shortest
        and the longest feasible design, with time and figure of merit scaled to [0, 1].

        Returns:
            The knee point, or ``None`` with fewer than three feasible points or if no point is
            farther than 0.01 from the line (a flat or straight front).
        """
        points = self.feasible_points()
        if len(points) < 3:
            return None
        t = np.array([p.coating_time for p in points])
        f = np.array([p.merit for p in points])
        if np.ptp(t) == 0 or np.ptp(f) == 0:
            return None
        x = (t - t[0]) / (t[-1] - t[0])
        y = (f - f.min()) / np.ptp(f)
        dx, dy = x[-1] - x[0], y[-1] - y[0]
        distance = np.abs(dy * (x - x[0]) - dx * (y - y[0])) / math.hypot(dx, dy)
        best = int(np.argmax(distance))
        if distance[best] < _KNEE_DISTANCE or not 0 < best < len(points) - 1:
            return None
        return points[best]


def default_limits(
    reference_time: float, points: int = 8, lowest: float = 0.2
) -> tuple[float, ...]:
    """Return time limits spaced evenly on a log scale below a reference time.

    Args:
        reference_time: The longest limit in s, e.g. the coating time of the design optimized
            without a time limit.
        points: Number of limits, at least 1.
        lowest: Shortest limit as a fraction of the reference time, in (0, 1].

    Returns:
        The limits in s in increasing order, ending with `reference_time`.

    Raises:
        ValueError: If an argument is out of range.

    Examples:
        >>> [round(t) for t in default_limits(3600.0, points=3, lowest=0.25)]
        [900, 1800, 3600]
    """
    if not reference_time > 0 or not math.isfinite(reference_time):
        raise ValueError("the reference time must be positive and finite")
    if points < 1:
        raise ValueError("at least one time limit is needed")
    if not 0 < lowest <= 1:
        raise ValueError("the shortest limit must be a fraction in (0, 1]")
    if points == 1:
        return (reference_time,)
    return tuple(float(t) for t in reference_time * np.geomspace(lowest, 1.0, points))


def time_tradeoff(
    problem: DesignProblem,
    limits: Sequence[float] | None = None,
    *,
    points: int = 8,
    lowest: float = 0.2,
    mode: TradeoffMode = "continuation",
    pipeline: Pipeline | None = None,
    refine: Optimizer | None = None,
    start: ParameterSet | None = None,
    events: EventSink | None = None,
    cancel: CancellationToken | None = None,
    on_point: Callable[[TradeoffPoint], None] | None = None,
) -> TradeoffResult:
    """Optimize a design problem for a sequence of maximum coating times.

    The limits are processed from the longest to the shortest. Every limit replaces the
    `MaxCoatingTime` constraints of the problem by a hard constraint with that maximum.
    Without explicit limits, the problem is first optimized without a time limit (a point with
    limit ``inf``), followed by `points` limits spaced evenly on a log scale from `lowest` times
    the coating time of that design up to just below it.

    In ``"continuation"`` mode, the longest limit is optimized with the full pipeline; every
    further limit starts from the design of the previous one, scans the free integer parameters
    (e.g. the number of bilayers) on a grid and refines locally. If this ends infeasible, the
    full pipeline is run for that limit. In ``"thorough"`` mode, every limit is optimized with
    the full pipeline, starting from the design of the previous limit.

    Finally, a design found for a shorter limit also satisfies every longer one; if it is
    better than the design found for a longer limit, it replaces that design (marked
    `TradeoffPoint.replaced`).

    Args:
        problem: The design problem; it needs a deposition process.
        limits: Maximum coating times in s, or ``None`` for the default limits.
        points: Number of default limits below the coating time of the unconstrained design.
        lowest: Shortest default limit as a fraction of the coating time of the unconstrained
            design.
        mode: ``"continuation"`` or ``"thorough"``.
        pipeline: Pipeline for full optimizations; defaults to `Pipeline.default` of each
            problem.
        refine: Local optimizer of the continuation steps; defaults to Nelder–Mead, which
            copes better than gradient-based methods with the jump of the hard time penalty
            at the limit.
        start: Start parameters. Defaults to the problem's parameters.
        events: Receiver of progress events; their task names are prefixed with the limit.
        cancel: Cancellation token; the current optimization stops and no further limit is
            processed.
        on_point: Called with every point when its optimization has finished.

    Returns:
        The result with one point per processed limit (and the unconstrained design without
        explicit limits).

    Raises:
        ValueError: If the problem has no deposition process, a limit is not positive, or a
            mode is unknown.
    """
    if problem.process is None:
        raise ValueError("a coating-time trade-off requires a deposition process")
    if mode not in ("continuation", "thorough"):
        raise ValueError(f"unknown trade-off mode {mode!r}")
    if limits is not None and any(not t > 0 for t in limits):
        raise ValueError("time limits must be positive")
    initial = start if start is not None else problem.parameters
    base = problem.evolve(
        constraints=tuple(c for c in problem.constraints if not isinstance(c, MaxCoatingTime))
    )
    sweep = _Sweep(base, initial, mode, pipeline, refine, events, cancel, on_point)
    params = initial
    if limits is None:
        free = sweep.run(math.inf, params, full=True)
        params = free.parameters
        if math.isfinite(free.coating_time) and free.coating_time > 0:
            todo = list(default_limits(free.coating_time, points + 1, lowest)[:-1])
        else:
            todo = []
    else:
        todo = sorted(set(limits))
    first = limits is not None
    while todo and not sweep.cancelled:
        point = sweep.run(todo.pop(), params, full=first or mode == "thorough")
        params = point.parameters
        first = False
    ordered = sorted(sweep.points, key=lambda p: p.limit)
    return TradeoffResult(_dominance_filtered(ordered), mode, sweep.cancelled)


class _Sweep:
    """Runs the optimizations of a trade-off and collects the points."""

    def __init__(
        self,
        base: DesignProblem,
        initial: ParameterSet,
        mode: TradeoffMode,
        pipeline: Pipeline | None,
        refine: Optimizer | None,
        events: EventSink | None,
        cancel: CancellationToken | None,
        on_point: Callable[[TradeoffPoint], None] | None,
    ) -> None:
        self.base = base
        self.initial = initial
        self.mode = mode
        self.pipeline = pipeline
        self.refine = refine or LmfitOptimizer("nelder", _REFINE_EVALUATIONS)
        self.events = events
        self.cancel = cancel
        self.on_point = on_point
        self.points: list[TradeoffPoint] = []

    @property
    def cancelled(self) -> bool:
        return self.cancel is not None and self.cancel.cancelled

    def problem(self, limit: float) -> DesignProblem:
        if math.isinf(limit):
            return self.base
        return self.base.evolve(constraints=(*self.base.constraints, MaxCoatingTime(limit)))

    def run(self, limit: float, start: ParameterSet, full: bool) -> TradeoffPoint:
        problem = self.problem(limit)
        label = "no limit" if math.isinf(limit) else f"{limit / 3600:.3g} h"

        def forward(event: ProgressEvent) -> None:
            if self.events is not None:
                self.events(replace(event, task=f"{label}: {event.task}"))

        found = self._continue(problem, start, forward) if not full else None
        if found is None or not found[1].feasible:
            pipeline = self.pipeline or Pipeline.default(problem)
            result = pipeline.run(problem, start, forward, self.cancel)
            found = _best(problem, (start, result.parameters))
        params, evaluation = found
        point = TradeoffPoint(
            limit,
            self.initial.with_values(
                {n: v for n, v in params.value_dict().items() if self.initial[n].expr is None}
            ),
            evaluation.objective,
            evaluation.merit,
            evaluation.coating_time if evaluation.coating_time is not None else math.nan,
            evaluation.feasible,
            tuple(problem.quantities(evaluation).items()),
        )
        self.points.append(point)
        if self.on_point is not None:
            self.on_point(point)
        return point

    def _continue(
        self, problem: DesignProblem, start: ParameterSet, forward: EventSink
    ) -> tuple[ParameterSet, Evaluation] | None:
        """Scan the integer parameters and refine locally, starting from a design."""
        stages = []
        grids = _integer_grids(start)
        if grids:
            stages.append(Stage("scan", GridScan(values=grids), free=tuple(n for n, _ in grids)))
        stages.append(Stage("refine", self.refine))
        try:
            result = Pipeline(tuple(stages)).run(problem, start, forward, self.cancel)
        except ValueError:
            return None
        return _best(problem, (start, result.parameters))


def _integer_grids(params: ParameterSet) -> tuple[tuple[str, tuple[float, ...]], ...]:
    """Grids over the free integer parameters with at most `_INTEGER_GRID_SIZE` points."""
    names = [n for n, p in params.items() if p.is_free and p.integer]
    finite = [n for n in names if math.isfinite(params[n].min) and math.isfinite(params[n].max)]
    if not finite:
        return ()
    per_axis = max(2, int(_INTEGER_GRID_SIZE ** (1 / len(finite))))
    grids = []
    for name in finite:
        p = params[name]
        lo, hi = math.ceil(p.min), math.floor(p.max)
        count = min(per_axis, hi - lo + 1)
        values = np.unique(np.round(np.linspace(lo, hi, count)))
        grids.append((name, tuple(float(v) for v in values)))
    return tuple(grids)


def _best(
    problem: DesignProblem, candidates: Sequence[ParameterSet]
) -> tuple[ParameterSet, Evaluation]:
    """Return the candidate with the lowest objective and its evaluation."""
    evaluated = [(p, problem.with_parameters(p).evaluate()) for p in candidates]
    return min(evaluated, key=lambda pe: pe[1].objective)


def _dominance_filtered(points: Sequence[TradeoffPoint]) -> tuple[TradeoffPoint, ...]:
    """Replace a point by a better feasible design found for a shorter limit."""
    result: list[TradeoffPoint] = []
    best: TradeoffPoint | None = None
    for point in points:
        if best is not None and best.objective < point.objective:
            point = replace(best, limit=point.limit, replaced=True)
        elif point.feasible:
            best = point
        result.append(point)
    return tuple(result)


register_serializable("TradeoffPoint", TradeoffPoint)
register_serializable("TradeoffResult", TradeoffResult)
