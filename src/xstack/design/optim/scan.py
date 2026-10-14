"""Design-space scans on regular parameter grids."""

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

from xstack.core.parameters import ParameterSet
from xstack.design.optim.events import CancellationToken, EventSink
from xstack.design.optim.optimizers import (
    OptimizationResult,
    Optimizer,
    Tracker,
    prepare_parameters,
    register_optimizer,
)
from xstack.design.optim.problem import DesignProblem

__all__ = ["GridScan", "ScanResult", "scan"]


@dataclass(frozen=True)
class ScanResult:
    """Objective values on a regular parameter grid.

    Attributes:
        names: Names of the scanned parameters, one per grid axis.
        axes: Grid values of each scanned parameter.
        objective: Objective values, shape (len(axes[0]), len(axes[1]), …); ``nan`` where not
            evaluated (cancelled scans).
        merit: Figure of merit without penalties, same shape; ``nan`` for designs that could not
            be built.
        feasible: Whether each design satisfies all hard constraints, same shape.
        optimization: Best design found, as an optimization result.
    """

    names: tuple[str, ...]
    axes: tuple[NDArray[np.float64], ...]
    objective: NDArray[np.float64]
    merit: NDArray[np.float64]
    feasible: NDArray[np.bool_]
    optimization: OptimizationResult

    def best_values(self) -> dict[str, float]:
        """Return the values of the scanned parameters at the best grid point."""
        index = np.unravel_index(np.nanargmin(self.objective), self.objective.shape)
        return {n: float(a[i]) for n, a, i in zip(self.names, self.axes, index, strict=True)}


def scan(
    problem: DesignProblem,
    axes: Mapping[str, ArrayLike],
    start: ParameterSet | None = None,
    events: EventSink | None = None,
    cancel: CancellationToken | None = None,
) -> ScanResult:
    """Evaluate a design problem on a regular grid of parameter values.

    Parameters not listed in `axes` keep their start values. Integer parameters are rounded.

    Args:
        problem: The design problem.
        axes: Grid values by parameter name.
        start: Start parameters. Defaults to the problem's parameters.
        events: Receiver of progress events.
        cancel: Token to cancel the scan; unevaluated grid points stay ``nan``.

    Returns:
        The scan result.

    Raises:
        ValueError: If no axis is given or an axis is empty.
        KeyError: If an axis refers to an unknown parameter.
    """
    params = start if start is not None else problem.parameters
    if not axes:
        raise ValueError("at least one scan axis is required")
    names = tuple(axes)
    grids = []
    for name in names:
        p = params[name]
        grid_values = np.unique(np.asarray(axes[name], dtype=np.float64))
        if p.integer:
            grid_values = np.unique(np.round(grid_values))
        if grid_values.size == 0:
            raise ValueError(f"scan axis {name!r} is empty")
        grids.append(grid_values)
    shape = tuple(len(g) for g in grids)
    objective = np.full(shape, np.nan)
    merit = np.full(shape, np.nan)
    feasible = np.zeros(shape, dtype=bool)
    tracker = Tracker(problem, "scan", events, cancel, int(np.prod(shape)))
    base = params.value_dict()
    for index in itertools.product(*(range(n) for n in shape)):
        if tracker.cancelled:
            break
        values = dict(base)
        for name, grid, i in zip(names, grids, index, strict=True):
            values[name] = float(grid[i])
        # Values outside the parameter bounds are allowed in a scan.
        widened = params
        for name in names:
            p = widened[name]
            if not p.min <= values[name] <= p.max:
                widened = widened.replace(
                    name, min=min(p.min, values[name]), max=max(p.max, values[name])
                )
        evaluation = problem.with_parameters(widened).evaluate(values)
        tracker.record(evaluation)
        objective[index] = evaluation.objective
        merit[index] = evaluation.merit
        feasible[index] = evaluation.feasible
    result = tracker.result(_widen(params, names, grids), "cancelled" if tracker.cancelled else "")
    return ScanResult(names, tuple(grids), objective, merit, feasible, result)


def _widen(
    params: ParameterSet, names: Sequence[str], grids: Sequence[NDArray[np.float64]]
) -> ParameterSet:
    for name, grid in zip(names, grids, strict=True):
        p = params[name]
        params = params.replace(name, min=min(p.min, grid.min()), max=max(p.max, grid.max()))
    return params


@register_optimizer
@dataclass(frozen=True)
class GridScan(Optimizer):
    """Grid search over the free parameters.

    Each free parameter is sampled at `points` equally spaced values within its bounds
    (integer parameters at all integers if fewer than `points`); parameters listed in
    `values` use the given grid instead. The best grid point is returned.

    Attributes:
        points: Number of grid points per free parameter.
        values: Explicit grids as (parameter name, values) pairs.
    """

    type_name: ClassVar[str] = "grid_scan"
    points: int = 11
    values: tuple[tuple[str, tuple[float, ...]], ...] = field(default=())

    def axes(self, params: ParameterSet) -> dict[str, NDArray[np.float64]]:
        """Return the scan grid for the free parameters of a parameter set.

        Args:
            params: The parameters.

        Returns:
            Grid values by parameter name.

        Raises:
            ValueError: If a free parameter without explicit grid has infinite bounds.
        """
        explicit = {name: np.asarray(v, dtype=np.float64) for name, v in self.values}
        axes: dict[str, NDArray[np.float64]] = {}
        for name in params.free_names():
            if name in explicit:
                axes[name] = explicit[name]
                continue
            p = params[name]
            if not (np.isfinite(p.min) and np.isfinite(p.max)):
                raise ValueError(f"grid scan requires finite bounds for {name!r}")
            if p.integer and p.max - p.min + 1 <= self.points:
                axes[name] = np.arange(np.ceil(p.min), np.floor(p.max) + 1)
            else:
                axes[name] = np.linspace(p.min, p.max, self.points)
        for name, grid in explicit.items():
            axes.setdefault(name, grid)
        return axes

    def run(
        self,
        problem: DesignProblem,
        start: ParameterSet | None = None,
        events: EventSink | None = None,
        cancel: CancellationToken | None = None,
    ) -> OptimizationResult:
        """Scan the free parameters and return the best grid point.

        Args:
            problem: The design problem.
            start: Start parameters. Defaults to the problem's parameters.
            events: Receiver of progress events.
            cancel: Token to cancel the scan.

        Returns:
            The result.
        """
        params = prepare_parameters(problem, start, vary_integers=True)
        return scan(problem, self.axes(params), params, events, cancel).optimization
