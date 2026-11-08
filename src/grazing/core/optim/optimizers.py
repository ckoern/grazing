"""Optimizers for problems with free parameters (see `grazing.core.optim.problem`)."""

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

import lmfit
import numpy as np
from numpy.typing import NDArray

from grazing.core.optim.events import CancellationToken, EventSink, ProgressEvent
from grazing.core.optim.problem import Evaluation, Problem
from grazing.core.parameters import ParameterSet
from grazing.core.plugins import Registry
from grazing.core.serialization import register_component_registry

__all__ = [
    "GLOBAL_METHODS",
    "LmfitOptimizer",
    "OptimizationResult",
    "Optimizer",
    "Tracker",
    "optimizers",
    "prepare_parameters",
    "register_optimizer",
]

GLOBAL_METHODS = frozenset(
    {"differential_evolution", "brute", "basinhopping", "ampgo", "dual_annealing", "shgo"}
)
"""lmfit methods that search globally. They vary integer parameters (rounded); local methods
keep integer parameters fixed."""

_RESIDUAL_METHODS = frozenset({"leastsq", "least_squares"})
_EVENT_INTERVAL = 0.2


@dataclass(frozen=True)
class OptimizationResult[E: Evaluation]:
    """Outcome of an optimization.

    Attributes:
        parameters: The parameter set with the values of the best evaluation.
        best: The best evaluation.
        evaluations: Number of objective evaluations.
        history: Best objective value after each evaluation.
        method: Name of the method.
        message: Message of the underlying optimizer.
        cancelled: Whether the optimization was cancelled.
        elapsed: Wall-clock time in s.
    """

    parameters: ParameterSet
    best: E
    evaluations: int
    history: tuple[float, ...]
    method: str
    message: str = ""
    cancelled: bool = False
    elapsed: float = 0.0


class Tracker[E: Evaluation]:
    """Keeps track of the best evaluation and reports progress during an optimization.

    Args:
        problem: The problem.
        task: Task name for progress events.
        events: Receiver of progress events.
        cancel: Cancellation token.
        total: Expected number of evaluations, if known.
    """

    def __init__(
        self,
        problem: Problem[E],
        task: str,
        events: EventSink | None = None,
        cancel: CancellationToken | None = None,
        total: int | None = None,
    ) -> None:
        self.problem = problem
        self.task = task
        self.events = events
        self.cancel = cancel
        self.total = total
        self.best: E | None = None
        self._best_quantities: dict[str, float] = {}
        self.history: list[float] = []
        self.start = time.perf_counter()
        self._last_event = -math.inf

    @property
    def cancelled(self) -> bool:
        """Whether cancellation has been requested."""
        return self.cancel is not None and self.cancel.cancelled

    @property
    def elapsed(self) -> float:
        """Elapsed time in s."""
        return time.perf_counter() - self.start

    def record(self, evaluation: E) -> None:
        """Record an evaluation and emit a progress event if due.

        Args:
            evaluation: The evaluation.
        """
        improved = self.best is None or evaluation.objective < self.best.objective
        if improved:
            self.best = evaluation
            self._best_quantities = self.problem.quantities(evaluation)
        assert self.best is not None
        self.history.append(self.best.objective)
        now = self.elapsed
        if self.events is not None and (improved or now - self._last_event > _EVENT_INTERVAL):
            self._last_event = now
            self.events(
                ProgressEvent(
                    self.task,
                    len(self.history),
                    self.total,
                    self.best.objective,
                    evaluation.objective,
                    dict(self.best.values),
                    now,
                    self.best.merit,
                    dict(self._best_quantities),
                )
            )

    def result(self, parameters: ParameterSet, message: str = "") -> OptimizationResult[E]:
        """Build the optimization result from the best recorded evaluation.

        Args:
            parameters: The parameter set used during the optimization (bounds and flags).
            message: Message of the underlying optimizer.

        Returns:
            The result.

        If nothing was evaluated (e.g. cancelled before the first evaluation), the start values
        given by `parameters` are evaluated.
        """
        if self.best is None:
            self.record(self.problem.with_parameters(parameters).evaluate())
            assert self.best is not None
        names = [n for n in parameters if n in self.best.values]
        values = {
            n: min(max(self.best.values[n], parameters[n].min), parameters[n].max) for n in names
        }
        free = {n: v for n, v in values.items() if parameters[n].expr is None}
        return OptimizationResult(
            parameters.with_values(free),
            self.best,
            len(self.history),
            tuple(self.history),
            self.task,
            message,
            self.cancelled,
            self.elapsed,
        )


@dataclass(frozen=True)
class Optimizer(ABC):
    """Base class of optimizers. Optimizers are immutable; their fields are the settings."""

    type_name: ClassVar[str]

    @abstractmethod
    def run[E: Evaluation](
        self,
        problem: Problem[E],
        start: ParameterSet | None = None,
        events: EventSink | None = None,
        cancel: CancellationToken | None = None,
    ) -> OptimizationResult[E]:
        """Optimize a problem.

        Args:
            problem: The problem.
            start: Start parameters (values, bounds, fixed/free). Defaults to the problem's
                parameters.
            events: Receiver of progress events.
            cancel: Token to cancel the optimization; the best evaluation so far is returned.

        Returns:
            The result.
        """


optimizers: Registry[type[Optimizer]] = Registry("optimizer")
"""Registry of optimizer classes by `Optimizer.type_name`."""


def register_optimizer[O: type[Optimizer]](cls: O) -> O:
    """Class decorator that registers an optimizer class under its `type_name`.

    Args:
        cls: The optimizer class.

    Returns:
        The unchanged class.
    """
    optimizers.register(cls.type_name, cls)
    return cls


def prepare_parameters(
    problem: Problem[Any], start: ParameterSet | None, vary_integers: bool
) -> ParameterSet:
    """Return the parameters to optimize, with constraint hints applied.

    Bounds of free parameters are intersected with the bounds hinted by the problem (if the
    intersection is not empty), and integer parameters are fixed unless `vary_integers` is set.

    Args:
        problem: The problem.
        start: Start parameters, or ``None`` for the problem's parameters.
        vary_integers: Whether integer parameters may be varied.

    Returns:
        The prepared parameter set.
    """
    params = start if start is not None else problem.parameters
    for name, (lo, hi) in problem.bounds_hints().items():
        if name not in params:
            continue
        p = params[name]
        new_lo, new_hi = max(p.min, lo), min(p.max, hi)
        if p.is_free and new_lo <= new_hi:
            params = params.replace(name, min=new_lo, max=new_hi)
    if not vary_integers:
        for name, p in params.items():
            if p.integer and p.vary:
                params = params.replace(name, vary=False)
    return params


def _seeded_population(params: lmfit.Parameters, options: dict[str, Any]) -> NDArray[np.float64]:
    """Initial differential-evolution population: a Latin hypercube plus the start point.

    lmfit runs differential evolution on internal coordinates in [-π/2, π/2] per varying
    parameter; the first member is the start point mapped to these coordinates.
    """
    varying = [p for p in params.values() if p.vary]
    popsize = int(options.get("popsize", 15))
    size = max(5, popsize * len(varying))
    rng = np.random.default_rng(options.get("seed"))
    strata = np.array([rng.permutation(size) for _ in varying]).T
    unit = (strata + rng.random(strata.shape)) / size
    population = -0.5 * np.pi + np.pi * unit
    population[0] = [p.setup_bounds() for p in varying]
    return population


@register_optimizer
@dataclass(frozen=True)
class LmfitOptimizer(Optimizer):
    """Optimization with any minimization method of lmfit.

    Least-squares methods (``"leastsq"``, ``"least_squares"``) minimize the residual vector of
    the problem; all other methods minimize the scalar objective. Global methods (see
    `GLOBAL_METHODS`) vary integer parameters (rounded when building the coating) and require
    finite bounds for all free parameters; local methods keep integer parameters fixed.

    The start values are evaluated first, so the result is never worse than the start.
    Differential evolution includes the start point in its initial population unless an
    ``init`` option is given.

    Attributes:
        method: lmfit method name, e.g. ``"leastsq"``, ``"nelder"``, ``"lbfgsb"``,
            ``"differential_evolution"``.
        max_evaluations: Maximum number of objective evaluations, or ``None`` for the method's
            default.
        options: Additional keyword arguments for the lmfit method as (name, value) pairs, e.g.
            ``(("popsize", 20), ("seed", 1))`` for differential evolution.
    """

    type_name: ClassVar[str] = "lmfit"
    method: str = "leastsq"
    max_evaluations: int | None = None
    options: tuple[tuple[str, Any], ...] = field(default=())

    def run[E: Evaluation](
        self,
        problem: Problem[E],
        start: ParameterSet | None = None,
        events: EventSink | None = None,
        cancel: CancellationToken | None = None,
    ) -> OptimizationResult[E]:
        """Optimize a problem.

        Args:
            problem: The problem.
            start: Start parameters. Defaults to the problem's parameters.
            events: Receiver of progress events.
            cancel: Token to cancel the optimization; the best evaluation so far is returned.

        Returns:
            The result.

        Raises:
            ValueError: If no parameter is free, or a global method lacks finite bounds.
        """
        is_global = self.method in GLOBAL_METHODS
        params = prepare_parameters(problem, start, vary_integers=is_global)
        free = params.free_names()
        if not free:
            raise ValueError("no free parameters to optimize")
        if is_global:
            unbounded = [
                n for n in free if not (np.isfinite(params[n].min) and np.isfinite(params[n].max))
            ]
            if unbounded:
                raise ValueError(f"{self.method} requires finite bounds for {unbounded}")
        tracker = Tracker(problem, self.method, events, cancel, self.max_evaluations)
        lm_params = params.to_lmfit()
        names = {n.replace(".", "__"): n for n in params}
        use_residuals = self.method in _RESIDUAL_METHODS
        first = problem.evaluate(params.value_dict())
        size = 0
        if use_residuals:
            if first.error is not None:
                raise ValueError(f"start values cannot be evaluated: {first.error}")
            size = len(problem.residuals(first, 1))
        tracker.record(first)
        options = dict(self.options)
        if self.method == "differential_evolution" and "init" not in options:
            options["init"] = _seeded_population(lm_params, options)

        expressions = {n for n, p in params.items() if p.expr is not None}

        def objective(p: lmfit.Parameters) -> float | NDArray[np.float64]:
            values = {names[k]: float(v.value) for k, v in p.items() if names[k] not in expressions}
            evaluation = problem.evaluate(values)
            tracker.record(evaluation)
            if use_residuals:
                return problem.residuals(evaluation, size)
            return evaluation.objective

        def iteration(*_: Any, **__: Any) -> bool:
            return tracker.cancelled

        minimizer = lmfit.Minimizer(
            objective, lm_params, iter_cb=iteration, max_nfev=self.max_evaluations
        )
        message = ""
        try:
            out = minimizer.minimize(method=self.method, **options)
            message = str(getattr(out, "message", ""))
        except lmfit.minimizer.AbortFitException:
            message = "aborted"
        return tracker.result(params, message)


register_component_registry("optimizer", optimizers)
