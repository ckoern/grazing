"""Design problem: everything needed to rate a set of parameter values."""

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from xstack.manufacturing.constraints import Constraint, ConstraintResult
from xstack.manufacturing.process import DepositionProcess
from xstack.materials import MaterialLibrary
from xstack.merit.base import EvaluationContext, FigureOfMerit
from xstack.parameters import ParameterSet
from xstack.physics.models import ParrattModel, ReflectivityModel, ReflectivityResult
from xstack.structure.coating import CoatingBuildError, Mirror
from xstack.structure.stack import Stack
from xstack.targets import TargetSet

__all__ = ["DesignProblem", "Evaluation"]


@dataclass(frozen=True)
class Evaluation:
    """Full diagnostic result of evaluating one design.

    Attributes:
        values: Evaluated parameter values.
        objective: Objective value including constraint penalties (lower is better).
        merit: Figure of merit without penalties, ``nan`` if the design could not be built.
        stack: The stack, or ``None`` if the design could not be built.
        results: Reflectivity results per target.
        coating_time: Coating time in s, or ``None`` without a process.
        constraints: Pairs of constraint and its result.
        error: Description of why the design could not be built, or ``None``.
    """

    values: Mapping[str, float]
    objective: float
    merit: float
    stack: Stack | None
    results: tuple[ReflectivityResult, ...] = ()
    coating_time: float | None = None
    constraints: tuple[tuple[Constraint, ConstraintResult], ...] = ()
    error: str | None = None

    @property
    def feasible(self) -> bool:
        """Whether the design could be built and satisfies all hard constraints."""
        return self.error is None and all(
            r.satisfied for c, r in self.constraints if c.severity == "hard"
        )


@dataclass(frozen=True)
class DesignProblem:
    """A mirror design problem.

    The objective of a design is

        objective = F + λ_soft·Σ v_soft² + (λ_hard·(1 + Σ v_hard) if any hard violation)

    with the figure of merit F and the normalized violations v of the soft and hard
    constraints. With λ_hard larger than the range of F, every feasible design is better than
    every infeasible one, while infeasible designs are still ranked by their violation.

    Attributes:
        mirror: Coating and substrate; its coating parameters are the design parameters.
        targets: The targets.
        merit: The figure of merit.
        constraints: Manufacturing constraints.
        process: Deposition process for coating times.
        model: Reflectivity model.
        points: Number of grid points per target, or ``None`` for the target defaults.
        hard_penalty: λ_hard.
        soft_penalty: λ_soft.
        library: Material library for optical constants, or ``None`` for the default library.
    """

    mirror: Mirror
    targets: TargetSet
    merit: FigureOfMerit
    constraints: tuple[Constraint, ...] = ()
    process: DepositionProcess | None = None
    model: ReflectivityModel = field(default_factory=ParrattModel)
    points: int | None = None
    hard_penalty: float = 10.0
    soft_penalty: float = 1.0
    library: MaterialLibrary | None = field(default=None, compare=False, hash=False)

    @property
    def parameters(self) -> ParameterSet:
        """The design parameters (coating parameters)."""
        return self.mirror.parameters

    def evolve(self, **changes: Any) -> "DesignProblem":
        """Return a copy with some fields changed.

        Args:
            **changes: New values for fields of the problem.

        Returns:
            The modified problem.
        """
        return replace(self, **changes)

    def with_parameters(self, parameters: ParameterSet) -> "DesignProblem":
        """Return a copy with different coating parameters.

        Args:
            parameters: The new parameter set (same names).

        Returns:
            The modified problem.
        """
        coating = self.mirror.coating.with_parameters(parameters)
        return replace(self, mirror=self.mirror.with_coating(coating))

    def bounds_hints(self) -> dict[str, tuple[float, float]]:
        """Return parameter bounds implied by the hard constraints.

        Returns:
            Intersection of the bounds hinted by all hard constraints, by parameter name.
        """
        hints: dict[str, tuple[float, float]] = {}
        for c in self.constraints:
            if c.severity != "hard":
                continue
            for name, (lo, hi) in c.bounds_hint(self.mirror.coating).items():
                old_lo, old_hi = hints.get(name, (-math.inf, math.inf))
                hints[name] = (max(lo, old_lo), min(hi, old_hi))
        return hints

    def evaluate(self, values: Mapping[str, float] | None = None) -> Evaluation:
        """Evaluate a design.

        Args:
            values: Parameter values overriding the current ones, by name.

        Returns:
            The evaluation. Designs that cannot be built, including values outside the
            parameter bounds, get the objective 2·λ_hard and an error message instead of
            raising.
        """
        params = self.parameters
        try:
            if values is not None:
                params = params.with_values(values)
            resolved = params.evaluate()
            stack = self.mirror.stack(resolved)
        except (CoatingBuildError, ValueError) as exc:
            return Evaluation(
                params.value_dict(), 2.0 * self.hard_penalty, math.nan, None, error=str(exc)
            )
        return self.evaluate_stack(stack, resolved)

    def evaluate_stack(self, stack: Stack, values: Mapping[str, float] | None = None) -> Evaluation:
        """Rate an arbitrary stack with the targets, figure of merit and constraints.

        Args:
            stack: The stack, e.g. a design with perturbed layer thicknesses.
            values: Parameter values to store in the evaluation, if known.

        Returns:
            The evaluation.
        """
        results = tuple(
            self.model.evaluate(stack, t, self.points, self.library) for t in self.targets.targets
        )
        time = self.process.time(stack) if self.process is not None else None
        checked = tuple((c, c.evaluate(stack, self.process)) for c in self.constraints)
        ctx = EvaluationContext(stack, self.targets, results, time)
        merit = self.merit.scalar(ctx)
        return Evaluation(
            dict(values or {}), merit + self._penalty(checked), merit, stack, results, time, checked
        )

    def _penalty(self, checked: tuple[tuple[Constraint, ConstraintResult], ...]) -> float:
        soft = sum(r.violation**2 for c, r in checked if c.severity == "soft")
        hard = sum(r.violation for c, r in checked if c.severity == "hard" and not r.satisfied)
        hard_violated = any(not r.satisfied for c, r in checked if c.severity == "hard")
        return self.soft_penalty * soft + (
            self.hard_penalty * (1.0 + hard) if hard_violated else 0.0
        )

    def residuals(self, values: Mapping[str, float] | None, size: int) -> NDArray[np.float64]:
        """Return the residual vector of a design for least-squares optimizers.

        The vector consists of the residuals of the figure of merit followed by two penalty
        entries whose squares are the soft and hard penalty terms.

        Args:
            values: Parameter values overriding the current ones, by name.
            size: Expected length of the vector, used for designs that cannot be built.

        Returns:
            Residual vector.
        """
        evaluation = self.evaluate(values)
        if evaluation.stack is None:
            return np.full(size, np.sqrt(2.0 * self.hard_penalty / size))
        ctx = EvaluationContext(
            evaluation.stack, self.targets, evaluation.results, evaluation.coating_time
        )
        penalty_soft = self.soft_penalty * sum(
            r.violation**2 for c, r in evaluation.constraints if c.severity == "soft"
        )
        penalty_hard = evaluation.objective - evaluation.merit - penalty_soft
        tail = np.sqrt(np.maximum([penalty_soft, penalty_hard], 0.0))
        return np.concatenate([self.merit.residuals(ctx), tail])
