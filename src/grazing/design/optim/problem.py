"""Design problem: everything needed to rate a set of parameter values."""

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from grazing.core.materials import MaterialLibrary
from grazing.core.parameters import ParameterSet
from grazing.core.physics.models import ParrattModel, ReflectivityModel, ReflectivityResult
from grazing.core.serialization import register_serializable
from grazing.core.structure.coating import CoatingBuildError, Mirror
from grazing.core.structure.stack import Stack
from grazing.core.targets import TargetSet
from grazing.design.manufacturing.constraints import Constraint, ConstraintResult
from grazing.design.manufacturing.process import DepositionProcess
from grazing.design.merit.base import EvaluationContext, FigureOfMerit

__all__ = ["DesignProblem", "Evaluation", "require_design_problem"]


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
    library: MaterialLibrary | None = field(
        default=None, compare=False, hash=False, metadata={"serialize": False}
    )

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
        """Return a copy with different design parameters (coating and density parameters).

        Args:
            parameters: The new parameter set (same names).

        Returns:
            The modified problem.
        """
        return replace(self, mirror=self.mirror.with_parameters(parameters))

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

    def evaluate_stack(
        self,
        stack: Stack,
        values: Mapping[str, float] | None = None,
        deposited: Stack | None = None,
    ) -> Evaluation:
        """Rate an arbitrary stack with the targets, figure of merit and constraints.

        Args:
            stack: The stack, e.g. a design with perturbed layer thicknesses.
            values: Parameter values to store in the evaluation, if known.
            deposited: The stack as deposited, used for the coating time and the constraints,
                if it differs from the physical `stack` (e.g. by interlayers that form at the
                interfaces). Defaults to `stack`.

        Returns:
            The evaluation.
        """
        results = tuple(
            self.model.evaluate(stack, t, self.points, self.library) for t in self.targets.targets
        )
        made = deposited if deposited is not None else stack
        time = self.process.time(made) if self.process is not None else None
        checked = tuple((c, c.evaluate(made, self.process)) for c in self.constraints)
        ctx = EvaluationContext(stack, self.targets, results, time)
        merit = self.merit.scalar(ctx)
        soft, hard = self._penalties(checked)
        return Evaluation(
            dict(values or {}), merit + (soft + hard), merit, stack, results, time, checked
        )

    def quantities(self, evaluation: Evaluation) -> dict[str, float]:
        """Return the quantities rated by the figure of merit for an evaluated design.

        Args:
            evaluation: An evaluation of this problem.

        Returns:
            Values by label (see `FigureOfMerit.quantities`); empty if the design could not
            be built.
        """
        if evaluation.stack is None or len(evaluation.results) != len(self.targets):
            return {}
        ctx = EvaluationContext(
            evaluation.stack, self.targets, evaluation.results, evaluation.coating_time
        )
        return self.merit.quantities(ctx)

    def _penalties(
        self, checked: tuple[tuple[Constraint, ConstraintResult], ...]
    ) -> tuple[float, float]:
        """Return the soft and hard penalty terms of checked constraints."""
        soft = sum(r.violation**2 for c, r in checked if c.severity == "soft")
        hard = sum(r.violation for c, r in checked if c.severity == "hard" and not r.satisfied)
        hard_violated = any(not r.satisfied for c, r in checked if c.severity == "hard")
        return (
            self.soft_penalty * soft,
            self.hard_penalty * (1.0 + hard) if hard_violated else 0.0,
        )

    def residuals(self, evaluation: Evaluation, size: int) -> NDArray[np.float64]:
        """Return the residual vector of an evaluated design for least-squares optimizers.

        The vector consists of the residuals of the figure of merit followed by two penalty
        entries whose squares are the soft and hard penalty terms, so its sum of squares is the
        objective of the evaluation. Designs that cannot be built get `size` equal entries whose
        sum of squares is 2·λ_hard.

        Args:
            evaluation: An evaluation of this problem.
            size: Length of the vector for designs that cannot be built.

        Returns:
            Residual vector.
        """
        if evaluation.stack is None:
            return np.full(size, np.sqrt(2.0 * self.hard_penalty / size))
        ctx = EvaluationContext(
            evaluation.stack, self.targets, evaluation.results, evaluation.coating_time
        )
        tail = np.sqrt(np.maximum(self._penalties(evaluation.constraints), 0.0))
        return np.concatenate([self.merit.residuals(ctx), tail])


register_serializable("DesignProblem", DesignProblem)


def require_design_problem(problem: object, user: str) -> DesignProblem:
    """Return a problem as `DesignProblem`, for optimizers that only work on designs.

    Args:
        problem: The problem passed to the optimizer.
        user: Name of the optimizer, used in the error message.

    Returns:
        The same problem.

    Raises:
        TypeError: If `problem` is not a `DesignProblem`.
    """
    if not isinstance(problem, DesignProblem):
        raise TypeError(f"{user} requires a DesignProblem, got {type(problem).__name__}")
    return problem
