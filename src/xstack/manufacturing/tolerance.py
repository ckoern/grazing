"""Monte Carlo analysis of the sensitivity of a design to thickness errors."""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from xstack.structure.stack import Stack

if TYPE_CHECKING:
    from xstack.optim.events import CancellationToken
    from xstack.optim.problem import DesignProblem, Evaluation

__all__ = ["ToleranceResult", "perturb_thicknesses", "thickness_tolerance"]


@dataclass(frozen=True)
class ToleranceResult:
    """Distribution of the figure of merit under random thickness errors.

    Attributes:
        nominal: Evaluation of the unperturbed design.
        merits: Figure of merit of each perturbed sample.
        objectives: Objective (including penalties) of each perturbed sample.
        feasible: Whether each perturbed sample satisfies all hard constraints.
        samples: Evaluations of the perturbed samples.
    """

    nominal: "Evaluation"
    merits: NDArray[np.float64]
    objectives: NDArray[np.float64]
    feasible: NDArray[np.bool_]
    samples: tuple["Evaluation", ...]

    def percentile(self, q: float) -> float:
        """Return a percentile of the figure of merit of the perturbed samples.

        Args:
            q: Percentile in [0, 100].

        Returns:
            The percentile.
        """
        return float(np.percentile(self.merits, q))


def perturb_thicknesses(
    stack: Stack,
    rng: np.random.Generator,
    random_error: float = 0.01,
    systematic_error: float = 0.0,
) -> Stack:
    """Return a copy of a stack with randomly perturbed layer thicknesses.

    Every thickness t is replaced by t·(1 + σ_r·ξ)·(1 + σ_s·η) with independent standard normal
    ξ per layer and one η per material and stack (a rate calibration error). Negative results
    are set to zero.

    Args:
        stack: The stack.
        rng: Random number generator.
        random_error: Relative standard deviation σ_r of independent per-layer errors.
        systematic_error: Relative standard deviation σ_s of the per-material rate error.

    Returns:
        The perturbed stack.
    """
    materials = {layer.material.name for layer in stack.layers}
    rate_error = {m: 1.0 + systematic_error * rng.standard_normal() for m in sorted(materials)}
    factors = 1.0 + random_error * rng.standard_normal(stack.num_layers)
    layers = [
        replace(layer, thickness=max(0.0, layer.thickness * f * rate_error[layer.material.name]))
        for layer, f in zip(stack.layers, factors, strict=True)
    ]
    return Stack(layers, stack.substrate, stack.substrate_roughness, stack.ambient)


def thickness_tolerance(
    problem: "DesignProblem",
    values: Mapping[str, float] | None = None,
    random_error: float = 0.01,
    systematic_error: float = 0.0,
    samples: int = 100,
    seed: int | None = None,
    cancel: "CancellationToken | None" = None,
) -> ToleranceResult:
    """Evaluate a design for randomly perturbed layer thicknesses.

    Args:
        problem: The design problem.
        values: Parameter values of the design. Defaults to the problem's current values.
        random_error: Relative standard deviation of independent per-layer errors.
        systematic_error: Relative standard deviation of per-material rate errors.
        samples: Number of perturbed samples.
        seed: Random seed.
        cancel: Cancellation token; stops after the current sample.

    Returns:
        The distribution of the figure of merit.

    Raises:
        ValueError: If the nominal design cannot be built.
    """
    nominal = problem.evaluate(values)
    if nominal.stack is None:
        raise ValueError(f"design cannot be built: {nominal.error}")
    rng = np.random.default_rng(seed)
    evaluations = []
    for _ in range(samples):
        if cancel is not None and cancel.cancelled:
            break
        stack = perturb_thicknesses(nominal.stack, rng, random_error, systematic_error)
        evaluations.append(problem.evaluate_stack(stack, nominal.values))
    return ToleranceResult(
        nominal,
        np.array([e.merit for e in evaluations]),
        np.array([e.objective for e in evaluations]),
        np.array([e.feasible for e in evaluations], dtype=bool),
        tuple(evaluations),
    )
