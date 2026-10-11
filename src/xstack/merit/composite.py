"""Figures of merit that combine other figures of merit."""

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from xstack.merit.base import EvaluationContext, FigureOfMerit, register_merit

__all__ = ["TimePenalized", "WeightedSum"]


@register_merit
@dataclass(frozen=True)
class WeightedSum(FigureOfMerit):
    """Weighted sum of several figures of merit.

    Value: Σ a_k·F_k. Residuals: the residuals of each term scaled by √a_k.

    Attributes:
        terms: Pairs of figure of merit and non-negative weight a_k.
    """

    type_name: ClassVar[str] = "weighted_sum"
    terms: tuple[tuple[FigureOfMerit, float], ...] = field(default=())

    def __post_init__(self) -> None:
        if not self.terms:
            raise ValueError("WeightedSum needs at least one term")
        if any(w < 0 for _, w in self.terms):
            raise ValueError("weights must not be negative")

    def scalar(self, ctx: EvaluationContext) -> float:
        """Return the weighted sum of the terms.

        Args:
            ctx: The evaluation context.

        Returns:
            The figure of merit.
        """
        return float(sum(w * m.scalar(ctx) for m, w in self.terms))

    def residuals(self, ctx: EvaluationContext) -> NDArray[np.float64]:
        """Return the concatenated, scaled residuals of all terms.

        Args:
            ctx: The evaluation context.

        Returns:
            One-dimensional residual vector.
        """
        return np.concatenate([np.sqrt(w) * m.residuals(ctx) for m, w in self.terms])


@register_merit
@dataclass(frozen=True)
class TimePenalized(FigureOfMerit):
    """A figure of merit with the coating time as an additional penalty.

    Value: (1 − w)·F + w·T / T_ref, with the optical figure of merit F, the coating time T and
    the reference time T_ref. Residuals: the residuals of F scaled by √(1 − w), followed by
    √(w·T / T_ref).

    Attributes:
        merit: The optical figure of merit F.
        weight: Weighting fraction w in [0, 1]; 0 ignores the time, 1 only minimizes the time.
        reference_time: T_ref in s, e.g. the maximum acceptable coating time.
    """

    type_name: ClassVar[str] = "time_penalized"
    merit: FigureOfMerit
    weight: float = 0.1
    reference_time: float = 3600.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError("weight must be in [0, 1]")
        if not self.reference_time > 0:
            raise ValueError("reference_time must be positive")

    def _time_term(self, ctx: EvaluationContext) -> float:
        if ctx.coating_time is None:
            raise ValueError("TimePenalized requires a coating time (define a deposition process)")
        return ctx.coating_time / self.reference_time

    def scalar(self, ctx: EvaluationContext) -> float:
        """Return (1 − w)·F + w·T / T_ref.

        Args:
            ctx: The evaluation context.

        Returns:
            The figure of merit.

        Raises:
            ValueError: If the context has no coating time.
        """
        return (1.0 - self.weight) * self.merit.scalar(ctx) + self.weight * self._time_term(ctx)

    def residuals(self, ctx: EvaluationContext) -> NDArray[np.float64]:
        """Return the scaled residuals of F followed by the time residual.

        Args:
            ctx: The evaluation context.

        Returns:
            One-dimensional residual vector.

        Raises:
            ValueError: If the context has no coating time.
        """
        time = np.sqrt(self.weight * self._time_term(ctx))
        return np.concatenate([np.sqrt(1.0 - self.weight) * self.merit.residuals(ctx), [time]])
