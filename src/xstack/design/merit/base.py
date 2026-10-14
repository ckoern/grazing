"""Interface of figures of merit and the data they are evaluated on."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from xstack.core.physics.models import ReflectivityResult
from xstack.core.plugins import Registry
from xstack.core.serialization import register_component_registry
from xstack.core.structure.stack import Stack
from xstack.core.targets import TargetSet

__all__ = [
    "CallableMerit",
    "Curve",
    "CurveMerit",
    "EvaluationContext",
    "FigureOfMerit",
    "merits",
    "register_merit",
]


@dataclass(frozen=True)
class Curve:
    """A reflectivity curve prepared for evaluating a figure of merit.

    Attributes:
        x: Values of the scanned variable, shape (P,).
        reflectivity: Reflectance, shape (P,).
        weights: Integration weights including the spectrum, summing to 1, shape (P,).
        profile: Desired reflectivity at `x`, or ``None``.
        weight: Weight of this curve among all curves, the weights of all curves sum to 1.
    """

    x: NDArray[np.float64]
    reflectivity: NDArray[np.float64]
    weights: NDArray[np.float64]
    profile: NDArray[np.float64] | None
    weight: float

    def mean(self) -> float:
        """Return the weighted mean reflectivity."""
        return float(np.dot(self.weights, self.reflectivity))


@dataclass(frozen=True)
class EvaluationContext:
    """Everything a figure of merit may use to rate a design.

    Attributes:
        stack: The evaluated stack.
        targets: The target set.
        results: Reflectivity results, one per target in the order of `targets`.
        coating_time: Estimated coating time in s, or ``None`` if no process model is set.
    """

    stack: Stack
    targets: TargetSet
    results: tuple[ReflectivityResult, ...]
    coating_time: float | None = None

    def __post_init__(self) -> None:
        if len(self.results) != len(self.targets):
            raise ValueError("one reflectivity result per target is required")

    @property
    def total_thickness(self) -> float:
        """Total coating thickness in nm."""
        return self.stack.total_thickness

    def curves(self) -> tuple[Curve, ...]:
        """Return the curves to rate, according to the combine mode of the target set.

        Returns:
            A single weighted-average curve for ``"average_reflectivity"``, one curve per target
            for ``"average_merit"``.
        """
        target_weights = self.targets.normalized_weights()
        prepared = []
        for result, w in zip(self.results, target_weights, strict=True):
            target = result.target
            profile = target.profile(result.x) if target.profile is not None else None
            prepared.append(
                Curve(result.x, result.reflectivity, target.weights(len(result.x)), profile, w)
            )
        if self.targets.combine == "average_merit" or len(prepared) == 1:
            return tuple(prepared)
        reflectivity = sum(
            (c.weight * c.reflectivity for c in prepared), np.zeros_like(prepared[0].x)
        )
        weights = sum((c.weight * c.weights for c in prepared), np.zeros_like(prepared[0].x))
        profile = next((c.profile for c in prepared if c.profile is not None), None)
        return (Curve(prepared[0].x, reflectivity, weights, profile, 1.0),)


@dataclass(frozen=True)
class FigureOfMerit(ABC):
    """Base class of all figures of merit. Lower values are better.

    Figures of merit are immutable dataclasses; their fields are the settings.
    """

    type_name: ClassVar[str]

    @abstractmethod
    def scalar(self, ctx: EvaluationContext) -> float:
        """Return the figure of merit (lower is better).

        Args:
            ctx: The evaluation context.

        Returns:
            The figure of merit.
        """

    def residuals(self, ctx: EvaluationContext) -> NDArray[np.float64]:
        """Return a residual vector for least-squares optimizers.

        Minimizing the sum of squares of the residuals approximately minimizes `scalar`.

        Args:
            ctx: The evaluation context.

        Returns:
            One-dimensional residual vector.
        """
        return np.array([np.sqrt(max(self.scalar(ctx), 0.0))])

    def quantities(self, ctx: EvaluationContext) -> dict[str, float]:
        """Return the quantities the figure of merit rates, for display.

        For example, the figure of merit 1 − ⟨R⟩ rates the mean reflectivity ⟨R⟩.

        Args:
            ctx: The evaluation context.

        Returns:
            Values by label, e.g. ``{"mean reflectivity": 0.42}``; empty if the figure of merit
            does not define any.
        """
        return {}


@dataclass(frozen=True)
class CurveMerit(FigureOfMerit):
    """Base class of figures of merit computed per reflectivity curve.

    The scalar value is the curve-weighted sum of `curve_value`; the residuals are the
    concatenated `curve_residuals`, each scaled with the square root of the curve weight.
    Subclasses that set `quantity_label` and implement `curve_quantity` report the
    curve-weighted sum of `curve_quantity` as their quantity.

    Attributes:
        quantity_label: Label of the quantity rated per curve, or ``None``.
    """

    quantity_label: ClassVar[str | None] = None

    def curve_quantity(self, curve: Curve) -> float:
        """Return the quantity rated for one curve, see `quantity_label`.

        Args:
            curve: The curve.

        Returns:
            The quantity.

        Raises:
            NotImplementedError: If the class defines no quantity.
        """
        raise NotImplementedError(f"{type(self).__name__} defines no quantity")

    @abstractmethod
    def curve_value(self, curve: Curve) -> float:
        """Return the figure of merit of one curve.

        Args:
            curve: The curve.

        Returns:
            The figure of merit (lower is better).
        """

    @abstractmethod
    def curve_residuals(self, curve: Curve) -> NDArray[np.float64]:
        """Return the residual vector of one curve.

        Args:
            curve: The curve.

        Returns:
            One-dimensional residual vector.
        """

    def scalar(self, ctx: EvaluationContext) -> float:
        """Return the figure of merit (lower is better).

        Args:
            ctx: The evaluation context.

        Returns:
            The figure of merit.
        """
        return float(sum(c.weight * self.curve_value(c) for c in ctx.curves()))

    def residuals(self, ctx: EvaluationContext) -> NDArray[np.float64]:
        """Return a residual vector for least-squares optimizers.

        Args:
            ctx: The evaluation context.

        Returns:
            One-dimensional residual vector.
        """
        return np.concatenate([np.sqrt(c.weight) * self.curve_residuals(c) for c in ctx.curves()])

    def quantities(self, ctx: EvaluationContext) -> dict[str, float]:
        """Return the curve-weighted sum of `curve_quantity` under `quantity_label`.

        Args:
            ctx: The evaluation context.

        Returns:
            Values by label; empty if `quantity_label` is not set.
        """
        if self.quantity_label is None:
            return {}
        value = sum(c.weight * self.curve_quantity(c) for c in ctx.curves())
        return {self.quantity_label: float(value)}


merits: Registry[type[FigureOfMerit]] = Registry("figure of merit")
"""Registry of figure-of-merit classes by `FigureOfMerit.type_name`."""


def register_merit[F: type[FigureOfMerit]](cls: F) -> F:
    """Class decorator that registers a figure-of-merit class under its `type_name`.

    Args:
        cls: The class.

    Returns:
        The unchanged class.
    """
    merits.register(cls.type_name, cls)
    return cls


@register_merit
@dataclass(frozen=True)
class CallableMerit(FigureOfMerit):
    """A figure of merit given by a user function.

    Attributes:
        function: Callable ``(EvaluationContext) -> float`` returning a value to minimize.
        label: Human-readable name.
    """

    type_name: ClassVar[str] = "callable"
    function: Callable[[EvaluationContext], float]
    label: str = "custom"

    def scalar(self, ctx: EvaluationContext) -> float:
        """Return the value of the user function.

        Args:
            ctx: The evaluation context.

        Returns:
            The figure of merit.
        """
        return float(self.function(ctx))


register_component_registry("merit", merits)
