"""Manufacturing constraints on layer stacks."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar, Literal

from xstack.core.plugins import Registry
from xstack.core.serialization import register_component_registry
from xstack.core.structure.coating import Coating
from xstack.core.structure.coatings.graded import DepthGraded
from xstack.core.structure.coatings.periodic import Periodic
from xstack.core.structure.stack import Stack
from xstack.design.manufacturing.process import DepositionProcess

__all__ = [
    "Constraint",
    "ConstraintResult",
    "MaxCoatingTime",
    "MaxLayers",
    "MaxTotalThickness",
    "MinLayerThickness",
    "Severity",
    "constraints",
    "register_constraint",
]

type Severity = Literal["hard", "soft"]
"""Hard constraints must be met; soft constraints are traded off against the figure of merit."""


@dataclass(frozen=True)
class ConstraintResult:
    """Outcome of checking a constraint.

    Attributes:
        satisfied: Whether the constraint is met.
        violation: Normalized amount of violation, 0 if satisfied, e.g. 0.1 for 10 % beyond
            the limit.
        message: Human-readable description.
        offending_layers: Indices (in `Stack.layers`) of layers that violate the constraint.
    """

    satisfied: bool
    violation: float
    message: str
    offending_layers: tuple[int, ...] = ()


@dataclass(frozen=True)
class Constraint(ABC):
    """Base class of all manufacturing constraints.

    Constraints are immutable dataclasses; their fields are the settings.

    Attributes:
        severity: ``"hard"`` or ``"soft"``.
    """

    type_name: ClassVar[str]
    severity: Severity = field(default="hard", kw_only=True)

    @abstractmethod
    def evaluate(self, stack: Stack, process: DepositionProcess | None = None) -> ConstraintResult:
        """Check the constraint.

        Args:
            stack: The stack.
            process: The deposition process, if defined.

        Returns:
            The result.
        """

    def bounds_hint(self, coating: Coating) -> dict[str, tuple[float, float]]:
        """Return parameter bounds that are necessary to satisfy the constraint.

        Optimizers can intersect these with the parameter bounds to exclude infeasible regions
        before evaluating them.

        Args:
            coating: The coating.

        Returns:
            Bounds (min, max) by parameter name; empty if no hint can be given.
        """
        return {}


constraints: Registry[type[Constraint]] = Registry("constraint")
"""Registry of constraint classes by `Constraint.type_name`."""


def register_constraint[C: type[Constraint]](cls: C) -> C:
    """Class decorator that registers a constraint class under its `type_name`.

    Args:
        cls: The constraint class.

    Returns:
        The unchanged class.
    """
    constraints.register(cls.type_name, cls)
    return cls


@register_constraint
@dataclass(frozen=True)
class MinLayerThickness(Constraint):
    """Every layer (of the given materials) must be at least a minimum thickness.

    Layers of zero thickness are ignored, since they are not deposited.

    Attributes:
        minimum: Minimum thickness in nm.
        materials: Material names the constraint applies to; empty means all materials.
    """

    type_name: ClassVar[str] = "min_layer_thickness"
    minimum: float = field(default=1.0, metadata={"unit": "nm"})
    materials: tuple[str, ...] = field(default=(), metadata={"label": "materials (empty = all)"})

    def __post_init__(self) -> None:
        if not self.minimum > 0:
            raise ValueError("minimum thickness must be positive")

    def _applies(self, material: str) -> bool:
        return not self.materials or material in self.materials

    def evaluate(self, stack: Stack, process: DepositionProcess | None = None) -> ConstraintResult:
        """Check the thickness of every layer.

        Args:
            stack: The stack.
            process: Not used.

        Returns:
            The result; the violation is the largest relative shortfall.
        """
        offending = []
        worst = 0.0
        for i, layer in enumerate(stack.layers):
            if 0 < layer.thickness < self.minimum and self._applies(layer.material.name):
                offending.append(i)
                worst = max(worst, (self.minimum - layer.thickness) / self.minimum)
        if not offending:
            return ConstraintResult(True, 0.0, f"all layers ≥ {self.minimum:g} nm")
        thinnest = min(stack.layers[i].thickness for i in offending)
        message = (
            f"{len(offending)} layer(s) thinner than {self.minimum:g} nm "
            f"(thinnest {thinnest:.3g} nm)"
        )
        return ConstraintResult(False, worst, message, tuple(offending))

    def bounds_hint(self, coating: Coating) -> dict[str, tuple[float, float]]:
        """Return Γ bounds implied by the minimum thickness when the smallest period is fixed.

        For a periodic multilayer with fixed ``d`` this is Γ ∈ [t_min/d, 1 − t_min/d]; for a
        depth-graded multilayer with fixed ``d_min`` the same holds for ``gamma_top`` and
        ``gamma_bottom`` with d = d_min. The hint assumes the constraint applies to both the
        absorber and the spacer material and that there are no interlayers.

        Args:
            coating: The coating.

        Returns:
            Bounds by parameter name, possibly empty.
        """
        gamma_names: tuple[str, ...]
        if isinstance(coating, Periodic):
            period_name, gamma_names = "d", ("gamma",)
        elif isinstance(coating, DepthGraded):
            period_name, gamma_names = "d_min", ("gamma_top", "gamma_bottom")
        else:
            return {}
        roles = coating.materials
        if not (self._applies(roles["H"].name) and self._applies(roles["L"].name)):
            return {}
        period = coating.parameters[period_name]
        if period.vary or period.expr is not None or period.value <= 0:
            return {}
        fraction = self.minimum / period.value
        if fraction > 0.5:
            return {}
        return {name: (fraction, 1.0 - fraction) for name in gamma_names}


@register_constraint
@dataclass(frozen=True)
class MaxTotalThickness(Constraint):
    """The total coating thickness must not exceed a maximum.

    Attributes:
        maximum: Maximum total thickness in nm.
    """

    type_name: ClassVar[str] = "max_total_thickness"
    maximum: float = field(default=1000.0, metadata={"unit": "nm"})

    def __post_init__(self) -> None:
        if not self.maximum > 0:
            raise ValueError("maximum thickness must be positive")

    def evaluate(self, stack: Stack, process: DepositionProcess | None = None) -> ConstraintResult:
        """Check the total thickness.

        Args:
            stack: The stack.
            process: Not used.

        Returns:
            The result; the violation is the relative excess.
        """
        total = stack.total_thickness
        violation = max(0.0, (total - self.maximum) / self.maximum)
        return ConstraintResult(
            violation == 0.0, violation, f"total thickness {total:.4g} nm (max {self.maximum:g} nm)"
        )


@register_constraint
@dataclass(frozen=True)
class MaxLayers(Constraint):
    """The number of deposited layers must not exceed a maximum.

    Attributes:
        maximum: Maximum number of layers of non-zero thickness.
    """

    type_name: ClassVar[str] = "max_layers"
    maximum: int = field(default=1000, metadata={"label": "maximum layers"})

    def __post_init__(self) -> None:
        if self.maximum < 1:
            raise ValueError("maximum number of layers must be at least 1")

    def evaluate(self, stack: Stack, process: DepositionProcess | None = None) -> ConstraintResult:
        """Check the number of layers.

        Args:
            stack: The stack.
            process: Not used.

        Returns:
            The result; the violation is the relative excess.
        """
        count = sum(1 for layer in stack.layers if layer.thickness > 0)
        violation = max(0.0, (count - self.maximum) / self.maximum)
        return ConstraintResult(violation == 0.0, violation, f"{count} layers (max {self.maximum})")


@register_constraint
@dataclass(frozen=True)
class MaxCoatingTime(Constraint):
    """The coating time must not exceed a maximum.

    Attributes:
        maximum: Maximum coating time in s.
    """

    type_name: ClassVar[str] = "max_coating_time"
    maximum: float = field(default=86400.0, metadata={"unit": "h", "scale": 1 / 3600})

    def __post_init__(self) -> None:
        if not self.maximum > 0:
            raise ValueError("maximum coating time must be positive")

    def evaluate(self, stack: Stack, process: DepositionProcess | None = None) -> ConstraintResult:
        """Check the coating time.

        Args:
            stack: The stack.
            process: The deposition process.

        Returns:
            The result; the violation is the relative excess.

        Raises:
            ValueError: If no process is given.
        """
        if process is None:
            raise ValueError("MaxCoatingTime requires a deposition process")
        time = process.time(stack)
        violation = max(0.0, (time - self.maximum) / self.maximum)
        return ConstraintResult(
            violation == 0.0,
            violation,
            f"coating time {time / 3600:.3g} h (max {self.maximum / 3600:.3g} h)",
        )


register_component_registry("constraint", constraints)
