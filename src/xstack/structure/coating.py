"""Coatings: parameterized generators of layer sequences."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, ClassVar, Self

from xstack.materials import VACUUM, Material
from xstack.parameters import ParameterSet
from xstack.plugins import Registry
from xstack.structure.stack import Layer, Stack

__all__ = ["Coating", "CoatingBuildError", "Mirror", "coatings", "register_coating"]


class CoatingBuildError(ValueError):
    """Raised when parameter values do not describe a physically valid layer sequence."""


@dataclass(frozen=True)
class Coating(ABC):
    """Base class of all coatings.

    A coating combines named material roles (e.g. ``"H"`` and ``"L"`` for the absorber and
    spacer of a multilayer) with a `ParameterSet` and generates the corresponding layers.
    Coatings are immutable; the ``with_*`` methods return modified copies.

    Subclasses define `type_name`, `roles` and `_build`, and are registered in `coatings`.

    Attributes:
        name: Name of the coating, used as namespace when coatings are combined.
        material_roles: Pairs of role name and material.
        parameters: The coating parameters.
    """

    type_name: ClassVar[str]
    roles: ClassVar[tuple[str, ...]]
    optional_roles: ClassVar[tuple[str, ...]] = ()

    name: str
    material_roles: tuple[tuple[str, Material], ...]
    parameters: ParameterSet = field(default_factory=ParameterSet)

    def __post_init__(self) -> None:
        if not self.name.isidentifier():
            raise ValueError(f"coating name {self.name!r} must be an identifier")
        given = dict(self.material_roles)
        missing = [r for r in self.roles if r not in given]
        if missing:
            raise ValueError(f"{type(self).__name__} requires materials for roles {missing}")
        unknown = [r for r in given if r not in self.roles + self.optional_roles]
        if unknown:
            raise ValueError(f"{type(self).__name__} has no material roles {unknown}")

    @property
    def materials(self) -> dict[str, Material]:
        """Materials by role."""
        return dict(self.material_roles)

    def material(self, role: str) -> Material:
        """Return the material of a role.

        Args:
            role: Role name.

        Returns:
            The material.

        Raises:
            KeyError: If the role has no material.
        """
        try:
            return self.materials[role]
        except KeyError:
            raise KeyError(f"{type(self).__name__} has no material for role {role!r}") from None

    def layers(self, values: Mapping[str, float] | None = None) -> tuple[Layer, ...]:
        """Generate the layers of the coating.

        Args:
            values: Parameter values that override the current ones, by parameter name, e.g.
                as proposed by an optimizer. Parameters with expressions are re-evaluated.

        Returns:
            Layers ordered from the top to the bottom.

        Raises:
            CoatingBuildError: If the values do not describe a valid layer sequence, including
                parameter expressions that cannot be evaluated or leave their bounds.
            KeyError: If `values` contains unknown names.
        """
        params = self.parameters if values is None else self.parameters.with_values(values)
        try:
            resolved = params.evaluate()
        except ValueError as exc:
            raise CoatingBuildError(f"{self.name}: {exc}") from None
        for name, p in params.items():
            if p.integer:
                resolved[name] = float(round(resolved[name]))
        return self._build(resolved)

    @abstractmethod
    def _build(self, values: dict[str, float]) -> tuple[Layer, ...]:
        """Generate layers from fully evaluated parameter values."""

    def with_parameters(self, parameters: ParameterSet) -> Self:
        """Return a copy with a different parameter set.

        Args:
            parameters: The new parameters. Must contain the same names.

        Returns:
            The modified coating.

        Raises:
            ValueError: If the parameter names differ.
        """
        if set(parameters) != set(self.parameters):
            raise ValueError("parameter names must not change")
        return replace(self, parameters=parameters)

    def with_values(self, **values: float) -> Self:
        """Return a copy with new parameter values.

        Args:
            **values: New values by parameter name.

        Returns:
            The modified coating.
        """
        return replace(self, parameters=self.parameters.with_values(values))

    def with_parameter(self, name: str, **changes: Any) -> Self:
        """Return a copy with one parameter modified.

        Args:
            name: Parameter name.
            **changes: New values for fields of the `Parameter`, e.g. ``vary=True, min=20``.

        Returns:
            The modified coating.
        """
        return replace(self, parameters=self.parameters.replace(name, **changes))

    def with_material(self, role: str, material: Material) -> Self:
        """Return a copy with the material of one role replaced.

        Args:
            role: Role name.
            material: New material.

        Returns:
            The modified coating.
        """
        if role not in self.roles + self.optional_roles:
            raise ValueError(f"{type(self).__name__} has no material role {role!r}")
        roles = dict(self.material_roles)
        roles[role] = material
        return replace(self, material_roles=tuple(roles.items()))

    def renamed(self, name: str) -> Self:
        """Return a copy with a different name.

        Args:
            name: The new name, an identifier.

        Returns:
            The renamed coating.
        """
        return replace(self, name=name)


coatings: Registry[type[Coating]] = Registry("coating type")
"""Registry of coating classes by `Coating.type_name`."""


def register_coating[C: type[Coating]](cls: C) -> C:
    """Class decorator that registers a coating class under its `type_name`.

    Args:
        cls: The coating class.

    Returns:
        The unchanged class.
    """
    coatings.register(cls.type_name, cls)
    return cls


@dataclass(frozen=True)
class Mirror:
    """A coating on a substrate, i.e. everything needed to build a `Stack`.

    Args:
        coating: The coating.
        substrate: Substrate material.
        substrate_roughness: Width σ in nm of the interface at the top of the substrate.
        ambient: Ambient material, vacuum by default.
    """

    coating: Coating
    substrate: Material
    substrate_roughness: float = 0.0
    ambient: Material = VACUUM

    @property
    def parameters(self) -> ParameterSet:
        """The coating parameters."""
        return self.coating.parameters

    def stack(self, values: Mapping[str, float] | None = None) -> Stack:
        """Build the stack for the current or the given parameter values.

        Args:
            values: Parameter values overriding the current ones, by name.

        Returns:
            The stack.

        Raises:
            CoatingBuildError: If the values do not describe a valid layer sequence.
        """
        return Stack(
            self.coating.layers(values), self.substrate, self.substrate_roughness, self.ambient
        )

    def with_coating(self, coating: Coating) -> "Mirror":
        """Return a copy with a different coating.

        Args:
            coating: The new coating.

        Returns:
            The modified mirror.
        """
        return replace(self, coating=coating)
