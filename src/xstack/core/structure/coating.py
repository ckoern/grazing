"""Coatings: parameterized generators of layer sequences."""

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, ClassVar, Self

from xstack.core.materials import VACUUM, Material
from xstack.core.parameters import Parameter, ParameterSet
from xstack.core.plugins import Registry
from xstack.core.serialization import register_component_registry, register_serializable
from xstack.core.structure.stack import Layer, Stack

__all__ = [
    "DENSITY_RANGE",
    "SUBSTRATE",
    "Coating",
    "CoatingBuildError",
    "Mirror",
    "coatings",
    "density_parameter_name",
    "register_coating",
]


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

    def material_map(self) -> dict[str, Material]:
        """Return the materials of all roles, including those of nested coatings.

        Returns:
            Materials by role; roles of nested coatings are prefixed with their name, e.g.
            ``"cap.H"``.
        """
        return self.materials

    def with_materials(self, materials: Mapping[str, Material]) -> Self:
        """Return a copy with other materials for some roles.

        Args:
            materials: New materials by role, as returned by `material_map`.

        Returns:
            The modified coating.
        """
        roles = tuple((r, materials.get(r, m)) for r, m in self.material_roles)
        return replace(self, material_roles=roles)

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

    def without_material(self, role: str) -> Self:
        """Return a copy without the material of an optional role (e.g. an interlayer).

        Args:
            role: Optional role name.

        Returns:
            The modified coating; unchanged if the role has no material.

        Raises:
            ValueError: If the role is required.
        """
        if role in self.roles:
            raise ValueError(f"{type(self).__name__} requires a material for role {role!r}")
        return replace(
            self, material_roles=tuple((r, m) for r, m in self.material_roles if r != role)
        )

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


SUBSTRATE = "substrate"
"""Role name of the substrate in `Mirror.density_roles`."""

DENSITY_RANGE = (0.5, 1.1)
"""Default bounds of density parameters relative to the material's density."""


def density_parameter_name(role: str) -> str:
    """Return the name of the density parameter of a material role.

    Args:
        role: Material role, e.g. ``"H"``, ``"cap.L"`` or ``"substrate"``.

    Returns:
        ``"rho_<role>"``; for roles of nested coatings the prefix is kept, e.g. ``"cap.rho_L"``.

    Examples:
        >>> density_parameter_name("H"), density_parameter_name("cap.L")
        ('rho_H', 'cap.rho_L')
    """
    prefix, _, last = role.rpartition(".")
    return f"{prefix}.rho_{last}" if prefix else f"rho_{last}"


@dataclass(frozen=True)
class Mirror:
    """A coating on a substrate, i.e. everything needed to build a `Stack`.

    The density of a material role (or of the substrate) can be made a parameter with
    `with_density_parameters`, e.g. to fit the density of a deposited film, which is often lower
    than the bulk density. A density parameter replaces the density of that role's material.

    Args:
        coating: The coating.
        substrate: Substrate material.
        substrate_roughness: Width σ in nm of the interface at the top of the substrate.
        ambient: Ambient material, vacuum by default.
        densities: Density parameters in g/cm³, named by `density_parameter_name`.

    Raises:
        ValueError: If a density parameter does not belong to a material role.
    """

    coating: Coating
    substrate: Material
    substrate_roughness: float = 0.0
    ambient: Material = VACUUM
    densities: ParameterSet = field(default_factory=ParameterSet)

    def __post_init__(self) -> None:
        known = {density_parameter_name(role) for role in self.density_roles()}
        unknown = sorted(set(self.densities) - known)
        if unknown:
            raise ValueError(f"density parameters {unknown} belong to no material role")

    def density_roles(self) -> dict[str, Material]:
        """Return the materials whose density can be a parameter.

        Returns:
            Materials by role: the coating's roles (see `Coating.material_map`) and
            `SUBSTRATE`.
        """
        return {**self.coating.material_map(), SUBSTRATE: self.substrate}

    @property
    def parameters(self) -> ParameterSet:
        """The coating parameters followed by the density parameters."""
        if not self.densities:
            return self.coating.parameters
        return self.coating.parameters.merge(self.densities)

    def stack(self, values: Mapping[str, float] | None = None) -> Stack:
        """Build the stack for the current or the given parameter values.

        Args:
            values: Parameter values overriding the current ones, by name (coating and
                density parameters).

        Returns:
            The stack.

        Raises:
            CoatingBuildError: If the values do not describe a valid layer sequence.
            KeyError: If `values` contains names that are not parameters of the mirror.
        """
        unknown = sorted(set(values or {}) - set(self.parameters))
        if unknown:
            raise KeyError(f"unknown parameters {unknown}")
        coating, substrate = self.coating, self.substrate
        if self.densities:
            density = self.densities.value_dict()
            if any(p.expr is not None for p in self.densities.values()):
                try:
                    density = self.parameters.with_values(
                        {n: v for n, v in (values or {}).items() if n in self.parameters}
                    ).evaluate()
                except ValueError as exc:
                    raise CoatingBuildError(f"density: {exc}") from None
            density |= {n: v for n, v in (values or {}).items() if n in self.densities}
            changed = {}
            for role, material in self.density_roles().items():
                name = density_parameter_name(role)
                if name in self.densities:
                    if not density[name] > 0.0:
                        raise CoatingBuildError(f"density {name} must be positive")
                    changed[role] = replace(material, density=density[name])
            substrate = changed.pop(SUBSTRATE, substrate)
            coating = coating.with_materials(changed)
        coating_values = None
        if values is not None:
            coating_values = {n: v for n, v in values.items() if n in self.coating.parameters}
        return Stack(
            coating.layers(coating_values), substrate, self.substrate_roughness, self.ambient
        )

    def with_coating(self, coating: Coating) -> "Mirror":
        """Return a copy with a different coating.

        Density parameters of roles that the new coating does not have are removed.

        Args:
            coating: The new coating.

        Returns:
            The modified mirror.
        """
        roles = {density_parameter_name(r) for r in [*coating.material_map(), SUBSTRATE]}
        densities = ParameterSet(p for n, p in self.densities.items() if n in roles)
        return replace(self, coating=coating, densities=densities)

    def with_parameters(self, parameters: ParameterSet) -> "Mirror":
        """Return a copy with different coating and density parameters.

        Args:
            parameters: The parameters, with the names of `parameters`.

        Returns:
            The modified mirror.

        Raises:
            ValueError: If the parameter names differ.
        """
        coating = ParameterSet(p for n, p in parameters.items() if n not in self.densities)
        densities = ParameterSet(p for n, p in parameters.items() if n in self.densities)
        if set(densities) != set(self.densities):
            raise ValueError("parameter names must not change")
        return replace(self, coating=self.coating.with_parameters(coating), densities=densities)

    def with_parameter(self, name: str, **changes: Any) -> "Mirror":
        """Return a copy with one coating or density parameter modified.

        Args:
            name: Parameter name.
            **changes: New values for fields of the `Parameter`.

        Returns:
            The modified mirror.
        """
        if name in self.densities:
            return replace(self, densities=self.densities.replace(name, **changes))
        return replace(self, coating=self.coating.with_parameter(name, **changes))

    def with_density_parameters(self, roles: Sequence[str] | None = None) -> "Mirror":
        """Return a copy with density parameters for some material roles.

        New parameters start at the material's density, are fixed and have the bounds
        `DENSITY_RANGE` relative to it; existing ones are kept.

        Args:
            roles: Material roles (see `density_roles`), or ``None`` for all roles.

        Returns:
            The modified mirror.

        Raises:
            KeyError: If a role is unknown.

        Examples:
            >>> from xstack.core.materials import get
            >>> from xstack.core.structure.coatings import Periodic
            >>> mirror = Mirror(Periodic.create(get("W"), get("Si")), get("Si"))
            >>> list(mirror.with_density_parameters(["H"]).densities)
            ['rho_H']
        """
        available = self.density_roles()
        chosen = list(available) if roles is None else list(roles)
        unknown = [r for r in chosen if r not in available]
        if unknown:
            raise KeyError(f"unknown material roles {unknown} (available: {list(available)})")
        lo, hi = DENSITY_RANGE
        params = list(self.densities.values())
        for role in chosen:
            name = density_parameter_name(role)
            if name in self.densities:
                continue
            material = available[role]
            params.append(
                Parameter(
                    name,
                    material.density,
                    min=lo * material.density,
                    max=hi * material.density,
                    unit="g/cm³",
                    description=f"Density of {material.name} ({role})",
                )
            )
        order = [density_parameter_name(r) for r in available]
        params.sort(key=lambda p: order.index(p.name))
        return replace(self, densities=ParameterSet(params))

    def without_density_parameters(self, roles: Sequence[str] | None = None) -> "Mirror":
        """Return a copy without the density parameters of some roles.

        The materials keep their own densities.

        Args:
            roles: Material roles, or ``None`` for all.

        Returns:
            The modified mirror.
        """
        names = None if roles is None else {density_parameter_name(r) for r in roles}
        kept = [] if names is None else [p for n, p in self.densities.items() if n not in names]
        return replace(self, densities=ParameterSet(kept))


register_serializable("Mirror", Mirror)
register_component_registry("coating", coatings)
