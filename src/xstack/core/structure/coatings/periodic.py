"""Periodic multilayer coating."""

from dataclasses import dataclass
from typing import ClassVar, Self

from xstack.core.materials import Material
from xstack.core.parameters import Parameter, ParameterSet, ParameterValue, resolve
from xstack.core.structure.coating import Coating, CoatingBuildError, register_coating
from xstack.core.structure.coatings.bilayer import (
    BILAYER_ROLES,
    INTERLAYER_PARAMETERS,
    INTERLAYER_ROLES,
    MAX_BILAYERS,
    MAX_THICKNESS,
    TopMaterial,
    active_interlayers,
    bilayer_layers,
    bilayer_parameters,
    check_interlayer_specs,
    set_interlayers,
)
from xstack.core.structure.stack import Layer

__all__ = ["Periodic"]


@register_coating
@dataclass(frozen=True)
class Periodic(Coating):
    """A periodic multilayer of N identical bilayers.

    Material roles ``"H"`` (absorber), ``"L"`` (spacer) and optionally the interlayers
    ``"interlayer_H_on_L"`` and ``"interlayer_L_on_H"`` (see
    `xstack.core.structure.coatings.bilayer`).

    Parameters: ``N`` (number of bilayers), ``d`` (period, nm), ``gamma`` (Γ = t_H / d) and the
    shared bilayer parameters ``sigma_H``, ``sigma_L``, ``t_interlayer_H_on_L`` and
    ``t_interlayer_L_on_H``.

    Attributes:
        top: Material of the bilayer facing the ambient medium.
    """

    type_name: ClassVar[str] = "periodic"
    roles: ClassVar[tuple[str, ...]] = BILAYER_ROLES
    optional_roles: ClassVar[tuple[str, ...]] = INTERLAYER_ROLES

    top: TopMaterial = "H"
    use_interlayers: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if any(role in self.materials for role in INTERLAYER_ROLES):
            object.__setattr__(self, "use_interlayers", True)

    def inactive_parameters(self) -> frozenset[str]:
        """Return the interlayer thicknesses if the coating has no interlayers.

        Returns:
            Parameter names.
        """
        return frozenset() if self.use_interlayers else frozenset(INTERLAYER_PARAMETERS)

    def with_interlayers(self, enabled: bool) -> Self:
        """Return a copy with interlayers switched on or off.

        Switching them off removes the interlayer materials and fixes the interlayer
        thicknesses at zero.

        Args:
            enabled: Whether the coating has interlayers.

        Returns:
            The modified coating.
        """
        return set_interlayers(self, enabled)

    @classmethod
    def create(
        cls,
        H: Material,
        L: Material,
        N: ParameterValue = 30,
        d: ParameterValue = 4.0,
        gamma: ParameterValue = 0.4,
        sigma_H: ParameterValue = 0.0,
        sigma_L: ParameterValue = 0.0,
        interlayers: dict[str, Material] | None = None,
        t_interlayer_H_on_L: ParameterValue = 0.0,
        t_interlayer_L_on_H: ParameterValue = 0.0,
        top: TopMaterial = "H",
        name: str = "periodic",
    ) -> "Periodic":
        """Create a periodic multilayer.

        Args:
            H: Absorber material.
            L: Spacer material.
            N: Number of bilayers.
            d: Period in nm.
            gamma: Absorber fraction Γ = t_H / d.
            sigma_H: Roughness at the top of the H layers in nm.
            sigma_L: Roughness at the top of the L layers in nm.
            interlayers: Interlayer materials by role (``"interlayer_H_on_L"``,
                ``"interlayer_L_on_H"``).
            t_interlayer_H_on_L: Thickness of the interlayer where H is deposited on L, in nm.
            t_interlayer_L_on_H: Thickness of the interlayer where L is deposited on H, in nm.
            top: Material of the bilayer facing the ambient medium.
            name: Coating name.

        Returns:
            The coating.
        """
        specs: dict[str, ParameterValue] = {
            "N": N,
            "d": d,
            "gamma": gamma,
            "sigma_H": sigma_H,
            "sigma_L": sigma_L,
            "t_interlayer_H_on_L": t_interlayer_H_on_L,
            "t_interlayer_L_on_H": t_interlayer_L_on_H,
        }
        templates = [
            Parameter(
                "N", 30, min=1, max=MAX_BILAYERS, integer=True, description="Number of bilayers"
            ),
            Parameter("d", 4.0, min=0.0, max=MAX_THICKNESS, unit="nm", description="Period"),
            Parameter("gamma", 0.4, min=0.0, max=1.0, description="Absorber fraction t_H/d"),
            *bilayer_parameters(),
        ]
        params = ParameterSet(resolve(t, specs[t.name]) for t in templates)
        check_interlayer_specs(params, interlayers)
        roles = (("H", H), ("L", L), *(interlayers or {}).items())
        return cls(name, roles, params, top=top)

    def _build(self, values: dict[str, float]) -> tuple[Layer, ...]:
        n = int(values["N"])
        if n < 1:
            raise CoatingBuildError(f"{self.name}: N must be at least 1")
        materials, values = active_interlayers(self.use_interlayers, self.materials, values)
        layers: list[Layer] = []
        for i in range(1, n + 1):
            layers += bilayer_layers(
                values["d"],
                values["gamma"],
                values,
                materials,
                self.top,
                f"{self.name}[{i}]",
                bottom=i == n,
            )
        return tuple(layers)
