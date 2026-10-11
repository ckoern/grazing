"""Periodic multilayer coating."""

from dataclasses import dataclass
from typing import ClassVar

from xstack.materials import Material
from xstack.parameters import Parameter, ParameterSet, ParameterValue, resolve
from xstack.structure.coating import Coating, CoatingBuildError, register_coating
from xstack.structure.coatings.bilayer import (
    BILAYER_ROLES,
    INTERLAYER_ROLES,
    TopMaterial,
    bilayer_layers,
    bilayer_parameters,
)
from xstack.structure.stack import Layer

__all__ = ["Periodic"]


@register_coating
@dataclass(frozen=True)
class Periodic(Coating):
    """A periodic multilayer of N identical bilayers.

    Material roles ``"H"`` (absorber), ``"L"`` (spacer) and optionally the interlayers
    ``"HonL"`` and ``"LonH"`` (see `xstack.structure.coatings.bilayer`).

    Parameters: ``N`` (number of bilayers), ``d`` (period, nm), ``gamma`` (Γ = t_H / d) and the
    shared bilayer parameters ``sigma_H``, ``sigma_L``, ``t_HonL``, ``t_LonH``.

    Attributes:
        top: Material of the bilayer facing the ambient medium.
    """

    type_name: ClassVar[str] = "periodic"
    roles: ClassVar[tuple[str, ...]] = BILAYER_ROLES
    optional_roles: ClassVar[tuple[str, ...]] = INTERLAYER_ROLES

    top: TopMaterial = "H"

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
        t_HonL: ParameterValue = 0.0,
        t_LonH: ParameterValue = 0.0,
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
            interlayers: Interlayer materials by role (``"HonL"``, ``"LonH"``).
            t_HonL: Thickness of the HonL interlayers in nm.
            t_LonH: Thickness of the LonH interlayers in nm.
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
            "t_HonL": t_HonL,
            "t_LonH": t_LonH,
        }
        templates = [
            Parameter("N", 30, min=1, integer=True, description="Number of bilayers"),
            Parameter("d", 4.0, min=0.0, unit="nm", description="Period"),
            Parameter("gamma", 0.4, min=0.0, max=1.0, description="Absorber fraction t_H/d"),
            *bilayer_parameters(),
        ]
        params = ParameterSet(resolve(t, specs[t.name]) for t in templates)
        roles = (("H", H), ("L", L), *(interlayers or {}).items())
        return cls(name, roles, params, top=top)

    def _build(self, values: dict[str, float]) -> tuple[Layer, ...]:
        n = int(values["N"])
        if n < 1:
            raise CoatingBuildError(f"{self.name}: N must be at least 1")
        materials = self.materials
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
