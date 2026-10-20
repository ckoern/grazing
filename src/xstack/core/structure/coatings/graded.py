"""Depth-graded multilayer coating and grading laws."""

from dataclasses import dataclass
from typing import ClassVar, Protocol, Self

import numpy as np
from numpy.typing import NDArray

from xstack.core.materials import Material
from xstack.core.parameters import Parameter, ParameterSet, ParameterValue, resolve
from xstack.core.plugins import Registry
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

__all__ = ["DepthGraded", "GradingLaw", "LinearLaw", "PowerLaw", "grading_laws"]


class GradingLaw(Protocol):
    """A rule for the period of each bilayer of a depth-graded multilayer."""

    @property
    def name(self) -> str:
        """Registry name of the law."""
        ...

    def parameters(self) -> list[Parameter]:
        """Return the templates of the parameters of the law."""
        ...

    def periods(self, count: int, values: dict[str, float]) -> NDArray[np.float64]:
        """Return the periods of all bilayers.

        Args:
            count: Number of bilayers N.
            values: Evaluated coating parameters.

        Returns:
            Periods in nm, shape (N,), from the top bilayer (index 1) to the bottom one.

        Raises:
            CoatingBuildError: If the parameter values are invalid.
        """
        ...


def _check_range(d_min: float, d_max: float) -> None:
    if not 0.0 < d_min <= d_max:
        raise CoatingBuildError(f"periods require 0 < d_min <= d_max, got {d_min:g}, {d_max:g}")


class PowerLaw:
    """Power-law grading d_i = a / (b + i)^c, i = 1 … N from the top.

    The constants a and b follow from d_1 = ``d_max`` and d_N = ``d_min``; the exponent is the
    parameter ``c`` (Joensen et al., Proc. SPIE 1736 (1993)).
    """

    @property
    def name(self) -> str:
        """Registry name of the law."""
        return "power_law"

    def parameters(self) -> list[Parameter]:
        """Return the templates of the parameters of the law."""
        return [
            Parameter(
                "d_min",
                3.0,
                min=0.0,
                max=MAX_THICKNESS,
                unit="nm",
                description="Period of the bottom bilayer",
            ),
            Parameter(
                "d_max",
                10.0,
                min=0.0,
                max=MAX_THICKNESS,
                unit="nm",
                description="Period of the top bilayer",
            ),
            Parameter("c", 0.25, min=1e-3, max=10.0, description="Power-law exponent"),
        ]

    def periods(self, count: int, values: dict[str, float]) -> NDArray[np.float64]:
        """Return the periods of all bilayers, from the top.

        Args:
            count: Number of bilayers N.
            values: Evaluated parameters containing ``d_min``, ``d_max`` and ``c``.

        Returns:
            Periods in nm, shape (N,).

        Raises:
            CoatingBuildError: If the parameter values are invalid.
        """
        d_min, d_max, c = values["d_min"], values["d_max"], values["c"]
        _check_range(d_min, d_max)
        if c <= 0:
            raise CoatingBuildError(f"power-law exponent must be positive, got {c:g}")
        if count == 1 or d_min == d_max:
            return np.full(count, d_max)
        k = (d_min / d_max) ** (1.0 / c)
        b = (count * k - 1.0) / (1.0 - k)
        a = d_max * (b + 1.0) ** c
        i = np.arange(1, count + 1, dtype=np.float64)
        return np.asarray(a / (b + i) ** c, dtype=np.float64)


class LinearLaw:
    """Periods decreasing linearly from ``d_max`` (top) to ``d_min`` (bottom)."""

    @property
    def name(self) -> str:
        """Registry name of the law."""
        return "linear"

    def parameters(self) -> list[Parameter]:
        """Return the templates of the parameters of the law."""
        return [
            Parameter(
                "d_min",
                3.0,
                min=0.0,
                max=MAX_THICKNESS,
                unit="nm",
                description="Period of the bottom bilayer",
            ),
            Parameter(
                "d_max",
                10.0,
                min=0.0,
                max=MAX_THICKNESS,
                unit="nm",
                description="Period of the top bilayer",
            ),
        ]

    def periods(self, count: int, values: dict[str, float]) -> NDArray[np.float64]:
        """Return the periods of all bilayers, from the top.

        Args:
            count: Number of bilayers N.
            values: Evaluated parameters containing ``d_min`` and ``d_max``.

        Returns:
            Periods in nm, shape (N,).

        Raises:
            CoatingBuildError: If the parameter values are invalid.
        """
        d_min, d_max = values["d_min"], values["d_max"]
        _check_range(d_min, d_max)
        return np.linspace(d_max, d_min, count)


grading_laws: Registry[GradingLaw] = Registry("grading law")
"""Registry of grading laws for depth-graded multilayers."""

grading_laws.register("power_law", PowerLaw())
grading_laws.register("linear", LinearLaw())


@register_coating
@dataclass(frozen=True)
class DepthGraded(Coating):
    """A depth-graded multilayer (supermirror) with an optional cap bilayer.

    Material roles ``"H"`` (absorber), ``"L"`` (spacer) and optionally the interlayers
    ``"interlayer_H_on_L"`` and ``"interlayer_L_on_H"`` (see
    `xstack.core.structure.coatings.bilayer`).

    Parameters:

    * ``N``: number of graded bilayers,
    * the parameters of the grading law (``d_min``, ``d_max`` and, for the power law, ``c``),
    * ``gamma_top``, ``gamma_bottom``: Γ of the top and bottom graded bilayer, linearly
      interpolated in between (constant Γ if equal),
    * ``cap_d``, ``cap_gamma``: period and Γ of the cap bilayer, present only if `cap` is set,
    * the shared bilayer parameters ``sigma_H``, ``sigma_L``, ``t_interlayer_H_on_L`` and
      ``t_interlayer_L_on_H``.

    Attributes:
        law: Name of the grading law in `grading_laws`.
        cap: Whether a cap bilayer with independent period and Γ sits on top.
        top: Material of each bilayer facing the ambient medium.
    """

    type_name: ClassVar[str] = "depth_graded"
    roles: ClassVar[tuple[str, ...]] = BILAYER_ROLES
    optional_roles: ClassVar[tuple[str, ...]] = INTERLAYER_ROLES

    law: str = "power_law"
    cap: bool = False
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
        N: ParameterValue = 100,
        gamma_top: ParameterValue = 0.4,
        gamma_bottom: ParameterValue | None = None,
        law: str = "power_law",
        cap: bool = False,
        cap_d: ParameterValue = 15.0,
        cap_gamma: ParameterValue = 0.4,
        sigma_H: ParameterValue = 0.0,
        sigma_L: ParameterValue = 0.0,
        interlayers: dict[str, Material] | None = None,
        t_interlayer_H_on_L: ParameterValue = 0.0,
        t_interlayer_L_on_H: ParameterValue = 0.0,
        top: TopMaterial = "H",
        name: str = "graded",
        **law_parameters: ParameterValue,
    ) -> "DepthGraded":
        """Create a depth-graded multilayer.

        Args:
            H: Absorber material.
            L: Spacer material.
            N: Number of graded bilayers.
            gamma_top: Γ of the top graded bilayer.
            gamma_bottom: Γ of the bottom graded bilayer. Defaults to an expression that ties it
                to ``gamma_top`` (constant Γ).
            law: Name of the grading law.
            cap: Add a cap bilayer on top.
            cap_d: Period of the cap bilayer in nm.
            cap_gamma: Γ of the cap bilayer.
            sigma_H: Roughness at the top of the H layers in nm.
            sigma_L: Roughness at the top of the L layers in nm.
            interlayers: Interlayer materials by role (``"interlayer_H_on_L"``,
                ``"interlayer_L_on_H"``).
            t_interlayer_H_on_L: Thickness of the interlayer where H is deposited on L, in nm.
            t_interlayer_L_on_H: Thickness of the interlayer where L is deposited on H, in nm.
            top: Material of each bilayer facing the ambient medium.
            name: Coating name.
            **law_parameters: Parameters of the grading law, e.g. ``d_min``, ``d_max``, ``c``.

        Returns:
            The coating.

        Raises:
            KeyError: If the grading law is unknown.
            ValueError: If a law parameter is unknown.
        """
        grading = grading_laws.get(law)
        law_templates = grading.parameters()
        unknown = set(law_parameters) - {t.name for t in law_templates}
        if unknown:
            raise ValueError(f"unknown parameters for grading law {law!r}: {sorted(unknown)}")
        specs: dict[str, ParameterValue] = {
            "N": N,
            "gamma_top": gamma_top,
            "gamma_bottom": "gamma_top" if gamma_bottom is None else gamma_bottom,
            "cap_d": cap_d,
            "cap_gamma": cap_gamma,
            "sigma_H": sigma_H,
            "sigma_L": sigma_L,
            "t_interlayer_H_on_L": t_interlayer_H_on_L,
            "t_interlayer_L_on_H": t_interlayer_L_on_H,
            **law_parameters,
        }
        templates = [
            Parameter(
                "N",
                100,
                min=1,
                max=MAX_BILAYERS,
                integer=True,
                description="Number of graded bilayers",
            ),
            *law_templates,
            Parameter("gamma_top", 0.4, min=0.0, max=1.0, description="Gamma of top bilayer"),
            Parameter("gamma_bottom", 0.4, min=0.0, max=1.0, description="Gamma of bottom bilayer"),
        ]
        if cap:
            templates += [
                Parameter(
                    "cap_d",
                    15.0,
                    min=0.0,
                    max=MAX_THICKNESS,
                    unit="nm",
                    description="Cap bilayer period",
                ),
                Parameter("cap_gamma", 0.4, min=0.0, max=1.0, description="Cap bilayer gamma"),
            ]
        templates += bilayer_parameters()
        params = ParameterSet(
            resolve(t, specs[t.name]) if t.name in specs else t for t in templates
        )
        check_interlayer_specs(params, interlayers)
        roles = (("H", H), ("L", L), *(interlayers or {}).items())
        return cls(name, roles, params, law=law, cap=cap, top=top)

    def periods(self, values: dict[str, float] | None = None) -> NDArray[np.float64]:
        """Return the periods of the graded bilayers (without cap), from the top.

        Args:
            values: Parameter values overriding the current ones.

        Returns:
            Periods in nm, shape (N,).
        """
        resolved = (
            self.parameters if values is None else self.parameters.with_values(values)
        ).evaluate()
        return grading_laws.get(self.law).periods(round(resolved["N"]), resolved)

    def gammas(self, values: dict[str, float] | None = None) -> NDArray[np.float64]:
        """Return Γ of the graded bilayers (without cap), from the top.

        Args:
            values: Parameter values overriding the current ones.

        Returns:
            Γ values, shape (N,).
        """
        resolved = (
            self.parameters if values is None else self.parameters.with_values(values)
        ).evaluate()
        return np.linspace(resolved["gamma_top"], resolved["gamma_bottom"], round(resolved["N"]))

    def _build(self, values: dict[str, float]) -> tuple[Layer, ...]:
        n = int(values["N"])
        if n < 1:
            raise CoatingBuildError(f"{self.name}: N must be at least 1")
        materials, values = active_interlayers(self.use_interlayers, self.materials, values)
        periods = grading_laws.get(self.law).periods(n, values)
        gammas = np.linspace(values["gamma_top"], values["gamma_bottom"], n)
        layers: list[Layer] = []
        if self.cap:
            layers += bilayer_layers(
                values["cap_d"],
                values["cap_gamma"],
                values,
                materials,
                self.top,
                f"{self.name}.cap",
            )
        for i in range(n):
            layers += bilayer_layers(
                float(periods[i]),
                float(gammas[i]),
                values,
                materials,
                self.top,
                f"{self.name}[{i + 1}]",
                bottom=i == n - 1,
            )
        return tuple(layers)
