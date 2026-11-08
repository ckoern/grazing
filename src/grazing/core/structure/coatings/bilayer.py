"""Common definitions for coatings made of absorber/spacer bilayers.

A bilayer of period d consists of an absorber layer (role ``"H"``, high Z) of thickness Γ·d,
a spacer layer (role ``"L"``, low Z) and optional interlayers that form at the interfaces:

* role ``"interlayer_H_on_L"``: interlayer that forms where H is deposited on L, i.e. between
  an H layer and the L layer below it (thickness parameter ``t_interlayer_H_on_L``),
* role ``"interlayer_L_on_H"``: interlayer that forms where L is deposited on H, i.e. between
  an L layer and the H layer below it (thickness parameter ``t_interlayer_L_on_H``).

Interlayers count towards the period: d = t_H + t_L + t_interlayer_H_on_L + t_interlayer_L_on_H,
with t_H = Γ·d. The interlayer below the lowest bilayer, which would face the substrate, is
omitted.

Roughness parameters refer to the interface at the top of a layer: ``sigma_H`` for the top of
every H layer, ``sigma_L`` for the top of every L layer. Interlayers inherit the roughness of
the interface they replace (``sigma_L`` for ``interlayer_H_on_L``, ``sigma_H`` for
``interlayer_L_on_H``).
"""

from collections.abc import Mapping
from dataclasses import replace
from typing import Literal

from grazing.core.materials import Material
from grazing.core.parameters import Parameter, ParameterSet
from grazing.core.structure.coating import Coating, CoatingBuildError
from grazing.core.structure.stack import Layer

__all__ = [
    "BILAYER_ROLES",
    "INTERLAYER_PARAMETERS",
    "INTERLAYER_ROLES",
    "MAX_BILAYERS",
    "MAX_ROUGHNESS",
    "MAX_THICKNESS",
    "TopMaterial",
    "active_interlayers",
    "bilayer_layers",
    "bilayer_parameters",
    "check_interlayer_specs",
    "set_interlayers",
]

MAX_THICKNESS = 100.0
"""Default upper bound of thickness and period parameters in nm."""

MAX_ROUGHNESS = 5.0
"""Default upper bound of roughness parameters in nm."""

MAX_BILAYERS = 1000
"""Default upper bound of the number of bilayers."""

BILAYER_ROLES = ("H", "L")
INTERLAYER_ROLES = ("interlayer_H_on_L", "interlayer_L_on_H")
INTERLAYER_PARAMETERS = ("t_interlayer_H_on_L", "t_interlayer_L_on_H")
"""Thickness parameters of the interlayers, in the order of `INTERLAYER_ROLES`."""

type TopMaterial = Literal["H", "L"]
"""Which material of a bilayer faces the ambient medium."""


def bilayer_parameters() -> list[Parameter]:
    """Return the templates of the roughness and interlayer parameters shared by bilayers."""
    return [
        Parameter(
            "sigma_H",
            0.0,
            min=0.0,
            max=MAX_ROUGHNESS,
            unit="nm",
            description="Roughness (interface width) at the top of every H layer",
        ),
        Parameter(
            "sigma_L",
            0.0,
            min=0.0,
            max=MAX_ROUGHNESS,
            unit="nm",
            description="Roughness (interface width) at the top of every L layer",
        ),
        Parameter(
            "t_interlayer_H_on_L",
            0.0,
            min=0.0,
            max=MAX_THICKNESS,
            unit="nm",
            description="Thickness of the interlayer where H is deposited on L",
        ),
        Parameter(
            "t_interlayer_L_on_H",
            0.0,
            min=0.0,
            max=MAX_THICKNESS,
            unit="nm",
            description="Thickness of the interlayer where L is deposited on H",
        ),
    ]


def bilayer_layers(
    period: float,
    gamma: float,
    values: Mapping[str, float],
    materials: Mapping[str, Material],
    top: TopMaterial,
    tag: str,
    bottom: bool = False,
) -> list[Layer]:
    """Return the layers of one bilayer, ordered from the top.

    Args:
        period: Period d in nm.
        gamma: Absorber fraction Γ = t_H / d.
        values: Evaluated parameters containing ``sigma_H``, ``sigma_L``,
            ``t_interlayer_H_on_L`` and ``t_interlayer_L_on_H``.
        materials: Materials by role; interlayer roles are optional.
        top: Material facing the ambient medium.
        tag: Prefix for the layer tags, e.g. ``"graded[3]"``.
        bottom: Whether this is the lowest bilayer of the coating. Its trailing interlayer,
            which would face the substrate, is omitted.

    Returns:
        Two to four layers.

    Raises:
        CoatingBuildError: If the thicknesses would be negative, or an interlayer has a positive
            thickness but no material.
    """
    for role in INTERLAYER_ROLES:
        if values[f"t_{role}"] > 0.0 and role not in materials:
            raise CoatingBuildError(
                f"{tag}: t_{role} is {values[f't_{role}']:g} nm, but the interlayer has no "
                "material; choose one or set the thickness to 0"
            )
    t_honl = values["t_interlayer_H_on_L"] if "interlayer_H_on_L" in materials else 0.0
    t_lonh = values["t_interlayer_L_on_H"] if "interlayer_L_on_H" in materials else 0.0
    t_h = gamma * period
    t_l = period - t_h - t_honl - t_lonh
    if not 0.0 <= gamma <= 1.0:
        raise CoatingBuildError(f"{tag}: gamma {gamma:g} outside [0, 1]")
    if period < 0 or t_l < 0:
        raise CoatingBuildError(
            f"{tag}: period {period:g} nm too small for gamma {gamma:g} and interlayers"
        )
    h = Layer(materials["H"], t_h, values["sigma_H"], f"{tag}.H")
    low = Layer(materials["L"], t_l, values["sigma_L"], f"{tag}.L")
    honl = (
        Layer(materials["interlayer_H_on_L"], t_honl, values["sigma_L"], f"{tag}.interlayer_H_on_L")
        if t_honl
        else None
    )
    lonh = (
        Layer(materials["interlayer_L_on_H"], t_lonh, values["sigma_H"], f"{tag}.interlayer_L_on_H")
        if t_lonh
        else None
    )
    # H above L needs the H-on-L interlayer between them; L above H needs the L-on-H one.
    sequence = [h, honl, low, lonh] if top == "H" else [low, lonh, h, honl]
    if bottom:
        sequence = sequence[:-1]
    return [layer for layer in sequence if layer is not None]


def active_interlayers(
    use: bool, materials: dict[str, Material], values: dict[str, float]
) -> tuple[dict[str, Material], dict[str, float]]:
    """Return the materials and values to build bilayers with, given the interlayer option.

    Args:
        use: Whether the coating uses interlayers.
        materials: Materials by role.
        values: Evaluated parameter values.

    Returns:
        Without interlayers, the materials without interlayer roles and the values with zero
        interlayer thicknesses; otherwise both unchanged.
    """
    if use:
        return materials, values
    kept = {r: m for r, m in materials.items() if r not in INTERLAYER_ROLES}
    return kept, values | dict.fromkeys(INTERLAYER_PARAMETERS, 0.0)


def check_interlayer_specs(
    parameters: ParameterSet, interlayers: dict[str, Material] | None
) -> None:
    """Check that interlayer thicknesses above zero come with interlayer materials.

    Args:
        parameters: The parameters of a new bilayer coating.
        interlayers: Interlayer materials by role, or ``None``.

    Raises:
        ValueError: If an interlayer thickness is positive but its role has no material.
    """
    given = interlayers or {}
    for role, name in zip(INTERLAYER_ROLES, INTERLAYER_PARAMETERS, strict=True):
        if parameters[name].value > 0.0 and role not in given:
            raise ValueError(f"{name} is positive, but no material is given for {role}")


def set_interlayers[C: Coating](coating: C, enabled: bool) -> C:
    """Switch the interlayers of a bilayer coating on or off.

    Switching them off removes the interlayer materials and fixes both interlayer thicknesses at
    zero.

    Args:
        coating: A bilayer coating with a ``use_interlayers`` field.
        enabled: Whether interlayers are used.

    Returns:
        The modified coating.
    """
    if enabled:
        return replace(coating, use_interlayers=True)  # type: ignore[call-arg]
    params = coating.parameters
    for name in INTERLAYER_PARAMETERS:
        params = params.replace(name, value=0.0, vary=False, expr=None)
    roles = tuple((r, m) for r, m in coating.material_roles if r not in INTERLAYER_ROLES)
    return replace(  # type: ignore[call-arg]
        coating, material_roles=roles, parameters=params, use_interlayers=False
    )
