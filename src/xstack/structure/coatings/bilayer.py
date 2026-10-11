"""Common definitions for coatings made of absorber/spacer bilayers.

A bilayer of period d consists of an absorber layer (role ``"H"``, high Z) of thickness Γ·d,
a spacer layer (role ``"L"``, low Z) and optional interlayers that form at the interfaces:

* role ``"HonL"``: interlayer between an H layer and the L layer below it,
* role ``"LonH"``: interlayer between an L layer and the H layer below it.

Interlayers count towards the period: d = t_H + t_L + t_HonL + t_LonH, with t_H = Γ·d. The
interlayer below the lowest bilayer, which would face the substrate, is omitted.

Roughness parameters refer to the interface at the top of a layer: ``sigma_H`` for the top of
every H layer, ``sigma_L`` for the top of every L layer. Interlayers inherit the roughness of
the interface they replace (``sigma_L`` for ``HonL``, ``sigma_H`` for ``LonH``).
"""

from collections.abc import Mapping
from typing import Literal

from xstack.materials import Material
from xstack.parameters import Parameter
from xstack.structure.coating import CoatingBuildError
from xstack.structure.stack import Layer

__all__ = [
    "BILAYER_ROLES",
    "INTERLAYER_ROLES",
    "TopMaterial",
    "bilayer_layers",
    "bilayer_parameters",
]

BILAYER_ROLES = ("H", "L")
INTERLAYER_ROLES = ("HonL", "LonH")

type TopMaterial = Literal["H", "L"]
"""Which material of a bilayer faces the ambient medium."""


def bilayer_parameters() -> list[Parameter]:
    """Return the templates of the roughness and interlayer parameters shared by bilayers."""
    return [
        Parameter("sigma_H", 0.0, min=0.0, unit="nm", description="Roughness at top of H layers"),
        Parameter("sigma_L", 0.0, min=0.0, unit="nm", description="Roughness at top of L layers"),
        Parameter("t_HonL", 0.0, min=0.0, unit="nm", description="Thickness of HonL interlayers"),
        Parameter("t_LonH", 0.0, min=0.0, unit="nm", description="Thickness of LonH interlayers"),
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
        values: Evaluated parameters containing ``sigma_H``, ``sigma_L``, ``t_HonL`` and
            ``t_LonH``.
        materials: Materials by role; interlayer roles are optional.
        top: Material facing the ambient medium.
        tag: Prefix for the layer tags, e.g. ``"graded[3]"``.
        bottom: Whether this is the lowest bilayer of the coating. Its trailing interlayer,
            which would face the substrate, is omitted.

    Returns:
        Two to four layers.

    Raises:
        CoatingBuildError: If the thicknesses would be negative.
    """
    t_honl = values["t_HonL"] if "HonL" in materials else 0.0
    t_lonh = values["t_LonH"] if "LonH" in materials else 0.0
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
    honl = Layer(materials["HonL"], t_honl, values["sigma_L"], f"{tag}.HonL") if t_honl else None
    lonh = Layer(materials["LonH"], t_lonh, values["sigma_H"], f"{tag}.LonH") if t_lonh else None
    # H above L needs the HonL interlayer between them; L above H needs LonH.
    sequence = [h, honl, low, lonh] if top == "H" else [low, lonh, h, honl]
    if bottom:
        sequence = sequence[:-1]
    return [layer for layer in sequence if layer is not None]
