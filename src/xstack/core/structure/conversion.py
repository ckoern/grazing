"""Conversion between coating types and edits of layer stacks.

Converting a coating keeps the materials and the settings of parameters that exist in both the
old and the new coating.
"""

import dataclasses
from collections.abc import Collection, Sequence
from typing import Any

from xstack.core.materials import Material, get
from xstack.core.parameters import Parameter, ParameterSet, expression_names, rename_in_expression
from xstack.core.structure.coating import Coating
from xstack.core.structure.coatings.composite import Composite
from xstack.core.structure.coatings.graded import DepthGraded
from xstack.core.structure.coatings.periodic import Periodic
from xstack.core.structure.coatings.simple import NLayer, SingleLayer

__all__ = [
    "constant_gamma_changes",
    "convert_coating",
    "has_constant_gamma",
    "restack_layers",
    "unique_block_name",
]

FALLBACK_PERIOD = 4.0
"""Period d in nm assumed when a multilayer is converted without a ``d`` or ``d_max`` parameter."""

FALLBACK_GAMMA = 0.4
"""Γ assumed when a multilayer is converted without a ``gamma`` or ``gamma_top`` parameter."""

DEFAULT_LAYER_THICKNESS = 5.0
"""Thickness in nm of a layer of a layer stack that has no counterpart in the old coating."""

FALLBACK_ABSORBER = "W"
"""Material name used for the absorber role when the old coating has no materials."""

FALLBACK_SPACER = "Si"
"""Material name used for the spacer role when the old coating has no spacer material."""

_BLOCK_NAMES = {
    "periodic": "periodic",
    "depth_graded": "graded",
    "single_layer": "layer",
    "n_layer": "stack",
}

_INTERLAYER_ROLES = ("interlayer_H_on_L", "interlayer_L_on_H")


def unique_block_name(type_name: str, used: Collection[str]) -> str:
    """Return a name for a new block of a combined coating.

    Args:
        type_name: Registry name of the block's coating type.
        used: Names of the existing blocks.

    Returns:
        A short name after the type (``"periodic"``, ``"graded"``, ``"layer"``, ``"stack"``),
        with a number appended if it is already used.
    """
    base = _BLOCK_NAMES.get(type_name, "block")
    name, number = base, 2
    while name in used:
        name, number = f"{base}{number}", number + 1
    return name


def _carry_over(new: Coating, old: Coating) -> Coating:
    """Return `new` with the settings of parameters that also exist in `old`."""
    params = ParameterSet(old.parameters.get(name, p) for name, p in new.parameters.items())
    try:
        return new.with_parameters(params)
    except ValueError:
        return new


def _bilayer_stack(old: Periodic | DepthGraded) -> tuple[list[Material], list[float], list[float]]:
    """Return materials, thicknesses (nm) and roughnesses (nm) of one period, top layer first.

    For a depth-graded multilayer the period is the largest one (``d_max``) and Γ is that of
    the top bilayer.
    """
    roles = old.materials
    params = old.parameters
    d = (
        params["d"].value
        if "d" in params
        else params["d_max"].value
        if "d_max" in params
        else FALLBACK_PERIOD
    )
    gamma = (
        params["gamma"].value
        if "gamma" in params
        else params["gamma_top"].value
        if "gamma_top" in params
        else FALLBACK_GAMMA
    )
    thickness = {"H": gamma * d, "L": (1.0 - gamma) * d}
    sigma = {r: params[f"sigma_{r}"].value if f"sigma_{r}" in params else 0.0 for r in "HL"}
    order = ["L", "H"] if old.top == "L" else ["H", "L"]
    return (
        [roles[r] for r in order],
        [thickness[r] for r in order],
        [sigma[r] for r in order],
    )


def _to_n_layer(old: Coating, materials: Sequence[Material] | None) -> NLayer:
    """Create a layer stack from `old` or, if given, from the new `materials`.

    A multilayer yields one period. Other coatings yield a single layer. With new `materials`,
    existing layers keep their thickness and roughness and further layers get the default
    thickness and no roughness.
    """
    if isinstance(old, NLayer):
        mats = list(materials) if materials is not None else [m for _, m in old.material_roles]
        return NLayer.create(mats, [DEFAULT_LAYER_THICKNESS] * len(mats))
    if isinstance(old, Periodic | DepthGraded):
        mats, thicknesses, sigmas = _bilayer_stack(old)
    else:
        params = old.parameters
        mats = [next(iter(old.materials.values()), get(FALLBACK_ABSORBER))]
        thicknesses = [params["t"].value if "t" in params else DEFAULT_LAYER_THICKNESS]
        sigmas = [params["sigma"].value if "sigma" in params else 0.0]
    if materials is not None:
        mats = list(materials)
        thicknesses = (thicknesses + [DEFAULT_LAYER_THICKNESS] * len(mats))[: len(mats)]
        sigmas = (sigmas + [0.0] * len(mats))[: len(mats)]
    return NLayer.create(mats, thicknesses, sigmas)


def convert_coating(old: Coating, type_name: str, **options: object) -> Coating:
    """Create a coating of another type or with other options from an existing one.

    Materials and the settings of parameters with the same name are taken over. Converting a
    multilayer to a layer stack yields the two layers of one period (thicknesses Γ·d and
    (1 - Γ)·d); converting a layer stack to a multilayer uses the materials of its first two
    layers as absorber and spacer. Converting to ``"composite"`` makes the coating the only
    block of a combined coating; converting a combined coating to another type converts its
    first block.

    Args:
        old: The current coating.
        type_name: Registry name of the new coating type.
        **options: Options of the new type (``top``, ``law``, ``cap``, ``interlayers``, and
            for ``"n_layer"`` the ``materials`` of all layers from the top).

    Returns:
        The new coating.

    Raises:
        ValueError: If the type is not supported.
    """
    if type_name == "composite":
        if isinstance(old, Composite):
            return old
        return Composite.create([old.renamed(unique_block_name(old.type_name, ()))])
    if isinstance(old, Composite):
        old = old.block(old.block_names()[0])
    roles = old.materials
    first = next(iter(roles.values()), get(FALLBACK_ABSORBER))
    # Absorber and spacer of the old coating, whichever role names it uses.
    h = roles.get("H", roles.get("M1", roles.get("M", first)))
    low = roles.get("L", roles.get("M2", get(FALLBACK_SPACER)))
    interlayers = {r: m for r, m in roles.items() if r in _INTERLAYER_ROLES}
    if "interlayers" in options:
        interlayers = options.pop("interlayers")  # type: ignore[assignment]
    new: Coating
    if type_name == "periodic":
        new = Periodic.create(h, low, interlayers=interlayers, **options)  # type: ignore[arg-type]
    elif type_name == "depth_graded":
        new = DepthGraded.create(h, low, interlayers=interlayers, **options)  # type: ignore[arg-type]
    elif type_name == "single_layer":
        new = SingleLayer.create(h)
    elif type_name == "n_layer":
        if isinstance(old, NLayer) and "materials" not in options:
            return old
        new = _to_n_layer(old, options.get("materials"))  # type: ignore[arg-type]
    else:
        raise ValueError(f"coating type {type_name!r} cannot be created here")
    new = _carry_over(new, old)
    if isinstance(new, Periodic | DepthGraded) and isinstance(old, Periodic | DepthGraded):
        new = new.with_interlayers(old.use_interlayers)
    return new


def has_constant_gamma(coating: DepthGraded) -> bool:
    """Return whether all graded bilayers of a coating share one Γ.

    Args:
        coating: The depth-graded coating.

    Returns:
        True if ``gamma_bottom`` is tied to ``gamma_top`` by an expression.
    """
    expr = coating.parameters["gamma_bottom"].expr
    return expr is not None and expr.replace(" ", "") == "gamma_top"


def constant_gamma_changes(coating: DepthGraded, constant: bool) -> dict[str, Any]:
    """Return the changes of ``gamma_bottom`` that switch between one Γ and a linear Γ gradient.

    With a constant Γ, ``gamma_bottom`` follows ``gamma_top`` through an expression. Without,
    ``gamma_bottom`` starts at the current value, bounds and vary setting of ``gamma_top``.

    Args:
        coating: The current depth-graded coating.
        constant: Use one Γ for all graded bilayers.

    Returns:
        Keyword arguments for the fields of `xstack.core.parameters.Parameter` to change on
        ``gamma_bottom``.
    """
    if constant:
        return {"expr": "gamma_top", "vary": False}
    top = coating.parameters["gamma_top"]
    value = coating.parameters.evaluate()["gamma_top"]
    return {"expr": None, "value": value, "vary": top.vary, "min": top.min, "max": top.max}


def restack_layers(
    old: NLayer, materials: Sequence[Material], sources: Sequence[int | None]
) -> NLayer:
    """Return a layer stack with new materials whose layers take over old parameter settings.

    The parameters of `old` are expected to list the thickness and roughness of layer k as
    the parameters 2k-1 and 2k (see `NLayer.thickness_name` and `NLayer.roughness_name`).
    Expressions of taken-over parameters are rewritten to the new names. An expression that
    refers to an old layer that is not taken over is dropped and the parameter keeps its value.

    Args:
        old: The current layer stack.
        materials: Materials of the new stack from the top.
        sources: For each new layer the 1-based index of the old layer whose thickness and
            roughness settings it takes over, or None for a new layer.

    Returns:
        The new layer stack with the name of `old`.
    """
    new = NLayer.create(materials, [DEFAULT_LAYER_THICKNESS] * len(materials), name=old.name)
    valid = [s is not None and 2 * s <= len(old.parameters) for s in sources]
    renamed: dict[str, str] = {}
    for i, (source, ok) in enumerate(zip(sources, valid, strict=True), start=1):
        if source is not None and ok:
            renamed[old.thickness_name(source)] = new.thickness_name(i)
            renamed[old.roughness_name(source)] = new.roughness_name(i)
    old_names = set(old.parameters)

    def translate(expr: str | None) -> str | None:
        """Rewrite an expression to the new names; None if it refers to a removed layer."""
        if expr is None or any(n in old_names and n not in renamed for n in expression_names(expr)):
            return None
        return rename_in_expression(expr, lambda n: renamed.get(n, n))

    params: list[Parameter] = []
    for i, (source, ok) in enumerate(zip(sources, valid, strict=True), start=1):
        pairs = (
            (new.thickness_name(i), old.thickness_name),
            (new.roughness_name(i), old.roughness_name),
        )
        for new_name, old_name_of in pairs:
            template = new.parameters[new_name]
            if source is None or not ok:
                params.append(template)
                continue
            carried = old.parameters[old_name_of(source)]
            params.append(
                dataclasses.replace(
                    carried,
                    name=template.name,
                    description=template.description,
                    expr=translate(carried.expr),
                )
            )
    try:
        return new.with_parameters(ParameterSet(params))
    except ValueError:
        return new
