"""Coatings without periodicity: a single layer and a sequence of N layers."""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

from xstack.core.materials import Material
from xstack.core.parameters import Parameter, ParameterSet, ParameterValue, resolve
from xstack.core.structure.coating import Coating, register_coating
from xstack.core.structure.coatings.bilayer import MAX_ROUGHNESS, MAX_THICKNESS
from xstack.core.structure.stack import Layer

__all__ = ["NLayer", "SingleLayer", "layer_labels"]


def _sanitize(name: str) -> str:
    """Turn a material name into a fragment of a parameter name."""
    label = re.sub(r"_+", "_", re.sub(r"\W", "_", name, flags=re.ASCII)).strip("_")
    if not label:
        return "layer"
    return f"m_{label}" if label[0].isdigit() else label


def layer_labels(materials: Sequence[Material]) -> list[str]:
    """Derive a unique label for every layer of a stack from its material.

    A material used once is labelled with its (sanitized) name, e.g. ``"Mo"``. A material used
    several times is labelled ``"<name>_<k>"`` with k counting its occurrences from the top,
    starting at 1, e.g. ``"Si_1"``, ``"Si_2"``. Characters that are not valid in identifiers
    become ``"_"`` and a leading digit gets the prefix ``"m_"``. If a label is already taken,
    a further ``"_<k>"`` is appended.

    Args:
        materials: Layer materials from the top.

    Returns:
        One label per layer; all labels are distinct identifiers without ``"__"``.
    """
    bases = [_sanitize(m.name) for m in materials]
    totals = {base: bases.count(base) for base in bases}
    labels: list[str | None] = [base if totals[base] == 1 else None for base in bases]
    used = {label for label in labels if label is not None}
    seen: dict[str, int] = {}
    for i, base in enumerate(bases):
        if labels[i] is not None:
            continue
        seen[base] = seen.get(base, 0) + 1
        label = f"{base}_{seen[base]}"
        extra = 1
        while label in used:
            extra += 1
            label = f"{base}_{seen[base]}_{extra}"
        used.add(label)
        labels[i] = label
    return [label for label in labels if label is not None]


@register_coating
@dataclass(frozen=True)
class SingleLayer(Coating):
    """A single layer, e.g. a total-reflection mirror coating.

    Material role ``"M"``. Parameters: ``t`` (thickness, nm) and ``sigma`` (roughness at the
    top of the layer, nm).
    """

    type_name: ClassVar[str] = "single_layer"
    roles: ClassVar[tuple[str, ...]] = ("M",)

    @classmethod
    def create(
        cls,
        material: Material,
        t: ParameterValue = 30.0,
        sigma: ParameterValue = 0.0,
        name: str = "layer",
    ) -> "SingleLayer":
        """Create a single-layer coating.

        Args:
            material: Layer material.
            t: Thickness in nm.
            sigma: Roughness at the top of the layer in nm.
            name: Coating name.

        Returns:
            The coating.
        """
        params = ParameterSet(
            [
                resolve(
                    Parameter(
                        "t", 30.0, min=0.0, max=MAX_THICKNESS, unit="nm", description="Thickness"
                    ),
                    t,
                ),
                resolve(
                    Parameter("sigma", 0.0, min=0.0, unit="nm", description="Roughness"), sigma
                ),
            ]
        )
        return cls(name, (("M", material),), params)

    def _build(self, values: dict[str, float]) -> tuple[Layer, ...]:
        return (Layer(self.material("M"), values["t"], values["sigma"], f"{self.name}.M"),)


@register_coating
@dataclass(frozen=True)
class NLayer(Coating):
    """A sequence of N layers of arbitrary materials without periodicity.

    Material roles ``"M1"`` … ``"MN"`` from the top. Each layer has the parameters ``t_<label>``
    (thickness, nm) and ``sigma_<label>`` (roughness at the top of the layer, nm), listed layer
    by layer from the top. The label is derived from the material by `layer_labels`: ``Mo`` for
    a material used once, ``Si_1``, ``Si_2``, … for a material used several times. The
    parameters of layer k are the ``2k-1``-th and ``2k``-th of the set, whatever their names.
    """

    type_name: ClassVar[str] = "n_layer"
    roles: ClassVar[tuple[str, ...]] = ()
    optional_roles: ClassVar[tuple[str, ...]] = tuple(f"M{i}" for i in range(1, 1000))

    def __post_init__(self) -> None:
        super().__post_init__()
        count = len(self.material_roles)
        expected = tuple(f"M{i}" for i in range(1, count + 1))
        if count == 0 or tuple(role for role, _ in self.material_roles) != expected:
            raise ValueError(f"NLayer roles must be {expected or ('M1', '...')} in order")

    @property
    def count(self) -> int:
        """Number of layers N."""
        return len(self.material_roles)

    @classmethod
    def create(
        cls,
        materials: Sequence[Material],
        thicknesses: Sequence[ParameterValue],
        roughnesses: Sequence[ParameterValue] | None = None,
        name: str = "layers",
    ) -> "NLayer":
        """Create an N-layer coating.

        Args:
            materials: Layer materials from the top.
            thicknesses: Layer thicknesses in nm from the top.
            roughnesses: Roughness at the top of each layer in nm. Defaults to zero.
            name: Coating name.

        Returns:
            The coating.

        Raises:
            ValueError: If the sequences have different lengths.
        """
        rough = list(roughnesses) if roughnesses is not None else [0.0] * len(materials)
        if not len(materials) == len(thicknesses) == len(rough):
            raise ValueError("materials, thicknesses and roughnesses must have equal length")
        params: list[Parameter] = []
        labels = layer_labels(materials)
        for i, (label, t, s) in enumerate(zip(labels, thicknesses, rough, strict=True), start=1):
            params.append(
                resolve(
                    Parameter(
                        f"t_{label}",
                        1.0,
                        min=0.0,
                        max=MAX_THICKNESS,
                        unit="nm",
                        description=f"Thickness of layer {i} ({label}, from the top)",
                    ),
                    t,
                )
            )
            params.append(
                resolve(
                    Parameter(
                        f"sigma_{label}",
                        0.0,
                        min=0.0,
                        max=MAX_ROUGHNESS,
                        unit="nm",
                        description=f"Roughness at the top of layer {i} ({label})",
                    ),
                    s,
                )
            )
        roles = tuple((f"M{i}", m) for i, m in enumerate(materials, start=1))
        return cls(name, roles, ParameterSet(params))

    def thickness_name(self, layer: int) -> str:
        """Name of the thickness parameter of a layer.

        Args:
            layer: 1-based layer index from the top.

        Returns:
            The parameter name.
        """
        return list(self.parameters)[2 * layer - 2]

    def roughness_name(self, layer: int) -> str:
        """Name of the roughness parameter of a layer.

        Args:
            layer: 1-based layer index from the top.

        Returns:
            The parameter name.
        """
        return list(self.parameters)[2 * layer - 1]

    def _build(self, values: dict[str, float]) -> tuple[Layer, ...]:
        names = list(self.parameters)
        return tuple(
            Layer(
                material,
                values[names[2 * i - 2]],
                values[names[2 * i - 1]],
                f"{self.name}.{role}",
            )
            for i, (role, material) in enumerate(self.material_roles, start=1)
        )
