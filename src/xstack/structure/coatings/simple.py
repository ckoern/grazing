"""Coatings without periodicity: a single layer and a sequence of N layers."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

from xstack.materials import Material
from xstack.parameters import Parameter, ParameterSet, ParameterValue, resolve
from xstack.structure.coating import Coating, register_coating
from xstack.structure.coatings.bilayer import MAX_ROUGHNESS, MAX_THICKNESS
from xstack.structure.stack import Layer

__all__ = ["NLayer", "SingleLayer"]


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

    Material roles ``"M1"`` … ``"MN"`` from the top. Parameters ``t1`` … ``tN`` (thickness, nm)
    and ``sigma1`` … ``sigmaN`` (roughness at the top of each layer, nm).
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
        for i, (t, s) in enumerate(zip(thicknesses, rough, strict=True), start=1):
            params.append(
                resolve(
                    Parameter(
                        f"t{i}",
                        1.0,
                        min=0.0,
                        max=MAX_THICKNESS,
                        unit="nm",
                        description=f"Thickness of layer {i} (from the top)",
                    ),
                    t,
                )
            )
            params.append(
                resolve(
                    Parameter(
                        f"sigma{i}",
                        0.0,
                        min=0.0,
                        max=MAX_ROUGHNESS,
                        unit="nm",
                        description=f"Roughness at the top of layer {i}",
                    ),
                    s,
                )
            )
        roles = tuple((f"M{i}", m) for i, m in enumerate(materials, start=1))
        return cls(name, roles, ParameterSet(params))

    def _build(self, values: dict[str, float]) -> tuple[Layer, ...]:
        return tuple(
            Layer(material, values[f"t{i}"], values[f"sigma{i}"], f"{self.name}.{role}")
            for i, (role, material) in enumerate(self.material_roles, start=1)
        )
