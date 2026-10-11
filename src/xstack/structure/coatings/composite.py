"""Composite coating made of several coatings stacked on top of each other."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import ClassVar

from xstack.parameters import ParameterSet
from xstack.structure.coating import Coating, register_coating
from xstack.structure.stack import Layer

__all__ = ["Composite"]


@register_coating
@dataclass(frozen=True)
class Composite(Coating):
    """Several coatings (blocks) stacked on top of each other.

    The first block faces the ambient medium. Parameters of each block appear in the composite
    with the block name as prefix, e.g. ``"cap.d"``. Use this for overcoats, adhesion or barrier
    layers, or for combining periodic and graded sections.

    Attributes:
        blocks: The blocks from the top. Their names must be unique.
    """

    type_name: ClassVar[str] = "composite"
    roles: ClassVar[tuple[str, ...]] = ()

    blocks: tuple[Coating, ...] = field(default=())

    def __post_init__(self) -> None:
        super().__post_init__()
        names = [b.name for b in self.blocks]
        if not names:
            raise ValueError("a composite coating needs at least one block")
        if len(set(names)) != len(names):
            raise ValueError(f"block names must be unique, got {names}")
        expected = ParameterSet().merge(*(b.parameters.prefixed(b.name) for b in self.blocks))
        if set(self.parameters) != set(expected):
            object.__setattr__(self, "parameters", expected)

    @classmethod
    def create(cls, blocks: Sequence[Coating], name: str = "composite") -> "Composite":
        """Create a composite coating.

        Args:
            blocks: The blocks from the top.
            name: Coating name.

        Returns:
            The coating.
        """
        return cls(name, (), ParameterSet(), blocks=tuple(blocks))

    def block(self, name: str) -> Coating:
        """Return a block by name, with the parameters currently set in the composite.

        Args:
            name: Block name.

        Returns:
            The block.

        Raises:
            KeyError: If there is no block with this name.
        """
        for b in self.blocks:
            if b.name == name:
                return b.with_parameters(self._block_parameters(b))
        raise KeyError(f"unknown block {name!r}")

    def _block_parameters(self, block: Coating) -> ParameterSet:
        prefix = block.name + "."
        updated = []
        for local_name in block.parameters:
            outer = self.parameters[prefix + local_name]
            expr = outer.expr.replace(prefix, "") if outer.expr is not None else None
            updated.append(outer.evolve(name=local_name, expr=expr))
        return ParameterSet(updated)

    def _build(self, values: dict[str, float]) -> tuple[Layer, ...]:
        layers: list[Layer] = []
        for b in self.blocks:
            prefix = b.name + "."
            local = {
                name[len(prefix) :]: v for name, v in values.items() if name.startswith(prefix)
            }
            layers += b._build(local)
        return tuple(layers)
