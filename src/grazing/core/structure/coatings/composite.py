"""Composite coating made of several coatings stacked on top of each other."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import ClassVar, Self

from grazing.core.materials import Material
from grazing.core.parameters import (
    Parameter,
    ParameterSet,
    expression_names,
    rename_in_expression,
)
from grazing.core.structure.coating import Coating, register_coating
from grazing.core.structure.stack import Layer

__all__ = ["Composite"]


def _rename_prefix(expr: str, old: str, new: str) -> str:
    """Replace the namespace `old` by `new` in the names of an expression (`""` strips it)."""
    head = old + "."

    def rename(name: str) -> str:
        return new + name[len(head) :] if name.startswith(head) else name

    return rename_in_expression(expr, rename)


@register_coating
@dataclass(frozen=True)
class Composite(Coating):
    """Several coatings (blocks) stacked on top of each other.

    The first block faces the ambient medium. Parameters and material roles of each block
    appear in the composite with the block name as prefix, e.g. ``"cap.d"`` and ``"cap.H"``.
    Expressions may refer to parameters of other blocks. Use this for overcoats, adhesion or
    barrier layers, or for combining periodic, depth-graded and single-layer sections.

    When a composite is created from blocks whose parameter names differ from the given
    parameters, the given settings (value, vary, bounds, expression) are kept for every name
    that still exists; the other parameters come from the blocks.

    The parameters of the composite are authoritative; `block` returns a block with them. The
    editing methods (`with_block`, `insert_block`, `remove_block`, `move_block`,
    `rename_block`, `with_material`, `without_material`) keep the settings of all other
    parameters.

    Attributes:
        blocks: The blocks from the top. Their names must be unique.

    Examples:
        >>> from grazing.core.materials import get
        >>> from grazing.core.structure import Periodic, SingleLayer
        >>> cap = SingleLayer.create(get("B4C"), t=2.0, name="cap")
        >>> ml = Periodic.create(get("W"), get("Si"), N=20, d=4.0, name="ml")
        >>> coating = Composite.create([cap, ml])
        >>> coating.parameters["ml.d"].value
        4.0
        >>> sorted(coating.materials)
        ['cap.M', 'ml.H', 'ml.L']
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
        if list(self.parameters) != list(expected):
            kept = ParameterSet(self.parameters.get(name, p) for name, p in expected.items())
            object.__setattr__(self, "parameters", kept)

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

    @property
    def materials(self) -> dict[str, Material]:
        """Materials of the blocks' roles, prefixed with the block names."""
        return self.material_map()

    def inactive_parameters(self) -> frozenset[str]:
        """Return the inactive parameters of the blocks, prefixed with the block names.

        Returns:
            Parameter names.
        """
        return frozenset(f"{b.name}.{n}" for b in self.blocks for n in b.inactive_parameters())

    def material_map(self) -> dict[str, Material]:
        """Return the materials of the blocks' roles, prefixed with the block names.

        Returns:
            Materials by role, e.g. ``{"cap.H": …, "ml.L": …}``.
        """
        return {
            f"{b.name}.{role}": material
            for b in self.blocks
            for role, material in b.material_map().items()
        }

    def with_materials(self, materials: Mapping[str, Material]) -> Self:
        """Return a copy with other materials for some roles of the blocks.

        Args:
            materials: New materials by prefixed role, as returned by `material_map`.

        Returns:
            The modified coating.
        """
        blocks = []
        for b in self.blocks:
            prefix = b.name + "."
            own = {r[len(prefix) :]: m for r, m in materials.items() if r.startswith(prefix)}
            blocks.append(b.with_materials(own) if own else b)
        return replace(self, blocks=tuple(blocks))

    def with_material(self, role: str, material: Material) -> Self:
        """Return a copy with the material of one prefixed role replaced.

        Args:
            role: Role with the block name as prefix, e.g. ``"ml.H"``; may be an optional role
                of the block such as an interlayer.
            material: New material.

        Returns:
            The modified coating.

        Raises:
            KeyError: If there is no block with the prefix.
            ValueError: If the block has no such role.
        """
        name, local = self._split_role(role)
        return self._replace_block(name, self.block(name).with_material(local, material))

    def without_material(self, role: str) -> Self:
        """Return a copy without the material of an optional prefixed role (e.g. an interlayer).

        Args:
            role: Optional role with the block name as prefix.

        Returns:
            The modified coating.

        Raises:
            KeyError: If there is no block with the prefix.
            ValueError: If the role is required.
        """
        name, local = self._split_role(role)
        return self._replace_block(name, self.block(name).without_material(local))

    def block_names(self) -> tuple[str, ...]:
        """Return the names of the blocks from the top."""
        return tuple(b.name for b in self.blocks)

    def block(self, name: str) -> Coating:
        """Return a block by name, with the parameters currently set in the composite.

        Parameter names and expressions lose the block prefix; references to other blocks keep
        their prefixes.

        Args:
            name: Block name.

        Returns:
            The block.

        Raises:
            KeyError: If there is no block with this name.
        """
        block = self._block(name)
        return block.with_parameters(self._block_parameters(block))

    def with_block(self, name: str, block: Coating) -> Self:
        """Return a copy with one block replaced.

        The parameters of the new block (with its settings) replace those of the old block;
        the parameters of the other blocks keep their settings. The new block may be of another
        type; it gets the name of the replaced block.

        Args:
            name: Name of the block to replace.
            block: The new block, e.g. a modified copy of `block` (name).

        Returns:
            The modified coating.

        Raises:
            KeyError: If there is no block with this name.
        """
        self._block(name)
        return self._replace_block(name, block.renamed(name))

    def insert_block(self, index: int, block: Coating) -> Self:
        """Return a copy with an additional block.

        Args:
            index: Position from the top; ``len(blocks)`` appends at the bottom.
            block: The new block, with a name not used by the other blocks.

        Returns:
            The modified coating.

        Raises:
            ValueError: If the block name is already used.
        """
        if block.name in self.block_names():
            raise ValueError(f"a block named {block.name!r} already exists")
        blocks = list(self.blocks)
        blocks.insert(index, block)
        own = block.parameters.prefixed(block.name)
        return self._with_blocks(blocks, self.parameters.merge(own))

    def remove_block(self, name: str) -> Self:
        """Return a copy without a block.

        Expressions of other blocks that refer to the removed parameters are cleared; these
        parameters keep their current values.

        Args:
            name: Name of the block.

        Returns:
            The modified coating.

        Raises:
            KeyError: If there is no block with this name.
            ValueError: If it is the only block.
        """
        self._block(name)
        if len(self.blocks) == 1:
            raise ValueError("a composite coating needs at least one block")
        removed = {n for n in self.parameters if n.startswith(name + ".")}
        try:
            values = self.parameters.evaluate()
        except ValueError:
            values = {n: p.value for n, p in self.parameters.items()}
        kept = []
        for p in self.parameters.values():
            if p.name in removed:
                continue
            if p.expr is not None and expression_names(p.expr) & removed:
                p = p.evolve(expr=None, value=values[p.name])
            kept.append(p)
        blocks = [b for b in self.blocks if b.name != name]
        return self._with_blocks(blocks, ParameterSet(kept))

    def move_block(self, name: str, index: int) -> Self:
        """Return a copy with a block at another position.

        Args:
            name: Name of the block.
            index: New position from the top.

        Returns:
            The modified coating.

        Raises:
            KeyError: If there is no block with this name.
        """
        block = self._block(name)
        blocks = [b for b in self.blocks if b.name != name]
        blocks.insert(index, block)
        return self._with_blocks(blocks, self.parameters)

    def rename_block(self, name: str, new_name: str) -> Self:
        """Return a copy with a block renamed.

        The prefixes of its parameters are renamed, also in the expressions of all blocks.

        Args:
            name: Current name of the block.
            new_name: New name, an identifier not used by another block.

        Returns:
            The modified coating.

        Raises:
            KeyError: If there is no block with this name.
            ValueError: If the new name is invalid or already used.
        """
        block = self._block(name)
        if new_name == name:
            return self
        if not new_name.isidentifier():
            raise ValueError(f"block name {new_name!r} must be an identifier")
        if new_name in self.block_names():
            raise ValueError(f"a block named {new_name!r} already exists")
        params = [
            p.evolve(
                name=_rename_prefix(p.name, name, new_name + "."),
                expr=_rename_prefix(p.expr, name, new_name + ".") if p.expr is not None else None,
            )
            for p in self.parameters.values()
        ]
        blocks = [block.renamed(new_name) if b.name == name else b for b in self.blocks]
        return self._with_blocks(blocks, ParameterSet(params))

    def _with_blocks(self, blocks: Sequence[Coating], parameters: ParameterSet) -> Self:
        """Return a copy with other blocks, keeping the given parameter settings.

        The blocks of the copy carry the composite's parameter settings, so that a block taken
        out with `block` and put back with `with_block` gives an equal composite.
        """
        new = replace(self, blocks=tuple(blocks), parameters=parameters)
        return replace(new, blocks=tuple(new.block(b.name) for b in new.blocks))

    def _replace_block(self, name: str, block: Coating) -> Self:
        """Return a copy with the block `name` replaced; its parameters come from `block`."""
        prefix = name + "."
        outer = [p for p in self.parameters.values() if not p.name.startswith(prefix)]
        own = block.parameters.prefixed(block.name)
        blocks = [block if b.name == name else b for b in self.blocks]
        return self._with_blocks(blocks, ParameterSet([*outer, *own.values()]))

    def _block(self, name: str) -> Coating:
        for b in self.blocks:
            if b.name == name:
                return b
        raise KeyError(f"unknown block {name!r}")

    def _split_role(self, role: str) -> tuple[str, str]:
        name, _, local = role.partition(".")
        self._block(name)
        return name, local

    def _block_parameters(self, block: Coating) -> ParameterSet:
        prefix = block.name + "."
        updated: list[Parameter] = []
        for local_name in block.parameters:
            outer = self.parameters[prefix + local_name]
            expr = _rename_prefix(outer.expr, block.name, "") if outer.expr is not None else None
            updated.append(outer.evolve(name=local_name, expr=expr))
        return ParameterSet(updated)

    def layers_by_block(
        self, values: Mapping[str, float] | None = None
    ) -> dict[str, tuple[Layer, ...]]:
        """Generate the layers of every block.

        Args:
            values: Parameter values that override the current ones, by prefixed name.

        Returns:
            The layers of each block from the top, by block name in the order of the blocks.

        Raises:
            CoatingBuildError: If the values do not describe a valid layer sequence.
            KeyError: If `values` contains unknown names.
        """
        return self._build_blocks(self._resolved(values))

    def _build_blocks(self, values: dict[str, float]) -> dict[str, tuple[Layer, ...]]:
        result: dict[str, tuple[Layer, ...]] = {}
        for b in self.blocks:
            prefix = b.name + "."
            local = {
                name[len(prefix) :]: v for name, v in values.items() if name.startswith(prefix)
            }
            result[b.name] = b._build(local)
        return result

    def _build(self, values: dict[str, float]) -> tuple[Layer, ...]:
        return tuple(layer for own in self._build_blocks(values).values() for layer in own)
