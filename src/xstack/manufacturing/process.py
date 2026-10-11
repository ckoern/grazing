"""Deposition process model: coating time of a stack."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from xstack.structure.stack import Stack

__all__ = ["DepositionProcess", "DepositionStep"]


@dataclass(frozen=True)
class DepositionStep:
    """One layer in deposition order.

    Attributes:
        step: Running number in deposition order, starting at 1 for the layer on the substrate.
        layer_index: Index of the layer in `Stack.layers` (0 = top layer).
        material: Name of the material.
        thickness: Thickness in nm.
        duration: Time for this layer including its overheads, in s.
        start_time: Time at which this layer starts, in s (after the fixed overhead).
        tag: Tag of the layer.
    """

    step: int
    layer_index: int
    material: str
    thickness: float
    duration: float
    start_time: float
    tag: str


@dataclass(frozen=True)
class DepositionProcess:
    """Time model of a sequential deposition process (e.g. magnetron sputtering).

    The coating time is

        T = t_fixed + Σ_layers (t / rate(material) + t_layer + t_material(material))

    Layers of zero thickness are skipped and cost no time.

    Args:
        rates: Deposition rate in nm/s by material name, as mapping or (name, rate) pairs.
        overhead_per_layer: Time per layer independent of its thickness and material (e.g.
            pre-sputtering, shutter movement), in s.
        overhead_per_material: Additional time per layer of a given material, in s, by material
            name, as mapping or (name, time) pairs.
        fixed_overhead: Time per coating run independent of the stack, in s.

    Raises:
        ValueError: If a rate is not positive or an overhead is negative.

    Examples:
        >>> from xstack.materials import get
        >>> from xstack.structure import Layer, Stack
        >>> process = DepositionProcess({"W": 0.1, "Si": 0.2}, overhead_per_layer=10.0)
        >>> stack = Stack([Layer(get("W"), 2.0), Layer(get("Si"), 3.0)] * 10, get("Si"))
        >>> process.time(stack)
        550.0
    """

    rates: tuple[tuple[str, float], ...]
    overhead_per_layer: float = 0.0
    overhead_per_material: tuple[tuple[str, float], ...] = ()
    fixed_overhead: float = 0.0

    def __init__(
        self,
        rates: Mapping[str, float] | Iterable[tuple[str, float]],
        overhead_per_layer: float = 0.0,
        overhead_per_material: Mapping[str, float] | Iterable[tuple[str, float]] | None = None,
        fixed_overhead: float = 0.0,
    ) -> None:
        rate_items = tuple((str(k), float(v)) for k, v in dict(rates).items())
        material_items = tuple(
            (str(k), float(v)) for k, v in dict(overhead_per_material or {}).items()
        )
        if any(v <= 0 for _, v in rate_items):
            raise ValueError("deposition rates must be positive")
        if overhead_per_layer < 0 or fixed_overhead < 0 or any(v < 0 for _, v in material_items):
            raise ValueError("overheads must not be negative")
        object.__setattr__(self, "rates", rate_items)
        object.__setattr__(self, "overhead_per_layer", float(overhead_per_layer))
        object.__setattr__(self, "overhead_per_material", material_items)
        object.__setattr__(self, "fixed_overhead", float(fixed_overhead))

    def rate(self, material: str) -> float:
        """Return the deposition rate of a material.

        Args:
            material: Material name.

        Returns:
            Rate in nm/s.

        Raises:
            KeyError: If the process has no rate for the material.
        """
        try:
            return dict(self.rates)[material]
        except KeyError:
            raise KeyError(f"no deposition rate defined for material {material!r}") from None

    def layer_times(self, stack: Stack) -> NDArray[np.float64]:
        """Return the time of each layer including its overheads.

        Args:
            stack: The stack.

        Returns:
            Times in s in the order of `Stack.layers` (top first); zero for layers of zero
            thickness.

        Raises:
            KeyError: If a material has no deposition rate.
        """
        material_overhead = dict(self.overhead_per_material)
        times = np.zeros(stack.num_layers)
        for i, layer in enumerate(stack.layers):
            if layer.thickness > 0:
                name = layer.material.name
                times[i] = (
                    layer.thickness / self.rate(name)
                    + self.overhead_per_layer
                    + material_overhead.get(name, 0.0)
                )
        return times

    def time(self, stack: Stack) -> float:
        """Return the total coating time.

        Args:
            stack: The stack.

        Returns:
            Time in s.

        Raises:
            KeyError: If a material has no deposition rate.
        """
        return self.fixed_overhead + float(self.layer_times(stack).sum())

    def schedule(self, stack: Stack) -> list[DepositionStep]:
        """Return the layers in deposition order (from the substrate up) with their timing.

        Layers of zero thickness are omitted.

        Args:
            stack: The stack.

        Returns:
            The deposition steps.

        Raises:
            KeyError: If a material has no deposition rate.
        """
        times = self.layer_times(stack)
        steps: list[DepositionStep] = []
        clock = self.fixed_overhead
        for index in range(stack.num_layers - 1, -1, -1):
            layer = stack.layers[index]
            if layer.thickness <= 0:
                continue
            steps.append(
                DepositionStep(
                    step=len(steps) + 1,
                    layer_index=index,
                    material=layer.material.name,
                    thickness=layer.thickness,
                    duration=float(times[index]),
                    start_time=clock,
                    tag=layer.tag,
                )
            )
            clock += float(times[index])
        return steps
