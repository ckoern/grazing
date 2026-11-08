"""Datasets: measured curves with their instrument and the selection of points to fit.

A dataset keeps all measured points. Which of them take part in a fit is given by a mask
computed from fit ranges and excluded ranges of the scanned variable, so that a selection can
be changed (and undone) without losing data, and excluded points can still be shown.

Examples:
    >>> from grazing.core.measurement import MeasuredCurve
    >>> curve = MeasuredCurve("angle", [0.01, 0.02, 0.03, 0.04, 0.05], [1.0] * 5, energy=8048.0)
    >>> data = Dataset.create("cu_ka", curve)
    >>> data = data.with_ranges([(0.0, 0.025), (0.035, 1.0)], excluded=[(0.045, 0.055)])
    >>> data.mask().tolist()
    [True, True, False, True, False]
    >>> data.parameters.free_names(), list(data.parameters)[:2]
    ((), ['cu_ka.scale', 'cu_ka.background'])
"""

from collections.abc import Iterable
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from grazing.core.measurement import MeasuredCurve
from grazing.core.parameters import ParameterSet, ParameterValue
from grazing.core.serialization import register_serializable
from grazing.fit.instrument import Instrument

__all__ = ["Dataset", "Interval"]

type Interval = tuple[float, float]
"""Closed range (lower, upper) of the scanned variable, in rad or eV."""


def _intervals(ranges: Iterable[Interval]) -> tuple[Interval, ...]:
    out = []
    for lo, hi in ranges:
        lo, hi = float(lo), float(hi)
        if not lo <= hi:
            raise ValueError(f"invalid range ({lo:g}, {hi:g}): lower bound above upper bound")
        out.append((lo, hi))
    return tuple(sorted(out))


@dataclass(frozen=True)
class Dataset:
    """A measured curve with its instrument model and point selection.

    Attributes:
        name: Identifier of the dataset, used as prefix of its instrument parameters
            (``"<name>.scale"``, ...).
        curve: The measured curve.
        instrument: Instrument model; its axis equals that of the curve.
        ranges: Fit ranges of the scanned variable (rad or eV). Points within any range are
            fitted; no ranges means all points.
        excluded: Ranges whose points are never fitted, e.g. outliers or detector artifacts.
        weight: Weight of the dataset in a simultaneous fit; its residuals are multiplied by
            √weight.
        enabled: Whether the dataset takes part in fits.

    Raises:
        ValueError: If the name is not an identifier, the axes differ, a range is invalid or
            the weight is not positive.
    """

    name: str
    curve: MeasuredCurve
    instrument: Instrument
    ranges: tuple[Interval, ...] = ()
    excluded: tuple[Interval, ...] = ()
    weight: float = 1.0
    enabled: bool = True

    def __post_init__(self) -> None:
        if not self.name.isidentifier():
            raise ValueError(f"dataset name {self.name!r} is not an identifier")
        if self.instrument.axis != self.curve.axis:
            raise ValueError("instrument and curve must have the same axis")
        if not self.weight > 0.0:
            raise ValueError("dataset weight must be positive")
        object.__setattr__(self, "ranges", _intervals(self.ranges))
        object.__setattr__(self, "excluded", _intervals(self.excluded))

    @classmethod
    def create(cls, name: str, curve: MeasuredCurve, **instrument: ParameterValue) -> "Dataset":
        """Create a dataset fitting all points.

        Args:
            name: Identifier of the dataset.
            curve: The measured curve.
            **instrument: Instrument parameter specifications for `Instrument.create`.

        Returns:
            The dataset.
        """
        return cls(name, curve, Instrument.create(curve.axis, **instrument))

    @property
    def parameters(self) -> ParameterSet:
        """The instrument parameters, prefixed with the dataset name."""
        return self.instrument.parameters.prefixed(self.name)

    def with_parameters(self, parameters: ParameterSet) -> "Dataset":
        """Return a copy with different instrument parameters.

        Args:
            parameters: Parameters named ``"<name>.<instrument parameter>"``.

        Returns:
            The modified dataset.
        """
        prefix = f"{self.name}."
        unprefixed = ParameterSet(
            p.evolve(name=p.name.removeprefix(prefix)) for p in parameters.values()
        )
        return replace(self, instrument=self.instrument.with_parameters(unprefixed))

    def with_ranges(
        self, ranges: Iterable[Interval], excluded: Iterable[Interval] | None = None
    ) -> "Dataset":
        """Return a copy with other fit ranges.

        Args:
            ranges: Fit ranges (rad or eV); empty for all points.
            excluded: Excluded ranges, or ``None`` to keep the current ones.

        Returns:
            The modified dataset.
        """
        return replace(
            self,
            ranges=tuple(ranges),
            excluded=self.excluded if excluded is None else tuple(excluded),
        )

    def mask(self) -> NDArray[np.bool_]:
        """Return which measured points are fitted.

        Returns:
            Boolean array of shape (curve.size,).
        """
        x = np.array(self.curve.x)
        if self.ranges:
            keep = np.zeros(x.shape, dtype=bool)
            for lo, hi in self.ranges:
                keep |= (x >= lo) & (x <= hi)
        else:
            keep = np.ones(x.shape, dtype=bool)
        for lo, hi in self.excluded:
            keep &= ~((x >= lo) & (x <= hi))
        return keep


register_serializable("Dataset", Dataset)
