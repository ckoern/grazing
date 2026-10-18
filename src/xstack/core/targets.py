"""Optimization targets: where in energy and angle a mirror shall reflect.

A target is either an energy scan at a fixed grazing angle (`EnergyScan`) or an angle scan at a
fixed energy (`AngleScan`). Each target has a weight, a polarization, an optional desired
reflectivity profile and an optional spectral weighting. Several targets form a `TargetSet`
for coupled optimization.

Examples:
    >>> from xstack.core.units import deg, kev
    >>> target = EnergyScan(angle=deg(0.2), energy_min=kev(15), energy_max=kev(70))
    >>> energy, theta = target.grid(5)
    >>> energy.round(0).tolist()
    [15000.0, 28750.0, 42500.0, 56250.0, 70000.0]
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from itertools import pairwise
from typing import Literal, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

from xstack.core.physics.fresnel import Polarization
from xstack.core.serialization import register_serializable

__all__ = [
    "AngleScan",
    "CombineMode",
    "EnergyScan",
    "Profile",
    "Target",
    "TargetSet",
]

type CombineMode = Literal["average_reflectivity", "average_merit"]
"""How the targets of a `TargetSet` are combined, see `TargetSet`."""


@dataclass(frozen=True)
class Profile:
    """A tabulated function of the scan variable, linearly interpolated.

    Used for desired reflectivity profiles and for spectral weightings.

    Args:
        x: Abscissa (eV for energy scans, rad for angle scans), strictly increasing.
        y: Function values.

    Raises:
        ValueError: If the table is malformed.
    """

    x: tuple[float, ...]
    y: tuple[float, ...]

    def __init__(self, x: ArrayLike, y: ArrayLike) -> None:
        xs = tuple(float(v) for v in np.atleast_1d(np.asarray(x, dtype=np.float64)))
        ys = tuple(float(v) for v in np.atleast_1d(np.asarray(y, dtype=np.float64)))
        if len(xs) != len(ys) or not xs:
            raise ValueError("x and y must have the same non-zero length")
        if any(b <= a for a, b in pairwise(xs)):
            raise ValueError("x must be strictly increasing")
        object.__setattr__(self, "x", xs)
        object.__setattr__(self, "y", ys)

    @classmethod
    def constant(cls, value: float) -> "Profile":
        """Return a profile with the same value everywhere.

        Args:
            value: The constant value.

        Returns:
            The profile.
        """
        return cls([0.0], [value])

    def __call__(self, x: ArrayLike) -> NDArray[np.float64]:
        """Evaluate the profile.

        Args:
            x: Abscissa values.

        Returns:
            Interpolated values; outside the table the end values are used.
        """
        return np.interp(np.asarray(x, dtype=np.float64), self.x, self.y)


@dataclass(frozen=True)
class Target(ABC):
    """Base class of energy and angle scans.

    Attributes:
        weight: Relative weight in a `TargetSet`, positive.
        points: Default number of grid points.
        polarization: ``"s"``, ``"p"`` or ``"unpolarized"``.
        profile: Desired reflectivity versus the scan variable. ``None`` means "as high as
            possible"; how it is used depends on the figure of merit.
        spectrum: Weighting versus the scan variable (e.g. source spectrum × detector
            efficiency). ``None`` means uniform weighting.
        resolution: Standard deviation of a Gaussian instrumental resolution in the units of
            the scan variable (eV or rad). Zero disables the convolution.
        reflections: Number of reflections n. The reflectivity of the target is Rⁿ, e.g. 2 for
            a double-reflection telescope; it is used by the figures of merit and in plots.
        name: Optional label.
    """

    weight: float = 1.0
    points: int = 200
    polarization: Polarization | Literal["unpolarized"] = "unpolarized"
    profile: Profile | None = None
    spectrum: Profile | None = None
    resolution: float = 0.0
    reflections: int = 1
    name: str = ""

    def __post_init__(self) -> None:
        if not self.weight > 0:
            raise ValueError("target weight must be positive")
        if self.points < 2:
            raise ValueError("a target needs at least 2 grid points")
        if self.polarization not in ("s", "p", "unpolarized"):
            raise ValueError(f"unknown polarization {self.polarization!r}")
        if self.resolution < 0:
            raise ValueError("resolution must not be negative")
        if self.reflections < 1:
            raise ValueError("the number of reflections must be at least 1")

    @property
    @abstractmethod
    def axis(self) -> Literal["energy", "angle"]:
        """The scanned variable."""

    def label(self, index: int) -> str:
        """Return the name of the target, or ``"target <index + 1>"`` if it has none.

        Args:
            index: Position of the target in its target set, starting at 0.

        Returns:
            The label.

        Examples:
            >>> EnergyScan(angle=0.01, energy_min=7e3, energy_max=9e3).label(0)
            'target 1'
        """
        return self.name or f"target {index + 1}"

    @property
    @abstractmethod
    def scan_range(self) -> tuple[float, float]:
        """Range of the scanned variable (eV or rad)."""

    @abstractmethod
    def grid(self, points: int | None = None) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return the evaluation grid.

        Args:
            points: Number of points. Defaults to `points`.

        Returns:
            Energies in eV and grazing angles in rad, both of shape (points,).
        """

    @abstractmethod
    def grid_at(
        self, values: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return energies and angles for given values of the scanned variable.

        Args:
            values: Energies in eV (energy scan) or angles in rad (angle scan).

        Returns:
            Energies in eV and grazing angles in rad, both of the shape of `values`.
        """

    def scan_values(self, points: int | None = None) -> NDArray[np.float64]:
        """Return the values of the scanned variable on the evaluation grid.

        Args:
            points: Number of points. Defaults to `points`.

        Returns:
            Energies in eV or angles in rad, shape (points,).
        """
        lo, hi = self.scan_range
        return np.linspace(lo, hi, points or self.points)

    def weights(self, points: int | None = None) -> NDArray[np.float64]:
        """Return the integration weights on the evaluation grid.

        The weights are trapezoidal-rule weights multiplied by the spectrum and normalized to a
        sum of 1, so that ``sum(weights * R)`` is the (spectrum-weighted) mean reflectivity.

        Args:
            points: Number of points. Defaults to `points`.

        Returns:
            Weights, shape (points,).
        """
        x = self.scan_values(points)
        w = np.ones_like(x)
        w[0] = w[-1] = 0.5
        if self.spectrum is not None:
            w = w * self.spectrum(x)
        total = w.sum()
        if not total > 0:
            raise ValueError("spectrum weighting is zero over the whole target range")
        return np.asarray(w / total, dtype=np.float64)

    def with_points(self, points: int) -> Self:
        """Return a copy with a different default number of grid points.

        Args:
            points: Number of grid points.

        Returns:
            The modified target.
        """
        return replace(self, points=points)


@dataclass(frozen=True)
class EnergyScan(Target):
    """An energy range at a fixed grazing angle.

    Attributes:
        angle: Grazing angle in rad.
        energy_min: Lower end of the energy range in eV.
        energy_max: Upper end of the energy range in eV.
    """

    angle: float = 0.0
    energy_min: float = 0.0
    energy_max: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0.0 < self.angle <= np.pi / 2:
            raise ValueError("angle must be in (0, pi/2]")
        if not 0.0 < self.energy_min < self.energy_max:
            raise ValueError("energy range must satisfy 0 < energy_min < energy_max")

    @property
    def axis(self) -> Literal["energy", "angle"]:
        """The scanned variable."""
        return "energy"

    @property
    def scan_range(self) -> tuple[float, float]:
        """Energy range in eV."""
        return (self.energy_min, self.energy_max)

    def grid(self, points: int | None = None) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return the evaluation grid.

        Args:
            points: Number of points. Defaults to `points`.

        Returns:
            Energies in eV and grazing angles in rad, both of shape (points,).
        """
        return self.grid_at(self.scan_values(points))

    def grid_at(
        self, values: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return energies and angles for given energies.

        Args:
            values: Energies in eV.

        Returns:
            Energies in eV and grazing angles in rad, both of the shape of `values`.
        """
        energy = np.asarray(values, dtype=np.float64)
        return energy, np.full_like(energy, self.angle)


@dataclass(frozen=True)
class AngleScan(Target):
    """A range of grazing angles at a fixed photon energy.

    Attributes:
        energy: Photon energy in eV.
        angle_min: Lower end of the angle range in rad.
        angle_max: Upper end of the angle range in rad.
    """

    energy: float = 0.0
    angle_min: float = 0.0
    angle_max: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.energy > 0.0:
            raise ValueError("energy must be positive")
        if not 0.0 < self.angle_min < self.angle_max <= np.pi / 2:
            raise ValueError("angle range must satisfy 0 < angle_min < angle_max <= pi/2")

    @property
    def axis(self) -> Literal["energy", "angle"]:
        """The scanned variable."""
        return "angle"

    @property
    def scan_range(self) -> tuple[float, float]:
        """Angle range in rad."""
        return (self.angle_min, self.angle_max)

    def grid(self, points: int | None = None) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return the evaluation grid.

        Args:
            points: Number of points. Defaults to `points`.

        Returns:
            Energies in eV and grazing angles in rad, both of shape (points,).
        """
        return self.grid_at(self.scan_values(points))

    def grid_at(
        self, values: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return energies and angles for given angles.

        Args:
            values: Grazing angles in rad.

        Returns:
            Energies in eV and grazing angles in rad, both of the shape of `values`.
        """
        theta = np.asarray(values, dtype=np.float64)
        return np.full_like(theta, self.energy), theta


@dataclass(frozen=True)
class TargetSet:
    """One or several targets that a design is optimized for together.

    With ``combine="average_reflectivity"``, the reflectivity curves of all targets are
    averaged with the target weights before the figure of merit is evaluated. This requires all
    targets to scan the same variable over the same range with the same number of points (e.g.
    one energy range at several angles). With ``combine="average_merit"``, the figure of merit
    is evaluated per target and the results are averaged with the target weights.

    Args:
        targets: The targets.
        combine: How the targets are combined.

    Raises:
        ValueError: If the set is empty or the targets are incompatible with `combine`.
    """

    targets: tuple[Target, ...] = field(default=())
    combine: CombineMode = "average_reflectivity"

    def __init__(
        self,
        targets: Target | tuple[Target, ...] | list[Target],
        combine: CombineMode = "average_reflectivity",
    ) -> None:
        items = (targets,) if isinstance(targets, Target) else tuple(targets)
        if not items:
            raise ValueError("a target set needs at least one target")
        if combine not in ("average_reflectivity", "average_merit"):
            raise ValueError(f"unknown combine mode {combine!r}")
        if combine == "average_reflectivity" and len(items) > 1:
            first = items[0]
            for t in items[1:]:
                if (t.axis, t.scan_range, t.points) != (first.axis, first.scan_range, first.points):
                    raise ValueError(
                        "average_reflectivity requires identical scan axes, ranges and points"
                    )
        object.__setattr__(self, "targets", items)
        object.__setattr__(self, "combine", combine)

    def __len__(self) -> int:
        return len(self.targets)

    def normalized_weights(self) -> NDArray[np.float64]:
        """Return the target weights normalized to a sum of 1."""
        w = np.array([t.weight for t in self.targets], dtype=np.float64)
        return np.asarray(w / w.sum(), dtype=np.float64)

    def with_points(self, points: int) -> "TargetSet":
        """Return a copy in which every target uses a different number of grid points.

        Args:
            points: Number of grid points.

        Returns:
            The modified target set.
        """
        return TargetSet(tuple(t.with_points(points) for t in self.targets), self.combine)


register_serializable("Profile", Profile)
register_serializable("EnergyScan", EnergyScan)
register_serializable("AngleScan", AngleScan)
register_serializable("TargetSet", TargetSet)
