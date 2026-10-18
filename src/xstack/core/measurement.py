"""Measured reflectivity curves.

A `MeasuredCurve` is a reflectivity measurement versus grazing angle at a fixed photon energy
(``axis="angle"``) or versus photon energy at a fixed grazing angle (``axis="energy"``), with
optional uncertainties. Curves are read from files with `xstack.core.io.read_curves`.

Examples:
    >>> from xstack.core.units import deg, kev
    >>> curve = MeasuredCurve(
    ...     "angle", deg(np.array([0.3, 0.1, 0.2])), [0.2, 0.9, 0.5], energy=kev(8.048)
    ... )
    >>> curve.size, curve.reflectivity
    (3, (0.9, 0.5, 0.2))
    >>> energy, theta = curve.grid()
    >>> energy.tolist()
    [8048.0, 8048.0, 8048.0]
"""

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from xstack.core.physics.fresnel import Polarization
from xstack.core.serialization import register_serializable

__all__ = ["Axis", "MeasuredCurve"]

type Axis = Literal["angle", "energy"]
"""The scanned variable of a curve."""


def _floats(values: ArrayLike) -> tuple[float, ...]:
    return tuple(float(v) for v in np.atleast_1d(np.asarray(values, dtype=np.float64)))


@dataclass(frozen=True, init=False)
class MeasuredCurve:
    """A measured reflectivity curve, sorted by the scanned variable.

    Args:
        axis: The scanned variable, ``"angle"`` or ``"energy"``.
        x: Grazing angles in rad (angle scans) or photon energies in eV (energy scans).
        reflectivity: Measured reflectivity at `x`, in any normalization.
        uncertainty: Standard uncertainty of each reflectivity value, or ``None`` if unknown.
        energy: Photon energy in eV of an angle scan, or ``None`` if not yet known. Unused for
            energy scans.
        angle: Grazing angle in rad of an energy scan, or ``None`` if not yet known. Unused for
            angle scans.
        polarization: ``"s"``, ``"p"`` or ``"unpolarized"``.
        name: Label of the curve.
        source: Where the curve comes from, e.g. the file name.

    Raises:
        ValueError: If the arrays are empty or differ in length, values are not finite, angles
            are outside (0, π/2], energies or the fixed energy or angle are not positive, or
            uncertainties are negative.
    """

    axis: Axis
    x: tuple[float, ...]
    reflectivity: tuple[float, ...]
    uncertainty: tuple[float, ...] | None
    energy: float | None
    angle: float | None
    polarization: Polarization | Literal["unpolarized"]
    name: str
    source: str

    def __init__(
        self,
        axis: Axis,
        x: ArrayLike,
        reflectivity: ArrayLike,
        uncertainty: ArrayLike | None = None,
        energy: float | None = None,
        angle: float | None = None,
        polarization: Polarization | Literal["unpolarized"] = "unpolarized",
        name: str = "",
        source: str = "",
    ) -> None:
        if axis not in ("angle", "energy"):
            raise ValueError(f"unknown axis {axis!r}")
        if polarization not in ("s", "p", "unpolarized"):
            raise ValueError(f"unknown polarization {polarization!r}")
        xs, rs = _floats(x), _floats(reflectivity)
        us = None if uncertainty is None else _floats(uncertainty)
        if not xs or len(rs) != len(xs) or (us is not None and len(us) != len(xs)):
            raise ValueError("x, reflectivity and uncertainty must have the same non-zero length")
        if not np.all(np.isfinite(xs + rs + (us or ()))):
            raise ValueError("measured values must be finite")
        if axis == "angle" and not all(0.0 < v <= np.pi / 2 for v in xs):
            raise ValueError("grazing angles must be in (0, pi/2]")
        if axis == "energy" and not all(v > 0.0 for v in xs):
            raise ValueError("energies must be positive")
        if us is not None and any(u < 0.0 for u in us):
            raise ValueError("uncertainties must not be negative")
        if energy is not None and not energy > 0.0:
            raise ValueError("energy must be positive")
        if angle is not None and not 0.0 < angle <= np.pi / 2:
            raise ValueError("angle must be in (0, pi/2]")
        order = np.argsort(xs, kind="stable")
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "x", tuple(xs[i] for i in order))
        object.__setattr__(self, "reflectivity", tuple(rs[i] for i in order))
        object.__setattr__(self, "uncertainty", None if us is None else tuple(us[i] for i in order))
        object.__setattr__(self, "energy", None if energy is None else float(energy))
        object.__setattr__(self, "angle", None if angle is None else float(angle))
        object.__setattr__(self, "polarization", polarization)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "source", source)

    @property
    def size(self) -> int:
        """Number of measured points."""
        return len(self.x)

    @property
    def scan_range(self) -> tuple[float, float]:
        """Smallest and largest value of the scanned variable (rad or eV)."""
        return (self.x[0], self.x[-1])

    @property
    def fixed(self) -> float | None:
        """The fixed variable: `energy` of an angle scan, `angle` of an energy scan."""
        return self.energy if self.axis == "angle" else self.angle

    def arrays(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64] | None]:
        """Return the measured values as arrays.

        Returns:
            Scanned variable (rad or eV), reflectivity and uncertainty (``None`` if unknown),
            each of shape (size,).
        """
        u = None if self.uncertainty is None else np.array(self.uncertainty)
        return np.array(self.x), np.array(self.reflectivity), u

    def grid(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return photon energy and grazing angle of every measured point.

        Returns:
            Energies in eV and grazing angles in rad, both of shape (size,).

        Raises:
            ValueError: If the fixed energy or angle is not known.
        """
        fixed = self.fixed
        if fixed is None:
            missing = "energy" if self.axis == "angle" else "angle"
            raise ValueError(f"the {missing} of curve {self.name or '(unnamed)'!r} is not set")
        x = np.array(self.x)
        constant = np.full_like(x, fixed)
        return (constant, x) if self.axis == "angle" else (x, constant)

    def crop(self, lo: float, hi: float) -> "MeasuredCurve":
        """Return the points with the scanned variable within a closed range.

        Args:
            lo: Lower bound (rad or eV).
            hi: Upper bound (rad or eV).

        Returns:
            The cropped curve.

        Raises:
            ValueError: If no point lies within the range.
        """
        keep = [i for i, v in enumerate(self.x) if lo <= v <= hi]
        if not keep:
            raise ValueError("no measured point within the range")
        u = self.uncertainty
        return replace(
            self,
            x=tuple(self.x[i] for i in keep),
            reflectivity=tuple(self.reflectivity[i] for i in keep),
            uncertainty=None if u is None else tuple(u[i] for i in keep),
        )

    def with_fixed(self, value: float) -> "MeasuredCurve":
        """Return a copy with the fixed variable set.

        Args:
            value: Photon energy in eV for an angle scan, grazing angle in rad for an energy scan.

        Returns:
            The modified curve.

        Raises:
            ValueError: If the value is out of range.
        """
        if self.axis == "angle":
            return replace(self, energy=value)
        return replace(self, angle=value)


register_serializable("MeasuredCurve", MeasuredCurve)
