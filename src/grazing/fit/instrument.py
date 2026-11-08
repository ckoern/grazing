"""Instrument model: how the reflectivity of a sample appears in a measurement.

The measured reflectivity at the nominal value x of the scanned variable (angle in rad or
energy in eV) is modeled as

    R_meas(x) = scale · F(θ) · (R ⊛ G_σ)(x + offset) + background

with the reflectivity R of the sample, a normalized Gaussian G_σ of standard deviation
σ = `resolution` and the footprint factor F of a Gaussian beam on a sample of finite length,

    F(θ) = erf(L · sin θ / (2√2 · σ_beam)),  σ_beam = beam_width / (2√(2 ln 2)),

where θ is the true grazing angle, L is `sample_length` and `beam_width` is the full width at
half maximum of the beam perpendicular to its direction. F = 1 when `sample_length` or
`beam_width` is zero.

All quantities are parameters of the instrument, so they can be fitted like structure
parameters.

Examples:
    >>> from grazing.core.parameters import free
    >>> instrument = Instrument.create("angle", scale=free(1.0, 0.5, 2.0), background=1e-7)
    >>> instrument.parameters.free_names()
    ('scale',)
    >>> instrument.apply(np.array([0.01, 0.02]), lambda x: 0.5 + 0 * x).tolist()
    [0.5000001, 0.5000001]
"""

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Self

import numpy as np
from numpy.typing import NDArray
from scipy.special import erf

from grazing.core.measurement import Axis
from grazing.core.parameters import Parameter, ParameterSet, ParameterValue, resolve
from grazing.core.serialization import register_serializable

__all__ = ["INSTRUMENT_PARAMETERS", "Instrument", "footprint_factor", "smear"]

INSTRUMENT_PARAMETERS = (
    "scale",
    "background",
    "offset",
    "resolution",
    "beam_width",
    "sample_length",
)
"""Names of the instrument parameters, in order."""

_FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))
_HERMITE_NODES = 15
_REACH = 5.0
_GRID_STEP = 0.1
_MIN_ANGLE = 1e-9

type PointFunction = Callable[[NDArray[np.float64]], NDArray[np.float64]]
"""Reflectivity as a function of the true value of the scanned variable."""


def _templates(axis: Axis) -> tuple[Parameter, ...]:
    unit = "rad" if axis == "angle" else "eV"
    return (
        Parameter("scale", 1.0, min=0.0, description="Intensity scale"),
        Parameter("background", 0.0, description="Constant background"),
        Parameter("offset", 0.0, unit=unit, description="Offset of the scan axis (true - nominal)"),
        Parameter("resolution", 0.0, min=0.0, unit=unit, description="Resolution (Gaussian σ)"),
        Parameter("beam_width", 0.0, min=0.0, unit="nm", description="Beam width (FWHM)"),
        Parameter("sample_length", 0.0, min=0.0, unit="nm", description="Sample length"),
    )


@dataclass(frozen=True)
class Instrument:
    """Instrument parameters of one measured curve.

    Attributes:
        axis: The scanned variable, ``"angle"`` or ``"energy"``; sets the units of `offset` and
            `resolution` (rad or eV).
        parameters: The parameters named in `INSTRUMENT_PARAMETERS`.

    Raises:
        ValueError: If the axis is unknown or the parameter names differ from
            `INSTRUMENT_PARAMETERS`.
    """

    axis: Axis
    parameters: ParameterSet

    def __post_init__(self) -> None:
        if self.axis not in ("angle", "energy"):
            raise ValueError(f"unknown axis {self.axis!r}")
        if tuple(self.parameters) != INSTRUMENT_PARAMETERS:
            raise ValueError(f"instrument parameters must be {INSTRUMENT_PARAMETERS}")

    @classmethod
    def create(
        cls,
        axis: Axis,
        scale: ParameterValue = 1.0,
        background: ParameterValue = 0.0,
        offset: ParameterValue = 0.0,
        resolution: ParameterValue = 0.0,
        beam_width: ParameterValue = 0.0,
        sample_length: ParameterValue = 0.0,
    ) -> "Instrument":
        """Create an instrument; parameters are fixed values or `free` ranges.

        Args:
            axis: The scanned variable.
            scale: Intensity scale.
            background: Constant background.
            offset: True minus nominal value of the scanned variable, in rad or eV.
            resolution: Standard deviation of the Gaussian resolution, in rad or eV.
            beam_width: Full width at half maximum of the beam, in nm.
            sample_length: Length of the sample along the beam, in nm.

        Returns:
            The instrument.
        """
        specs = (scale, background, offset, resolution, beam_width, sample_length)
        templates = _templates(axis)
        return cls(axis, ParameterSet(resolve(t, s) for t, s in zip(templates, specs, strict=True)))

    def with_parameters(self, parameters: ParameterSet) -> Self:
        """Return a copy with a different parameter set.

        Args:
            parameters: The new parameters, named as in `INSTRUMENT_PARAMETERS`.

        Returns:
            The modified instrument.
        """
        return replace(self, parameters=parameters)

    def apply(
        self,
        x: NDArray[np.float64],
        reflectivity: PointFunction,
        fixed_angle: float | None = None,
        values: Mapping[str, float] | None = None,
    ) -> NDArray[np.float64]:
        """Return the modeled measurement at nominal values of the scanned variable.

        Args:
            x: Nominal angles in rad or energies in eV, increasing.
            reflectivity: Reflectivity of the sample as a function of true angle in rad or
                energy in eV; it is evaluated at positive values only.
            fixed_angle: Grazing angle in rad of an energy scan, for the footprint factor.
            values: Parameter values overriding the current ones, by name (expressions must be
                evaluated already).

        Returns:
            Modeled measured reflectivity, same shape as `x`.
        """
        v = self.parameters.value_dict() | dict(values or {})
        true_x = x + v["offset"]
        r = smear(reflectivity, true_x, v["resolution"], self.axis)
        theta = true_x if self.axis == "angle" else np.full_like(x, fixed_angle or math.pi / 2)
        r = r * footprint_factor(theta, v["beam_width"], v["sample_length"])
        return np.asarray(v["scale"] * r + v["background"], dtype=np.float64)


def footprint_factor(
    theta: NDArray[np.float64], beam_width: float, sample_length: float
) -> NDArray[np.float64]:
    """Return the fraction of a Gaussian beam that hits a sample of finite length.

    Args:
        theta: Grazing angles in rad.
        beam_width: Full width at half maximum of the beam, in nm. Zero disables the factor.
        sample_length: Length of the sample along the beam, in nm. Zero disables the factor.

    Returns:
        The factor erf(L sin θ / (2√2 σ_beam)), or ones if disabled.
    """
    if beam_width <= 0.0 or sample_length <= 0.0:
        return np.ones_like(theta)
    sigma = beam_width / _FWHM_PER_SIGMA
    return np.asarray(
        erf(sample_length * np.sin(np.abs(theta)) / (2.0 * math.sqrt(2.0) * sigma)),
        dtype=np.float64,
    )


def smear(
    reflectivity: PointFunction, x: NDArray[np.float64], sigma: float, axis: Axis = "angle"
) -> NDArray[np.float64]:
    """Convolve a reflectivity curve with a Gaussian at arbitrary increasing points.

    Points are grouped into clusters separated by gaps larger than the reach of the Gaussian.
    Each cluster is evaluated on a uniform grid with a spacing of σ/10, or with Gauss–Hermite
    quadrature at every point if that needs fewer evaluations. Values of the scanned variable
    below zero (and angles above π/2) are clipped to the valid range.

    Args:
        reflectivity: Reflectivity as a function of angle in rad or energy in eV.
        x: Points at which to evaluate the convolution, increasing.
        sigma: Standard deviation of the Gaussian, in the units of `x`. Zero evaluates
            `reflectivity` at `x` directly.
        axis: The scanned variable, for the valid range.

    Returns:
        The convolved reflectivity at `x`.
    """
    upper = math.pi / 2 if axis == "angle" else math.inf

    def evaluate(points: NDArray[np.float64]) -> NDArray[np.float64]:
        return reflectivity(np.clip(points, _MIN_ANGLE, upper))

    if sigma <= 0.0 or x.size == 0:
        return evaluate(x)
    out = np.empty_like(x)
    breaks = np.flatnonzero(np.diff(x) > 2.0 * _REACH * sigma) + 1
    for part in np.split(np.arange(x.size), breaks):
        out[part] = _smear_cluster(evaluate, x[part], sigma)
    return out


def _smear_cluster(
    evaluate: PointFunction, x: NDArray[np.float64], sigma: float
) -> NDArray[np.float64]:
    step = _GRID_STEP * sigma
    grid_size = math.ceil((x[-1] - x[0]) / step) + 1 + 2 * math.ceil(_REACH / _GRID_STEP)
    if grid_size <= _HERMITE_NODES * x.size:
        pad = _REACH * sigma
        grid = np.arange(x[0] - pad, x[-1] + pad + step, step)
        half = math.ceil(_REACH * sigma / step)
        offsets = step * np.arange(-half, half + 1)
        kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
        smeared = np.convolve(evaluate(grid), kernel / kernel.sum(), mode="same")
        return np.asarray(np.interp(x, grid, smeared), dtype=np.float64)
    nodes, weights = np.polynomial.hermite_e.hermegauss(_HERMITE_NODES)
    values = evaluate((x[:, None] + sigma * nodes[None, :]).ravel()).reshape(x.size, -1)
    return np.asarray(values @ (weights / weights.sum()), dtype=np.float64)


register_serializable("Instrument", Instrument)
