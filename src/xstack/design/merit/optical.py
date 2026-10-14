"""Figures of merit that rate the reflectivity curves."""

from dataclasses import dataclass
from typing import ClassVar, Literal

import numpy as np
from numpy.typing import NDArray

from xstack.design.merit.base import Curve, CurveMerit, register_merit

__all__ = [
    "Flatness",
    "MeanReflectivity",
    "PeakReflectivity",
    "ProfileDeviation",
    "SoftMinReflectivity",
]


@register_merit
@dataclass(frozen=True)
class MeanReflectivity(CurveMerit):
    """Maximize the (spectrum-weighted) mean reflectivity over the target range.

    Value: 1 − ⟨R⟩. Residuals: √w_i·(1 − R_i).
    """

    type_name: ClassVar[str] = "mean_reflectivity"
    quantity_label: ClassVar[str | None] = "mean reflectivity"

    def curve_quantity(self, curve: Curve) -> float:
        """Return the weighted mean reflectivity ⟨R⟩.

        Args:
            curve: The curve.

        Returns:
            The quantity.
        """
        return curve.mean()

    def curve_value(self, curve: Curve) -> float:
        """Return 1 − mean reflectivity of one curve.

        Args:
            curve: The curve.

        Returns:
            The figure of merit.
        """
        return 1.0 - curve.mean()

    def curve_residuals(self, curve: Curve) -> NDArray[np.float64]:
        """Return √w_i·(1 − R_i).

        Args:
            curve: The curve.

        Returns:
            Residual vector.
        """
        return np.sqrt(curve.weights) * (1.0 - curve.reflectivity)


@register_merit
@dataclass(frozen=True)
class SoftMinReflectivity(CurveMerit):
    """Maximize the minimum reflectivity over the target range (flat, gap-free coverage).

    Uses the smooth minimum m = −τ·ln Σ w_i·exp(−R_i/τ), which approaches min(R) for τ → 0.
    Value: 1 − m. Residuals: √p_i·(1 − R_i) with the soft-min weights p_i.

    Attributes:
        softness: Smoothing parameter τ in units of reflectivity.
    """

    type_name: ClassVar[str] = "soft_min_reflectivity"
    quantity_label: ClassVar[str | None] = "minimum reflectivity"
    softness: float = 0.01

    def __post_init__(self) -> None:
        if not self.softness > 0:
            raise ValueError("softness must be positive")

    def _weights(self, curve: Curve) -> tuple[float, NDArray[np.float64]]:
        z = -curve.reflectivity / self.softness
        shift = z.max()
        e = curve.weights * np.exp(z - shift)
        total = e.sum()
        return float(-self.softness * (np.log(total) + shift)), e / total

    def curve_quantity(self, curve: Curve) -> float:
        """Return the minimum reflectivity.

        Args:
            curve: The curve.

        Returns:
            The quantity.
        """
        return float(curve.reflectivity.min())

    def curve_value(self, curve: Curve) -> float:
        """Return 1 − soft minimum of the reflectivity of one curve.

        Args:
            curve: The curve.

        Returns:
            The figure of merit.
        """
        return 1.0 - self._weights(curve)[0]

    def curve_residuals(self, curve: Curve) -> NDArray[np.float64]:
        """Return √p_i·(1 − R_i).

        Args:
            curve: The curve.

        Returns:
            Residual vector.
        """
        return np.sqrt(self._weights(curve)[1]) * (1.0 - curve.reflectivity)


@register_merit
@dataclass(frozen=True)
class ProfileDeviation(CurveMerit):
    """Match a desired reflectivity profile given by the targets.

    Value: weighted RMS deviation √(Σ w_i·Δ_i²). Residuals: √w_i·Δ_i. With ``mode="below"``
    only reflectivities below the profile count (Δ_i = max(P_i − R_i, 0)), so the profile is a
    lower limit; with ``"two_sided"`` Δ_i = R_i − P_i.

    Attributes:
        mode: ``"two_sided"`` or ``"below"``.
    """

    type_name: ClassVar[str] = "profile_deviation"
    quantity_label: ClassVar[str | None] = "RMS deviation from profile"
    mode: Literal["two_sided", "below"] = "two_sided"

    def __post_init__(self) -> None:
        if self.mode not in ("two_sided", "below"):
            raise ValueError(f"unknown mode {self.mode!r}")

    def _deviation(self, curve: Curve) -> NDArray[np.float64]:
        if curve.profile is None:
            raise ValueError("ProfileDeviation requires targets with a profile")
        diff = curve.reflectivity - curve.profile
        return np.minimum(diff, 0.0) if self.mode == "below" else diff

    def curve_quantity(self, curve: Curve) -> float:
        """Return the weighted RMS deviation from the profile.

        Args:
            curve: The curve.

        Returns:
            The quantity.
        """
        return self.curve_value(curve)

    def curve_value(self, curve: Curve) -> float:
        """Return the weighted RMS deviation from the profile.

        Args:
            curve: The curve.

        Returns:
            The figure of merit.

        Raises:
            ValueError: If the curve has no profile.
        """
        return float(np.sqrt(np.dot(curve.weights, self._deviation(curve) ** 2)))

    def curve_residuals(self, curve: Curve) -> NDArray[np.float64]:
        """Return √w_i·Δ_i.

        Args:
            curve: The curve.

        Returns:
            Residual vector.

        Raises:
            ValueError: If the curve has no profile.
        """
        return np.sqrt(curve.weights) * self._deviation(curve)


@register_merit
@dataclass(frozen=True)
class Flatness(CurveMerit):
    """Minimize the relative variation of the reflectivity over the target range.

    Value: weighted standard deviation divided by the weighted mean. Residuals:
    √w_i·(R_i − ⟨R⟩)/⟨R⟩. Usually combined with a term that rewards high reflectivity.
    """

    type_name: ClassVar[str] = "flatness"
    quantity_label: ClassVar[str | None] = "relative standard deviation"

    def curve_quantity(self, curve: Curve) -> float:
        """Return the relative standard deviation of the reflectivity.

        Args:
            curve: The curve.

        Returns:
            The quantity.
        """
        return self.curve_value(curve)

    def curve_value(self, curve: Curve) -> float:
        """Return the relative standard deviation of the reflectivity.

        Args:
            curve: The curve.

        Returns:
            The figure of merit.
        """
        return float(np.linalg.norm(self.curve_residuals(curve)))

    def curve_residuals(self, curve: Curve) -> NDArray[np.float64]:
        """Return √w_i·(R_i − ⟨R⟩)/⟨R⟩.

        Args:
            curve: The curve.

        Returns:
            Residual vector.
        """
        mean = max(curve.mean(), 1e-12)
        return np.sqrt(curve.weights) * (curve.reflectivity - mean) / mean


@register_merit
@dataclass(frozen=True)
class PeakReflectivity(CurveMerit):
    """Maximize the highest reflectivity within the target range.

    Value and single residual: 1 − max(R).
    """

    type_name: ClassVar[str] = "peak_reflectivity"
    quantity_label: ClassVar[str | None] = "peak reflectivity"

    def curve_quantity(self, curve: Curve) -> float:
        """Return the maximum reflectivity.

        Args:
            curve: The curve.

        Returns:
            The quantity.
        """
        return float(curve.reflectivity.max())

    def curve_value(self, curve: Curve) -> float:
        """Return 1 − the maximum reflectivity.

        Args:
            curve: The curve.

        Returns:
            The figure of merit.
        """
        return 1.0 - float(curve.reflectivity.max())

    def curve_residuals(self, curve: Curve) -> NDArray[np.float64]:
        """Return [1 − max(R)].

        Args:
            curve: The curve.

        Returns:
            Residual vector with one element.
        """
        return np.array([self.curve_value(curve)])
