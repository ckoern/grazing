"""Reflectivity models of different fidelity.

A reflectivity model evaluates a `Stack` on the grid of a `Target`. All models share the same
interface, so that design stages can switch between a fast approximate model and the full
dynamical calculation.

* `ParrattModel`: exact dynamical theory (Parratt recursion) with roughness, polarization and
  instrumental resolution.
* `KinematicModel`: single-scattering approximation, i.e. the sum of all interface reflection
  amplitudes with their phases, without multiple reflections. It is evaluated without a loop
  over layers and is therefore much faster. With ``refraction=False`` it corresponds to the
  plain Bragg law (no refraction shift).
"""

from dataclasses import dataclass
from typing import ClassVar, Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from xstack.materials import MaterialLibrary
from xstack.physics.fresnel import (
    Polarization,
    fresnel_coefficients,
    interface_models,
    wavevector_z,
)
from xstack.physics.kernel import reflectance
from xstack.physics.resolution import extended_grid, gaussian_smear
from xstack.plugins import Registry
from xstack.structure.stack import Stack
from xstack.targets import Target
from xstack.units import energy_to_wavelength

__all__ = [
    "KinematicModel",
    "ParrattModel",
    "ReflectivityModel",
    "ReflectivityResult",
    "evaluate_on_target",
    "models",
]


@dataclass(frozen=True)
class ReflectivityResult:
    """Reflectivity of a stack on the grid of one target.

    Attributes:
        target: The target.
        x: Values of the scanned variable (eV or rad), shape (P,).
        energy: Photon energies in eV, shape (P,).
        theta: Grazing angles in rad, shape (P,).
        reflectivity: Reflectance after the target's number of reflections n, i.e. Rⁿ,
            shape (P,).
    """

    target: Target
    x: NDArray[np.float64]
    energy: NDArray[np.float64]
    theta: NDArray[np.float64]
    reflectivity: NDArray[np.float64]


class ReflectivityModel(Protocol):
    """Interface of all reflectivity models."""

    @property
    def name(self) -> str:
        """Short name of the model."""
        ...

    def evaluate(
        self,
        stack: Stack,
        target: Target,
        points: int | None = None,
        library: MaterialLibrary | None = None,
    ) -> ReflectivityResult:
        """Compute the reflectivity of a stack on the grid of a target.

        Args:
            stack: The stack.
            target: The target defining grid, polarization and resolution.
            points: Number of grid points. Defaults to the target's own setting.
            library: Material library for optical constants.

        Returns:
            The reflectivity on the target grid.
        """
        ...


class _PointModel(Protocol):
    def _reflectance(
        self,
        stack: Stack,
        energy: NDArray[np.float64],
        theta: NDArray[np.float64],
        polarization: Polarization | Literal["unpolarized"],
        library: MaterialLibrary | None,
    ) -> NDArray[np.float64]: ...


def evaluate_on_target(
    model: _PointModel,
    stack: Stack,
    target: Target,
    points: int | None,
    library: MaterialLibrary | None,
) -> ReflectivityResult:
    """Evaluate a point-wise reflectance function on a target grid.

    The result includes the target's number of reflections (Rⁿ) and its resolution.

    Args:
        model: Object providing ``_reflectance(stack, energy, theta, polarization, library)``.
        stack: The stack.
        target: The target.
        points: Number of grid points, or ``None`` for the target default.
        library: Material library for optical constants.

    Returns:
        The reflectivity on the target grid.
    """
    n = points or target.points
    x = target.scan_values(n)
    if target.resolution > 0.0:
        lo, hi = target.scan_range
        fine, _ = extended_grid(lo, hi, n, target.resolution)
        fine = fine[fine > 0.0]
        e_fine, th_fine = target.grid_at(fine)
        if target.axis == "angle":
            keep = th_fine <= np.pi / 2
            fine, e_fine, th_fine = fine[keep], e_fine[keep], th_fine[keep]
        r_fine = model._reflectance(stack, e_fine, th_fine, target.polarization, library)
        r_fine = r_fine**target.reflections
        smeared = gaussian_smear(r_fine, float(fine[1] - fine[0]), target.resolution)
        r = np.interp(x, fine, smeared)
    else:
        energy, theta = target.grid_at(x)
        r = model._reflectance(stack, energy, theta, target.polarization, library)
        r = r**target.reflections
    energy, theta = target.grid_at(x)
    return ReflectivityResult(target, x, energy, theta, np.asarray(r, dtype=np.float64))


@dataclass(frozen=True)
class ParrattModel:
    """Exact dynamical reflectivity (Parratt recursion).

    Attributes:
        interface_model: Name of a registered interface roughness model.
    """

    type_name: ClassVar[str] = "parratt"
    interface_model: str = "nevot_croce"

    @property
    def name(self) -> str:
        """Short name of the model."""
        return self.type_name

    def evaluate(
        self,
        stack: Stack,
        target: Target,
        points: int | None = None,
        library: MaterialLibrary | None = None,
    ) -> ReflectivityResult:
        """Compute the reflectivity of a stack on the grid of a target.

        Args:
            stack: The stack.
            target: The target defining grid, polarization and resolution.
            points: Number of grid points. Defaults to the target's own setting.
            library: Material library for optical constants.

        Returns:
            The reflectivity on the target grid.
        """
        return evaluate_on_target(self, stack, target, points, library)

    def _reflectance(
        self,
        stack: Stack,
        energy: NDArray[np.float64],
        theta: NDArray[np.float64],
        polarization: Polarization | Literal["unpolarized"],
        library: MaterialLibrary | None,
    ) -> NDArray[np.float64]:
        n, media = stack.distinct_refractive_indices(energy, library)
        return reflectance(
            n,
            stack.thicknesses(),
            stack.roughnesses(),
            theta,
            energy_to_wavelength(energy),
            polarization,
            self.interface_model,
            media,
        )


@dataclass(frozen=True)
class KinematicModel:
    """Single-scattering (kinematic) approximation of the reflectivity.

    The reflection amplitude is the sum of the reflection amplitudes of all interfaces,
    each with the phase accumulated on the way down and up again; multiple reflections are
    neglected. The intensity is computed as tanh²(|r|), which equals |r|² for weak reflection
    and reproduces the peak reflectivity of a non-absorbing periodic multilayer.

    Attributes:
        refraction: Include refraction. If False, waves propagate with the vacuum wavevector
            inside the stack (absorption is kept) and interface amplitudes follow the
            small-angle Born approximation; Bragg peaks then appear at the angles of the plain
            Bragg law.
        interface_model: Name of a registered interface roughness model.
    """

    type_name: ClassVar[str] = "kinematic"
    refraction: bool = True
    interface_model: str = "nevot_croce"

    @property
    def name(self) -> str:
        """Short name of the model."""
        return self.type_name if self.refraction else f"{self.type_name}_no_refraction"

    def evaluate(
        self,
        stack: Stack,
        target: Target,
        points: int | None = None,
        library: MaterialLibrary | None = None,
    ) -> ReflectivityResult:
        """Compute the reflectivity of a stack on the grid of a target.

        Args:
            stack: The stack.
            target: The target defining grid, polarization and resolution.
            points: Number of grid points. Defaults to the target's own setting.
            library: Material library for optical constants.

        Returns:
            The reflectivity on the target grid.
        """
        return evaluate_on_target(self, stack, target, points, library)

    def _reflectance(
        self,
        stack: Stack,
        energy: NDArray[np.float64],
        theta: NDArray[np.float64],
        polarization: Polarization | Literal["unpolarized"],
        library: MaterialLibrary | None,
    ) -> NDArray[np.float64]:
        if polarization == "unpolarized":
            r_s = self._reflectance(stack, energy, theta, "s", library)
            r_p = self._reflectance(stack, energy, theta, "p", library)
            return 0.5 * (r_s + r_p)
        n = stack.refractive_indices(energy, library)
        wavelength = energy_to_wavelength(energy)
        k = 2.0 * np.pi / wavelength
        if self.refraction:
            kz = wavevector_z(n, theta, wavelength)
            r_int = fresnel_coefficients(kz, n, polarization)
        else:
            sin = np.sin(theta)[:, None]
            # Vacuum propagation with absorption: kz ≈ k·(sinθ + iβ/sinθ).
            kz = k[:, None] * (sin + 1j * n.imag / sin)
            eps = n**2
            r_int = (eps[:, :-1] - eps[:, 1:]) / (4.0 * sin**2)
            if polarization == "p":
                r_int = r_int * np.cos(2.0 * theta)[:, None]
        r_factor = interface_models.get(self.interface_model).reflection_factor(
            kz[:, :-1], kz[:, 1:], stack.roughnesses()
        )
        # Phase of interface j: path through media 1 … j and back.
        path = np.cumsum(kz[:, 1:-1] * stack.thicknesses(), axis=1)
        phase = np.exp(2j * np.concatenate([np.zeros((len(energy), 1)), path], axis=1))
        amplitude = np.abs(np.sum(r_int * r_factor * phase, axis=1))
        return np.asarray(np.tanh(amplitude) ** 2, dtype=np.float64)


models: Registry[ReflectivityModel] = Registry("reflectivity model")
"""Registry of reflectivity models by name."""

for _model in (ParrattModel(), KinematicModel(), KinematicModel(refraction=False)):
    models.register(_model.name, _model)
