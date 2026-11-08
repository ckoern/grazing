"""Reflectivity of a `Stack` for arbitrary energies and angles."""

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from grazing.core.materials import MaterialLibrary
from grazing.core.physics.fresnel import Polarization
from grazing.core.physics.kernel import KernelResult, parratt, reflectance
from grazing.core.structure.stack import Stack
from grazing.core.units import energy_to_wavelength

__all__ = ["reflectivity", "stack_response"]


def _prepare(
    stack: Stack, energy: ArrayLike, theta: ArrayLike, library: MaterialLibrary | None
) -> tuple[
    tuple[int, ...],
    NDArray[np.float64],
    NDArray[np.float64],
    tuple[NDArray[np.complex128], NDArray[np.intp]],
]:
    e, th = np.broadcast_arrays(
        np.asarray(energy, dtype=np.float64), np.asarray(theta, dtype=np.float64)
    )
    shape = e.shape
    e_flat, th_flat = e.ravel(), th.ravel()
    return shape, e_flat, th_flat, stack.distinct_refractive_indices(e_flat, library)


def reflectivity(
    stack: Stack,
    energy: ArrayLike,
    theta: ArrayLike,
    polarization: Polarization | Literal["unpolarized"] = "unpolarized",
    interface_model: str = "nevot_croce",
    library: MaterialLibrary | None = None,
) -> NDArray[np.float64]:
    """Compute the reflectance of a stack.

    `energy` and `theta` are broadcast against each other, so an energy scan at fixed angle,
    an angle scan at fixed energy, or a 2-D map can be computed with one call.

    Args:
        stack: The layer stack.
        energy: Photon energies in eV.
        theta: Grazing angles of incidence in rad, in (0, π/2].
        polarization: ``"s"``, ``"p"`` or ``"unpolarized"``.
        interface_model: Name of a registered interface roughness model.
        library: Material library for optical constants. Defaults to the shared library.

    Returns:
        Reflectance with the broadcast shape of `energy` and `theta`.

    Examples:
        >>> from grazing.core.materials import get
        >>> from grazing.core.structure.stack import Stack
        >>> r = reflectivity(Stack([], get("Si")), 8048.0, np.array([1e-3, 20e-3]))
        >>> bool(r[0] > 0.9 and r[1] < 1e-3)
        True
    """
    shape, e, th, (n, media) = _prepare(stack, energy, theta, library)
    r = reflectance(
        n,
        stack.thicknesses(),
        stack.roughnesses(),
        th,
        energy_to_wavelength(e),
        polarization,
        interface_model,
        media,
    )
    return r.reshape(shape)


def stack_response(
    stack: Stack,
    energy: ArrayLike,
    theta: ArrayLike,
    polarization: Polarization = "s",
    interface_model: str = "nevot_croce",
    library: MaterialLibrary | None = None,
) -> KernelResult:
    """Compute complex reflection and transmission amplitudes of a stack.

    Args:
        stack: The layer stack.
        energy: Photon energies in eV.
        theta: Grazing angles of incidence in rad, in (0, π/2].
        polarization: ``"s"`` or ``"p"``.
        interface_model: Name of a registered interface roughness model.
        library: Material library for optical constants. Defaults to the shared library.

    Returns:
        Kernel result with all arrays flattened to one dimension in the broadcast order of
        `energy` and `theta`.
    """
    _, e, th, (n, media) = _prepare(stack, energy, theta, library)
    return parratt(
        n,
        stack.thicknesses(),
        stack.roughnesses(),
        th,
        energy_to_wavelength(e),
        polarization,
        interface_model,
        media,
    )
