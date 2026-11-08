"""Direct reflectivity calculations for simple scripts.

The functions take a mirror and the scanned values and return the reflectivity, without models,
targets or evaluation contexts to set up: `reflectivity_vs_angle` scans the grazing angle,
`reflectivity_vs_energy` the photon energy, and `reflectivity_map` gives a two-dimensional grid of
energies and angles. They compute a single reflection at the given points,
without instrument resolution. For these, and for figures of merit, use `grazing.core.targets`
with a reflectivity model.

Examples:
    >>> import numpy as np
    >>> from grazing.core.materials import get
    >>> from grazing.core.structure import Mirror, Periodic
    >>> from grazing.core.units import deg
    >>> mirror = Mirror(Periodic.create(get("W"), get("Si"), N=30, d=4.0), get("Si"))
    >>> angles = deg(np.linspace(0.2, 3.0, 100))
    >>> reflectivity_vs_angle(mirror, angles, energy=8048.0).shape
    (100,)
    >>> reflectivity_map(mirror, energy=[8048.0, 17479.0], angle=angles).shape
    (2, 100)
"""

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from grazing.core.materials import provider_library
from grazing.core.physics.models import ParrattModel, ReflectivityModel, models
from grazing.core.structure.coating import Mirror
from grazing.core.structure.stack import Stack

__all__ = ["reflectivity_map", "reflectivity_vs_angle", "reflectivity_vs_energy"]

type _Polarization = Literal["s", "p", "unpolarized"]


def _calculate(
    mirror: Mirror | Stack,
    energy: ArrayLike,
    angle: ArrayLike,
    polarization: _Polarization,
    model: ReflectivityModel | str | None,
    provider: str | None,
) -> NDArray[np.float64]:
    if polarization not in ("s", "p", "unpolarized"):
        raise ValueError(f"unknown polarization {polarization!r}")
    e, theta = np.broadcast_arrays(
        np.asarray(energy, dtype=np.float64), np.asarray(angle, dtype=np.float64)
    )
    if np.any(e <= 0.0):
        raise ValueError("photon energies must be positive (in eV)")
    if np.any(theta <= 0.0) or np.any(theta > np.pi / 2.0):
        raise ValueError("grazing angles must be in (0, π/2] rad; convert degrees with units.deg")
    if model is None:
        chosen: ReflectivityModel = ParrattModel()
    elif isinstance(model, str):
        chosen = models.get(model)
    else:
        chosen = model
    library = None if provider is None else provider_library(provider)
    stack = mirror if isinstance(mirror, Stack) else mirror.stack()
    result = chosen.reflectance(stack, e.flatten(), theta.flatten(), polarization, library)
    return result.reshape(e.shape)


def reflectivity_vs_angle(
    mirror: Mirror | Stack,
    angle: ArrayLike,
    *,
    energy: ArrayLike,
    polarization: _Polarization = "unpolarized",
    model: ReflectivityModel | str | None = None,
    provider: str | None = None,
) -> NDArray[np.float64]:
    """Return the reflectivity of a mirror as a function of the grazing angle.

    Args:
        mirror: The mirror or an already built layer stack.
        angle: Grazing angles in rad, in (0, π/2]. See `grazing.core.units.deg`.
        energy: Photon energy in eV; broadcast against `angle`. For a grid of energies and
            angles, `reflectivity_map` is clearer.
        polarization: ``"s"``, ``"p"`` or ``"unpolarized"``.
        model: A reflectivity model or the name of a registered one (``"parratt"``,
            ``"kinematic"``, ``"kinematic_no_refraction"``). Defaults to `ParrattModel`.
        provider: Name of the optical-constants provider for materials without their own.
            Defaults to the default provider.

    Returns:
        Reflectivity of a single reflection, with the broadcast shape of `angle` and `energy`.

    Raises:
        CoatingBuildError: If the parameters of the mirror do not describe a valid stack.
        KeyError: If the model or provider is not registered.
        ValueError: If an energy or angle is out of range, or the polarization is unknown.

    Examples:
        >>> from grazing.core.materials import get
        >>> from grazing.core.structure.stack import Stack
        >>> r = reflectivity_vs_angle(Stack([], get("Si")), 1e-3, energy=8048.0)
        >>> bool(r > 0.9)
        True
    """
    return _calculate(mirror, energy, angle, polarization, model, provider)


def reflectivity_vs_energy(
    mirror: Mirror | Stack,
    energy: ArrayLike,
    *,
    angle: ArrayLike,
    polarization: _Polarization = "unpolarized",
    model: ReflectivityModel | str | None = None,
    provider: str | None = None,
) -> NDArray[np.float64]:
    """Return the reflectivity of a mirror as a function of the photon energy.

    Args:
        mirror: The mirror or an already built layer stack.
        energy: Photon energies in eV. See `grazing.core.units.kev`.
        angle: Grazing angle in rad, in (0, π/2]; broadcast against `energy`. For a grid of
            energies and angles, `reflectivity_map` is clearer.
        polarization: ``"s"``, ``"p"`` or ``"unpolarized"``.
        model: A reflectivity model or the name of a registered one (``"parratt"``,
            ``"kinematic"``, ``"kinematic_no_refraction"``). Defaults to `ParrattModel`.
        provider: Name of the optical-constants provider for materials without their own.
            Defaults to the default provider.

    Returns:
        Reflectivity of a single reflection, with the broadcast shape of `energy` and `angle`.

    Raises:
        CoatingBuildError: If the parameters of the mirror do not describe a valid stack.
        KeyError: If the model or provider is not registered.
        ValueError: If an energy or angle is out of range, or the polarization is unknown.

    Examples:
        >>> from grazing.core.materials import get
        >>> from grazing.core.structure.stack import Stack
        >>> r = reflectivity_vs_energy(Stack([], get("Si")), 8048.0, angle=1e-3)
        >>> bool(r > 0.9)
        True
    """
    return _calculate(mirror, energy, angle, polarization, model, provider)


def reflectivity_map(
    mirror: Mirror | Stack,
    *,
    energy: ArrayLike,
    angle: ArrayLike,
    polarization: _Polarization = "unpolarized",
    model: ReflectivityModel | str | None = None,
    provider: str | None = None,
) -> NDArray[np.float64]:
    """Return the reflectivity of a mirror on a grid of photon energies and grazing angles.

    Every energy is combined with every angle. The result always has two dimensions, with one
    row per energy and one column per angle, also when one of the axes has a single value.

    Args:
        mirror: The mirror or an already built layer stack.
        energy: Photon energies in eV, a scalar or a one-dimensional sequence. See
            `grazing.core.units.kev`.
        angle: Grazing angles in rad, in (0, π/2], a scalar or a one-dimensional sequence. See
            `grazing.core.units.deg`.
        polarization: ``"s"``, ``"p"`` or ``"unpolarized"``.
        model: A reflectivity model or the name of a registered one (``"parratt"``,
            ``"kinematic"``, ``"kinematic_no_refraction"``). Defaults to `ParrattModel`.
        provider: Name of the optical-constants provider for materials without their own.
            Defaults to the default provider.

    Returns:
        Reflectivity of a single reflection with shape ``(len(energy), len(angle))``: element
        ``[i, j]`` belongs to ``energy[i]`` and ``angle[j]``.

    Raises:
        CoatingBuildError: If the parameters of the mirror do not describe a valid stack.
        KeyError: If the model or provider is not registered.
        ValueError: If `energy` or `angle` has more than one dimension, an energy or angle is out
            of range, or the polarization is unknown.

    Examples:
        >>> from grazing.core.materials import get
        >>> from grazing.core.structure.stack import Stack
        >>> r = reflectivity_map(Stack([], get("Si")), energy=[8048.0, 17479.0], angle=1e-3)
        >>> r.shape
        (2, 1)
    """
    e = np.asarray(energy, dtype=np.float64)
    theta = np.asarray(angle, dtype=np.float64)
    if e.ndim > 1 or theta.ndim > 1:
        raise ValueError("energy and angle must be scalars or one-dimensional sequences")
    e, theta = np.atleast_1d(e), np.atleast_1d(theta)
    return _calculate(mirror, e[:, None], theta[None, :], polarization, model, provider)
