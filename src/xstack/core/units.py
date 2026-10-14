"""Physical constants and unit conversions.

xstack uses the following internal units throughout the library:

* length: nm
* energy: eV
* angle: rad
* time: s
* density: g/cm³

The functions in this module convert between these internal units and other common units.
They accept scalars or numpy arrays and return the same kind of object.

Examples:
    >>> from xstack.core import units
    >>> round(units.energy_to_wavelength(units.kev(8.048)), 5)
    0.15406
    >>> round(units.deg(0.5), 6)
    0.008727
"""

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "AVOGADRO",
    "CLASSICAL_ELECTRON_RADIUS",
    "HC",
    "angstrom",
    "deg",
    "energy_to_wavelength",
    "kev",
    "mrad",
    "to_angstrom",
    "to_deg",
    "to_kev",
    "to_mrad",
    "to_um",
    "um",
    "wavelength_to_energy",
]

HC: float = 1239.8419843320026
"""Planck constant times speed of light, in eV·nm."""

CLASSICAL_ELECTRON_RADIUS: float = 2.8179403262e-6
"""Classical electron radius r_e, in nm."""

AVOGADRO: float = 6.02214076e23
"""Avogadro constant, in 1/mol."""

_DEG = np.pi / 180.0


def kev[T: (float, NDArray[np.float64])](value: T) -> T:
    """Convert an energy from keV to eV.

    Args:
        value: Energy in keV.

    Returns:
        Energy in eV.
    """
    return value * 1e3


def to_kev[T: (float, NDArray[np.float64])](value: T) -> T:
    """Convert an energy from eV to keV.

    Args:
        value: Energy in eV.

    Returns:
        Energy in keV.
    """
    return value * 1e-3


def deg[T: (float, NDArray[np.float64])](value: T) -> T:
    """Convert an angle from degrees to radians.

    Args:
        value: Angle in degrees.

    Returns:
        Angle in rad.
    """
    return value * _DEG


def to_deg[T: (float, NDArray[np.float64])](value: T) -> T:
    """Convert an angle from radians to degrees.

    Args:
        value: Angle in rad.

    Returns:
        Angle in degrees.
    """
    return value / _DEG


def mrad[T: (float, NDArray[np.float64])](value: T) -> T:
    """Convert an angle from mrad to rad.

    Args:
        value: Angle in mrad.

    Returns:
        Angle in rad.
    """
    return value * 1e-3


def to_mrad[T: (float, NDArray[np.float64])](value: T) -> T:
    """Convert an angle from rad to mrad.

    Args:
        value: Angle in rad.

    Returns:
        Angle in mrad.
    """
    return value * 1e3


def angstrom[T: (float, NDArray[np.float64])](value: T) -> T:
    """Convert a length from Å to nm.

    Args:
        value: Length in Å.

    Returns:
        Length in nm.
    """
    return value * 0.1


def to_angstrom[T: (float, NDArray[np.float64])](value: T) -> T:
    """Convert a length from nm to Å.

    Args:
        value: Length in nm.

    Returns:
        Length in Å.
    """
    return value * 10.0


def um[T: (float, NDArray[np.float64])](value: T) -> T:
    """Convert a length from µm to nm.

    Args:
        value: Length in µm.

    Returns:
        Length in nm.
    """
    return value * 1e3


def to_um[T: (float, NDArray[np.float64])](value: T) -> T:
    """Convert a length from nm to µm.

    Args:
        value: Length in nm.

    Returns:
        Length in µm.
    """
    return value * 1e-3


def energy_to_wavelength[T: (float, NDArray[np.float64])](energy: T) -> T:
    """Convert a photon energy to its vacuum wavelength.

    Args:
        energy: Photon energy in eV. Must be positive.

    Returns:
        Wavelength in nm.
    """
    return HC / energy


def wavelength_to_energy[T: (float, NDArray[np.float64])](wavelength: T) -> T:
    """Convert a vacuum wavelength to its photon energy.

    Args:
        wavelength: Wavelength in nm. Must be positive.

    Returns:
        Photon energy in eV.
    """
    return HC / wavelength
