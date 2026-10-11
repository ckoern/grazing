"""Optical constants and material definitions.

Examples:
    >>> from xstack import materials
    >>> w = materials.get("W")
    >>> n = materials.refractive_index(w, 30000.0)
    >>> bool(n.real < 1.0)
    True
"""

from xstack.materials.library import PRESETS, MaterialLibrary, default_library
from xstack.materials.material import VACUUM, Material
from xstack.materials.providers import (
    DEFAULT_PROVIDER,
    CombinedProvider,
    HenkeProvider,
    OpticalConstantsProvider,
    ScatteringFactorProvider,
    TabulatedProvider,
    XraylibProvider,
    delta_beta,
    providers,
    refractive_index,
)

__all__ = [
    "DEFAULT_PROVIDER",
    "PRESETS",
    "VACUUM",
    "CombinedProvider",
    "HenkeProvider",
    "Material",
    "MaterialLibrary",
    "OpticalConstantsProvider",
    "ScatteringFactorProvider",
    "TabulatedProvider",
    "XraylibProvider",
    "default_library",
    "delta_beta",
    "get",
    "providers",
    "refractive_index",
]


def get(name: str) -> Material:
    """Return a material from the default library.

    Args:
        name: Name of the material, e.g. ``"W"`` or ``"B4C"``.

    Returns:
        The material.

    Raises:
        KeyError: If the default library has no material with this name.
    """
    return default_library().get(name)
