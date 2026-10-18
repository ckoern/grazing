"""Ready-made fit project states with a starting structure.

The structures are typical starting points for fitting: the parameters that a reflectivity
measurement determines well (thicknesses, period, Γ, roughness) are free within generous
bounds, layer counts are fixed.

Examples:
    >>> state = new_state("periodic")
    >>> state.mirror.parameters.free_names()
    ('d', 'gamma', 'sigma_H', 'sigma_L')
"""

from collections.abc import Callable

from xstack.core.materials import get
from xstack.core.parameters import free
from xstack.core.structure.coating import Mirror
from xstack.core.structure.coatings.periodic import Periodic
from xstack.core.structure.coatings.simple import SingleLayer
from xstack.fit.project import FitState

__all__ = ["TEMPLATES", "new_state"]


def _periodic() -> FitState:
    w, si = get("W"), get("Si")
    coating = Periodic.create(
        w,
        si,
        N=20,
        d=free(4.0, 2.0, 8.0),
        gamma=free(0.4, 0.15, 0.7),
        sigma_H=free(0.3, 0.0, 1.5),
        sigma_L=free(0.3, 0.0, 1.5),
    )
    return FitState(name="Periodic W/Si", mirror=Mirror(coating, si))


def _single_layer() -> FitState:
    coating = SingleLayer.create(get("Pt"), t=free(30.0, 5.0, 100.0), sigma=free(0.5, 0.0, 3.0))
    return FitState(name="Single-layer Pt", mirror=Mirror(coating, get("SiO2")))


def _empty() -> FitState:
    return FitState()


TEMPLATES: dict[str, Callable[[], FitState]] = {
    "periodic": _periodic,
    "single_layer": _single_layer,
    "empty": _empty,
}
"""Factories of template states by name."""


def new_state(template: str = "periodic") -> FitState:
    """Return a new fit state from a template.

    Args:
        template: Name of a template in `TEMPLATES`.

    Returns:
        The state, without measured data.

    Raises:
        KeyError: If the template is unknown.
    """
    try:
        return TEMPLATES[template]()
    except KeyError:
        known = ", ".join(TEMPLATES)
        raise KeyError(f"unknown template {template!r} (known: {known})") from None
