"""Ready-made project states to start a new design from."""

from collections.abc import Callable

from grazing.core.materials import get
from grazing.core.parameters import free
from grazing.core.structure.coating import Mirror
from grazing.core.structure.coatings.graded import DepthGraded
from grazing.core.structure.coatings.periodic import Periodic
from grazing.core.structure.coatings.simple import SingleLayer
from grazing.core.targets import AngleScan, EnergyScan, TargetSet
from grazing.core.units import deg, kev
from grazing.design.manufacturing.constraints import MaxTotalThickness, MinLayerThickness
from grazing.design.merit.optical import MeanReflectivity
from grazing.design.project.project import ProjectState

__all__ = ["TEMPLATES", "new_state"]


def _periodic() -> ProjectState:
    w, si = get("W"), get("Si")
    coating = Periodic.create(
        w, si, N=free(40, 10, 200), d=free(4.0, 2.0, 8.0), gamma=free(0.4, 0.15, 0.7)
    )
    return ProjectState(
        name="Periodic W/Si",
        mirror=Mirror(coating, si),
        targets=TargetSet(EnergyScan(angle=deg(1.1), energy_min=7900.0, energy_max=8200.0)),
        merit=MeanReflectivity(),
        constraints=(MinLayerThickness(0.8), MaxTotalThickness(1000.0)),
    )


def _depth_graded() -> ProjectState:
    w, si = get("W"), get("Si")
    coating = DepthGraded.create(
        w,
        si,
        N=free(150, 20, 400),
        d_min=free(3.0, 1.5, 6.0),
        d_max=free(10.0, 5.0, 25.0),
        c=0.25,
        gamma_top=free(0.4, 0.15, 0.7),
        gamma_bottom=free(0.4, 0.15, 0.7),
    )
    return ProjectState(
        name="Depth-graded W/Si",
        mirror=Mirror(coating, si),
        targets=TargetSet(EnergyScan(angle=deg(0.2), energy_min=kev(15), energy_max=kev(70))),
        merit=MeanReflectivity(),
        constraints=(MinLayerThickness(0.8), MaxTotalThickness(1500.0)),
    )


def _single_layer() -> ProjectState:
    coating = SingleLayer.create(get("Pt"), t=free(30.0, 5.0, 100.0))
    return ProjectState(
        name="Single-layer Pt mirror",
        mirror=Mirror(coating, get("SiO2")),
        targets=TargetSet(AngleScan(energy=8048.0, angle_min=1e-3, angle_max=deg(0.5))),
        merit=MeanReflectivity(),
    )


TEMPLATES: dict[str, Callable[[], ProjectState]] = {
    "periodic": _periodic,
    "depth_graded": _depth_graded,
    "single_layer": _single_layer,
}
"""Factories of template states by name."""


def new_state(template: str = "periodic") -> ProjectState:
    """Return a new project state from a template.

    Args:
        template: Name of a template in `TEMPLATES`.

    Returns:
        The project state.

    Raises:
        KeyError: If the template is unknown.
    """
    try:
        return TEMPLATES[template]()
    except KeyError:
        raise KeyError(
            f"unknown template {template!r} (available: {', '.join(TEMPLATES)})"
        ) from None
