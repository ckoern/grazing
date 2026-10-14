"""Evaluation of the input design and of candidates for live display."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from functools import cache

import numpy as np

from xstack.core.materials import Material, default_library
from xstack.core.physics.models import ReflectivityModel, ReflectivityResult
from xstack.core.structure.coating import CoatingBuildError, Mirror
from xstack.core.structure.stack import Stack
from xstack.core.targets import AngleScan, EnergyScan, Target
from xstack.design.manufacturing.tolerance import ToleranceResult
from xstack.design.optim.problem import Evaluation
from xstack.design.project import ProjectState

__all__ = [
    "DisplayRanges",
    "Preview",
    "ToleranceCurves",
    "compute_curves",
    "compute_preview",
    "display_targets",
    "lowest_energy",
    "tolerance_curves",
]

_MAX_POINTS = 400
_MAX_DISPLAY_POINTS = 1000

type DisplayRanges = Mapping[str, tuple[float, float]]
"""Plot ranges by scan axis (``"energy"`` in eV, ``"angle"`` in rad)."""


@dataclass(frozen=True)
class Preview:
    """Everything shown about the input design and the displayed candidates.

    Attributes:
        state: The evaluated project state.
        stack: Stack of the input design, or ``None`` if it cannot be built.
        results: Reflectivity of the input design on the target grids.
        evaluation: Full evaluation of the input design if the project defines a complete
            problem.
        error: Why the input design could not be evaluated, if it could not.
        curves: Reflectivity of the input design on the display grids (equal to `results`
            without display ranges).
        overlays: Reflectivity of candidates on the display grids, as (name, results) pairs.
    """

    state: ProjectState
    stack: Stack | None
    results: tuple[ReflectivityResult, ...] = ()
    evaluation: Evaluation | None = None
    error: str | None = None
    curves: tuple[ReflectivityResult, ...] = ()
    overlays: tuple[tuple[str, tuple[ReflectivityResult, ...]], ...] = ()


@cache
def lowest_energy(materials: frozenset[Material]) -> float:
    """Return the lowest photon energy at which the optical constants of all materials exist.

    The energy is found on a logarithmic grid from 1 eV to 100 keV (about 2 % spacing) using
    the default material library.

    Args:
        materials: The materials.

    Returns:
        The energy in eV; 0 if all materials are vacuum.
    """
    library = default_library()
    real = [m for m in materials if not m.is_vacuum]
    if not real:
        return 0.0
    for energy in np.geomspace(1.0, 1e5, 600):
        try:
            for material in real:
                library.delta_beta(material, float(energy))
        except (ValueError, KeyError):
            continue
        return float(energy)
    return 1e5


def _stack_materials(stacks: Sequence[Stack | None]) -> frozenset[Material]:
    return frozenset(m for stack in stacks if stack is not None for m in stack.media)


def display_targets(
    targets: Sequence[Target], display: DisplayRanges | None, energy_floor: float = 0.0
) -> tuple[Target, ...]:
    """Return the targets with their scan range replaced by a display range.

    The number of points is scaled with the range, so the sampling density stays similar. A
    lower end of zero or below (where no reflectivity can be calculated) is moved to the first
    grid point above zero; for energy scans, it is in addition raised to `energy_floor`.

    Args:
        targets: The targets.
        display: Plot range by scan axis; axes without an entry keep the target range.
        energy_floor: Lowest energy in eV for which reflectivity can be calculated.

    Returns:
        Targets for plotting.
    """
    if not display:
        return tuple(targets)
    shown = []
    for target in targets:
        if target.axis not in display:
            shown.append(target)
            continue
        lo, hi = display[target.axis]
        t_lo, t_hi = target.scan_range
        points = int(
            np.clip(round(target.points * (hi - lo) / (t_hi - t_lo)), 50, _MAX_DISPLAY_POINTS)
        )
        if lo <= 0:
            lo = lo + (hi - lo) / (points - 1)
        if target.axis == "energy":
            lo = max(lo, energy_floor)
        if not lo < hi:
            shown.append(target)
            continue
        if isinstance(target, EnergyScan):
            shown.append(replace(target, energy_min=lo, energy_max=hi, points=points))
        elif isinstance(target, AngleScan):
            shown.append(replace(target, angle_min=lo, angle_max=hi, points=points))
        else:
            shown.append(target)
    return tuple(shown)


def _curves(
    stack: Stack, targets: Sequence[Target], model: ReflectivityModel
) -> tuple[ReflectivityResult, ...]:
    return tuple(model.evaluate(stack, t, min(t.points, _MAX_DISPLAY_POINTS)) for t in targets)


def compute_curves(
    state: ProjectState,
    mirror: Mirror,
    values: Mapping[str, float] | None = None,
    display: DisplayRanges | None = None,
) -> tuple[ReflectivityResult, ...]:
    """Return the reflectivity of a design on the display grids of a project's targets.

    Args:
        state: The project state (targets and model).
        mirror: The design.
        values: Parameter values overriding those of the design.
        display: Plot ranges by scan axis.

    Returns:
        One result per target; empty if the design cannot be built or there are no targets.
    """
    if state.targets is None:
        return ()
    try:
        stack = mirror.stack(values)
    except (CoatingBuildError, ValueError, KeyError):
        return ()
    floor = lowest_energy(_stack_materials([stack]))
    return _curves(stack, display_targets(state.targets.targets, display, floor), state.model)


def compute_preview(
    state: ProjectState,
    overlays: Sequence[str] = (),
    display: DisplayRanges | None = None,
) -> Preview:
    """Evaluate the input design of a project state and the reflectivity of candidates.

    Args:
        state: The project state.
        overlays: Names of stored candidates whose reflectivity is computed as well.
        display: Plot ranges by scan axis; the figure of merit is always evaluated on the
            target ranges.

    Returns:
        The preview; problems are reported in `Preview.error` instead of raising.
    """
    preview = _current(state)
    if state.targets is None:
        return preview
    stacks: list[Stack | None] = [preview.stack]
    for name in overlays:
        try:
            stacks.append(state.candidate(name).mirror.stack())
        except (KeyError, CoatingBuildError, ValueError):
            continue
    floor = lowest_energy(_stack_materials(stacks)) if display else 0.0
    shown = display_targets(state.targets.targets, display, floor)
    curves = preview.results
    if display and preview.stack is not None:
        curves = _curves(preview.stack, shown, state.model)
    computed = []
    for name in overlays:
        try:
            stack = state.candidate(name).mirror.stack()
        except (KeyError, CoatingBuildError, ValueError):
            continue
        computed.append((name, _curves(stack, shown, state.model)))
    return replace(preview, curves=curves, overlays=tuple(computed))


def _current(state: ProjectState) -> Preview:
    if state.mirror is None:
        return Preview(state, None, error="no design defined")
    try:
        stack = state.mirror.stack()
    except (CoatingBuildError, ValueError) as exc:
        return Preview(state, None, error=str(exc))
    if state.targets is None:
        return Preview(state, stack)
    points = min(_MAX_POINTS, max(t.points for t in state.targets.targets))
    if state.merit is None:
        results = tuple(state.model.evaluate(stack, t, points) for t in state.targets.targets)
        return Preview(state, stack, results)
    problem = state.problem().evolve(points=points)
    evaluation = problem.evaluate()
    if evaluation.stack is None:
        return Preview(state, stack, error=evaluation.error)
    return Preview(state, stack, evaluation.results, evaluation)


@dataclass(frozen=True)
class ToleranceCurves:
    """Reflectivity of the nominal design and of the Monte Carlo samples for display.

    Attributes:
        nominal: Reflectivity of the nominal design per target.
        samples: Reflectivity of each sample per target.
    """

    nominal: tuple[ReflectivityResult, ...]
    samples: tuple[tuple[ReflectivityResult, ...], ...]


def tolerance_curves(
    state: ProjectState, result: ToleranceResult, display: DisplayRanges | None = None
) -> ToleranceCurves:
    """Return the reflectivity of a tolerance analysis on the display grids.

    Without display ranges, the curves of the analysis itself (on the target grids) are
    returned; otherwise the reflectivity of the nominal and every perturbed stack is
    calculated on the display grids.

    Args:
        state: The project state (targets and model).
        result: The tolerance analysis result.
        display: Plot ranges by scan axis.

    Returns:
        The curves.
    """
    if not display or state.targets is None:
        return ToleranceCurves(
            result.nominal.results, tuple(sample.results for sample in result.samples)
        )
    stacks = [result.nominal.stack, *(sample.stack for sample in result.samples)]
    floor = lowest_energy(_stack_materials(stacks))
    shown = display_targets(state.targets.targets, display, floor)
    nominal = _curves(result.nominal.stack, shown, state.model) if result.nominal.stack else ()
    samples = tuple(
        _curves(sample.stack, shown, state.model)
        for sample in result.samples
        if sample.stack is not None
    )
    return ToleranceCurves(nominal, samples)
