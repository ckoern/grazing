"""Evaluation of the input design and of candidates for live display."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np

from xstack.optim.problem import Evaluation
from xstack.physics.models import ReflectivityModel, ReflectivityResult
from xstack.project import ProjectState
from xstack.structure.coating import CoatingBuildError, Mirror
from xstack.structure.stack import Stack
from xstack.targets import AngleScan, EnergyScan, Target

__all__ = ["DisplayRanges", "Preview", "compute_curves", "compute_preview", "display_targets"]

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


def display_targets(targets: Sequence[Target], display: DisplayRanges | None) -> tuple[Target, ...]:
    """Return the targets with their scan range replaced by a display range.

    The number of points is scaled with the range, so the sampling density stays similar.

    Args:
        targets: The targets.
        display: Plot range by scan axis; axes without an entry keep the target range.

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
    return _curves(stack, display_targets(state.targets.targets, display), state.model)


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
    shown = display_targets(state.targets.targets, display)
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
