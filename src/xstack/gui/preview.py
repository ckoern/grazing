"""Evaluation of the current design for live display."""

from collections.abc import Sequence
from dataclasses import dataclass, replace

from xstack.optim.problem import Evaluation
from xstack.physics.models import ReflectivityResult
from xstack.project import ProjectState
from xstack.structure.coating import CoatingBuildError
from xstack.structure.stack import Stack

__all__ = ["Preview", "compute_preview"]

_MAX_POINTS = 400


@dataclass(frozen=True)
class Preview:
    """Everything shown about the current design.

    Attributes:
        state: The evaluated project state.
        stack: The stack, or ``None`` if it cannot be built.
        results: Reflectivity per target.
        evaluation: Full evaluation if the project defines a complete problem.
        error: Why the design could not be evaluated, if it could not.
        overlays: Reflectivity of stored candidates per target, as (name, results) pairs.
    """

    state: ProjectState
    stack: Stack | None
    results: tuple[ReflectivityResult, ...] = ()
    evaluation: Evaluation | None = None
    error: str | None = None
    overlays: tuple[tuple[str, tuple[ReflectivityResult, ...]], ...] = ()


def compute_preview(state: ProjectState, overlays: Sequence[str] = ()) -> Preview:
    """Evaluate the current design of a project state.

    Args:
        state: The project state.
        overlays: Names of stored candidates whose reflectivity is computed as well.

    Returns:
        The preview; problems are reported in `Preview.error` instead of raising.
    """
    preview = _current(state)
    if not overlays or state.targets is None:
        return preview
    points = min(_MAX_POINTS, max(t.points for t in state.targets.targets))
    computed = []
    for name in overlays:
        try:
            stack = state.candidate(name).mirror.stack()
        except (KeyError, CoatingBuildError, ValueError):
            continue
        results = tuple(state.model.evaluate(stack, t, points) for t in state.targets.targets)
        computed.append((name, results))
    return replace(preview, overlays=tuple(computed))


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
