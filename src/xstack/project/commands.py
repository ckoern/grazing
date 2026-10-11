"""Commands for common project changes and project-level actions.

Each function returns a `Command` to pass to `Project.execute`, so the change can be undone
and is recorded in the journal.

Examples:
    >>> from xstack.materials import get
    >>> from xstack.structure import Mirror, Periodic
    >>> from xstack.project import Project
    >>> project = Project()
    >>> project.execute(set_mirror(Mirror(Periodic.create(get("W"), get("Si")), get("Si"))))
    >>> project.execute(set_parameter("N", vary=True, min=20, max=80))
    >>> project.state.mirror.parameters["N"].vary
    True
"""

from typing import Any

from xstack.manufacturing.constraints import Constraint
from xstack.manufacturing.process import DepositionProcess
from xstack.merit.base import FigureOfMerit
from xstack.optim.events import CancellationToken, EventSink
from xstack.optim.pipeline import Pipeline, PipelineResult
from xstack.optim.problem import DesignProblem, Evaluation
from xstack.parameters import ParameterSet
from xstack.physics.models import ReflectivityModel
from xstack.project.journal import Journal
from xstack.project.project import Candidate, Command, Project, ProjectState
from xstack.structure.coating import Coating, Mirror
from xstack.targets import TargetSet

__all__ = [
    "add_candidate",
    "add_constraint",
    "apply_result",
    "promote_candidate",
    "record_run",
    "remove_candidate",
    "remove_constraint",
    "rename_project",
    "replace_constraint",
    "run_pipeline",
    "set_coating",
    "set_merit",
    "set_mirror",
    "set_model",
    "set_notes",
    "set_parameter",
    "set_parameters",
    "set_pipeline",
    "set_process",
    "set_targets",
]


def _require_mirror(state: ProjectState) -> Mirror:
    if state.mirror is None:
        raise ValueError("the project has no design yet")
    return state.mirror


def rename_project(name: str) -> Command:
    """Rename the project.

    Args:
        name: New project name.

    Returns:
        The command.
    """
    return Command(f"Rename project to {name!r}", lambda s: s.evolve(name=name))


def set_notes(notes: str) -> Command:
    """Replace the project notes.

    Args:
        notes: The notes.

    Returns:
        The command.
    """
    return Command("Edit notes", lambda s: s.evolve(notes=notes), merge_key="notes")


def set_mirror(mirror: Mirror) -> Command:
    """Replace the design (coating and substrate).

    Args:
        mirror: The new design.

    Returns:
        The command.
    """
    return Command(f"Set design ({mirror.coating.type_name})", lambda s: s.evolve(mirror=mirror))


def set_coating(coating: Coating) -> Command:
    """Replace the coating of the current design.

    Args:
        coating: The new coating.

    Returns:
        The command.
    """

    def transform(state: ProjectState) -> ProjectState:
        return state.evolve(mirror=_require_mirror(state).with_coating(coating))

    return Command(f"Set coating ({coating.type_name})", transform)


def set_parameter(name: str, **changes: Any) -> Command:
    """Modify one coating parameter (value, free/fixed, bounds, expression, ...).

    Consecutive edits of the same parameter form a single undo step.

    Args:
        name: Parameter name.
        **changes: New values for fields of the `xstack.parameters.Parameter`.

    Returns:
        The command.
    """
    summary = ", ".join(f"{k}={v!r}" for k, v in changes.items())

    def transform(state: ProjectState) -> ProjectState:
        mirror = _require_mirror(state)
        coating = mirror.coating.with_parameter(name, **changes)
        return state.evolve(mirror=mirror.with_coating(coating))

    return Command(f"Set {name}: {summary}", transform, merge_key=f"parameter:{name}")


def set_parameters(parameters: ParameterSet, description: str = "Set parameters") -> Command:
    """Replace all coating parameters.

    Args:
        parameters: The new parameter set (same names).
        description: Description for the journal.

    Returns:
        The command.
    """

    def transform(state: ProjectState) -> ProjectState:
        mirror = _require_mirror(state)
        return state.evolve(mirror=mirror.with_coating(mirror.coating.with_parameters(parameters)))

    return Command(description, transform)


def set_targets(targets: TargetSet) -> Command:
    """Replace the targets.

    Args:
        targets: The new targets.

    Returns:
        The command.
    """
    return Command(f"Set {len(targets)} target(s)", lambda s: s.evolve(targets=targets))


def set_merit(merit: FigureOfMerit) -> Command:
    """Replace the figure of merit.

    Args:
        merit: The new figure of merit.

    Returns:
        The command.
    """
    return Command(f"Set figure of merit ({merit.type_name})", lambda s: s.evolve(merit=merit))


def set_process(process: DepositionProcess | None) -> Command:
    """Replace or remove the deposition process.

    Args:
        process: The new process, or ``None`` to remove it.

    Returns:
        The command.
    """
    text = "Remove deposition process" if process is None else "Set deposition process"
    return Command(text, lambda s: s.evolve(process=process))


def set_model(model: ReflectivityModel) -> Command:
    """Replace the reflectivity model.

    Args:
        model: The new model.

    Returns:
        The command.
    """
    return Command(f"Set reflectivity model ({model.name})", lambda s: s.evolve(model=model))


def set_pipeline(pipeline: Pipeline | None) -> Command:
    """Replace the pipeline, or use the default pipeline with ``None``.

    Args:
        pipeline: The new pipeline.

    Returns:
        The command.
    """
    return Command("Set pipeline", lambda s: s.evolve(pipeline=pipeline))


def add_constraint(constraint: Constraint) -> Command:
    """Add a constraint.

    Args:
        constraint: The constraint.

    Returns:
        The command.
    """
    return Command(
        f"Add constraint ({constraint.type_name})",
        lambda s: s.evolve(constraints=(*s.constraints, constraint)),
    )


def remove_constraint(index: int) -> Command:
    """Remove a constraint.

    Args:
        index: Position of the constraint.

    Returns:
        The command.
    """

    def transform(state: ProjectState) -> ProjectState:
        items = list(state.constraints)
        del items[index]
        return state.evolve(constraints=tuple(items))

    return Command(f"Remove constraint {index}", transform)


def replace_constraint(index: int, constraint: Constraint) -> Command:
    """Replace a constraint, e.g. with modified settings.

    Args:
        index: Position of the constraint.
        constraint: The new constraint.

    Returns:
        The command.
    """

    def transform(state: ProjectState) -> ProjectState:
        items = list(state.constraints)
        items[index] = constraint
        return state.evolve(constraints=tuple(items))

    return Command(
        f"Edit constraint {index} ({constraint.type_name})", transform, f"constraint:{index}"
    )


def _unique_name(state: ProjectState, name: str) -> str:
    existing = {c.name for c in state.candidates}
    if name not in existing:
        return name
    i = 2
    while f"{name} ({i})" in existing:
        i += 1
    return f"{name} ({i})"


def add_candidate(candidate: Candidate) -> Command:
    """Store a candidate design; its name is made unique if necessary.

    Args:
        candidate: The candidate.

    Returns:
        The command.
    """

    def transform(state: ProjectState) -> ProjectState:
        stored = Candidate(**{**candidate.__dict__, "name": _unique_name(state, candidate.name)})
        return state.evolve(candidates=(*state.candidates, stored))

    return Command(f"Store candidate {candidate.name!r}", transform)


def remove_candidate(name: str) -> Command:
    """Delete a stored candidate.

    Args:
        name: Candidate name.

    Returns:
        The command.
    """

    def transform(state: ProjectState) -> ProjectState:
        state.candidate(name)
        return state.evolve(candidates=tuple(c for c in state.candidates if c.name != name))

    return Command(f"Delete candidate {name!r}", transform)


def promote_candidate(name: str) -> Command:
    """Make a stored candidate the current design.

    Args:
        name: Candidate name.

    Returns:
        The command.
    """
    return Command(
        f"Use candidate {name!r} as current design",
        lambda s: s.evolve(mirror=s.candidate(name).mirror),
    )


def record_run(journal: Journal, result: PipelineResult) -> None:
    """Add a journal entry summarizing a pipeline run.

    Args:
        journal: The journal.
        result: The pipeline result.
    """
    summary = {
        "stages": [
            {
                "name": s.stage.name,
                "evaluations": s.result.evaluations,
                "objective": s.result.best.objective,
                "elapsed_s": round(s.result.elapsed, 3),
            }
            for s in result.stages
        ],
        "objective": result.final.objective,
        "merit": result.final.merit,
        "cancelled": result.cancelled,
    }
    journal.add(
        "run",
        f"Ran pipeline ({len(result.stages)} stages): objective {result.final.objective:.5g}"
        + (" (cancelled)" if result.cancelled else ""),
        **summary,
    )


def apply_result(
    problem: DesignProblem,
    parameters: ParameterSet,
    evaluation: Evaluation,
    candidate_name: str,
    source: str,
) -> Command:
    """Make an optimized design the current design and store it as a candidate.

    Args:
        problem: The design problem that was optimized.
        parameters: The optimized parameters.
        evaluation: Evaluation of the optimized design.
        candidate_name: Name of the stored candidate (made unique if necessary).
        source: Origin of the design, e.g. ``"pipeline"`` or a stage name.

    Returns:
        The command.
    """
    mirror = problem.with_parameters(parameters).mirror
    candidate = Candidate.from_evaluation(candidate_name, mirror, evaluation, source)

    def transform(s: ProjectState) -> ProjectState:
        stored = Candidate(**{**candidate.__dict__, "name": _unique_name(s, candidate.name)})
        return s.evolve(mirror=mirror, candidates=(*s.candidates, stored))

    return Command(f"Apply result {candidate_name!r}", transform)


def run_pipeline(
    project: Project,
    pipeline: Pipeline | None = None,
    events: EventSink | None = None,
    cancel: CancellationToken | None = None,
    candidate_name: str = "optimized",
) -> PipelineResult:
    """Run a pipeline on the project and store the result.

    The final design becomes the current design and is stored as a candidate (one undoable
    step); the run is recorded in the journal.

    Args:
        project: The project.
        pipeline: Pipeline to run. Defaults to the project's pipeline or the default pipeline.
        events: Receiver of progress events.
        cancel: Cancellation token.
        candidate_name: Name of the stored candidate.

    Returns:
        The pipeline result.

    Raises:
        ValueError: If the project does not define a complete design problem.
    """
    problem = project.state.problem()
    chosen = pipeline or project.state.pipeline or Pipeline.default(problem)
    result = chosen.run(problem, events=events, cancel=cancel)
    record_run(project.journal, result)
    project.execute(
        apply_result(problem, result.parameters, result.final, candidate_name, "pipeline")
    )
    return result
