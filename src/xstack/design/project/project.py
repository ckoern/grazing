"""Project state, candidate designs and undo/redo history."""

import json
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from xstack.core.physics.models import ParrattModel, ReflectivityModel
from xstack.core.serialization import from_data, register_serializable, to_data
from xstack.core.structure.coating import Mirror
from xstack.core.targets import TargetSet
from xstack.design.manufacturing.constraints import Constraint
from xstack.design.manufacturing.process import DepositionProcess
from xstack.design.merit.base import FigureOfMerit
from xstack.design.optim.pipeline import Pipeline
from xstack.design.optim.problem import DesignProblem, Evaluation
from xstack.design.optim.tradeoff import TradeoffResult
from xstack.design.project.journal import Journal

__all__ = ["SCHEMA_VERSION", "Candidate", "Command", "History", "Project", "ProjectState"]

SCHEMA_VERSION = 1
"""Version of the project file format."""


@dataclass(frozen=True)
class Candidate:
    """A stored design.

    Attributes:
        name: Unique name within the project.
        mirror: The design (coating with its parameter values, substrate).
        source: Where the design comes from, e.g. ``"refine"`` or ``"manual"``.
        created: Creation time, ISO 8601 in UTC.
        objective: Objective of the design when it was stored, if evaluated.
        merit: Figure of merit of the design when it was stored, if evaluated.
        coating_time: Coating time in s, if evaluated with a process.
        notes: Free-form notes.
    """

    name: str
    mirror: Mirror
    source: str = "manual"
    created: str = ""
    objective: float | None = None
    merit: float | None = None
    coating_time: float | None = None
    notes: str = ""

    @classmethod
    def from_evaluation(
        cls, name: str, mirror: Mirror, evaluation: Evaluation, source: str
    ) -> "Candidate":
        """Create a candidate with the summary of an evaluation.

        Args:
            name: Candidate name.
            mirror: The design.
            evaluation: Its evaluation.
            source: Origin of the design.

        Returns:
            The candidate.
        """
        return cls(
            name,
            mirror,
            source,
            datetime.now(UTC).isoformat(timespec="seconds"),
            evaluation.objective,
            evaluation.merit,
            evaluation.coating_time,
        )


@dataclass(frozen=True)
class ProjectState:
    """Complete, immutable description of a design project.

    Attributes:
        name: Project name.
        mirror: The input design: the coating as defined by the user and the start point of
            optimizations. Optimization results never change it; they are stored as
            candidates.
        targets: The targets, or ``None`` if not yet defined.
        merit: The figure of merit, or ``None`` if not yet defined.
        constraints: Manufacturing constraints.
        process: Deposition process, or ``None``.
        model: Reflectivity model used for evaluations.
        pipeline: Stages to run, or ``None`` for the default pipeline.
        candidates: Stored designs, e.g. optimization results.
        selected: Name of the selected candidate, shown and analyzed next to the input design,
            or ``None``.
        notes: Free-form notes.
        tradeoff: The last coating-time trade-off, or ``None``.
    """

    name: str = "untitled"
    mirror: Mirror | None = None
    targets: TargetSet | None = None
    merit: FigureOfMerit | None = None
    constraints: tuple[Constraint, ...] = ()
    process: DepositionProcess | None = None
    model: ReflectivityModel = field(default_factory=ParrattModel)
    pipeline: Pipeline | None = None
    candidates: tuple[Candidate, ...] = ()
    selected: str | None = None
    notes: str = ""
    tradeoff: TradeoffResult | None = None

    def evolve(self, **changes: Any) -> "ProjectState":
        """Return a copy with some fields changed.

        Args:
            **changes: New values for fields of the state.

        Returns:
            The modified state.
        """
        return replace(self, **changes)

    def problem(self) -> DesignProblem:
        """Return the design problem described by this state.

        Returns:
            The design problem.

        Raises:
            ValueError: If mirror, targets or figure of merit are not defined.
        """
        mirror, targets, merit = self.mirror, self.targets, self.merit
        if mirror is None or targets is None or merit is None:
            missing = [n for n in ("mirror", "targets", "merit") if getattr(self, n) is None]
            raise ValueError(f"the project does not define {', '.join(missing)}")
        return DesignProblem(mirror, targets, merit, self.constraints, self.process, self.model)

    def selected_candidate(self) -> Candidate | None:
        """Return the selected candidate, if any."""
        if self.selected is None:
            return None
        try:
            return self.candidate(self.selected)
        except KeyError:
            return None

    def candidate(self, name: str) -> Candidate:
        """Return a candidate by name.

        Args:
            name: Candidate name.

        Returns:
            The candidate.

        Raises:
            KeyError: If there is no candidate with this name.
        """
        for c in self.candidates:
            if c.name == name:
                return c
        raise KeyError(f"unknown candidate {name!r}")


register_serializable("Candidate", Candidate)
register_serializable("ProjectState", ProjectState)


@dataclass(frozen=True)
class Command:
    """A described change of the project state.

    Attributes:
        description: Human-readable description, shown in the journal and undo history.
        transform: Function computing the new state from the current one.
        merge_key: Consecutive commands with the same non-empty key are merged into one undo
            step (e.g. repeated edits of the same parameter).
    """

    description: str
    transform: Callable[[ProjectState], ProjectState]
    merge_key: str = ""


@dataclass(frozen=True)
class _Step:
    description: str
    before: ProjectState
    after: ProjectState
    merge_key: str


class History:
    """Undo and redo stacks of project states."""

    def __init__(self) -> None:
        self._undo: list[_Step] = []
        self._redo: list[_Step] = []

    @property
    def can_undo(self) -> bool:
        """Whether there is a step to undo."""
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        """Whether there is a step to redo."""
        return bool(self._redo)

    def undo_descriptions(self) -> tuple[str, ...]:
        """Return the descriptions of the undoable steps, oldest first."""
        return tuple(s.description for s in self._undo)

    def redo_descriptions(self) -> tuple[str, ...]:
        """Return the descriptions of the redoable steps, next first."""
        return tuple(s.description for s in reversed(self._redo))

    def push(
        self, description: str, before: ProjectState, after: ProjectState, merge_key: str = ""
    ) -> None:
        """Record a step and clear the redo stack.

        Args:
            description: Description of the step.
            before: State before the step.
            after: State after the step.
            merge_key: Merge with the previous step if it has the same non-empty key.
        """
        self._redo.clear()
        if merge_key and self._undo and self._undo[-1].merge_key == merge_key:
            previous = self._undo.pop()
            before = previous.before
        self._undo.append(_Step(description, before, after, merge_key))

    def undo(self) -> tuple[str, ProjectState]:
        """Undo the last step.

        Returns:
            Description of the step and the state to restore.

        Raises:
            IndexError: If there is nothing to undo.
        """
        if not self._undo:
            raise IndexError("nothing to undo")
        step = self._undo.pop()
        self._redo.append(step)
        return step.description, step.before

    def redo(self) -> tuple[str, ProjectState]:
        """Redo the last undone step.

        Returns:
            Description of the step and the state to restore.

        Raises:
            IndexError: If there is nothing to redo.
        """
        if not self._redo:
            raise IndexError("nothing to redo")
        step = self._redo.pop()
        self._undo.append(step)
        return step.description, step.after

    def clear(self) -> None:
        """Forget all steps."""
        self._undo.clear()
        self._redo.clear()


class Project:
    """A design project: the current state, its undo history and the journal.

    All changes of the state go through `execute`, so that they can be undone and are recorded
    in the journal.

    Args:
        state: Initial state.
        journal: Existing journal, e.g. loaded from a file.

    Examples:
        >>> project = Project()
        >>> project.execute(Command("Rename", lambda s: s.evolve(name="mirror A")))
        >>> project.state.name
        'mirror A'
        >>> project.undo()
        >>> project.state.name
        'untitled'
    """

    def __init__(self, state: ProjectState | None = None, journal: Journal | None = None) -> None:
        self._state = state or ProjectState()
        self.history = History()
        self.journal = journal or Journal()
        self._listeners: list[Callable[[ProjectState], None]] = []

    @property
    def state(self) -> ProjectState:
        """The current state."""
        return self._state

    def subscribe(self, listener: Callable[[ProjectState], None]) -> None:
        """Register a function that is called with the new state after every change.

        Args:
            listener: The function.
        """
        self._listeners.append(listener)

    def _set(self, state: ProjectState) -> None:
        self._state = state
        for listener in self._listeners:
            listener(state)

    def execute(self, command: Command) -> None:
        """Apply a command, record it for undo and in the journal.

        Args:
            command: The command.

        Raises:
            Exception: Any exception raised by the command's transform; the state is unchanged.
        """
        before = self._state
        after = command.transform(before)
        if after == before:
            return
        self.history.push(command.description, before, after, command.merge_key)
        self.journal.add("command", command.description)
        self._set(after)

    def undo(self) -> None:
        """Undo the last change.

        Raises:
            IndexError: If there is nothing to undo.
        """
        description, state = self.history.undo()
        self.journal.add("undo", f"Undo: {description}")
        self._set(state)

    def redo(self) -> None:
        """Redo the last undone change.

        Raises:
            IndexError: If there is nothing to redo.
        """
        description, state = self.history.redo()
        self.journal.add("redo", f"Redo: {description}")
        self._set(state)

    def to_data(self) -> dict[str, Any]:
        """Return the project (state and journal) as JSON-compatible data."""
        return {
            "format": "xstack-project",
            "schema_version": SCHEMA_VERSION,
            "state": to_data(self._state),
            "journal": self.journal.to_data(),
        }

    @classmethod
    def from_data(cls, data: dict[str, Any]) -> "Project":
        """Create a project from data returned by `to_data`.

        Args:
            data: The data.

        Returns:
            The project with an empty undo history.

        Raises:
            ValueError: If the data is not an xstack project of a supported version.
        """
        if data.get("format") != "xstack-project":
            raise ValueError("not an xstack project file")
        version = data.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(f"unsupported project file version {version}")
        state = from_data(data["state"])
        if not isinstance(state, ProjectState):
            raise ValueError("project file does not contain a project state")
        return cls(state, Journal.from_data(data.get("journal", [])))

    def save(self, path: Path | str) -> None:
        """Save the project to a JSON file.

        Args:
            path: Output file, conventionally with the suffix ``.xstack``.
        """
        self.journal.add("file", f"Saved to {Path(path).name}")
        Path(path).write_text(json.dumps(self.to_data(), indent=1))

    @classmethod
    def load(cls, path: Path | str) -> "Project":
        """Load a project from a JSON file.

        Args:
            path: The project file.

        Returns:
            The project.
        """
        project = cls.from_data(json.loads(Path(path).read_text()))
        project.journal.add("file", f"Loaded {Path(path).name}")
        return project
