"""Design project: its state, candidate designs and commands."""

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any

from xstack.core import document
from xstack.core.document import Document
from xstack.core.physics.models import ParrattModel, ReflectivityModel
from xstack.core.serialization import register_serializable
from xstack.core.structure.coating import Mirror
from xstack.core.targets import TargetSet
from xstack.design.manufacturing.constraints import Constraint
from xstack.design.manufacturing.process import DepositionProcess
from xstack.design.merit.base import FigureOfMerit
from xstack.design.optim.pipeline import Pipeline
from xstack.design.optim.problem import DesignProblem, Evaluation
from xstack.design.optim.tradeoff import TradeoffResult

__all__ = ["SCHEMA_VERSION", "Candidate", "Command", "Project", "ProjectState"]

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
class Command(document.Command[ProjectState]):
    """A described change of the project state.

    Attributes:
        description: Human-readable description, shown in the journal and undo history.
        transform: Function computing the new state from the current one.
        merge_key: Consecutive commands with the same non-empty key are merged into one undo
            step (e.g. repeated edits of the same parameter).
    """


class Project(Document[ProjectState]):
    """A design project: the current state, its undo history and the journal.

    All changes of the state go through `execute`, so that they can be undone and are recorded
    in the journal. Project files conventionally have the suffix ``.xstack``.

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

    file_format = "xstack-project"
    file_description = "xstack project"
    schema_version = SCHEMA_VERSION
    state_type = ProjectState
