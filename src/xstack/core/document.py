"""Documents with an immutable state, commands, undo/redo history, journal and file storage.

A `Document` holds the current state of an application (e.g. a design or a fit project). All
changes go through `Document.execute` with a `Command`, so that they can be undone and are
recorded in the journal. Subclasses define the state type and the file format.

Examples:
    >>> from dataclasses import dataclass
    >>> @dataclass(frozen=True)
    ... class Counter:
    ...     value: int = 0
    >>> class CounterDocument(Document[Counter]):
    ...     file_format = "counter"
    ...     file_description = "counter"
    ...     schema_version = 1
    ...     state_type = Counter
    >>> doc = CounterDocument()
    >>> doc.execute(Command("Increment", lambda s: Counter(s.value + 1)))
    >>> doc.state.value
    1
    >>> doc.undo()
    >>> doc.state.value
    0
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Self, cast

from xstack.core.journal import Journal
from xstack.core.serialization import from_data, to_data

__all__ = ["Command", "Document", "History"]


@dataclass(frozen=True)
class Command[S]:
    """A described change of a document state.

    Attributes:
        description: Human-readable description, shown in the journal and undo history.
        transform: Function computing the new state from the current one.
        merge_key: Consecutive commands with the same non-empty key are merged into one undo
            step (e.g. repeated edits of the same parameter), unless the document was saved
            between them (see `History.push`).
    """

    description: str
    transform: Callable[[S], S]
    merge_key: str = ""


@dataclass(frozen=True)
class _Step[S]:
    description: str
    before: S
    after: S
    merge_key: str


class History[S]:
    """Undo and redo stacks of document states with a save point.

    The save point is the number of undoable steps when the document was last saved (0 for a
    new or loaded document). A step is merged into the previous one only if both have the same
    non-empty merge key and the history is not at its save point, so that undo can always
    return to the saved state. If the steps being discarded by a push contain the save point,
    the saved state can no longer be reached and the save point is lost. These rules match
    Qt's ``QUndoStack`` and its clean index.
    """

    def __init__(self) -> None:
        self._undo: list[_Step[S]] = []
        self._redo: list[_Step[S]] = []
        self._save_point: int | None = 0

    @property
    def can_undo(self) -> bool:
        """Whether there is a step to undo."""
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        """Whether there is a step to redo."""
        return bool(self._redo)

    @property
    def save_point(self) -> int | None:
        """Number of undoable steps at the last save, or ``None`` if that state is unreachable."""
        return self._save_point

    @property
    def is_at_save_point(self) -> bool:
        """Whether the current state is the saved state."""
        return self._save_point == len(self._undo)

    def mark_saved(self) -> None:
        """Set the save point to the current state."""
        self._save_point = len(self._undo)

    def undo_descriptions(self) -> tuple[str, ...]:
        """Return the descriptions of the undoable steps, oldest first."""
        return tuple(s.description for s in self._undo)

    def redo_descriptions(self) -> tuple[str, ...]:
        """Return the descriptions of the redoable steps, next first."""
        return tuple(s.description for s in reversed(self._redo))

    def push(self, description: str, before: S, after: S, merge_key: str = "") -> None:
        """Record a step and clear the redo stack.

        The step is merged into the previous one if that has the same non-empty merge key and
        the history is not at its save point. If the cleared redo steps contain the save point,
        the save point becomes unreachable (``None``).

        Args:
            description: Description of the step.
            before: State before the step.
            after: State after the step.
            merge_key: Key for merging with the previous step; empty for no merging.
        """
        if self._redo and self._save_point is not None and self._save_point > len(self._undo):
            self._save_point = None
        self._redo.clear()
        if (
            merge_key
            and self._undo
            and self._undo[-1].merge_key == merge_key
            and not self.is_at_save_point
        ):
            previous = self._undo.pop()
            before = previous.before
        self._undo.append(_Step(description, before, after, merge_key))

    def undo(self) -> tuple[str, S]:
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

    def redo(self) -> tuple[str, S]:
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
        """Forget all steps and set the save point to the empty history."""
        self._undo.clear()
        self._redo.clear()
        self._save_point = 0


class Document[S]:
    """The current state of a document, its undo history and its journal.

    Subclasses set the class attributes below. The state must be serializable with
    `xstack.core.serialization`.

    Args:
        state: Initial state. Defaults to ``state_type()``.
        journal: Existing journal, e.g. loaded from a file.

    Attributes:
        file_format: Name of the file format, stored in and checked when reading files.
        file_description: Human-readable name of the document kind for error messages, e.g.
            ``"xstack project"``.
        schema_version: Version of the file format.
        state_type: Class of the state.
        history: Undo and redo history.
        journal: The journal of the document.
    """

    file_format: ClassVar[str]
    file_description: ClassVar[str]
    schema_version: ClassVar[int]
    state_type: ClassVar[type]

    def __init__(self, state: S | None = None, journal: Journal | None = None) -> None:
        self._state: S = state if state is not None else self.state_type()
        self.history: History[S] = History()
        self.journal = journal or Journal()
        self._listeners: list[Callable[[S], None]] = []

    @property
    def state(self) -> S:
        """The current state."""
        return self._state

    def subscribe(self, listener: Callable[[S], None]) -> None:
        """Register a function that is called with the new state after every change.

        Args:
            listener: The function.
        """
        self._listeners.append(listener)

    def _set(self, state: S) -> None:
        self._state = state
        for listener in self._listeners:
            listener(state)

    def execute(self, command: Command[S]) -> None:
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
        """Return the document (state and journal) as JSON-compatible data."""
        return {
            "format": self.file_format,
            "schema_version": self.schema_version,
            "state": to_data(self._state),
            "journal": self.journal.to_data(),
        }

    @classmethod
    def from_data(cls, data: dict[str, Any]) -> Self:
        """Create a document from data returned by `to_data`.

        Args:
            data: The data.

        Returns:
            The document with an empty undo history.

        Raises:
            ValueError: If the data has another format, an unsupported version or no state of
                the expected type.
        """
        kind = cls.file_description
        if data.get("format") != cls.file_format:
            raise ValueError(f"not an {kind} file")
        version = data.get("schema_version")
        if version != cls.schema_version:
            raise ValueError(f"unsupported {kind} file version {version}")
        state = from_data(data["state"])
        if not isinstance(state, cls.state_type):
            raise ValueError(f"{kind} file does not contain a {kind} state")
        return cls(cast(S, state), Journal.from_data(data.get("journal", [])))

    def save(self, path: Path | str) -> None:
        """Save the document to a JSON file and mark the save point of the history.

        The next command is not merged into the last undo step, even if it has the same merge
        key, so that undo can return to the saved state.

        Args:
            path: Output file.
        """
        self.journal.add("file", f"Saved to {Path(path).name}")
        Path(path).write_text(json.dumps(self.to_data(), indent=1))
        self.history.mark_saved()

    @classmethod
    def load(cls, path: Path | str) -> Self:
        """Load a document from a JSON file.

        Args:
            path: The file.

        Returns:
            The document.

        Raises:
            ValueError: If the file is not a valid document of this class.
        """
        document = cls.from_data(json.loads(Path(path).read_text()))
        document.journal.add("file", f"Loaded {Path(path).name}")
        return document
