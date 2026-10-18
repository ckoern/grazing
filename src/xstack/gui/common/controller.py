"""Bridge between a core `Document` (design or fit project) and the Qt user interface."""

from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QUndoCommand, QUndoStack

from xstack.core.document import Command, Document

__all__ = ["CommandAdapter", "DocumentController"]


class CommandAdapter(QUndoCommand):
    """Qt undo command that mirrors one step of the core project history.

    The core command has already been executed when the adapter is pushed; redo and undo
    delegate to the core project's history.

    Args:
        controller: The project controller.
        command: The executed core command.
    """

    def __init__(self, controller: "DocumentController", command: Command[Any]) -> None:
        super().__init__(command.description)
        self._controller = controller
        self._merge_key = command.merge_key
        self._first = True

    def redo(self) -> None:
        """Redo the step (the first call is a no-op, the command is already applied)."""
        if self._first:
            self._first = False
            return
        self._controller.project.redo()
        self._controller._changed()

    def undo(self) -> None:
        """Undo the step."""
        self._controller.project.undo()
        self._controller._changed()

    def id(self) -> int:
        """Return a merge id; commands with the same non-empty merge key share an id."""
        return (hash(self._merge_key) & 0x7FFFFFFF) if self._merge_key else -1

    def mergeWith(self, other: QUndoCommand) -> bool:
        """Merge with a following command of the same merge key.

        The core history has merged the steps already; the newer command's text is kept.

        Args:
            other: The following command.

        Returns:
            Whether the commands were merged.
        """
        if (
            isinstance(other, CommandAdapter)
            and self._merge_key
            and other._merge_key == self._merge_key
        ):
            self.setText(other.text())
            return True
        return False


class DocumentController(QObject):
    """Owns the current document and its Qt undo stack and notifies views of changes.

    Signals:
        stateChanged(object): The document state changed.
        journalChanged(): The journal has new entries.
        fileChanged(object): The file path changed (``Path`` or ``None``).
        errorOccurred(str): A command failed; the message describes why.

    Args:
        document_type: Class of the documents, used to load files.
        project: Initial document.
        parent: Parent object.
    """

    stateChanged = Signal(object)
    journalChanged = Signal()
    fileChanged = Signal(object)
    errorOccurred = Signal(str)

    def __init__(
        self,
        document_type: type[Document[Any]],
        project: Document[Any],
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.undo_stack = QUndoStack(self)
        self._document_type = document_type
        self._project = project
        self._path: Path | None = None

    @property
    def project(self) -> Any:
        """The current document."""
        return self._project

    @property
    def state(self) -> Any:
        """The current document state."""
        return self._project.state

    @property
    def path(self) -> Path | None:
        """The file the project was loaded from or saved to, if any."""
        return self._path

    @property
    def is_dirty(self) -> bool:
        """Whether there are changes since the project was created, loaded or saved."""
        return not self.undo_stack.isClean()

    def _changed(self) -> None:
        self.stateChanged.emit(self._project.state)
        self.journalChanged.emit()

    def execute(self, command: Command[Any]) -> bool:
        """Execute a core command as an undoable step.

        Args:
            command: The command.

        Returns:
            Whether the state changed. Failures are reported through `errorOccurred`.
        """
        before = self._project.state
        try:
            self._project.execute(command)
        except Exception as exc:  # any failure of a command is reported to the user
            self.errorOccurred.emit(f"{command.description}: {exc}")
            return False
        if self._project.state is before:
            return False
        self.undo_stack.push(CommandAdapter(self, command))
        self._changed()
        return True

    def undo(self) -> None:
        """Undo the last step."""
        self.undo_stack.undo()

    def redo(self) -> None:
        """Redo the last undone step."""
        self.undo_stack.redo()

    def replace_project(self, project: Document[Any], path: Path | None = None) -> None:
        """Replace the current document, clearing the undo history.

        Args:
            project: The new document.
            path: Its file, or ``None`` for a new document.
        """
        self._project = project
        self._path = path
        self.undo_stack.clear()
        self.undo_stack.setClean()
        self.fileChanged.emit(path)
        self._changed()

    def open_project(self, path: Path | str) -> bool:
        """Load a project file.

        Args:
            path: The file.

        Returns:
            Whether the document was loaded. Failures are reported through `errorOccurred`.
        """
        try:
            project = self._document_type.load(path)
        except (OSError, ValueError) as exc:
            self.errorOccurred.emit(f"Cannot open {path}: {exc}")
            return False
        self.replace_project(project, Path(path))
        return True

    def save_project(self, path: Path | str | None = None) -> bool:
        """Save the project.

        Args:
            path: Target file. Defaults to the current file.

        Returns:
            Whether the project was saved. Failures are reported through `errorOccurred`.

        Raises:
            ValueError: If no path is given and the project has no file yet.
        """
        target = Path(path) if path is not None else self._path
        if target is None:
            raise ValueError("no file name given")
        try:
            self._project.save(target)
        except (OSError, ValueError) as exc:
            self.errorOccurred.emit(f"Cannot save {target}: {exc}")
            return False
        self.undo_stack.setClean()
        if target != self._path:
            self._path = target
            self.fileChanged.emit(target)
        self.journalChanged.emit()
        return True
