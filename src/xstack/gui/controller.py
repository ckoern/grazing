"""Bridge between the core `Project` and the Qt user interface."""

from pathlib import Path

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QUndoCommand, QUndoStack

from xstack.project import Command, Project, ProjectState
from xstack.project.templates import new_state

__all__ = ["CommandAdapter", "ProjectController"]


class CommandAdapter(QUndoCommand):
    """Qt undo command that mirrors one step of the core project history.

    The core command has already been executed when the adapter is pushed; redo and undo
    delegate to the core project's history.

    Args:
        controller: The project controller.
        command: The executed core command.
    """

    def __init__(self, controller: "ProjectController", command: Command) -> None:
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


class ProjectController(QObject):
    """Owns the current project and its Qt undo stack and notifies views of changes.

    Signals:
        stateChanged(ProjectState): The project state changed.
        journalChanged(): The journal has new entries.
        fileChanged(object): The project file path changed (``Path`` or ``None``).
        errorOccurred(str): A command failed; the message describes why.

    Args:
        project: Initial project. Defaults to a new project from the periodic template.
        parent: Parent object.
    """

    stateChanged = Signal(object)
    journalChanged = Signal()
    fileChanged = Signal(object)
    errorOccurred = Signal(str)

    def __init__(self, project: Project | None = None, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.undo_stack = QUndoStack(self)
        self._project = project or Project(new_state("periodic"))
        self._path: Path | None = None

    @property
    def project(self) -> Project:
        """The current project."""
        return self._project

    @property
    def state(self) -> ProjectState:
        """The current project state."""
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

    def execute(self, command: Command) -> bool:
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

    def _replace_project(self, project: Project, path: Path | None) -> None:
        self._project = project
        self._path = path
        self.undo_stack.clear()
        self.undo_stack.setClean()
        self.fileChanged.emit(path)
        self._changed()

    def new_project(self, template: str = "periodic") -> None:
        """Replace the current project with a new one from a template.

        Args:
            template: Name of a template in `xstack.project.templates.TEMPLATES`.
        """
        project = Project(new_state(template))
        project.journal.add("file", f"New project from template {template!r}")
        self._replace_project(project, None)

    def open_project(self, path: Path | str) -> bool:
        """Load a project file.

        Args:
            path: The project file.

        Returns:
            Whether the project was loaded. Failures are reported through `errorOccurred`.
        """
        try:
            project = Project.load(path)
        except (OSError, ValueError) as exc:
            self.errorOccurred.emit(f"Cannot open {path}: {exc}")
            return False
        self._replace_project(project, Path(path))
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
