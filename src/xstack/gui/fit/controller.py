"""Controller of the fit project."""

from PySide6.QtCore import QObject

from xstack.fit.project import FitProject, FitState
from xstack.fit.templates import new_state
from xstack.gui.common.controller import DocumentController

__all__ = ["FitController"]


class FitController(DocumentController):
    """Owns the current fit project and its Qt undo stack.

    Args:
        project: Initial project. Defaults to a new project from the periodic template.
        parent: Parent object.
    """

    def __init__(self, project: FitProject | None = None, parent: QObject | None = None) -> None:
        super().__init__(FitProject, project or FitProject(new_state("periodic")), parent)

    @property
    def project(self) -> FitProject:
        """The current project."""
        assert isinstance(self._project, FitProject)
        return self._project

    @property
    def state(self) -> FitState:
        """The current project state."""
        return self.project.state

    def new_project(self, template: str = "periodic") -> None:
        """Replace the current project with a new one from a template.

        Args:
            template: Name of a template in `xstack.fit.templates.TEMPLATES`.
        """
        project = FitProject(new_state(template))
        project.journal.add("file", f"New fit project from template {template!r}")
        self.replace_project(project, None)
