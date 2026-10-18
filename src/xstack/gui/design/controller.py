"""Controller of the design project."""

from PySide6.QtCore import QObject

from xstack.design.project import Project, ProjectState
from xstack.design.project.templates import new_state
from xstack.gui.common.controller import CommandAdapter, DocumentController

__all__ = ["CommandAdapter", "ProjectController"]


class ProjectController(DocumentController):
    """Owns the current design project and its Qt undo stack.

    Args:
        project: Initial project. Defaults to a new project from the periodic template.
        parent: Parent object.
    """

    def __init__(self, project: Project | None = None, parent: QObject | None = None) -> None:
        super().__init__(Project, project or Project(new_state("periodic")), parent)

    @property
    def project(self) -> Project:
        """The current project."""
        assert isinstance(self._project, Project)
        return self._project

    @property
    def state(self) -> ProjectState:
        """The current project state."""
        state: ProjectState = self._project.state
        return state

    def new_project(self, template: str = "periodic") -> None:
        """Replace the current project with a new one from a template.

        Args:
            template: Name of a template in `xstack.design.project.templates.TEMPLATES`.
        """
        project = Project(new_state(template))
        project.journal.add("file", f"New project from template {template!r}")
        self.replace_project(project, None)
