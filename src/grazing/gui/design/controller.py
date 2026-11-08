"""Controller of the design project."""

from PySide6.QtCore import QObject

from grazing.core.structure.presets import COATING_KINDS, build_mirror
from grazing.design.project import Project, ProjectState
from grazing.design.project.templates import new_state
from grazing.gui.common.controller import CommandAdapter, DocumentController
from grazing.gui.common.wizard import ProjectSetup

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

    def new_from_setup(self, setup: ProjectSetup) -> None:
        """Replace the current project with a new one from the choices of the project wizard.

        Targets, figure of merit and constraints come from the template matching the coating
        kind (single layer and individual layers: ``"single_layer"``; periodic: ``"periodic"``;
        depth-graded: ``"depth_graded"``); the structure and the name from the wizard.

        Args:
            setup: The wizard's choices.

        Raises:
            KeyError: If a material of the structure is not in the library.
        """
        kind = setup.structure.kind
        template = {
            "periodic": "periodic",
            "single_layer": "single_layer",
            "layers": "single_layer",
        }
        state = new_state(template.get(kind, "depth_graded"))
        state = state.evolve(name=setup.name, mirror=build_mirror(setup.structure))
        project = Project(state)
        project.journal.add("file", f"New project {setup.name!r} ({COATING_KINDS[kind]})")
        self.replace_project(project, None)

    def new_project(self, template: str = "periodic") -> None:
        """Replace the current project with a new one from a template.

        Args:
            template: Name of a template in `grazing.design.project.templates.TEMPLATES`.
        """
        project = Project(new_state(template))
        project.journal.add("file", f"New project from template {template!r}")
        self.replace_project(project, None)
