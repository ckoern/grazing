"""Controller of the fit project."""

from PySide6.QtCore import QObject

from xstack.core.structure.presets import COATING_KINDS, build_mirror
from xstack.fit.project import FitProject, FitState
from xstack.fit.templates import new_state
from xstack.gui.common.controller import DocumentController
from xstack.gui.common.wizard import ProjectSetup

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

    def new_from_setup(self, setup: ProjectSetup) -> None:
        """Replace the current project with a new one from the choices of the project wizard.

        Args:
            setup: The wizard's choices: name and structure.

        Raises:
            KeyError: If a material of the structure is not in the library.
        """
        state = FitState(name=setup.name, mirror=build_mirror(setup.structure))
        project = FitProject(state)
        kind = COATING_KINDS[setup.structure.kind]
        project.journal.add("file", f"New fit project {setup.name!r} ({kind})")
        self.replace_project(project, None)

    def new_project(self, template: str = "periodic") -> None:
        """Replace the current project with a new one from a template.

        Args:
            template: Name of a template in `xstack.fit.templates.TEMPLATES`.
        """
        project = FitProject(new_state(template))
        project.journal.add("file", f"New fit project from template {template!r}")
        self.replace_project(project, None)
