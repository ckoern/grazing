"""Common base of the workflow step pages."""

from typing import ClassVar

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget

from xstack.design.project import ProjectState
from xstack.gui.design.preview import Preview

__all__ = ["StepPage"]


class StepPage(QWidget):
    """A page of the workflow that shows and edits part of the project state.

    Pages never modify the project directly; they emit core commands.

    Signals:
        commandRequested(Command): An edit to execute.
        errorOccurred(str): An entry was invalid; the message describes why.
        jobRequested(Job): A long-running calculation to run in the background.

    Attributes:
        title: Name of the page in the workflow.
        help_text: Rich-text explanation of the step, shown by the main window above the page.
    """

    commandRequested = Signal(object)
    errorOccurred = Signal(str)
    jobRequested = Signal(object)

    title: str = ""
    help_text: ClassVar[str] = ""

    def set_state(self, state: ProjectState) -> None:
        """Show a project state.

        Args:
            state: The current project state.
        """

    def set_preview(self, preview: Preview) -> None:
        """Show the evaluation of the current design.

        Args:
            preview: The latest preview.
        """
