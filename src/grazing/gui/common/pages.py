"""Common base of workflow pages and the commands that structure pages emit."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget

from grazing.core.structure.coating import Coating, Mirror

__all__ = ["StepPage", "StructureCommands"]


@dataclass(frozen=True)
class StructureCommands:
    """Factories of the application's commands that edit the mirror of its document.

    Structure pages (coating, materials, parameter table) are shared by the applications; each
    application passes the factories that create its own command type.

    Attributes:
        set_mirror: Command replacing the mirror (substrate, roughness, coating).
        set_coating: Command replacing the coating.
        set_parameter: Command changing fields of one coating parameter, called as
            ``set_parameter(name, **changes)``.
        set_optical_constants: Command choosing the optical-constants provider of the project,
            called with the provider name.
    """

    set_mirror: Callable[[Mirror], Any]
    set_coating: Callable[[Coating], Any]
    set_parameter: Callable[..., Any]
    set_optical_constants: Callable[[str], Any]


class StepPage(QWidget):
    """A page of a workflow that shows and edits part of a document state.

    Pages never modify the document directly; they emit commands.

    Signals:
        commandRequested(Command): An edit to execute.
        errorOccurred(str): An entry was invalid; the message describes why.
        jobRequested(Job): A long-running calculation to run in the background.

    Attributes:
        title: Name of the page in the workflow.
        help_text: Rich-text explanation of the step, shown above the page.
    """

    commandRequested = Signal(object)
    errorOccurred = Signal(str)
    jobRequested = Signal(object)

    title: str = ""
    help_text: ClassVar[str] = ""

    def set_state(self, state: Any) -> None:
        """Show a document state.

        Args:
            state: The current state of the application's document.
        """

    def set_preview(self, preview: Any) -> None:
        """Show the evaluation of the current state.

        Args:
            preview: The latest preview of the application.
        """
