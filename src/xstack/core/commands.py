"""Commands for the project fields shared by all document states.

The states of the design and fit projects both have a name, notes and an optical-constants
provider. The functions here change these fields of any state with such fields. Each returns a
`xstack.core.document.Command` to pass to `Document.execute`.

Examples:
    >>> from dataclasses import dataclass
    >>> @dataclass(frozen=True)
    ... class State:
    ...     name: str = "untitled"
    ...     notes: str = ""
    ...     optical_constants: str = "henke"
    >>> rename_project("mirror A").transform(State()).name
    'mirror A'
"""

from dataclasses import replace
from typing import Any, ClassVar, Protocol

from xstack.core.document import Command
from xstack.core.materials import providers

__all__ = ["ProjectFields", "rename_project", "set_notes", "set_optical_constants"]


class _Dataclass(Protocol):
    __dataclass_fields__: ClassVar[dict[str, Any]]


class ProjectFields(_Dataclass, Protocol):
    """A dataclass state with the fields common to all projects.

    Attributes:
        name: Project name.
        notes: Free-form notes.
        optical_constants: Name of the optical-constants provider.
    """

    name: str
    notes: str
    optical_constants: str


def rename_project[S: ProjectFields](name: str) -> Command[S]:
    """Rename the project.

    Args:
        name: New project name.

    Returns:
        The command.
    """
    return Command(f"Rename project to {name!r}", lambda s: replace(s, name=name))


def set_notes[S: ProjectFields](notes: str) -> Command[S]:
    """Replace the project notes.

    Consecutive edits form a single undo step.

    Args:
        notes: The notes.

    Returns:
        The command.
    """
    return Command("Edit notes", lambda s: replace(s, notes=notes), merge_key="notes")


def set_optical_constants[S: ProjectFields](provider: str) -> Command[S]:
    """Choose the optical-constants provider of the project.

    Args:
        provider: Name of a registered optical-constants provider.

    Returns:
        The command.

    Raises:
        KeyError: If the provider is not registered.
    """
    providers.get(provider)
    return Command(
        f"Set optical constants ({provider})", lambda s: replace(s, optical_constants=provider)
    )
