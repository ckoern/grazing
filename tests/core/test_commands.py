from dataclasses import dataclass

import pytest

from grazing.core.commands import rename_project, set_notes, set_optical_constants
from grazing.core.document import Command, Document


@dataclass(frozen=True)
class State:
    name: str = "untitled"
    notes: str = ""
    optical_constants: str = "henke"


class StateDocument(Document[State]):
    file_format = "test-state"
    file_description = "test state"
    schema_version = 1
    state_type = State


def test_rename_project() -> None:
    document = StateDocument()
    command: Command[State] = rename_project("mirror A")
    assert command.description == "Rename project to 'mirror A'"
    document.execute(command)
    assert document.state.name == "mirror A"
    document.undo()
    assert document.state.name == "untitled"


def test_set_notes_merges_edits() -> None:
    document = StateDocument()
    document.execute(set_notes("a"))
    document.execute(set_notes("ab"))
    assert document.state.notes == "ab"
    assert document.history.undo_descriptions() == ("Edit notes",)
    document.undo()
    assert document.state.notes == ""


def test_set_optical_constants() -> None:
    document = StateDocument()
    command: Command[State] = set_optical_constants("chantler")
    assert command.description == "Set optical constants (chantler)"
    document.execute(command)
    assert document.state.optical_constants == "chantler"


def test_set_optical_constants_unknown_provider() -> None:
    with pytest.raises(KeyError):
        set_optical_constants("nonexistent")
