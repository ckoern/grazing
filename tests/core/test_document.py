"""Tests for the generic document, command and history machinery."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import pytest

from xstack.core.document import Command, Document, History
from xstack.core.parameters import Parameter
from xstack.core.serialization import register_serializable, to_data


@dataclass(frozen=True)
class Note:
    text: str = ""
    count: int = 0


register_serializable("tests.Note", Note)


class NoteDocument(Document[Note]):
    file_format: ClassVar[str] = "test-note"
    file_description: ClassVar[str] = "test note"
    schema_version: ClassVar[int] = 3
    state_type: ClassVar[type] = Note


def set_text(text: str, merge_key: str = "") -> Command[Note]:
    return Command(f"Set text {text}", lambda s: Note(text, s.count), merge_key)


def increment(merge_key: str = "") -> Command[Note]:
    return Command("Increment", lambda s: Note(s.text, s.count + 1), merge_key)


class TestHistory:
    def test_empty(self) -> None:
        history: History[int] = History()
        assert not history.can_undo
        assert not history.can_redo
        with pytest.raises(IndexError):
            history.undo()
        with pytest.raises(IndexError):
            history.redo()

    def test_push_undo_redo(self) -> None:
        history: History[int] = History()
        history.push("a", 0, 1)
        history.push("b", 1, 2)
        assert history.undo_descriptions() == ("a", "b")
        assert history.undo() == ("b", 1)
        assert history.redo_descriptions() == ("b",)
        assert history.redo() == ("b", 2)

    def test_push_clears_redo(self) -> None:
        history: History[int] = History()
        history.push("a", 0, 1)
        history.undo()
        assert history.can_redo
        history.push("b", 0, 5)
        assert not history.can_redo

    def test_clear(self) -> None:
        history: History[int] = History()
        history.push("a", 0, 1)
        history.undo()
        history.clear()
        assert not history.can_undo
        assert not history.can_redo


class TestDocument:
    def test_default_state(self) -> None:
        assert NoteDocument().state == Note()

    def test_initial_state(self) -> None:
        assert NoteDocument(Note("x", 2)).state == Note("x", 2)

    def test_execute(self) -> None:
        doc = NoteDocument()
        seen: list[Note] = []
        doc.subscribe(seen.append)
        doc.execute(set_text("hello"))
        assert doc.state == Note("hello", 0)
        assert seen == [Note("hello", 0)]
        assert doc.history.undo_descriptions() == ("Set text hello",)
        entries = doc.journal.entries("command")
        assert [e.message for e in entries] == ["Set text hello"]

    def test_equal_state_is_noop(self) -> None:
        doc = NoteDocument(Note("a", 1))
        seen: list[Note] = []
        doc.subscribe(seen.append)
        doc.execute(set_text("a"))
        assert doc.state == Note("a", 1)
        assert not doc.history.can_undo
        assert len(doc.journal) == 0
        assert seen == []

    def test_undo_redo(self) -> None:
        doc = NoteDocument()
        seen: list[Note] = []
        doc.execute(set_text("a"))
        doc.execute(increment())
        doc.subscribe(seen.append)
        doc.undo()
        assert doc.state == Note("a", 0)
        doc.undo()
        assert doc.state == Note()
        doc.redo()
        assert doc.state == Note("a", 0)
        doc.redo()
        assert doc.state == Note("a", 1)
        assert seen == [Note("a", 0), Note(), Note("a", 0), Note("a", 1)]
        assert [e.message for e in doc.journal.entries("undo")] == [
            "Undo: Increment",
            "Undo: Set text a",
        ]
        assert [e.message for e in doc.journal.entries("redo")] == [
            "Redo: Set text a",
            "Redo: Increment",
        ]

    def test_new_command_clears_redo(self) -> None:
        doc = NoteDocument()
        doc.execute(increment())
        doc.undo()
        doc.execute(set_text("b"))
        assert not doc.history.can_redo

    def test_merge_key(self) -> None:
        doc = NoteDocument()
        doc.execute(set_text("a", merge_key="text"))
        doc.execute(set_text("ab", merge_key="text"))
        doc.execute(set_text("abc", merge_key="text"))
        assert len(doc.history.undo_descriptions()) == 1
        assert len(doc.journal.entries("command")) == 3
        doc.undo()
        assert doc.state == Note()
        assert not doc.history.can_undo

    def test_different_or_empty_merge_keys_do_not_merge(self) -> None:
        doc = NoteDocument()
        doc.execute(set_text("a", merge_key="x"))
        doc.execute(set_text("b", merge_key="y"))
        doc.execute(increment())
        doc.execute(increment())
        assert len(doc.history.undo_descriptions()) == 4

    def test_undo_redo_without_history(self) -> None:
        doc = NoteDocument()
        with pytest.raises(IndexError):
            doc.undo()
        with pytest.raises(IndexError):
            doc.redo()
        assert len(doc.journal) == 0

    def test_failing_transform_leaves_document_unchanged(self) -> None:
        doc = NoteDocument()
        doc.execute(set_text("a"))
        seen: list[Note] = []
        doc.subscribe(seen.append)

        def fail(state: Note) -> Note:
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            doc.execute(Command("Fail", fail))
        assert doc.state == Note("a", 0)
        assert doc.history.undo_descriptions() == ("Set text a",)
        assert len(doc.journal) == 1
        assert seen == []


class TestFiles:
    def test_round_trip(self, tmp_path: Path) -> None:
        doc = NoteDocument()
        doc.execute(set_text("persist"))
        doc.execute(increment())
        path = tmp_path / "doc.json"
        doc.save(path)
        loaded = NoteDocument.load(path)
        assert loaded.state == Note("persist", 1)
        assert not loaded.history.can_undo
        messages = [e.message for e in loaded.journal.entries("file")]
        assert messages == ["Saved to doc.json", "Loaded doc.json"]
        assert len(loaded.journal.entries("command")) == 2

    def test_to_data_header(self) -> None:
        data = NoteDocument(Note("x", 1)).to_data()
        assert data["format"] == "test-note"
        assert data["schema_version"] == 3

    def test_wrong_format(self) -> None:
        data = NoteDocument().to_data()
        data["format"] = "other"
        with pytest.raises(ValueError, match="test note"):
            NoteDocument.from_data(data)

    def test_wrong_version(self) -> None:
        data = NoteDocument().to_data()
        data["schema_version"] = 99
        with pytest.raises(ValueError, match="version"):
            NoteDocument.from_data(data)

    def test_wrong_state_type(self) -> None:
        data: dict[str, Any] = NoteDocument().to_data()
        data["state"] = to_data(Parameter("a", 1.0))
        with pytest.raises(ValueError, match="state"):
            NoteDocument.from_data(data)
