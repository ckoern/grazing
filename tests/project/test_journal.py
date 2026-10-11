"""Tests for the project journal."""

import json
from datetime import UTC, datetime

from xstack.project import Journal, JournalEntry


class TestJournal:
    def test_add(self) -> None:
        journal = Journal()
        entry = journal.add("note", "hello", a=1, b="x")
        assert entry.kind == "note"
        assert entry.message == "hello"
        assert entry.data == {"a": 1, "b": "x"}
        assert journal.entries() == (entry,)

    def test_timestamp_is_iso_utc(self) -> None:
        before = datetime.now(UTC).replace(microsecond=0)
        entry = Journal().add("note", "x")
        parsed = datetime.fromisoformat(entry.timestamp)
        assert parsed.utcoffset() is not None
        assert parsed.utcoffset().total_seconds() == 0  # type: ignore[union-attr]
        assert before <= parsed <= datetime.now(UTC)

    def test_entries_filter(self) -> None:
        journal = Journal()
        journal.add("command", "c1")
        journal.add("undo", "u1")
        journal.add("command", "c2")
        assert [e.message for e in journal.entries("command")] == ["c1", "c2"]
        assert [e.message for e in journal.entries("undo")] == ["u1"]
        assert journal.entries("run") == ()
        assert len(journal.entries()) == 3

    def test_len_and_iter(self) -> None:
        journal = Journal()
        assert len(journal) == 0
        journal.add("note", "a")
        journal.add("note", "b")
        assert len(journal) == 2
        assert [e.message for e in journal] == ["a", "b"]

    def test_initial_entries_are_copied(self) -> None:
        entries = [JournalEntry("2026-01-01T00:00:00+00:00", "note", "x")]
        journal = Journal(entries)
        journal.add("note", "y")
        assert len(entries) == 1
        assert len(journal) == 2

    def test_to_markdown(self) -> None:
        journal = Journal([JournalEntry("2026-01-01T00:00:00+00:00", "run", "ran it")])
        journal.add("note", "second")
        lines = journal.to_markdown().splitlines()
        assert lines[0] == "- 2026-01-01T00:00:00+00:00 [run] ran it"
        assert lines[1].startswith("- ")
        assert lines[1].endswith("[note] second")

    def test_empty_markdown(self) -> None:
        assert Journal().to_markdown() == ""

    def test_data_round_trip(self) -> None:
        journal = Journal()
        journal.add("command", "c", n=3, items=[1, 2])
        journal.add("file", "saved")
        data = json.loads(json.dumps(journal.to_data()))
        restored = Journal.from_data(data)
        assert restored.entries() == (
            JournalEntry(journal.entries()[0].timestamp, "command", "c", {"n": 3, "items": [1, 2]}),
            JournalEntry(journal.entries()[1].timestamp, "file", "saved", {}),
        )
        assert restored.to_data() == journal.to_data()

    def test_from_data_without_data_field(self) -> None:
        journal = Journal.from_data([{"timestamp": "t", "kind": "note", "message": "m"}])
        assert journal.entries()[0].data == {}
