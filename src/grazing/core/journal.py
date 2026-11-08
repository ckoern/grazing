"""Append-only journal of user actions and calculations."""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

__all__ = ["EntryKind", "Journal", "JournalEntry"]

type EntryKind = Literal["command", "undo", "redo", "run", "note", "file"]
"""Kind of a journal entry."""


@dataclass(frozen=True)
class JournalEntry:
    """One journal entry.

    Attributes:
        timestamp: Time of the entry, ISO 8601 in UTC.
        kind: Kind of entry.
        message: Human-readable description.
        data: Additional JSON-compatible details.
    """

    timestamp: str
    kind: EntryKind
    message: str
    data: Mapping[str, Any] = field(default_factory=dict)


class Journal:
    """An append-only, chronological list of journal entries.

    Args:
        entries: Initial entries, e.g. loaded from a project file.
    """

    def __init__(self, entries: list[JournalEntry] | None = None) -> None:
        self._entries: list[JournalEntry] = list(entries or [])

    def add(self, kind: EntryKind, message: str, **data: Any) -> JournalEntry:
        """Append an entry with the current time.

        Args:
            kind: Kind of entry.
            message: Human-readable description.
            **data: Additional JSON-compatible details.

        Returns:
            The new entry.
        """
        entry = JournalEntry(datetime.now(UTC).isoformat(timespec="seconds"), kind, message, data)
        self._entries.append(entry)
        return entry

    def entries(self, kind: EntryKind | None = None) -> tuple[JournalEntry, ...]:
        """Return all entries, optionally only those of one kind.

        Args:
            kind: Kind to filter by.

        Returns:
            The entries in chronological order.
        """
        return tuple(e for e in self._entries if kind is None or e.kind == kind)

    def __iter__(self) -> Iterator[JournalEntry]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def to_markdown(self) -> str:
        """Return the journal as a Markdown list.

        Returns:
            One line per entry.
        """
        return "\n".join(f"- {e.timestamp} [{e.kind}] {e.message}" for e in self._entries)

    def to_data(self) -> list[dict[str, Any]]:
        """Return the entries as JSON-compatible data."""
        return [
            {"timestamp": e.timestamp, "kind": e.kind, "message": e.message, "data": dict(e.data)}
            for e in self._entries
        ]

    @classmethod
    def from_data(cls, data: list[Mapping[str, Any]]) -> "Journal":
        """Create a journal from data returned by `to_data`.

        Args:
            data: The entries as dictionaries.

        Returns:
            The journal.
        """
        return cls(
            [JournalEntry(d["timestamp"], d["kind"], d["message"], d.get("data", {})) for d in data]
        )
