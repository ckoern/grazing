"""Read-only view of the project journal."""

from PySide6.QtWidgets import QListWidget, QListWidgetItem, QWidget

from grazing.core.journal import Journal

__all__ = ["JournalView"]


class JournalView(QListWidget):
    """List of journal entries, newest at the bottom.

    Args:
        parent: Parent widget.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAlternatingRowColors(True)

    def set_journal(self, journal: Journal) -> None:
        """Show the entries of a journal.

        Args:
            journal: The journal.
        """
        entries = journal.entries()
        if self.count() > len(entries):
            self.clear()
        for entry in entries[self.count() :]:
            time = entry.timestamp[11:19]
            item = QListWidgetItem(f"{time}  [{entry.kind}]  {entry.message}")
            item.setToolTip(entry.timestamp)
            self.addItem(item)
        self.scrollToBottom()
