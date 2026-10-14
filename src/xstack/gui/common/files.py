"""File dialogs that complete file names."""

from PySide6.QtWidgets import QFileDialog, QWidget

__all__ = ["ask_save_path", "with_suffix"]


def with_suffix(name: str, suffix: str) -> str:
    """Return a file name that ends with a suffix, appending it if missing.

    The comparison ignores case, so ``"data.CSV"`` keeps its name.

    Args:
        name: File name or path.
        suffix: Suffix including the dot, e.g. ``".csv"``.

    Returns:
        The file name with the suffix.

    Examples:
        >>> with_suffix("curves", ".csv")
        'curves.csv'
        >>> with_suffix("curves.csv", ".csv")
        'curves.csv'
    """
    return name if name.lower().endswith(suffix.lower()) else name + suffix


def ask_save_path(parent: QWidget | None, title: str, name_filter: str, suffix: str) -> str | None:
    """Ask for the name of a file to write and complete it with a suffix.

    Args:
        parent: Parent widget of the dialog.
        title: Dialog title.
        name_filter: File type filter, e.g. ``"CSV files (*.csv)"``.
        suffix: Suffix appended if the chosen name lacks it, e.g. ``".csv"``.

    Returns:
        The file name, or None if the dialog was cancelled.
    """
    chosen, _ = QFileDialog.getSaveFileName(parent, title, "", name_filter)
    return with_suffix(chosen, suffix) if chosen else None
