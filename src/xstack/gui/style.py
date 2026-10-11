"""Application-wide look of the graphical user interface."""

from PySide6.QtWidgets import QApplication

__all__ = ["HELP_STYLE", "apply_style"]

HELP_STYLE = (
    "background-color: #eef4fb; border: 1px solid #c5d6ea; border-radius: 6px; "
    "padding: 8px 10px; color: #1f2d3d;"
)
"""Stylesheet for labels that explain a workflow step."""

_STYLESHEET = """
QTableView, QTableWidget, QListWidget {
    gridline-color: transparent;
    border: none;
    alternate-background-color: #f6f8fa;
    selection-background-color: #dbe8f7;
    selection-color: #1f2d3d;
}
QTableView::item, QTableWidget::item, QListWidget::item {
    padding: 4px 6px;
}
QHeaderView::section {
    background-color: #f3f5f7;
    color: #4a5562;
    border: none;
    border-bottom: 1px solid #d5dbe1;
    padding: 5px 6px;
    font-weight: bold;
}
QTableCornerButton::section {
    background-color: #f3f5f7;
    border: none;
}
QGroupBox {
    margin-top: 1.2em;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 8px;
    font-weight: bold;
}
"""


def apply_style(app: QApplication) -> None:
    """Install the application stylesheet.

    Tables and lists are drawn as clean lists without grid lines, group box titles are bold.

    Args:
        app: The application to style.
    """
    app.setStyleSheet(_STYLESHEET)
