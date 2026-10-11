"""Design language of the graphical user interface.

All colors, spacing and component styles are defined here: a light, neutral surface, a single
indigo accent for primary actions and selections, white cards with rounded corners for grouped
content, and a categorical palette for plots that matches the interface colors.

Use `apply_style` once for the application, `set_variant` to mark primary buttons, and
`PLOT_PALETTE` for plot colors.
"""

from typing import Literal

import pyqtgraph as pg
from PySide6.QtGui import QColor, QFont, QIcon, QPalette, QPixmap
from PySide6.QtWidgets import QApplication, QStyleFactory, QWidget

__all__ = [
    "COLORS",
    "HELP_STYLE",
    "PLOT_PALETTE",
    "Variant",
    "apply_style",
    "icon",
    "set_variant",
]

COLORS: dict[str, str] = {
    "background": "#f4f5f8",
    "surface": "#ffffff",
    "surface_alt": "#f8f9fb",
    "border": "#e1e4ea",
    "border_strong": "#c9ced8",
    "text": "#1d2433",
    "text_muted": "#5f6b7c",
    "accent": "#4f46e5",
    "accent_hover": "#4338ca",
    "accent_soft": "#eef0fe",
    "accent_border": "#c7cbfa",
    "success": "#15803d",
    "success_soft": "#e9f7ef",
    "danger": "#b91c1c",
}
"""Color tokens of the interface."""

PLOT_PALETTE: tuple[str, ...] = (
    "#4f46e5",
    "#e11d48",
    "#0891b2",
    "#d97706",
    "#16a34a",
    "#9333ea",
    "#475569",
    "#db2777",
)
"""Categorical colors for plot curves, in the order of use."""

HELP_STYLE = (
    f"background-color: {COLORS['accent_soft']}; border: none; "
    f"border-left: 3px solid {COLORS['accent']}; border-radius: 6px; "
    f"padding: 10px 12px; color: {COLORS['text']};"
)
"""Stylesheet for labels that explain a workflow step."""

type Variant = Literal["primary", "secondary"]
"""Button variants: one primary action per area, secondary for everything else."""

_C = COLORS

_STYLESHEET = f"""
QMainWindow, #step_panel {{
    background: {_C["background"]};
}}
QDockWidget::title {{
    background: {_C["background"]};
    padding: 6px 8px;
    font-weight: 600;
    color: {_C["text_muted"]};
}}
QToolBar {{
    background: {_C["surface"]};
    border: none;
    border-bottom: 1px solid {_C["border"]};
    spacing: 4px;
    padding: 4px 8px;
}}
QToolButton {{
    border: 1px solid transparent;
    border-radius: 6px;
    padding: 4px 8px;
}}
QToolButton:hover {{
    background: {_C["accent_soft"]};
    border-color: {_C["accent_border"]};
}}
QToolButton:disabled {{
    color: {_C["border_strong"]};
}}
QStatusBar {{
    background: {_C["surface"]};
    border-top: 1px solid {_C["border"]};
    color: {_C["text_muted"]};
}}
QMenuBar {{
    background: {_C["surface"]};
}}
QMenuBar::item:selected, QMenu::item:selected {{
    background: {_C["accent_soft"]};
    color: {_C["text"]};
}}
QMenu {{
    background: {_C["surface"]};
    border: 1px solid {_C["border"]};
    padding: 4px;
}}
QMenu::item {{
    padding: 5px 18px;
    border-radius: 4px;
}}

QTabWidget::pane {{
    border: 1px solid {_C["border"]};
    border-radius: 8px;
    background: {_C["surface"]};
    top: -1px;
}}
QTabBar {{
    qproperty-drawBase: 0;
}}
QTabBar::tab {{
    background: transparent;
    color: {_C["text_muted"]};
    padding: 6px 11px;
    margin: 0 2px 4px 0;
    border-radius: 14px;
    font-weight: 600;
}}
QTabBar::tab:hover {{
    background: {_C["surface"]};
    color: {_C["text"]};
}}
QTabBar::tab:selected {{
    background: {_C["accent"]};
    color: white;
}}
#step_tabs QTabBar::tab {{
    padding: 5px 8px;
    margin-right: 0;
    border-radius: 12px;
    font-size: 9pt;
}}
#plot_tabs QTabBar::tab {{
    padding: 5px 10px;
    border-radius: 12px;
    font-size: 9pt;
}}
#primary_tool {{
    background: {_C["accent"]};
    color: white;
    font-weight: 600;
    padding: 4px 12px;
}}
#primary_tool:hover {{
    background: {_C["accent_hover"]};
}}
#primary_tool:disabled {{
    background: {_C["accent_border"]};
}}

QPushButton {{
    background: {_C["surface"]};
    border: 1px solid {_C["border_strong"]};
    border-radius: 6px;
    padding: 6px 14px;
    font-weight: 500;
}}
QPushButton:hover {{
    border-color: {_C["accent"]};
    color: {_C["accent"]};
}}
QPushButton:pressed {{
    background: {_C["accent_soft"]};
}}
QPushButton:disabled {{
    color: {_C["border_strong"]};
    border-color: {_C["border"]};
    background: {_C["surface_alt"]};
}}
QPushButton[variant="primary"] {{
    background: {_C["accent"]};
    border: 1px solid {_C["accent"]};
    color: white;
    font-weight: 600;
}}
QPushButton[variant="primary"]:hover {{
    background: {_C["accent_hover"]};
    border-color: {_C["accent_hover"]};
    color: white;
}}
QPushButton[variant="primary"]:disabled {{
    background: {_C["accent_border"]};
    border-color: {_C["accent_border"]};
    color: white;
}}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {_C["surface"]};
    border: 1px solid {_C["border_strong"]};
    border-radius: 6px;
    padding: 4px 8px;
    min-height: 20px;
    selection-background-color: {_C["accent"]};
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border: 1px solid {_C["accent"]};
}}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {{
    background: {_C["surface_alt"]};
    color: {_C["text_muted"]};
    border-color: {_C["border"]};
}}
QComboBox::drop-down {{
    border: none;
    width: 22px;
}}
QComboBox QAbstractItemView {{
    background: {_C["surface"]};
    border: 1px solid {_C["border"]};
    selection-background-color: {_C["accent_soft"]};
    selection-color: {_C["text"]};
}}
QCheckBox, QRadioButton {{
    spacing: 6px;
}}

QGroupBox {{
    background: {_C["surface"]};
    border: 1px solid {_C["border"]};
    border-radius: 8px;
    margin-top: 14px;
    padding: 12px 10px 10px 10px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: {_C["text"]};
    font-weight: 600;
}}

QTableView, QTableWidget, QListWidget, QTreeView {{
    background: {_C["surface"]};
    gridline-color: transparent;
    border: none;
    alternate-background-color: {_C["surface_alt"]};
    selection-background-color: {_C["accent_soft"]};
    selection-color: {_C["text"]};
    outline: 0;
}}
QTableView::item, QTableWidget::item, QListWidget::item {{
    padding: 4px 6px;
    border: none;
}}
QListWidget::item {{
    border-radius: 6px;
    margin: 1px 2px;
}}
QListWidget::item:hover, QTableView::item:hover {{
    background: {_C["surface_alt"]};
}}
QListWidget::item:selected, QTableView::item:selected {{
    background: {_C["accent_soft"]};
    color: {_C["text"]};
}}
QHeaderView::section {{
    background: {_C["surface"]};
    color: {_C["text_muted"]};
    border: none;
    border-bottom: 1px solid {_C["border"]};
    padding: 6px 6px;
    font-weight: 600;
}}
QTableCornerButton::section {{
    background: {_C["surface"]};
    border: none;
}}

QScrollArea {{
    background: transparent;
}}
QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 2px;
}}
QScrollBar:horizontal {{
    background: transparent;
    height: 10px;
    margin: 2px;
}}
QScrollBar::handle {{
    background: {_C["border_strong"]};
    border-radius: 4px;
    min-height: 24px;
    min-width: 24px;
}}
QScrollBar::handle:hover {{
    background: {_C["text_muted"]};
}}
QScrollBar::add-line, QScrollBar::sub-line, QScrollBar::add-page, QScrollBar::sub-page {{
    background: none;
    border: none;
    width: 0;
    height: 0;
}}
QSplitter::handle {{
    background: transparent;
}}
QProgressBar {{
    border: 1px solid {_C["border"]};
    border-radius: 6px;
    background: {_C["surface_alt"]};
    text-align: center;
    height: 14px;
}}
QProgressBar::chunk {{
    background: {_C["accent"]};
    border-radius: 5px;
}}
QToolTip {{
    background: {_C["text"]};
    color: white;
    border: none;
    padding: 6px 8px;
}}

#design_header {{
    background: {_C["surface"]};
    border: 1px solid {_C["border"]};
    border-radius: 8px;
    padding: 10px 12px;
}}
#confirmation {{
    background: {_C["success_soft"]};
    color: {_C["success"]};
    border-radius: 6px;
    padding: 6px 10px;
}}
"""


_ICONS: dict[str, str] = {
    "new": '<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/>'
    '<path d="M14 3v5h5"/><path d="M12 11v6M9 14h6"/>',
    "open": '<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>',
    "save": '<path d="M5 3h11l3 3v13a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2z"/>'
    '<path d="M7 3v5h8V3"/><path d="M7 21v-7h10v7"/>',
    "undo": '<path d="M9 14 4 9l5-5"/><path d="M4 9h10a6 6 0 0 1 0 12h-3"/>',
    "redo": '<path d="m15 14 5-5-5-5"/><path d="M20 9H10a6 6 0 0 0 0 12h3"/>',
    "run": '<path d="M7 4v16l13-8z"/>',
    "stop": '<rect x="6" y="6" width="12" height="12" rx="1.5" fill="{color}"/>',
}


def icon(name: str, color: str | None = None) -> QIcon:
    """Return a line icon of the design language.

    Args:
        name: One of ``"new"``, ``"open"``, ``"save"``, ``"undo"``, ``"redo"``, ``"run"``,
            ``"stop"``.
        color: Stroke color; defaults to the text color.

    Returns:
        The icon.

    Raises:
        KeyError: If the icon name is unknown.
    """
    stroke = color or COLORS["text"]
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="48" height="48" '
        f'fill="none" stroke="{stroke}" stroke-width="2" stroke-linecap="round" '
        f'stroke-linejoin="round">{_ICONS[name].format(color=stroke)}</svg>'
    )
    pixmap = QPixmap()
    pixmap.loadFromData(svg.encode(), "SVG")
    return QIcon(pixmap)


def set_variant(widget: QWidget, variant: Variant) -> None:
    """Give a button the look of a primary or secondary action.

    Args:
        widget: The button.
        variant: ``"primary"`` for the main action of an area, ``"secondary"`` otherwise.
    """
    widget.setProperty("variant", variant)
    widget.style().unpolish(widget)
    widget.style().polish(widget)


def _palette() -> QPalette:
    palette = QPalette()
    roles = {
        QPalette.ColorRole.Window: COLORS["background"],
        QPalette.ColorRole.Base: COLORS["surface"],
        QPalette.ColorRole.AlternateBase: COLORS["surface_alt"],
        QPalette.ColorRole.Button: COLORS["surface"],
        QPalette.ColorRole.Text: COLORS["text"],
        QPalette.ColorRole.WindowText: COLORS["text"],
        QPalette.ColorRole.ButtonText: COLORS["text"],
        QPalette.ColorRole.Highlight: COLORS["accent"],
        QPalette.ColorRole.HighlightedText: "#ffffff",
        QPalette.ColorRole.PlaceholderText: COLORS["text_muted"],
        QPalette.ColorRole.ToolTipBase: COLORS["text"],
        QPalette.ColorRole.ToolTipText: "#ffffff",
    }
    for role, color in roles.items():
        palette.setColor(role, QColor(color))
    return palette


def apply_style(app: QApplication) -> None:
    """Install the design language for the application and the plots.

    Sets the Fusion style with a light palette, the application stylesheet and the
    pyqtgraph defaults (white background, muted axes, antialiasing). Calling it again for the
    same application has no effect.

    Args:
        app: The application to style.
    """
    if app.property("xstack_styled"):
        return
    app.setProperty("xstack_styled", True)
    fusion = QStyleFactory.create("Fusion")
    if fusion is not None:
        app.setStyle(fusion)
    app.setPalette(_palette())
    font = QFont(app.font())
    font.setPointSizeF(max(font.pointSizeF(), 9.5))
    app.setFont(font)
    app.setStyleSheet(_STYLESHEET)
    pg.setConfigOptions(
        antialias=True, background=COLORS["surface"], foreground=COLORS["text_muted"]
    )
