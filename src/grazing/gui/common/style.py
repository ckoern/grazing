"""Design language of the graphical user interface.

All colors, spacing and component styles are defined here: a neutral surface, a single indigo
accent for primary actions and selections, cards with rounded corners for grouped content, and
a categorical palette for plots that reads well on light and dark backgrounds.

Two color themes exist, ``"light"`` and ``"dark"``; ``"system"`` follows the operating system.
Use `apply_style` for the application (again to switch the theme), `set_variant` to mark primary
buttons, `restyle_plots` to update existing plots after a theme change, `COLORS` for the colors
of the active theme and `PLOT_PALETTE` for plot colors.
"""

from typing import Literal

import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QIcon, QPalette, QPixmap
from PySide6.QtWidgets import QApplication, QStyleFactory, QWidget

__all__ = [
    "COLORS",
    "PLOT_PALETTE",
    "THEMES",
    "Theme",
    "Variant",
    "apply_style",
    "current_theme",
    "icon",
    "resolve_theme",
    "restyle_plots",
    "set_variant",
]

type Theme = Literal["light", "dark", "system"]
"""Color theme; ``"system"`` follows the color scheme of the operating system."""

THEMES: dict[str, dict[str, str]] = {
    "light": {
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
    },
    "dark": {
        "background": "#14161c",
        "surface": "#1d2029",
        "surface_alt": "#242833",
        "border": "#2f3442",
        "border_strong": "#424959",
        "text": "#e6e8ee",
        "text_muted": "#9aa3b2",
        "accent": "#6366f1",
        "accent_hover": "#818cf8",
        "accent_soft": "#272b4d",
        "accent_border": "#3f4683",
        "success": "#4ade80",
        "success_soft": "#173222",
        "danger": "#f87171",
    },
}
"""Color tokens of the themes."""

COLORS: dict[str, str] = dict(THEMES["light"])
"""Color tokens of the active theme; updated in place by `apply_style`."""

PLOT_PALETTE: tuple[str, ...] = (
    "#6366f1",
    "#f43f5e",
    "#06b6d4",
    "#f59e0b",
    "#22c55e",
    "#a855f7",
    "#94a3b8",
    "#ec4899",
)
"""Categorical colors for plot curves, in the order of use, for both themes."""

type Variant = Literal["primary", "secondary"]
"""Button variants: one primary action per area, secondary for everything else."""


def _stylesheet(c: dict[str, str]) -> str:
    return f"""
QMainWindow, #step_panel {{
    background: {c["background"]};
}}
QDockWidget::title {{
    background: {c["background"]};
    padding: 6px 8px;
    font-weight: 600;
    color: {c["text_muted"]};
}}
QToolBar {{
    background: {c["surface"]};
    border: none;
    border-bottom: 1px solid {c["border"]};
    spacing: 4px;
    padding: 4px 8px;
}}
QToolButton {{
    border: 1px solid transparent;
    border-radius: 6px;
    padding: 4px 8px;
}}
QToolButton:hover {{
    background: {c["accent_soft"]};
    border-color: {c["accent_border"]};
}}
QToolButton:disabled {{
    color: {c["border_strong"]};
}}
QStatusBar {{
    background: {c["surface"]};
    border-top: 1px solid {c["border"]};
    color: {c["text_muted"]};
}}
QMenuBar {{
    background: {c["surface"]};
}}
QMenuBar::item:selected, QMenu::item:selected {{
    background: {c["accent_soft"]};
    color: {c["text"]};
}}
QMenu {{
    background: {c["surface"]};
    border: 1px solid {c["border"]};
    padding: 4px;
}}
QMenu::item {{
    padding: 5px 18px;
    border-radius: 4px;
}}

QTabWidget::pane {{
    border: 1px solid {c["border"]};
    border-radius: 8px;
    background: {c["surface"]};
    top: -1px;
}}
QTabBar {{
    qproperty-drawBase: 0;
}}
QTabBar::tab {{
    background: transparent;
    color: {c["text_muted"]};
    padding: 6px 11px;
    margin: 0 2px 4px 0;
    border-radius: 14px;
    font-weight: 600;
}}
QTabBar::tab:hover {{
    background: {c["surface"]};
    color: {c["text"]};
}}
QTabBar::tab:selected {{
    background: {c["accent"]};
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
    background: {c["accent"]};
    color: white;
    font-weight: 600;
    padding: 4px 12px;
}}
#primary_tool:hover {{
    background: {c["accent_hover"]};
}}
#primary_tool:disabled {{
    background: {c["accent_border"]};
}}

QPushButton {{
    background: {c["surface"]};
    border: 1px solid {c["border_strong"]};
    border-radius: 6px;
    padding: 6px 14px;
    font-weight: 500;
}}
QPushButton:hover {{
    border-color: {c["accent"]};
    color: {c["accent"]};
}}
QPushButton:pressed {{
    background: {c["accent_soft"]};
}}
QPushButton:disabled {{
    color: {c["border_strong"]};
    border-color: {c["border"]};
    background: {c["surface_alt"]};
}}
QPushButton[variant="primary"] {{
    background: {c["accent"]};
    border: 1px solid {c["accent"]};
    color: white;
    font-weight: 600;
}}
QPushButton[variant="primary"]:hover {{
    background: {c["accent_hover"]};
    border-color: {c["accent_hover"]};
    color: white;
}}
QPushButton[variant="primary"]:disabled {{
    background: {c["accent_border"]};
    border-color: {c["accent_border"]};
    color: white;
}}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {c["surface"]};
    border: 1px solid {c["border_strong"]};
    border-radius: 6px;
    padding: 4px 8px;
    min-height: 20px;
    selection-background-color: {c["accent"]};
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border: 1px solid {c["accent"]};
}}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {{
    background: {c["surface_alt"]};
    color: {c["text_muted"]};
    border-color: {c["border"]};
}}
QTableView QLineEdit, QTableView QSpinBox, QTableView QDoubleSpinBox, QTableView QComboBox {{
    padding: 2px 4px;
    border-radius: 4px;
}}
QComboBox::drop-down {{
    border: none;
    width: 22px;
}}
QComboBox QAbstractItemView {{
    background: {c["surface"]};
    border: 1px solid {c["border"]};
    selection-background-color: {c["accent_soft"]};
    selection-color: {c["text"]};
}}
QCheckBox, QRadioButton {{
    spacing: 6px;
}}

QGroupBox {{
    background: {c["surface"]};
    border: 1px solid {c["border"]};
    border-radius: 8px;
    margin-top: 14px;
    padding: 12px 10px 10px 10px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: {c["text"]};
    font-weight: 600;
}}

QTableView, QTableWidget, QListWidget, QTreeView {{
    background: {c["surface"]};
    gridline-color: transparent;
    border: none;
    alternate-background-color: {c["surface_alt"]};
    selection-background-color: {c["accent_soft"]};
    selection-color: {c["text"]};
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
    background: {c["surface_alt"]};
}}
QListWidget::item:selected, QTableView::item:selected {{
    background: {c["accent_soft"]};
    color: {c["text"]};
}}
QHeaderView::section {{
    background: {c["surface"]};
    color: {c["text_muted"]};
    border: none;
    border-bottom: 1px solid {c["border"]};
    padding: 6px 6px;
    font-weight: 600;
}}
QTableCornerButton::section {{
    background: {c["surface"]};
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
    background: {c["border_strong"]};
    border-radius: 4px;
    min-height: 24px;
    min-width: 24px;
}}
QScrollBar::handle:hover {{
    background: {c["text_muted"]};
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
    border: 1px solid {c["border"]};
    border-radius: 6px;
    background: {c["surface_alt"]};
    text-align: center;
    height: 14px;
}}
QProgressBar::chunk {{
    background: {c["accent"]};
    border-radius: 5px;
}}
QToolTip {{
    background: {c["text"]};
    color: {c["background"]};
    border: none;
    padding: 6px 8px;
}}

#design_header {{
    background: {c["surface"]};
    border: 1px solid {c["border"]};
    border-radius: 8px;
    padding: 10px 12px;
}}
#confirmation {{
    background: {c["success_soft"]};
    color: {c["success"]};
    border-radius: 6px;
    padding: 6px 10px;
}}
QLabel[role="help"] {{
    background: {c["accent_soft"]};
    border: none;
    border-left: 3px solid {c["accent"]};
    border-radius: 6px;
    padding: 10px 12px;
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


def _palette(colors: dict[str, str]) -> QPalette:
    palette = QPalette()
    roles = {
        QPalette.ColorRole.Window: colors["background"],
        QPalette.ColorRole.Base: colors["surface"],
        QPalette.ColorRole.AlternateBase: colors["surface_alt"],
        QPalette.ColorRole.Button: colors["surface"],
        QPalette.ColorRole.Text: colors["text"],
        QPalette.ColorRole.WindowText: colors["text"],
        QPalette.ColorRole.ButtonText: colors["text"],
        QPalette.ColorRole.BrightText: colors["text"],
        QPalette.ColorRole.Highlight: colors["accent"],
        QPalette.ColorRole.HighlightedText: "#ffffff",
        QPalette.ColorRole.PlaceholderText: colors["text_muted"],
        QPalette.ColorRole.ToolTipBase: colors["text"],
        QPalette.ColorRole.ToolTipText: colors["background"],
        QPalette.ColorRole.Light: colors["border_strong"],
        QPalette.ColorRole.Mid: colors["border"],
        QPalette.ColorRole.Dark: colors["border_strong"],
    }
    for role, color in roles.items():
        palette.setColor(role, QColor(color))
    disabled = QColor(colors["text_muted"])
    for role in (
        QPalette.ColorRole.Text,
        QPalette.ColorRole.WindowText,
        QPalette.ColorRole.ButtonText,
    ):
        palette.setColor(QPalette.ColorGroup.Disabled, role, disabled)
    return palette


def resolve_theme(theme: Theme) -> Literal["light", "dark"]:
    """Return the concrete theme for a theme setting.

    Args:
        theme: ``"light"``, ``"dark"`` or ``"system"``.

    Returns:
        ``"light"`` or ``"dark"``; ``"system"`` follows the color scheme reported by Qt.
    """
    if theme in ("light", "dark"):
        return theme
    app = QApplication.instance()
    if isinstance(app, QApplication) and app.styleHints().colorScheme() == Qt.ColorScheme.Dark:
        return "dark"
    return "light"


def current_theme() -> Literal["light", "dark"]:
    """Return the active concrete theme."""
    return "dark" if COLORS["background"] == THEMES["dark"]["background"] else "light"


def apply_style(app: QApplication, theme: Theme = "light") -> None:
    """Install the design language for the application and new plots.

    Sets the Fusion style with the palette of the theme, the application stylesheet and the
    pyqtgraph defaults. Calling it again with the same concrete theme has no effect; calling
    it with another theme switches the theme (existing plots are updated by `restyle_plots`).

    Args:
        app: The application to style.
        theme: ``"light"``, ``"dark"`` or ``"system"``.
    """
    concrete = resolve_theme(theme)
    if app.property("grazing_theme") == concrete:
        return
    app.setProperty("grazing_theme", concrete)
    COLORS.clear()
    COLORS.update(THEMES[concrete])
    fusion = QStyleFactory.create("Fusion")
    if fusion is not None:
        app.setStyle(fusion)
    app.setPalette(_palette(COLORS))
    font = QFont(app.font())
    font.setPointSizeF(max(font.pointSizeF(), 9.5))
    app.setFont(font)
    app.setStyleSheet(_stylesheet(COLORS))
    pg.setConfigOptions(
        antialias=True, background=COLORS["surface"], foreground=COLORS["text_muted"]
    )


def restyle_plots(root: QWidget) -> None:
    """Apply the colors of the active theme to all existing pyqtgraph plots below a widget.

    Args:
        root: Widget whose plots are updated.
    """
    pen = pg.mkPen(COLORS["text_muted"])
    for view in root.findChildren(pg.GraphicsView):
        view.setBackground(COLORS["surface"])
        scene = view.scene()
        if scene is None:
            continue
        for item in scene.items():
            if isinstance(item, pg.AxisItem):
                item.setPen(pen)
                item.setTextPen(pen)
            elif isinstance(item, pg.LegendItem):
                item.setLabelTextColor(COLORS["text"])
