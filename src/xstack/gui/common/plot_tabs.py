"""Plot tabs that can be detached into their own windows.

`PlotTabs` is a `QTabWidget` whose tabs can be moved into separate windows (context menu of a tab,
the *Detach* button next to the tabs, or `detach`). Closing such a window puts the tab back at its
original position. `show_view` brings a view to the front, in the tab widget or in its window, so
that the applications can switch the shown plot when the workflow step changes.
"""

from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QMenu, QTabWidget, QToolButton, QVBoxLayout, QWidget

__all__ = ["DetachedView", "PlotTabs"]


class DetachedView(QWidget):
    """Window that shows one detached plot tab.

    Signals:
        closed(): The window was closed; the plot goes back into its tab widget.

    Args:
        widget: The plot widget.
        title: Window title.
    """

    closed = Signal()

    def __init__(self, widget: QWidget, title: str) -> None:
        super().__init__(None, Qt.WindowType.Window)
        self.setWindowTitle(title)
        self.setObjectName("detached_view")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(widget)
        self.widget = widget

    def closeEvent(self, event: QCloseEvent) -> None:
        """Hand the plot back before the window closes."""
        self.closed.emit()
        event.accept()


class PlotTabs(QTabWidget):
    """Tab widget of plots whose tabs can be detached into windows.

    Signals:
        detachedChanged(): A tab was detached or attached again.

    Args:
        application: Name of the application, used in the titles of detached windows.
        parent: Parent widget.
    """

    detachedChanged = Signal()

    def __init__(self, application: str = "xstack", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("plot_tabs")
        self.setDocumentMode(True)
        self._application = application
        self._order: list[tuple[QWidget, str]] = []
        self._windows: dict[QWidget, DetachedView] = {}
        self.tabBar().setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tabBar().customContextMenuRequested.connect(self._tab_menu)
        self.detach_button = QToolButton()
        self.detach_button.setText("Detach")
        self.detach_button.setObjectName("detach_plot")
        self.detach_button.setToolTip("Show the current plot in its own window")
        self.detach_button.clicked.connect(self.detach_current)
        self.setCornerWidget(self.detach_button, Qt.Corner.TopRightCorner)

    def addTab(self, widget: QWidget, title: str) -> int:
        """Add a plot tab and remember its position for re-attaching.

        Args:
            widget: The plot widget.
            title: Tab title.

        Returns:
            The index of the new tab.
        """
        self._order.append((widget, title))
        return super().addTab(widget, title)

    def title(self, widget: QWidget) -> str:
        """Return the tab title of a plot widget.

        Args:
            widget: A widget added with `addTab`.

        Returns:
            Its title.
        """
        return next(t for w, t in self._order if w is widget)

    def is_detached(self, widget: QWidget) -> bool:
        """Whether a plot is shown in its own window.

        Args:
            widget: A widget added with `addTab`.

        Returns:
            True if detached.
        """
        return widget in self._windows

    def window_of(self, widget: QWidget) -> DetachedView | None:
        """Return the window of a detached plot, or ``None``.

        Args:
            widget: A widget added with `addTab`.

        Returns:
            The window, or ``None`` if the plot is in the tab widget.
        """
        return self._windows.get(widget)

    def detach_current(self) -> DetachedView | None:
        """Detach the current tab into its own window.

        Returns:
            The window, or ``None`` if there is no tab.
        """
        widget = self.currentWidget()
        return self.detach(widget) if widget is not None else None

    def detach(self, widget: QWidget) -> DetachedView:
        """Move a plot into its own window, or bring its window to the front if detached.

        Args:
            widget: A widget added with `addTab`.

        Returns:
            The window.
        """
        window = self._windows.get(widget)
        if window is None:
            title = self.title(widget)
            self.removeTab(self.indexOf(widget))
            window = DetachedView(widget, f"{title} — {self._application}")
            window.closed.connect(lambda w=widget: self.attach(w))
            window.resize(max(widget.width(), 700), max(widget.height(), 500))
            self._windows[widget] = window
            widget.show()
            window.show()
            self.detachedChanged.emit()
        window.raise_()
        return window

    def attach(self, widget: QWidget) -> None:
        """Put a detached plot back into its tab at its original position.

        Args:
            widget: A widget added with `addTab`.
        """
        window = self._windows.pop(widget, None)
        if window is None:
            return
        position = 0
        for other, _title in self._order:
            if other is widget:
                break
            if other not in self._windows:
                position += 1
        self.insertTab(position, widget, self.title(widget))
        widget.show()
        window.hide()
        window.deleteLater()
        self.detachedChanged.emit()

    def attach_all(self) -> None:
        """Put all detached plots back into their tabs."""
        for widget in list(self._windows):
            self.attach(widget)

    def show_view(self, widget: QWidget) -> None:
        """Bring a plot to the front: select its tab, or raise its window if detached.

        Args:
            widget: A widget added with `addTab`.
        """
        window = self._windows.get(widget)
        if window is not None:
            window.raise_()
        else:
            self.setCurrentWidget(widget)

    def _tab_menu(self, position: QPoint) -> None:
        index = self.tabBar().tabAt(position)
        if index < 0:
            return
        widget = self.widget(index)
        menu = QMenu(self)
        action = menu.addAction(f"Detach “{self.tabText(index)}”")
        action.triggered.connect(lambda: self.detach(widget))
        menu.exec(self.tabBar().mapToGlobal(position))
