"""Main window of the xstack application."""

from pathlib import Path

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QAction, QCloseEvent, QKeySequence
from PySide6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QTabWidget,
    QUndoView,
    QVBoxLayout,
    QWidget,
)

from xstack.gui.controller import ProjectController
from xstack.gui.widgets import JournalView, ParameterTable
from xstack.project import ProjectState
from xstack.project.templates import TEMPLATES

__all__ = ["WORKFLOW_STEPS", "MainWindow"]

WORKFLOW_STEPS = (
    "Goal",
    "Materials",
    "Coating",
    "Constraints",
    "Process",
    "Scan",
    "Optimize",
    "Analyze",
)
"""Steps of the design workflow shown in the workflow list."""

PROJECT_FILTER = "xstack projects (*.xstack);;All files (*)"
_MAX_RECENT = 8


def _titled(title: str, widget: QWidget) -> QWidget:
    """Return a widget with a bold title label above `widget`."""
    container = QWidget()
    layout = QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    label = QLabel(f"<b>{title}</b>")
    layout.addWidget(label)
    layout.addWidget(widget)
    return container


class MainWindow(QMainWindow):
    """Main window with workflow, plots, properties, journal and history.

    Args:
        controller: The project controller. A new one is created if not given.
        settings: Settings store for window geometry and recent files.
        confirm_close: Ask whether to save unsaved changes when closing or replacing the
            project.
    """

    def __init__(
        self,
        controller: ProjectController | None = None,
        settings: QSettings | None = None,
        confirm_close: bool = True,
    ) -> None:
        super().__init__()
        self.controller = controller or ProjectController(parent=self)
        self.settings = settings or QSettings("xstack", "xstack")
        self.confirm_close = confirm_close
        self.setDockNestingEnabled(True)
        self._build_central()
        self._build_docks()
        self._build_actions()
        self._build_menus()
        self._connect()
        self.statusBar().showMessage("Ready", 3000)
        self._on_state_changed(self.controller.state)
        self._on_journal_changed()
        self._update_title()
        geometry = self.settings.value("main_window/geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        else:
            self.resize(1400, 900)
        state = self.settings.value("main_window/state")
        if state is not None:
            self.restoreState(state)

    def _build_central(self) -> None:
        self.plot_tabs = QTabWidget()
        for title in ("Reflectivity", "Layer profile", "Design map", "Convergence"):
            placeholder = QLabel(f"{title} plot")
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            placeholder.setObjectName(f"placeholder_{title}")
            self.plot_tabs.addTab(placeholder, title)
        self.setCentralWidget(self.plot_tabs)

    def _dock(self, title: str, widget: QWidget, area: Qt.DockWidgetArea) -> QDockWidget:
        dock = QDockWidget(title, self)
        dock.setObjectName(f"dock_{title.lower().replace(' ', '_')}")
        dock.setWidget(widget)
        self.addDockWidget(area, dock)
        return dock

    def _build_docks(self) -> None:
        self.workflow_list = QListWidget()
        self.workflow_list.addItems(list(WORKFLOW_STEPS))
        self.candidate_list = QListWidget()
        left = QSplitter(Qt.Orientation.Vertical)
        left.addWidget(_titled("Workflow", self.workflow_list))
        left.addWidget(_titled("Candidates", self.candidate_list))
        self.workflow_dock = self._dock("Project", left, Qt.DockWidgetArea.LeftDockWidgetArea)

        properties = QWidget()
        layout = QVBoxLayout(properties)
        self.design_label = QLabel()
        self.design_label.setWordWrap(True)
        self.parameter_table = ParameterTable()
        layout.addWidget(self.design_label)
        layout.addWidget(self.parameter_table)
        self.properties_dock = self._dock(
            "Properties", properties, Qt.DockWidgetArea.RightDockWidgetArea
        )

        self.journal_view = JournalView()
        self.undo_view = QUndoView(self.controller.undo_stack)
        self.undo_view.setEmptyLabel("<initial state>")
        self.journal_dock = self._dock(
            "Journal", self.journal_view, Qt.DockWidgetArea.BottomDockWidgetArea
        )
        self.history_dock = self._dock(
            "History", self.undo_view, Qt.DockWidgetArea.BottomDockWidgetArea
        )
        self.tabifyDockWidget(self.journal_dock, self.history_dock)
        self.journal_dock.raise_()
        self.resizeDocks(
            [self.workflow_dock, self.properties_dock], [220, 420], Qt.Orientation.Horizontal
        )
        self.resizeDocks([self.journal_dock], [160], Qt.Orientation.Vertical)

    def _build_actions(self) -> None:
        self.new_actions: dict[str, QAction] = {}
        for name in TEMPLATES:
            action = QAction(name.replace("_", " ").capitalize(), self)
            action.triggered.connect(lambda _=False, n=name: self.new_project(n))
            self.new_actions[name] = action
        self.new_action = QAction("&New", self)
        self.new_action.setShortcut(QKeySequence.StandardKey.New)
        self.new_action.setToolTip("New project from the periodic multilayer template")
        self.new_action.triggered.connect(lambda: self.new_project("periodic"))
        self.open_action = QAction("&Open…", self)
        self.open_action.setShortcut(QKeySequence.StandardKey.Open)
        self.open_action.triggered.connect(lambda: self.open_project())
        self.save_action = QAction("&Save", self)
        self.save_action.setShortcut(QKeySequence.StandardKey.Save)
        self.save_action.triggered.connect(lambda: self.save_project())
        self.save_as_action = QAction("Save &as…", self)
        self.save_as_action.setShortcut(QKeySequence.StandardKey.SaveAs)
        self.save_as_action.triggered.connect(lambda: self.save_project_as())
        self.quit_action = QAction("&Quit", self)
        self.quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        self.quit_action.triggered.connect(self.close)
        self.undo_action = self.controller.undo_stack.createUndoAction(self, "&Undo")
        self.undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        self.redo_action = self.controller.undo_stack.createRedoAction(self, "&Redo")
        self.redo_action.setShortcut(QKeySequence.StandardKey.Redo)

    def _build_menus(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        new_menu = file_menu.addMenu("&New project")
        for action in self.new_actions.values():
            new_menu.addAction(action)
        file_menu.addAction(self.open_action)
        self.recent_menu = file_menu.addMenu("Open &recent")
        file_menu.addSeparator()
        file_menu.addAction(self.save_action)
        file_menu.addAction(self.save_as_action)
        file_menu.addSeparator()
        file_menu.addAction(self.quit_action)
        edit_menu = self.menuBar().addMenu("&Edit")
        edit_menu.addAction(self.undo_action)
        edit_menu.addAction(self.redo_action)
        view_menu = self.menuBar().addMenu("&View")
        for dock in (
            self.workflow_dock,
            self.properties_dock,
            self.journal_dock,
            self.history_dock,
        ):
            view_menu.addAction(dock.toggleViewAction())
        toolbar = self.addToolBar("Main")
        toolbar.setObjectName("toolbar_main")
        toolbar.addAction(self.new_action)
        toolbar.addAction(self.open_action)
        toolbar.addAction(self.save_action)
        toolbar.addSeparator()
        toolbar.addAction(self.undo_action)
        toolbar.addAction(self.redo_action)
        self._update_recent_menu()

    def _connect(self) -> None:
        c = self.controller
        c.stateChanged.connect(self._on_state_changed)
        c.journalChanged.connect(self._on_journal_changed)
        c.fileChanged.connect(lambda _: self._update_title())
        c.errorOccurred.connect(self._show_error)
        c.undo_stack.cleanChanged.connect(lambda _: self._update_title())
        self.parameter_table.commandRequested.connect(c.execute)

    def _on_state_changed(self, state: ProjectState) -> None:
        if state.mirror is None:
            self.design_label.setText("No design defined.")
            self.parameter_table.setEnabled(False)
        else:
            coating = state.mirror.coating
            materials = ", ".join(f"{r}: {m.name}" for r, m in coating.materials.items())
            self.design_label.setText(
                f"<b>{type(coating).__name__}</b> ({materials}) on {state.mirror.substrate.name}"
            )
            self.parameter_table.setEnabled(True)
            self.parameter_table.set_parameters(coating.parameters)
        self.candidate_list.clear()
        self.candidate_list.addItems([c.name for c in state.candidates])
        self._update_title()

    def _on_journal_changed(self) -> None:
        self.journal_view.set_journal(self.controller.project.journal)

    def _show_error(self, message: str) -> None:
        self.statusBar().showMessage(message, 8000)

    def _update_title(self) -> None:
        path = self.controller.path
        name = self.controller.state.name
        location = path.name if path is not None else "unsaved"
        self.setWindowTitle(f"{name} ({location})[*] — xstack")
        self.setWindowModified(self.controller.is_dirty)

    def _recent_files(self) -> list[str]:
        value = self.settings.value("recent_files", [])
        if isinstance(value, str):
            return [value]
        return [str(v) for v in value or []]

    def _add_recent(self, path: Path) -> None:
        files = [str(path)] + [f for f in self._recent_files() if f != str(path)]
        self.settings.setValue("recent_files", files[:_MAX_RECENT])
        self._update_recent_menu()

    def _update_recent_menu(self) -> None:
        self.recent_menu.clear()
        files = self._recent_files()
        for f in files:
            action = self.recent_menu.addAction(f)
            action.triggered.connect(lambda _=False, p=f: self.open_project(p))
        self.recent_menu.setEnabled(bool(files))

    def _maybe_save(self) -> bool:
        """Ask to save unsaved changes. Returns False if the user cancels."""
        if not self.confirm_close or not self.controller.is_dirty:
            return True
        answer = QMessageBox.question(
            self,
            "Unsaved changes",
            "Save changes to the current project?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
        )
        if answer == QMessageBox.StandardButton.Save:
            return self.save_project()
        return answer == QMessageBox.StandardButton.Discard

    def new_project(self, template: str = "periodic") -> None:
        """Start a new project from a template.

        Args:
            template: Name of a project template.
        """
        if self._maybe_save():
            self.controller.new_project(template)

    def open_project(self, path: Path | str | None = None) -> bool:
        """Open a project file, asking for the file name if not given.

        Args:
            path: The project file.

        Returns:
            Whether a project was opened.
        """
        if not self._maybe_save():
            return False
        if path is None:
            chosen, _ = QFileDialog.getOpenFileName(self, "Open project", "", PROJECT_FILTER)
            if not chosen:
                return False
            path = chosen
        if self.controller.open_project(path):
            self._add_recent(Path(path))
            self.statusBar().showMessage(f"Opened {path}", 5000)
            return True
        return False

    def save_project(self) -> bool:
        """Save the project to its file, asking for a file name if it has none.

        Returns:
            Whether the project was saved.
        """
        if self.controller.path is None:
            return self.save_project_as()
        saved = self.controller.save_project()
        if saved:
            self.statusBar().showMessage(f"Saved {self.controller.path}", 5000)
        return saved

    def save_project_as(self, path: Path | str | None = None) -> bool:
        """Save the project under a new file name.

        Args:
            path: The new file. Asked for if not given.

        Returns:
            Whether the project was saved.
        """
        if path is None:
            chosen, _ = QFileDialog.getSaveFileName(self, "Save project", "", PROJECT_FILTER)
            if not chosen:
                return False
            path = chosen if chosen.endswith(".xstack") else chosen + ".xstack"
        saved = self.controller.save_project(path)
        if saved:
            self._add_recent(Path(path))
        return saved

    def closeEvent(self, event: QCloseEvent) -> None:
        """Ask to save unsaved changes and store the window layout."""
        if not self._maybe_save():
            event.ignore()
            return
        self.settings.setValue("main_window/geometry", self.saveGeometry())
        self.settings.setValue("main_window/state", self.saveState())
        event.accept()
