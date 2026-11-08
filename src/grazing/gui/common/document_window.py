"""Base class of the main windows of the grazing applications."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QAction, QCloseEvent
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QFileDialog,
    QInputDialog,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QWidget,
)

from grazing.core.commands import rename_project
from grazing.gui.common.controller import DocumentController
from grazing.gui.common.files import ask_save_path
from grazing.gui.common.journal_view import JournalView
from grazing.gui.common.plot_tabs import PlotTabs
from grazing.gui.common.style import Theme, apply_style, icon, restyle_plots
from grazing.gui.common.wizard import ProjectWizard
from grazing.gui.common.workers import Job, JobRunner

__all__ = ["DocumentWindow"]

MAX_RECENT_FILES = 8
"""Number of recently opened files that are remembered."""


class DocumentWindow(QMainWindow):
    """Main window of an application working on one document (a project file).

    The class implements the document handling shared by the applications: opening, saving and
    renaming the project, the recent-files list, the window title, closing with a prompt for
    unsaved changes, theme switching, docks, and starting jobs and showing their progress.
    Subclasses build the widgets and implement the hooks below.

    Subclasses set the class attributes, create the widgets listed below in their constructor
    and implement the hooks `_theme_changed`, `_job_state_changed`, `_before_close` and
    `show_step_view`.

    Attributes:
        project_kind: Name of the kind of document in dialog texts, e.g. ``"project"``.
        file_filter: File dialog filter of project files, e.g. ``"grazing projects (*.grazing)"``.
        file_suffix: Suffix of project files including the dot, e.g. ``".grazing"``.
        app_name: Application name shown in the window title.
        controller: The document controller.
        settings: Settings store for geometry, theme and recent files.
        confirm_close: Ask whether to save unsaved changes when closing or replacing the
            project.
        wizard_factory: Creates the new-project wizard.
        name_edit: Field showing the project name (created by the subclass).
        recent_menu: Menu listing the recent files (created by the subclass).
        theme_actions: Checkable theme action by theme name (created by the subclass).
        new_action: Toolbar action that gets its icon from `_apply_icons`.
        open_action: Toolbar action that gets its icon from `_apply_icons`.
        save_action: Toolbar action that gets its icon from `_apply_icons`.
        undo_action: Toolbar action that gets its icon from `_apply_icons`.
        redo_action: Toolbar action that gets its icon from `_apply_icons`.
        run_action: Toolbar action that gets its icon from `_apply_icons`.
        stop_action: Toolbar action that gets its icon from `_apply_icons`.
        auto_views_action: Checkable action switching the plots with the workflow step.
        plot_tabs: The plot tabs, which can be detached.
        convergence_plot: Plot of the optimization progress, with a ``clear`` method.
        progress_bar: Progress bar of running jobs.
        job_runner: Runs the background jobs.
        journal_view: Display of the project journal.
    """

    project_kind = "project"
    file_filter = "All files (*)"
    file_suffix = ""
    app_name = "grazing"

    controller: DocumentController
    settings: QSettings
    confirm_close: bool
    wizard_factory: Callable[[], ProjectWizard]
    name_edit: QLineEdit
    recent_menu: QMenu
    theme_actions: dict[str, QAction]
    new_action: QAction
    open_action: QAction
    save_action: QAction
    undo_action: QAction
    redo_action: QAction
    run_action: QAction
    stop_action: QAction
    auto_views_action: QAction
    plot_tabs: PlotTabs
    convergence_plot: Any
    progress_bar: QProgressBar
    job_runner: JobRunner
    journal_view: JournalView
    _job_state: Any
    _live_values: dict[str, float] | None

    # ------------------------------------------------------------------ hooks
    def _theme_changed(self) -> None:
        """Refresh the views after a theme change. The default does nothing."""

    def _job_state_changed(self, running: bool) -> None:
        """Adapt the application after a job started or ended. The default does nothing.

        Args:
            running: Whether a job is running now.
        """

    def _before_close(self) -> None:
        """Release resources when the window closes. The default does nothing."""

    def show_step_view(self) -> None:
        """Show the plot of the current workflow step, if plots switch automatically."""
        raise NotImplementedError

    # ------------------------------------------------------------------ layout
    def _dock(self, title: str, widget: QWidget, area: Qt.DockWidgetArea) -> QDockWidget:
        dock = QDockWidget(title, self)
        dock.setObjectName(f"dock_{title.lower().replace(' ', '_')}")
        dock.setWidget(widget)
        self.addDockWidget(area, dock)
        return dock

    def _apply_icons(self) -> None:
        for action, name in (
            (self.new_action, "new"),
            (self.open_action, "open"),
            (self.save_action, "save"),
            (self.undo_action, "undo"),
            (self.redo_action, "redo"),
            (self.stop_action, "stop"),
        ):
            action.setIcon(icon(name))
        self.run_action.setIcon(icon("run", "#ffffff"))

    def set_theme(self, theme: Theme) -> None:
        """Switch the color theme of the application and remember it in the settings.

        Args:
            theme: ``"light"``, ``"dark"`` or ``"system"``.
        """
        app = QApplication.instance()
        if isinstance(app, QApplication):
            apply_style(app, theme)
        self.settings.setValue("theme", theme)
        self.theme_actions[theme].setChecked(True)
        self._apply_icons()
        restyle_plots(self)
        self._theme_changed()

    # ------------------------------------------------------------------ jobs
    def start_job(self, job: Job) -> bool:
        """Run a long calculation in the background.

        Args:
            job: The job.

        Returns:
            Whether the job was started (only one job runs at a time).
        """
        if not self.job_runner.start(job):
            self._show_error("Another calculation is running")
            return False
        return True

    def _job_started(self, name: str) -> None:
        if self.auto_views_action.isChecked():
            self.plot_tabs.show_view(self.convergence_plot)
        self.convergence_plot.clear()
        self._job_state = self.controller.state
        self._live_values = None
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(True)
        self.stop_action.setEnabled(True)
        self.run_action.setEnabled(False)
        self._job_state_changed(True)
        self.statusBar().showMessage(f"{name} running…")

    def _job_finished(self, name: str) -> None:
        self._job_done(f"{name} finished")

    def _job_done(self, message: str) -> None:
        self._job_state = None
        self.progress_bar.setVisible(False)
        self.stop_action.setEnabled(False)
        self.run_action.setEnabled(True)
        self.statusBar().showMessage(message, 8000)
        if self.plot_tabs.currentWidget() is self.convergence_plot:
            self.show_step_view()
        self._job_state_changed(False)

    def _show_error(self, message: str) -> None:
        self.statusBar().showMessage(message, 8000)

    # ------------------------------------------------------------------ project name and title
    def _on_journal_changed(self) -> None:
        self.journal_view.set_journal(self.controller.project.journal)

    def _on_file_changed(self, _path: object) -> None:
        self.name_edit.setText(self.controller.state.name)
        self._update_title()

    def _on_clean_changed(self, _clean: bool) -> None:
        self._update_title()

    def _commit_project_name(self) -> None:
        """Apply the text of the name field as the project name; empty names are rejected."""
        name = self.name_edit.text().strip()
        current = self.controller.state.name
        if not name:
            self.name_edit.setText(current)
            return
        if name != current:
            self.controller.execute(rename_project(name))
        self.name_edit.setText(name)

    def rename_project(self) -> None:
        """Ask for a new project name and apply it."""
        text, accepted = QInputDialog.getText(
            self, "Rename project", "Project name:", text=self.controller.state.name
        )
        if accepted:
            self.name_edit.setText(text)
            self._commit_project_name()

    def _update_title(self) -> None:
        path = self.controller.path
        name = self.controller.state.name
        location = path.name if path is not None else "unsaved"
        self.setWindowTitle(f"{name} ({location})[*] — {self.app_name}")
        self.setWindowModified(self.controller.is_dirty)

    # ------------------------------------------------------------------ files
    def _recent_files(self) -> list[str]:
        value = self.settings.value("recent_files", [])
        if isinstance(value, str):
            return [value]
        return [str(v) for v in value or []]

    def _add_recent(self, path: Path) -> None:
        files = [str(path)] + [f for f in self._recent_files() if f != str(path)]
        self.settings.setValue("recent_files", files[:MAX_RECENT_FILES])
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
            f"Save changes to the current {self.project_kind}?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
        )
        if answer == QMessageBox.StandardButton.Save:
            return self.save_project()
        return answer == QMessageBox.StandardButton.Discard

    def new_project_wizard(self) -> bool:
        """Create a new project with the project wizard and save it if a location was chosen.

        Returns:
            Whether a new project was created.
        """
        if not self._maybe_save():
            return False
        wizard = self.wizard_factory()
        if not wizard.exec():
            return False
        setup = wizard.setup()
        try:
            self.controller.new_from_setup(setup)
        except (KeyError, ValueError) as exc:
            self._show_error(f"Cannot create the project: {exc}")
            return False
        if setup.path is not None:
            self.save_project_as(setup.path)
        return True

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
            chosen, _ = QFileDialog.getOpenFileName(
                self, f"Open {self.project_kind}", "", self.file_filter
            )
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
            chosen = ask_save_path(
                self, f"Save {self.project_kind}", self.file_filter, self.file_suffix
            )
            if chosen is None:
                return False
            path = chosen
        saved = self.controller.save_project(path)
        if saved:
            self._add_recent(Path(path))
        return saved

    def closeEvent(self, event: QCloseEvent) -> None:
        """Ask to save unsaved changes and store the window layout."""
        if not self._maybe_save():
            event.ignore()
            return
        self._before_close()
        self.plot_tabs.attach_all()
        self.settings.setValue("main_window/geometry", self.saveGeometry())
        self.settings.setValue("main_window/state", self.saveState())
        event.accept()
