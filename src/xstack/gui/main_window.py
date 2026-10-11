"""Main window of the xstack application."""

import functools
from pathlib import Path

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QAction, QCloseEvent, QKeySequence
from PySide6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QUndoView,
    QVBoxLayout,
    QWidget,
)

from xstack.gui.controller import ProjectController
from xstack.gui.plots import (
    ConvergencePlot,
    Curve,
    DesignMapPlot,
    HistogramPlot,
    LayerProfilePlot,
    ReflectivityPlot,
)
from xstack.gui.preview import Preview, compute_preview
from xstack.gui.steps import (
    AnalyzePage,
    CoatingPage,
    ConstraintsPage,
    GoalPage,
    MaterialsPage,
    OptimizePage,
    ProcessPage,
    ScanPage,
    StepPage,
)
from xstack.gui.widgets import JournalView
from xstack.gui.workers import Job, JobRunner, LatestRunner
from xstack.manufacturing.tolerance import ToleranceResult
from xstack.optim.events import ProgressEvent
from xstack.optim.pipeline import PipelineResult
from xstack.optim.problem import DesignProblem
from xstack.optim.scan import ScanResult
from xstack.project import Candidate, ProjectState
from xstack.project.commands import (
    add_candidate,
    apply_result,
    promote_candidate,
    record_run,
    remove_candidate,
)
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


def _summary(preview: Preview) -> str:
    """Return a one-line summary of a preview for the status bar."""
    if preview.error is not None:
        return f"⚠ {preview.error}"
    parts = []
    evaluation = preview.evaluation
    if evaluation is not None:
        parts.append(f"FOM {evaluation.merit:.4g}")
        if evaluation.coating_time is not None:
            parts.append(f"time {evaluation.coating_time / 3600:.3g} h")
    if preview.results:
        mean = sum(float(r.reflectivity.mean()) for r in preview.results) / len(preview.results)
        parts.append(f"R̄ {mean:.3f}")
    if preview.stack is not None:
        parts.append(f"{preview.stack.num_layers} layers, {preview.stack.total_thickness:.4g} nm")
    if evaluation is not None and evaluation.constraints:
        parts.append("constraints ✓" if evaluation.feasible else "constraints ✗")
    return " · ".join(parts)


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
        self.reflectivity_plot = ReflectivityPlot()
        self.layer_plot = LayerProfilePlot()
        self.plot_tabs.addTab(self.reflectivity_plot, "Reflectivity")
        self.plot_tabs.addTab(self.layer_plot, "Layer profile")
        self.design_map = DesignMapPlot()
        self.convergence_plot = ConvergencePlot()
        self.tolerance_plot = HistogramPlot()
        self.plot_tabs.addTab(self.design_map, "Design map")
        self.plot_tabs.addTab(self.convergence_plot, "Convergence")
        self.plot_tabs.addTab(self.tolerance_plot, "Tolerance")
        self.setCentralWidget(self.plot_tabs)
        self.job_runner = JobRunner(parent=self)
        self.overlays: list[str] = []
        self.preview_runner = LatestRunner(parent=self)
        self.preview_runner.finished.connect(self._on_preview)
        self.preview_runner.failed.connect(self._show_error)
        self.preview: Preview | None = None

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
        self.candidate_list.itemChanged.connect(self._overlay_toggled)
        candidates = QWidget()
        candidate_layout = QVBoxLayout(candidates)
        candidate_layout.setContentsMargins(0, 0, 0, 0)
        candidate_layout.addWidget(self.candidate_list)
        buttons = QHBoxLayout()
        self.store_button = QPushButton("Store")
        self.store_button.setToolTip("Store the current design as candidate")
        self.use_button = QPushButton("Use")
        self.use_button.setToolTip("Make the selected candidate the current design")
        self.delete_button = QPushButton("Delete")
        for button in (self.store_button, self.use_button, self.delete_button):
            buttons.addWidget(button)
        candidate_layout.addLayout(buttons)
        hint = QLabel("Check a candidate to overlay it in the reflectivity plot.")
        hint.setWordWrap(True)
        candidate_layout.addWidget(hint)
        left = QSplitter(Qt.Orientation.Vertical)
        left.addWidget(_titled("Workflow", self.workflow_list))
        left.addWidget(_titled("Candidates", candidates))
        self.workflow_dock = self._dock("Project", left, Qt.DockWidgetArea.LeftDockWidgetArea)

        properties = QWidget()
        layout = QVBoxLayout(properties)
        self.design_label = QLabel()
        self.design_label.setWordWrap(True)
        layout.addWidget(self.design_label)
        self.goal_page = GoalPage()
        self.materials_page = MaterialsPage()
        self.coating_page = CoatingPage()
        self.constraints_page = ConstraintsPage()
        self.process_page = ProcessPage()
        self.scan_page = ScanPage()
        self.optimize_page = OptimizePage()
        self.analyze_page = AnalyzePage()
        self.pages: dict[str, QWidget] = {
            "Goal": self.goal_page,
            "Materials": self.materials_page,
            "Coating": self.coating_page,
            "Constraints": self.constraints_page,
            "Process": self.process_page,
            "Scan": self.scan_page,
            "Optimize": self.optimize_page,
            "Analyze": self.analyze_page,
        }
        self.page_stack = QStackedWidget()
        for step in WORKFLOW_STEPS:
            page = self.pages.get(step)
            if page is None:
                page = QLabel(f"{step}: not available yet")
                page.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self.pages[step] = page
            self.page_stack.addWidget(page)
        layout.addWidget(self.page_stack)
        self.parameter_table = self.coating_page.parameter_table
        self.workflow_list.currentRowChanged.connect(self.page_stack.setCurrentIndex)
        self.workflow_list.setCurrentRow(WORKFLOW_STEPS.index("Coating"))
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
        self.run_action = QAction("&Run pipeline", self)
        self.run_action.setShortcut(QKeySequence("Ctrl+R"))
        self.run_action.triggered.connect(lambda: self.optimize_page.run())
        self.stop_action = QAction("&Stop", self)
        self.stop_action.setShortcut(QKeySequence("Ctrl+."))
        self.stop_action.setEnabled(False)
        self.stop_action.triggered.connect(self.job_runner.cancel)

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
        run_menu = self.menuBar().addMenu("&Run")
        run_menu.addAction(self.run_action)
        run_menu.addAction(self.stop_action)
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
        toolbar.addSeparator()
        toolbar.addAction(self.run_action)
        toolbar.addAction(self.stop_action)
        self._update_recent_menu()

    def _connect(self) -> None:
        c = self.controller
        c.stateChanged.connect(self._on_state_changed)
        c.journalChanged.connect(self._on_journal_changed)
        c.fileChanged.connect(self._on_file_changed)
        c.errorOccurred.connect(self._show_error)
        c.undo_stack.cleanChanged.connect(self._on_clean_changed)
        for page in self.step_pages():
            page.commandRequested.connect(c.execute)
            page.errorOccurred.connect(self._show_error)
            page.jobRequested.connect(self.start_job)
        self.summary_label = QLabel()
        self.statusBar().addPermanentWidget(self.summary_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress_bar)
        runner = self.job_runner
        runner.started.connect(self._job_started)
        runner.progress.connect(self._job_progress)
        runner.finished.connect(self._job_finished)
        runner.failed.connect(self._job_done)
        self.scan_page.scanFinished.connect(self._show_scan)
        self.design_map.pointSelected.connect(self.scan_page.apply_values)
        self.optimize_page.pipelineFinished.connect(self._apply_pipeline_result)
        self.analyze_page.toleranceFinished.connect(self._show_tolerance)
        self.store_button.clicked.connect(self.store_candidate)
        self.use_button.clicked.connect(self._use_candidate)
        self.delete_button.clicked.connect(self._delete_candidate)

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
        for page in self.step_pages():
            page.set_state(state)
        names = [c.name for c in state.candidates]
        self.overlays = [n for n in self.overlays if n in names]
        self.candidate_list.blockSignals(True)
        self.candidate_list.clear()
        for name in names:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            checked = name in self.overlays
            item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
            self.candidate_list.addItem(item)
        self.candidate_list.blockSignals(False)
        self._update_title()
        self._request_preview()

    def _request_preview(self) -> None:
        state = self.controller.state
        self.preview_runner.request(functools.partial(compute_preview, state, tuple(self.overlays)))

    def step_pages(self) -> list[StepPage]:
        """Return the workflow pages that edit the project."""
        return [p for p in self.pages.values() if isinstance(p, StepPage)]

    def _on_preview(self, preview: Preview) -> None:
        if preview.state is not self.controller.state:
            return
        self.preview = preview
        targets = preview.state.targets.targets if preview.state.targets else ()
        curves = [
            Curve(t.name or f"target {i + 1}", r)
            for i, (t, r) in enumerate(zip(targets, preview.results, strict=False))
        ]
        for name, results in preview.overlays:
            curves += [Curve(name, r, dashed=True) for r in results]
        self.reflectivity_plot.set_curves(curves, targets)
        self.layer_plot.set_stack(preview.stack)
        for page in self.step_pages():
            page.set_preview(preview)
        self.summary_label.setText(_summary(preview))

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
        self.convergence_plot.clear()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(True)
        self.stop_action.setEnabled(True)
        self.run_action.setEnabled(False)
        self.statusBar().showMessage(f"{name} running…")

    def _job_progress(self, event: object) -> None:
        if not isinstance(event, ProgressEvent):
            return
        self.convergence_plot.add_point(event.task, event.evaluations, event.best_objective)
        if event.total:
            self.progress_bar.setRange(0, event.total)
            self.progress_bar.setValue(min(event.evaluations, event.total))
        self.statusBar().showMessage(
            f"{event.task}: {event.evaluations} evaluations, best {event.best_objective:.5g}"
        )

    def _on_file_changed(self, _path: object) -> None:
        self._update_title()

    def _on_clean_changed(self, _clean: bool) -> None:
        self._update_title()

    def _job_finished(self, name: str) -> None:
        self._job_done(f"{name} finished")

    def _job_done(self, message: str) -> None:
        self.progress_bar.setVisible(False)
        self.stop_action.setEnabled(False)
        self.run_action.setEnabled(True)
        self.statusBar().showMessage(message, 8000)

    def _show_scan(self, result: ScanResult) -> None:
        self.design_map.set_scan(
            result.names, result.axes, result.merit, result.feasible, "Figure of merit"
        )
        self.plot_tabs.setCurrentWidget(self.design_map)

    def _show_tolerance(self, result: ToleranceResult) -> None:
        self.tolerance_plot.set_samples(result.merits, result.nominal.merit, "Figure of merit")
        self.plot_tabs.setCurrentWidget(self.tolerance_plot)

    def _apply_pipeline_result(self, problem: DesignProblem, result: PipelineResult) -> None:
        record_run(self.controller.project.journal, result)
        runs = len(self.controller.project.journal.entries("run"))
        command = apply_result(problem, result.parameters, result.final, f"run {runs}", "pipeline")
        self.controller.execute(command)

    def store_candidate(self, name: str | None = None) -> None:
        """Store the current design as a candidate.

        Args:
            name: Candidate name; defaults to a numbered name.
        """
        state = self.controller.state
        if state.mirror is None:
            return
        name = name or f"design {len(state.candidates) + 1}"
        evaluation = (
            self.preview.evaluation if self.preview and self.preview.state is state else None
        )
        candidate = (
            Candidate.from_evaluation(name, state.mirror, evaluation, "manual")
            if evaluation is not None
            else Candidate(name, state.mirror, "manual")
        )
        self.controller.execute(add_candidate(candidate))

    def _selected_candidate(self) -> str | None:
        item = self.candidate_list.currentItem()
        return item.text() if item is not None else None

    def _use_candidate(self) -> None:
        name = self._selected_candidate()
        if name is not None:
            self.controller.execute(promote_candidate(name))

    def _delete_candidate(self) -> None:
        name = self._selected_candidate()
        if name is not None:
            self.controller.execute(remove_candidate(name))

    def _overlay_toggled(self, item: QListWidgetItem) -> None:
        name = item.text()
        if item.checkState() == Qt.CheckState.Checked and name not in self.overlays:
            self.overlays.append(name)
        elif item.checkState() != Qt.CheckState.Checked and name in self.overlays:
            self.overlays.remove(name)
        self._request_preview()

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
