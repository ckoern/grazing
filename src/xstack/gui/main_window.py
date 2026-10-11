"""Main window of the xstack application."""

import functools
from pathlib import Path

import numpy as np
from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QAction, QCloseEvent, QKeySequence
from PySide6.QtWidgets import (
    QCheckBox,
    QDockWidget,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
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
from xstack.gui.preview import Preview, compute_curves, compute_preview
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
from xstack.gui.style import HELP_STYLE
from xstack.gui.widgets import JournalView
from xstack.gui.workers import Job, JobRunner, LatestRunner
from xstack.manufacturing.tolerance import ToleranceResult
from xstack.optim.events import ProgressEvent
from xstack.optim.pipeline import PipelineResult
from xstack.optim.problem import DesignProblem, Evaluation
from xstack.optim.scan import ScanResult
from xstack.physics.models import ReflectivityResult
from xstack.project import Candidate, ProjectState
from xstack.project.commands import (
    add_candidate,
    promote_candidate,
    record_run,
    remove_candidate,
    rename_candidate,
    select_candidate,
    store_pipeline_result,
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


def _candidate_details(state: ProjectState) -> str:
    """Return a description of the selected candidate."""
    candidate = state.selected_candidate()
    if candidate is None:
        return "No candidate selected. The plots show the input design."
    parts = [f"<b>{candidate.name}</b> ({candidate.source})"]
    if candidate.merit is not None:
        parts.append(f"figure of merit {candidate.merit:.4g}")
    if candidate.objective is not None:
        parts.append(f"objective {candidate.objective:.4g}")
    if candidate.coating_time is not None:
        parts.append(f"coating time {candidate.coating_time / 3600:.3g} h")
    if candidate.created:
        parts.append(f"created {candidate.created.replace('T', ' ')[:19]}")
    return "<br>".join(parts)


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
        self.plot_tabs.addTab(self._reflectivity_tab(), "Reflectivity")
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
        self.live_runner = LatestRunner(delay_ms=300, parent=self)
        self.live_runner.finished.connect(self._on_live_curves)
        self.live_curves: tuple[ReflectivityResult, ...] = ()
        self._live_values: dict[str, float] | None = None
        self._job_state: ProjectState | None = None

    def _reflectivity_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        bar = QHBoxLayout()
        bar.addWidget(QLabel("Plot range:"))
        self.range_edits: dict[str, tuple[QLineEdit, QLineEdit]] = {}
        for axis, label, unit in (("energy", "energy", "keV"), ("angle", "angle", "deg")):
            low, high = QLineEdit(), QLineEdit()
            for edit, end in ((low, "min"), (high, "max")):
                edit.setPlaceholderText(f"{end} {label} ({unit}), auto")
                edit.setToolTip(
                    f"Range of the {label} axis in the reflectivity plot. Leave empty to show "
                    "the target range. The figure of merit always uses the target range."
                )
                edit.setMaximumWidth(170)
                edit.editingFinished.connect(self._range_changed)
                bar.addWidget(edit)
            self.range_edits[axis] = (low, high)
        self.log_check = QCheckBox("log scale")
        self.log_check.setToolTip("Show the reflectivity on a logarithmic axis")
        self.log_check.toggled.connect(self._log_toggled)
        bar.addWidget(self.log_check)
        bar.addStretch()
        layout.addLayout(bar)
        layout.addWidget(self.reflectivity_plot)
        self.display_ranges: dict[str, tuple[float, float]] = {}
        return tab

    def _range_changed(self) -> None:
        ranges: dict[str, tuple[float, float]] = {}
        for axis, (low, high) in self.range_edits.items():
            if not low.text().strip() and not high.text().strip():
                continue
            scale = 1e3 if axis == "energy" else np.pi / 180.0
            try:
                lo, hi = float(low.text()) * scale, float(high.text()) * scale
            except ValueError:
                self._show_error(f"Plot range of the {axis} axis needs a min and a max value")
                continue
            if not 0 < lo < hi or (axis == "angle" and hi > np.pi / 2):
                self._show_error(f"Invalid plot range of the {axis} axis")
                continue
            ranges[axis] = (lo, hi)
        if ranges != self.display_ranges:
            self.display_ranges = ranges
            self._request_preview()

    def _log_toggled(self, checked: bool) -> None:
        self.reflectivity_plot.log_scale = checked
        self._redraw()

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
        self.candidate_list.setToolTip(
            "Stored designs. Optimization runs are stored as “run N” (final design) and "
            "“run N · stage” (design after each stage). Click to select a candidate: it is "
            "shown next to the input design and can be analyzed or exported. Double-click to "
            "rename. Check to overlay it in the reflectivity plot."
        )
        self.candidate_list.itemChanged.connect(self._candidate_item_changed)
        self.candidate_list.currentItemChanged.connect(self._candidate_clicked)
        candidates = QWidget()
        candidate_layout = QVBoxLayout(candidates)
        candidate_layout.setContentsMargins(0, 0, 0, 0)
        candidate_layout.addWidget(self.candidate_list)
        buttons = QHBoxLayout()
        self.store_button = QPushButton("Store input…")
        self.store_button.setToolTip("Store the input design (Coating step) as a named candidate")
        self.use_button = QPushButton("Use as start")
        self.use_button.setToolTip(
            "Copy the selected candidate into the input design, e.g. to continue optimizing from it"
        )
        self.delete_button = QPushButton("Delete")
        self.delete_button.setToolTip("Delete the selected candidate")
        for button in (self.store_button, self.use_button, self.delete_button):
            buttons.addWidget(button)
        candidate_layout.addLayout(buttons)
        self.candidate_details = QLabel("No candidate selected.")
        self.candidate_details.setWordWrap(True)
        candidate_layout.addWidget(self.candidate_details)
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
        for index, step in enumerate(WORKFLOW_STEPS):
            self.page_stack.addWidget(self._step_container(index, self.pages[step]))
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
            [self.workflow_dock, self.properties_dock], [220, 520], Qt.Orientation.Horizontal
        )
        self.resizeDocks([self.journal_dock], [160], Qt.Orientation.Vertical)

    def _step_container(self, index: int, page: QWidget) -> QWidget:
        """Wrap a workflow page with its help text and Back/Next buttons."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        help_text = getattr(page, "help_text", "")
        if help_text:
            help_label = QLabel(help_text)
            help_label.setWordWrap(True)
            help_label.setTextFormat(Qt.TextFormat.RichText)
            help_label.setStyleSheet(HELP_STYLE)
            help_label.setObjectName(f"help_{WORKFLOW_STEPS[index].lower()}")
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        if help_text:
            content_layout.addWidget(help_label)
        content_layout.addWidget(page)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(content)
        layout.addWidget(scroll, 1)
        navigation = QHBoxLayout()
        navigation.addStretch()
        back = QPushButton("← Back")
        back.setEnabled(index > 0)
        back.clicked.connect(self.previous_step)
        navigation.addWidget(back)
        if index < len(WORKFLOW_STEPS) - 1:
            forward = QPushButton(f"Next: {WORKFLOW_STEPS[index + 1]} →")
            forward.setObjectName(f"next_{WORKFLOW_STEPS[index].lower()}")
            forward.clicked.connect(self.next_step)
            navigation.addWidget(forward)
        layout.addLayout(navigation)
        return container

    def next_step(self) -> None:
        """Show the next workflow step."""
        row = self.workflow_list.currentRow()
        self.workflow_list.setCurrentRow(min(row + 1, len(WORKFLOW_STEPS) - 1))

    def previous_step(self) -> None:
        """Show the previous workflow step."""
        self.workflow_list.setCurrentRow(max(self.workflow_list.currentRow() - 1, 0))

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
        self.store_button.clicked.connect(lambda: self.store_candidate())
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
            item.setData(Qt.ItemDataRole.UserRole, name)
            item.setFlags(
                item.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEditable
            )
            checked = name in self.overlays
            item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
            self.candidate_list.addItem(item)
            if name == state.selected:
                self.candidate_list.setCurrentItem(item)
        self.candidate_list.blockSignals(False)
        self.candidate_details.setText(_candidate_details(state))
        self._update_title()
        self._request_preview()

    def _request_preview(self) -> None:
        state = self.controller.state
        shown = [state.selected] if state.selected else []
        shown += [n for n in self.overlays if n not in shown]
        self.preview_runner.request(
            functools.partial(compute_preview, state, tuple(shown), dict(self.display_ranges))
        )

    def step_pages(self) -> list[StepPage]:
        """Return the workflow pages that edit the project."""
        return [p for p in self.pages.values() if isinstance(p, StepPage)]

    def _on_preview(self, preview: Preview) -> None:
        if preview.state is not self.controller.state:
            return
        self.preview = preview
        self._redraw()
        self.layer_plot.set_stack(preview.stack)
        for page in self.step_pages():
            page.set_preview(preview)
        self.summary_label.setText(_summary(preview))

    def _redraw(self) -> None:
        """Draw the input design, the shown candidates and the live best design."""
        preview = self.preview
        if preview is None:
            return
        targets = preview.state.targets.targets if preview.state.targets else ()
        multiple = len(targets) > 1

        def label(name: str, i: int) -> str:
            target = targets[i] if i < len(targets) else None
            suffix = f" – {target.name or f'target {i + 1}'}" if multiple and target else ""
            return name + suffix

        curves = [
            Curve(label("input design", i), r, group="input") for i, r in enumerate(preview.curves)
        ]
        for name, results in preview.overlays:
            curves += [
                Curve(label(name, i), r, dashed=True, group=name) for i, r in enumerate(results)
            ]
        curves += [
            Curve(label("best so far", i), r, group="live") for i, r in enumerate(self.live_curves)
        ]
        self.reflectivity_plot.set_curves(curves, targets)

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
        self._job_state = self.controller.state
        self._live_values = None
        self.live_curves = ()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(True)
        self.stop_action.setEnabled(True)
        self.run_action.setEnabled(False)
        self.statusBar().showMessage(f"{name} running…")

    def _job_progress(self, event: object) -> None:
        if not isinstance(event, ProgressEvent):
            return
        self.convergence_plot.add_point(
            event.task, event.evaluations, event.best_objective, event.best_merit
        )
        state = self._job_state
        values = dict(event.best_values)
        if state is not None and state.mirror is not None and values != self._live_values:
            self._live_values = values
            self.live_runner.request(
                functools.partial(
                    compute_curves, state, state.mirror, values, dict(self.display_ranges)
                )
            )
        if event.total:
            self.progress_bar.setRange(0, event.total)
            self.progress_bar.setValue(min(event.evaluations, event.total))
        self.statusBar().showMessage(
            f"{event.task}: {event.evaluations} evaluations, best objective "
            f"{event.best_objective:.5g}, figure of merit {event.best_merit:.5g}"
        )

    def _on_live_curves(self, curves: tuple[ReflectivityResult, ...]) -> None:
        if self.job_runner.running:
            self.live_curves = curves
            self._redraw()

    def _on_file_changed(self, _path: object) -> None:
        self._update_title()

    def _on_clean_changed(self, _clean: bool) -> None:
        self._update_title()

    def _job_finished(self, name: str) -> None:
        self._job_done(f"{name} finished")

    def _job_done(self, message: str) -> None:
        self.live_curves = ()
        self._job_state = None
        self._redraw()
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

    def _apply_pipeline_result(
        self,
        problem: DesignProblem,
        result: PipelineResult,
        stage_evaluations: tuple[Evaluation, ...] | None = None,
    ) -> None:
        record_run(self.controller.project.journal, result)
        runs = len(self.controller.project.journal.entries("run"))
        self.controller.execute(
            store_pipeline_result(problem, result, f"run {runs}", stage_evaluations)
        )

    def store_candidate(self, name: str | None = None) -> None:
        """Store the input design as a candidate.

        Args:
            name: Candidate name; asked for in a dialog if not given.
        """
        state = self.controller.state
        if state.mirror is None:
            return
        if name is None:
            proposal = f"input {len(state.candidates) + 1}"
            text, accepted = QInputDialog.getText(
                self, "Store input design", "Name of the candidate:", text=proposal
            )
            if not accepted or not text.strip():
                return
            name = text.strip()
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
        return item.data(Qt.ItemDataRole.UserRole) if item is not None else None

    def _use_candidate(self) -> None:
        name = self._selected_candidate()
        if name is not None:
            self.controller.execute(promote_candidate(name))

    def _delete_candidate(self) -> None:
        name = self._selected_candidate()
        if name is not None:
            self.controller.execute(remove_candidate(name))

    def _candidate_clicked(self, current: QListWidgetItem | None, _previous: object) -> None:
        name = current.data(Qt.ItemDataRole.UserRole) if current is not None else None
        if name != self.controller.state.selected:
            self.controller.execute(select_candidate(name))

    def _candidate_item_changed(self, item: QListWidgetItem) -> None:
        old = item.data(Qt.ItemDataRole.UserRole)
        if item.text() != old:
            if not self.controller.execute(rename_candidate(old, item.text().strip())):
                self.candidate_list.blockSignals(True)
                item.setText(old)
                self.candidate_list.blockSignals(False)
            return
        self._overlay_toggled(item)

    def _overlay_toggled(self, item: QListWidgetItem) -> None:
        name = item.data(Qt.ItemDataRole.UserRole)
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
