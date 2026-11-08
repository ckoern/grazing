"""Main window of the xstack application."""

import functools
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PySide6.QtCore import QItemSelectionModel, QPoint, QSettings, Qt
from PySide6.QtGui import QAction, QActionGroup, QKeySequence, QShowEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QToolButton,
    QUndoView,
    QVBoxLayout,
    QWidget,
)

from xstack.core.io.reflectivity import write_reflectivity_csv
from xstack.core.optim.events import ProgressEvent
from xstack.core.physics.models import ReflectivityResult
from xstack.core.structure.coating import CoatingBuildError
from xstack.design.manufacturing.tolerance import ToleranceResult
from xstack.design.optim.pipeline import PipelineResult
from xstack.design.optim.problem import DesignProblem
from xstack.design.optim.scan import ScanResult
from xstack.design.optim.tradeoff import TradeoffPoint
from xstack.design.project import Candidate, ProjectState
from xstack.design.project.commands import (
    add_candidate,
    promote_candidate,
    record_run,
    remove_candidates,
    rename_candidate,
    select_candidate,
    set_display_ranges,
    store_pipeline_result,
)
from xstack.design.project.templates import TEMPLATES
from xstack.gui.common.document_window import DocumentWindow
from xstack.gui.common.files import ask_save_path
from xstack.gui.common.plot_tabs import PlotTabs
from xstack.gui.common.style import (
    COLORS,
    set_variant,
)
from xstack.gui.common.wizard import ProjectWizard
from xstack.gui.common.workers import JobRunner, LatestRunner
from xstack.gui.design.controller import ProjectController
from xstack.gui.design.materials_summary import coating_title, materials_text
from xstack.gui.design.plots import (
    ConvergencePlot,
    Curve,
    DesignMapPlot,
    LayerProfilePlot,
    ReferenceDesign,
    ReflectivityPlot,
    TradeoffPlot,
)
from xstack.gui.design.preview import (
    Preview,
    ToleranceCurves,
    compute_curves,
    compute_preview,
    display_targets,
    lowest_energy,
    tolerance_curves,
)
from xstack.gui.design.steps import (
    AnalyzePage,
    CoatingPage,
    ConstraintsPage,
    EstimatePage,
    GoalPage,
    MaterialsPage,
    OptimizePage,
    ProcessPage,
    ScanPage,
    StepPage,
    TradeoffPage,
)
from xstack.gui.design.steps.tradeoff import limit_label
from xstack.gui.design.widgets import JournalView
from xstack.gui.design.widgets.candidate_dialog import CandidateParametersDialog
from xstack.gui.design.widgets.design_comparison import DesignComparisonView
from xstack.gui.design.widgets.tolerance_view import ToleranceView

__all__ = ["WORKFLOW_STEPS", "MainWindow"]

WORKFLOW_STEPS = (
    "Goal",
    "Materials",
    "Coating",
    "Estimate",
    "Constraints",
    "Process",
    "Scan",
    "Optimize",
    "Trade-off",
    "Analyze",
)
"""Steps of the design workflow, shown as tabs in this order."""

PROJECT_FILTER = "xstack projects (*.xstack);;All files (*)"

INPUT_DESIGN_LABEL = "input design"
"""Label of the input design in the candidates list."""


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


class MainWindow(DocumentWindow):
    """Main window with workflow, plots, properties, journal and history.

    Args:
        controller: The project controller. A new one is created if not given.
        settings: Settings store for window geometry and recent files.
        confirm_close: Ask whether to save unsaved changes when closing or replacing the
            project.
    """

    project_kind = "project"
    file_filter = PROJECT_FILTER
    controller: ProjectController
    file_suffix = ".xstack"
    app_name = "xstack"

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
        self.wizard_factory: Callable[[], ProjectWizard] = lambda: ProjectWizard(
            "design project", ".xstack", PROJECT_FILTER, parent=self
        )
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
        self.plot_tabs = PlotTabs("xstack-design")
        self.reflectivity_plot = ReflectivityPlot()
        self.layer_plot = LayerProfilePlot()
        self.comparison_view = DesignComparisonView()
        self.reflectivity_view = self._reflectivity_tab()
        self.layer_view = self._layer_tab()
        self.plot_tabs.addTab(self.reflectivity_view, "Reflectivity")
        self.plot_tabs.addTab(self.layer_view, "Layer profile")
        self.plot_tabs.addTab(self.comparison_view, "Parameters")
        self.design_map = DesignMapPlot()
        self.convergence_plot = ConvergencePlot()
        self.tolerance_view = ToleranceView()
        self.plot_tabs.addTab(self.design_map, "Design map")
        self.plot_tabs.addTab(self.convergence_plot, "Convergence")
        self.plot_tabs.addTab(self.tolerance_view, "Tolerance")
        self.tradeoff_plot = TradeoffPlot()
        self.plot_tabs.addTab(self.tradeoff_plot, "Trade-off")
        self.plot_tabs.setTabToolTip(
            2, "Coating parameters of the selected candidate compared with the input design"
        )
        self.step_views: dict[str, QWidget] = {
            "Goal": self.reflectivity_view,
            "Materials": self.reflectivity_view,
            "Coating": self.layer_view,
            "Estimate": self.reflectivity_view,
            "Constraints": self.layer_view,
            "Process": self.layer_view,
            "Scan": self.design_map,
            "Optimize": self.reflectivity_view,
            "Trade-off": self.tradeoff_plot,
            "Analyze": self.tolerance_view,
        }
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._step_panel())
        plot_area = QWidget()
        plot_layout = QVBoxLayout(plot_area)
        plot_layout.setContentsMargins(4, 8, 8, 4)
        plot_layout.addLayout(self._range_bar())
        plot_layout.addWidget(self.plot_tabs, 1)
        splitter.addWidget(plot_area)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([640, 860])
        self.setCentralWidget(splitter)
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
        self.tolerance_result: ToleranceResult | None = None
        self.tolerance_runner = LatestRunner(delay_ms=200, parent=self)
        self.tolerance_runner.finished.connect(self._on_tolerance_curves)
        self.tolerance_runner.failed.connect(self._show_error)
        self.tradeoff_runner = LatestRunner(delay_ms=100, parent=self)
        self.tradeoff_runner.finished.connect(self._on_tradeoff_curves)
        self.tradeoff_runner.failed.connect(self._show_error)
        self.tradeoff_point: TradeoffPoint | None = None
        self.tradeoff_curves: tuple[ReflectivityResult, ...] = ()

    def _step_panel(self) -> QWidget:
        """Create the workflow panel: design summary and one tab per workflow step."""
        panel = QWidget()
        panel.setObjectName("step_panel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 4, 8)
        self.name_edit = QLineEdit()
        self.name_edit.setObjectName("project_name")
        self.name_edit.setToolTip("Project name (press Enter to apply)")
        self.name_edit.setStyleSheet("font-size: 13pt; font-weight: 600;")
        self.name_edit.editingFinished.connect(self._commit_project_name)
        layout.addWidget(self.name_edit)
        self.design_label = QLabel()
        self.design_label.setObjectName("design_header")
        self.design_label.setWordWrap(True)
        layout.addWidget(self.design_label)
        self.goal_page = GoalPage()
        self.materials_page = MaterialsPage()
        self.coating_page = CoatingPage()
        self.estimate_page = EstimatePage()
        self.constraints_page = ConstraintsPage()
        self.process_page = ProcessPage()
        self.scan_page = ScanPage()
        self.optimize_page = OptimizePage()
        self.tradeoff_page = TradeoffPage()
        self.tradeoff_page.pipeline_provider = self.optimize_page.pipeline
        self.analyze_page = AnalyzePage()
        self.pages: dict[str, QWidget] = {
            "Goal": self.goal_page,
            "Materials": self.materials_page,
            "Coating": self.coating_page,
            "Estimate": self.estimate_page,
            "Constraints": self.constraints_page,
            "Process": self.process_page,
            "Scan": self.scan_page,
            "Optimize": self.optimize_page,
            "Trade-off": self.tradeoff_page,
            "Analyze": self.analyze_page,
        }
        self.step_tabs = QTabWidget()
        self.step_tabs.setObjectName("step_tabs")
        self.step_tabs.setDocumentMode(True)
        self.step_tabs.setUsesScrollButtons(True)
        for index, step in enumerate(WORKFLOW_STEPS):
            container = self._step_container(index, self.pages[step])
            self.step_tabs.addTab(container, f"{index + 1} {step}")
            self.step_tabs.setTabToolTip(index, f"Step {index + 1}: {step}")
        self.step_tabs.setCurrentIndex(WORKFLOW_STEPS.index("Coating"))
        self.step_tabs.currentChanged.connect(lambda _: self.show_step_view())
        layout.addWidget(self.step_tabs, 1)
        self._step_panel_widget = panel
        self.parameter_table = self.coating_page.parameter_table
        for button in (
            self.estimate_page.estimate_button,
            self.scan_page.run_button,
            self.optimize_page.run_button,
            self.tradeoff_page.run_button,
            self.analyze_page.run_button,
        ):
            set_variant(button, "primary")
        return panel

    def showEvent(self, event: QShowEvent) -> None:
        """Size the workflow panel so that all step tabs are visible once styles apply."""
        super().showEvent(event)
        panel = self._step_panel_widget
        margins = panel.layout().contentsMargins()
        width = self.step_tabs.tabBar().sizeHint().width() + margins.left() + margins.right()
        panel.setMinimumWidth(width + 8)

    def _layer_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        bar = QHBoxLayout()
        bar.addWidget(QLabel("Show:"))
        self.layer_source = QComboBox()
        self.layer_source.addItem("Input design", "input")
        self.layer_source.addItem("Selected candidate", "selected")
        self.layer_source.setToolTip(
            "Show the layers of the input design or of the selected candidate"
        )
        self.layer_source.activated.connect(self._show_layers)
        bar.addWidget(self.layer_source)
        bar.addStretch()
        layout.addLayout(bar)
        layout.addWidget(self.layer_plot)
        return tab

    def _show_layers(self, _index: int = 0) -> None:
        state = self.controller.state
        stack = self.preview.stack if self.preview is not None else None
        if self.layer_source.currentData() == "selected":
            candidate = state.selected_candidate()
            stack = None
            if candidate is not None:
                try:
                    stack = candidate.mirror.stack()
                except ValueError:
                    stack = None
        self.layer_plot.set_stack(stack)

    def _reflectivity_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        bar = QHBoxLayout()
        self.log_check = QCheckBox("log scale")
        self.log_check.setToolTip("Show the reflectivity on a logarithmic axis")
        self.log_check.toggled.connect(self._log_toggled)
        bar.addWidget(self.log_check)
        bar.addStretch()
        layout.addLayout(bar)
        layout.addWidget(self.reflectivity_plot)
        return tab

    def _range_bar(self) -> QHBoxLayout:
        """Create the plot-range fields shared by the reflectivity and tolerance plots."""
        bar = QHBoxLayout()
        bar.addWidget(QLabel("Plot range:"))
        self.range_edits: dict[str, tuple[QLineEdit, QLineEdit]] = {}
        for axis, label, unit in (("energy", "energy", "keV"), ("angle", "angle", "deg")):
            low, high = QLineEdit(), QLineEdit()
            for edit, end in ((low, "min"), (high, "max")):
                edit.setPlaceholderText(f"{end} {label} ({unit})")
                edit.setToolTip(
                    f"Range of the {label} axis in the Reflectivity and Tolerance plots. Leave "
                    "both fields empty to show the target range. A minimum of 0 starts at the "
                    "first grid point for which the reflectivity can be calculated. The figure "
                    "of merit always uses the target range."
                )
                edit.setMaximumWidth(150)
                edit.editingFinished.connect(self._range_changed)
                bar.addWidget(edit)
            self.range_edits[axis] = (low, high)
        bar.addStretch()
        self.display_ranges: dict[str, tuple[float, float]] = {}
        return bar

    def _range_changed(self) -> None:
        ranges: dict[str, tuple[float, float]] = {}
        for axis, (low, high) in self.range_edits.items():
            if not low.text().strip() and not high.text().strip():
                continue
            scale = 1e3 if axis == "energy" else np.pi / 180.0
            previous = self.display_ranges.get(axis)
            try:
                lo, hi = float(low.text()) * scale, float(high.text()) * scale
            except ValueError:
                self._show_error(f"Plot range of the {axis} axis needs a min and a max value")
                lo, hi = -1.0, -1.0
            else:
                if not 0 <= lo < hi or (axis == "angle" and hi > np.pi / 2):
                    self._show_error(f"Invalid plot range of the {axis} axis")
                    lo, hi = -1.0, -1.0
            if lo >= 0:
                ranges[axis] = (lo, hi)
            elif previous is not None:
                # An invalid entry keeps the last valid range of this axis.
                ranges[axis] = previous
        if ranges != self.controller.state.display_ranges:
            self.controller.execute(set_display_ranges(ranges))

    def _sync_display_ranges(self, ranges: dict[str, tuple[float, float]]) -> bool:
        """Show the plot ranges of the project state in the fields and the plots.

        Args:
            ranges: Plot range by scan axis in eV or rad.

        Returns:
            Whether the ranges differed from the displayed ones.
        """
        if ranges == self.display_ranges:
            return False
        self.display_ranges = dict(ranges)
        self.analyze_page.display_ranges = dict(ranges)
        for axis, (low, high) in self.range_edits.items():
            scale = 1e3 if axis == "energy" else np.pi / 180.0
            lo, hi = ranges.get(axis, (None, None))
            low.setText("" if lo is None else f"{lo / scale:g}")
            high.setText("" if hi is None else f"{hi / scale:g}")
        self._request_tolerance_curves()
        return True

    def _log_toggled(self, checked: bool) -> None:
        self.reflectivity_plot.log_scale = checked
        self._redraw()

    def _build_docks(self) -> None:
        self.candidate_list = QListWidget()
        self.candidate_list.setToolTip(
            "Stored designs. An optimization run is stored as “run N”: the final design of a "
            "full pipeline run, or the result of a single stage run. Click to select a candidate: "
            "it is shown next to the input design and can be analyzed or exported. Shift-click "
            "selects a range, Ctrl-click toggles single entries. Double-click to rename. Check "
            "to overlay it in the reflectivity plot."
        )
        self.candidate_list.itemChanged.connect(self._candidate_item_changed)
        self.candidate_list.currentItemChanged.connect(self._candidate_clicked)
        self.candidate_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.candidate_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.candidate_list.customContextMenuRequested.connect(self._candidate_menu)
        candidates = QWidget()
        candidate_layout = QVBoxLayout(candidates)
        candidate_layout.setContentsMargins(0, 0, 0, 0)
        candidate_layout.addWidget(self.candidate_list)
        buttons = QGridLayout()
        self.store_button = QPushButton("Store input…")
        self.store_button.setToolTip("Store the input design (Coating step) as a named candidate")
        self.use_button = QPushButton("Use as start")
        self.use_button.setToolTip(
            "Copy the selected candidate into the input design, e.g. to continue optimizing from it"
        )
        self.delete_button = QPushButton("Delete")
        self.delete_button.setToolTip(
            "Delete the selected candidates (Ctrl- or Shift-click to select several)"
        )
        buttons.addWidget(self.store_button, 0, 0, 1, 2)
        buttons.addWidget(self.use_button, 1, 0)
        buttons.addWidget(self.delete_button, 1, 1)
        candidate_layout.addLayout(buttons)
        self.candidate_details = QLabel("No candidate selected.")
        self.candidate_details.setWordWrap(True)
        self.candidate_details.setContentsMargins(6, 4, 6, 6)
        candidate_layout.addWidget(self.candidate_details)
        self.candidates_dock = self._dock(
            "Candidates", candidates, Qt.DockWidgetArea.LeftDockWidgetArea
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
        self.resizeDocks([self.candidates_dock], [200], Qt.Orientation.Horizontal)
        self.resizeDocks([self.journal_dock], [130], Qt.Orientation.Vertical)

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
            help_label.setProperty("role", "help")
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
            set_variant(forward, "primary")
            forward.clicked.connect(self.next_step)
            navigation.addWidget(forward)
        layout.addLayout(navigation)
        return container

    def next_step(self) -> None:
        """Show the next workflow step."""
        self.step_tabs.setCurrentIndex(
            min(self.step_tabs.currentIndex() + 1, len(WORKFLOW_STEPS) - 1)
        )

    def previous_step(self) -> None:
        """Show the previous workflow step."""
        self.step_tabs.setCurrentIndex(max(self.step_tabs.currentIndex() - 1, 0))

    def _build_actions(self) -> None:
        self.new_actions: dict[str, QAction] = {}
        for name in TEMPLATES:
            action = QAction(name.replace("_", " ").capitalize(), self)
            action.triggered.connect(lambda _=False, n=name: self.new_project(n))
            self.new_actions[name] = action
        self.new_action = QAction("&New…", self)
        self.new_action.setShortcut(QKeySequence.StandardKey.New)
        self.new_action.setToolTip("New project: name, location and coating chosen in a wizard")
        self.new_action.triggered.connect(lambda: self.new_project_wizard())
        self.open_action = QAction("&Open…", self)
        self.open_action.setShortcut(QKeySequence.StandardKey.Open)
        self.open_action.triggered.connect(lambda: self.open_project())
        self.save_action = QAction("&Save", self)
        self.save_action.setShortcut(QKeySequence.StandardKey.Save)
        self.save_action.triggered.connect(lambda: self.save_project())
        self.save_as_action = QAction("Save &as…", self)
        self.save_as_action.setShortcut(QKeySequence.StandardKey.SaveAs)
        self.save_as_action.triggered.connect(lambda: self.save_project_as())
        self.rename_action = QAction("&Rename project…", self)
        self.rename_action.triggered.connect(lambda: self.rename_project())
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
        self.auto_views_action = QAction("Switch plots &automatically", self, checkable=True)
        self.auto_views_action.setToolTip(
            "Show the plot that belongs to the current step, and the convergence during runs"
        )
        self.auto_views_action.setChecked(
            str(self.settings.value("plots/auto_switch", "true")).lower() == "true"
        )
        self.auto_views_action.toggled.connect(self._auto_views_toggled)
        self.detach_action = QAction("&Detach current plot", self)
        self.detach_action.setShortcut(QKeySequence("Ctrl+Shift+D"))
        self.detach_action.triggered.connect(self.plot_tabs.detach_current)
        self.stop_action = QAction("&Stop", self)
        self.stop_action.setShortcut(QKeySequence("Ctrl+."))
        self.stop_action.setEnabled(False)
        self.stop_action.triggered.connect(self.job_runner.cancel)

    def _build_menus(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction(self.new_action)
        new_menu = file_menu.addMenu("New from &template")
        for action in self.new_actions.values():
            new_menu.addAction(action)
        file_menu.addAction(self.open_action)
        self.recent_menu = file_menu.addMenu("Open &recent")
        file_menu.addSeparator()
        file_menu.addAction(self.save_action)
        file_menu.addAction(self.save_as_action)
        file_menu.addAction(self.rename_action)
        file_menu.addSeparator()
        file_menu.addAction(self.quit_action)
        edit_menu = self.menuBar().addMenu("&Edit")
        edit_menu.addAction(self.undo_action)
        edit_menu.addAction(self.redo_action)
        run_menu = self.menuBar().addMenu("&Run")
        run_menu.addAction(self.run_action)
        run_menu.addAction(self.stop_action)
        view_menu = self.menuBar().addMenu("&View")
        theme_menu = view_menu.addMenu("&Theme")
        self.theme_group = QActionGroup(self)
        self.theme_actions: dict[str, QAction] = {}
        saved = str(self.settings.value("theme", "system"))
        for theme, label in (("light", "&Light"), ("dark", "&Dark"), ("system", "&System")):
            action = QAction(label, self, checkable=True)
            action.setChecked(theme == saved)
            action.triggered.connect(lambda _=False, t=theme: self.set_theme(t))
            self.theme_group.addAction(action)
            theme_menu.addAction(action)
            self.theme_actions[theme] = action
        view_menu.addSeparator()
        view_menu.addAction(self.auto_views_action)
        view_menu.addAction(self.detach_action)
        view_menu.addSeparator()
        for dock in (
            self.candidates_dock,
            self.journal_dock,
            self.history_dock,
        ):
            view_menu.addAction(dock.toggleViewAction())
        toolbar = self.addToolBar("Main")
        toolbar.setObjectName("toolbar_main")
        toolbar.setMovable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._apply_icons()
        toolbar.addAction(self.new_action)
        toolbar.addAction(self.open_action)
        toolbar.addAction(self.save_action)
        toolbar.addSeparator()
        toolbar.addAction(self.undo_action)
        toolbar.addAction(self.redo_action)
        toolbar.addSeparator()
        toolbar.addAction(self.run_action)
        toolbar.addAction(self.stop_action)
        for action in (self.undo_action, self.redo_action):
            button = toolbar.widgetForAction(action)
            if isinstance(button, QToolButton):
                button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        run_button = toolbar.widgetForAction(self.run_action)
        if run_button is not None:
            run_button.setObjectName("primary_tool")
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
        self.tradeoff_page.quantityChanged.connect(self._update_tradeoff_plot)
        self.tradeoff_page.pointSelected.connect(self._tradeoff_point_selected)
        self.tradeoff_page.tradeoffFinished.connect(self._show_tradeoff)
        self.tradeoff_plot.pointSelected.connect(self.tradeoff_page.select_point)
        self.store_button.clicked.connect(lambda: self.store_candidate())
        self.comparison_view.commandRequested.connect(c.execute)
        self.use_button.clicked.connect(self._use_candidate)
        self.delete_button.clicked.connect(self._delete_candidate)

    def _on_state_changed(self, state: ProjectState) -> None:
        if self.name_edit.text() != state.name:
            self.name_edit.setText(state.name)
        if state.mirror is None:
            self.design_label.setText("No design defined.")
            self.parameter_table.setEnabled(False)
        else:
            coating = state.mirror.coating
            materials = materials_text(coating)
            self.design_label.setText(
                f"<span style='color:{COLORS['text_muted']}'>Input design: "
                f"<b>{coating_title(coating)}</b> · {materials} · substrate "
                f"{state.mirror.substrate.name}</span>"
            )
            self.parameter_table.setEnabled(True)
        for page in self.step_pages():
            page.set_state(state)
        self.comparison_view.set_state(state)
        names = [c.name for c in state.candidates]
        self.overlays = [n for n in self.overlays if n in names]
        self._sync_candidate_list(state, names)
        self.candidate_details.setText(_candidate_details(state))
        self._show_layers()
        self._update_title()
        self._sync_display_ranges(state.display_ranges)
        self._request_preview()
        self._update_tradeoff_plot()

    def _sync_candidate_list(self, state: ProjectState, names: list[str]) -> None:
        """Bring the candidate list in line with the state, keeping scroll position and selection.

        The entries are only rebuilt if the candidate names changed; otherwise the check states
        and the current entry are updated in place, which leaves a Shift/Ctrl multi-selection
        intact.
        """
        view = self.candidate_list
        view.blockSignals(True)
        try:
            existing = [view.item(r).data(Qt.ItemDataRole.UserRole) for r in range(view.count())]
            if existing != [None, *names]:
                self._rebuild_candidate_list(state, names)
                return
            for row, name in enumerate(names, start=1):
                item = view.item(row)
                checked = name in self.overlays
                check = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
                if item.checkState() != check:
                    item.setCheckState(check)
                if item.text() != name:
                    item.setText(name)
            current = view.currentItem()
            current_name = current.data(Qt.ItemDataRole.UserRole) if current is not None else None
            if current is None or current_name != state.selected:
                row = 0 if state.selected is None else 1 + names.index(state.selected)
                view.setCurrentRow(row)
        finally:
            view.blockSignals(False)

    def _rebuild_candidate_list(self, state: ProjectState, names: list[str]) -> None:
        view = self.candidate_list
        scroll = view.verticalScrollBar().value()
        kept = {i.data(Qt.ItemDataRole.UserRole) for i in view.selectedItems()}
        view.clear()
        view.addItem(self._input_item())
        for name in names:
            item = QListWidgetItem(name)
            item.setData(Qt.ItemDataRole.UserRole, name)
            item.setFlags(
                item.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEditable
            )
            checked = name in self.overlays
            item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
            view.addItem(item)
        row = 0 if state.selected is None else 1 + names.index(state.selected)
        if state.selected in kept and len(kept) > 1:
            for r in range(view.count()):
                if view.item(r).data(Qt.ItemDataRole.UserRole) in kept:
                    view.item(r).setSelected(True)
            view.selectionModel().setCurrentIndex(
                view.model().index(row, 0), QItemSelectionModel.SelectionFlag.NoUpdate
            )
        else:
            view.setCurrentRow(row)
        view.doItemsLayout()
        view.verticalScrollBar().setValue(scroll)

    def _input_item(self) -> QListWidgetItem:
        """Return the list entry of the input design (no candidate name)."""
        item = QListWidgetItem(INPUT_DESIGN_LABEL)
        item.setData(Qt.ItemDataRole.UserRole, None)
        item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
        item.setToolTip(
            "The input design defined in the Coating step; always shown in the plots. "
            "Selecting it deselects the candidates."
        )
        font = item.font()
        font.setItalic(True)
        item.setFont(font)
        return item

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
        self._show_layers()
        for page in self.step_pages():
            page.set_preview(preview)
        self._update_tradeoff_plot()
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
            suffix = f" – {target.label(i)}" if multiple and target else ""
            return name + suffix

        curves = [Curve(label("input design", i), r) for i, r in enumerate(preview.curves)]
        for name, results in preview.overlays:
            curves += [Curve(label(name, i), r, dashed=True) for i, r in enumerate(results)]
        curves += [Curve(label("best so far", i), r) for i, r in enumerate(self.live_curves)]
        if self.tradeoff_point is not None:
            name = f"trade-off {limit_label(self.tradeoff_point.limit, unit=True)}"
            curves += [
                Curve(label(name, i), r, dashed=True) for i, r in enumerate(self.tradeoff_curves)
            ]
        self.reflectivity_plot.set_curves(curves, targets)

    def show_step_view(self) -> None:
        """Show the plot of the current workflow step, if plots switch automatically."""
        if not self.auto_views_action.isChecked():
            return
        step = WORKFLOW_STEPS[self.step_tabs.currentIndex()]
        view = self.step_views.get(step)
        if view is not None:
            self.plot_tabs.show_view(view)

    def _theme_changed(self) -> None:
        self._on_state_changed(self.controller.state)
        self._redraw()
        self.tolerance_view.refresh()
        self._update_tradeoff_plot()

    def _job_state_changed(self, running: bool) -> None:
        self.live_curves = ()
        if not running:
            self._redraw()

    def _auto_views_toggled(self, checked: bool) -> None:
        self.settings.setValue("plots/auto_switch", checked)
        if checked:
            self.show_step_view()

    def _job_progress(self, event: object) -> None:
        if not isinstance(event, ProgressEvent):
            return
        self.convergence_plot.add_point(
            event.task,
            event.evaluations,
            event.best_objective,
            event.best_merit,
            event.best_quantities,
        )
        state = self._job_state
        values = dict(event.best_values)
        if (
            values
            and state is not None
            and state.mirror is not None
            and values != self._live_values
        ):
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

    def _show_scan(self, result: ScanResult) -> None:
        self.design_map.set_scan(
            result.names, result.axes, result.merit, result.feasible, "Figure of merit"
        )
        self.plot_tabs.show_view(self.design_map)

    def _request_tolerance_curves(self) -> None:
        if self.tolerance_result is None:
            return
        if not self.display_ranges:
            self.tolerance_view.set_result(self.tolerance_result)
            return
        result, ranges, state = (
            self.tolerance_result,
            dict(self.display_ranges),
            self.controller.state,
        )

        def compute() -> tuple[ToleranceResult, dict[str, tuple[float, float]], ToleranceCurves]:
            return result, ranges, tolerance_curves(state, result, ranges)

        self.statusBar().showMessage("Updating tolerance curves for the plot range…")
        self.tolerance_runner.request(compute)

    def _on_tolerance_curves(
        self, outcome: tuple[ToleranceResult, dict[str, tuple[float, float]], ToleranceCurves]
    ) -> None:
        result, ranges, curves = outcome
        # Curves of an older result or range are ignored.
        if result is self.tolerance_result and ranges == self.display_ranges:
            self.tolerance_view.set_result(result, curves)
            self.statusBar().showMessage("Tolerance curves updated", 3000)

    def _show_tolerance(
        self,
        result: ToleranceResult,
        curves: ToleranceCurves | None = None,
        ranges: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        self.tolerance_result = result
        if curves is not None and ranges == self.display_ranges:
            self.tolerance_view.set_result(result, curves)
        else:
            self.tolerance_view.set_result(result)
            self._request_tolerance_curves()
        self.plot_tabs.show_view(self.tolerance_view)

    def _update_tradeoff_plot(self) -> None:
        """Draw the stored trade-off with the input design and the candidates for reference."""
        state = self.controller.state
        references = [
            ReferenceDesign(c.name, c.coating_time, c.merit)
            for c in state.candidates
            if c.coating_time is not None and c.merit is not None
        ]
        preview = self.preview
        if preview is not None and preview.state is state and preview.evaluation is not None:
            evaluation = preview.evaluation
            if evaluation.coating_time is not None:
                references.insert(
                    0,
                    ReferenceDesign(INPUT_DESIGN_LABEL, evaluation.coating_time, evaluation.merit),
                )
        self.tradeoff_plot.set_result(
            self.tradeoff_page.result(), self.tradeoff_page.quantity(), references
        )
        if self.tradeoff_point is not None and not any(
            p is self.tradeoff_point for p in (state.tradeoff.points if state.tradeoff else ())
        ):
            self._clear_tradeoff_curves()

    def _clear_tradeoff_curves(self) -> None:
        self.tradeoff_point = None
        self.tradeoff_curves = ()
        self._redraw()

    def _tradeoff_point_selected(self, index: object) -> None:
        """Show the reflectivity of a selected trade-off point as a dashed overlay."""
        state = self.controller.state
        result = state.tradeoff
        self.tradeoff_plot.select(index if isinstance(index, int) else None)
        if not isinstance(index, int) or result is None or state.mirror is None:
            self._clear_tradeoff_curves()
            return
        point = result.points[index]
        mirror = state.mirror
        ranges = dict(self.display_ranges)
        self.tradeoff_point = point
        self.tradeoff_curves = ()

        def compute() -> tuple[TradeoffPoint, tuple[ReflectivityResult, ...]]:
            design = mirror.with_parameters(point.parameters)
            return point, compute_curves(state, design, None, ranges)

        self.tradeoff_runner.request(compute)

    def _on_tradeoff_curves(
        self, outcome: tuple[TradeoffPoint, tuple[ReflectivityResult, ...]]
    ) -> None:
        point, curves = outcome
        if point is self.tradeoff_point:
            self.tradeoff_curves = curves
            self._redraw()
            self.statusBar().showMessage(
                f"Reflectivity of the trade-off design for {limit_label(point.limit, unit=True)} "
                "is shown dashed in the Reflectivity tab",
                5000,
            )

    def _show_tradeoff(self, _result: object) -> None:
        self.plot_tabs.show_view(self.tradeoff_plot)

    def _apply_pipeline_result(
        self,
        problem: DesignProblem,
        result: PipelineResult,
        name: str | None = None,
    ) -> None:
        record_run(self.controller.project.journal, result)
        runs = len(self.controller.project.journal.entries("run"))
        self.controller.execute(store_pipeline_result(problem, result, name or f"run {runs}"))

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

    def selected_candidates(self) -> list[str]:
        """Return the names of the candidates selected in the list, in list order."""
        rows = sorted(self.candidate_list.row(i) for i in self.candidate_list.selectedItems())
        names = [self.candidate_list.item(r).data(Qt.ItemDataRole.UserRole) for r in rows]
        return [n for n in names if n is not None]

    def _input_is_current(self) -> bool:
        item = self.candidate_list.currentItem()
        return item is not None and item.data(Qt.ItemDataRole.UserRole) is None

    def export_reflectivity(self, name: str | None = None, path: str | Path | None = None) -> bool:
        """Write the reflectivity of one design over the plotted range to a CSV file.

        Each target is exported on its plot range (see the plot range fields) with the
        reflectivity R of one reflection and Rᴺ after the target's N reflections.

        Args:
            name: Candidate name; None exports the input design.
            path: Output file; asks for a file name if None.

        Returns:
            Whether the file was written.
        """
        state = self.controller.state
        try:
            mirror = state.mirror if name is None else state.candidate(name).mirror
        except KeyError:
            mirror = None
        if mirror is None or state.targets is None:
            self._show_error("No design or no targets to export")
            return False
        try:
            stack = mirror.stack()
        except (CoatingBuildError, ValueError) as exc:
            self._show_error(f"Cannot export: {exc}")
            return False
        floor = lowest_energy(frozenset(stack.media), state.optical_constants)
        targets = display_targets(state.targets.targets, self.display_ranges, floor)
        if path is None:
            label = INPUT_DESIGN_LABEL if name is None else name
            chosen = ask_save_path(
                self, f"Export reflectivity of {label}", "CSV files (*.csv)", ".csv"
            )
            if chosen is None:
                return False
            path = chosen
        try:
            write_reflectivity_csv(path, stack, targets, state.model, state.library())
        except (OSError, ValueError, KeyError) as exc:
            self._show_error(f"Cannot export: {exc}")
            return False
        self.statusBar().showMessage(f"Exported reflectivity to {path}", 5000)
        return True

    def _use_candidate(self) -> None:
        name = self._selected_candidate()
        if name is not None:
            self.controller.execute(promote_candidate(name))

    def _delete_candidate(self) -> None:
        names = self.selected_candidates()
        if not names:
            current = self._selected_candidate()
            names = [current] if current is not None else []
        if names:
            self.controller.execute(remove_candidates(names))

    def _rename_candidate(self) -> None:
        item = self.candidate_list.currentItem()
        if item is not None:
            self.candidate_list.editItem(item)

    def show_candidate_parameters(
        self, name: str | None = None
    ) -> CandidateParametersDialog | None:
        """Open a window with the coating parameters of a candidate.

        Args:
            name: Candidate name; defaults to the current candidate of the list.

        Returns:
            The dialog, or ``None`` if there is no such candidate.
        """
        name = name or self._selected_candidate()
        if name is None:
            return None
        try:
            candidate = self.controller.state.candidate(name)
        except KeyError:
            return None
        dialog = CandidateParametersDialog(candidate, self)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.show()
        return dialog

    def candidate_menu(self) -> QMenu:
        """Return the context menu of the candidates list for the current selection."""
        menu = QMenu(self.candidate_list)
        names = self.selected_candidates()
        single = len(names) <= 1 and self._selected_candidate() is not None
        use = menu.addAction("Use as start design", self._use_candidate)
        rename = menu.addAction("Rename", self._rename_candidate)
        show = menu.addAction("Show parameters…", lambda: self.show_candidate_parameters())
        for action in (use, rename, show):
            action.setEnabled(single)
        export = menu.addAction(
            "Export reflectivity…",
            lambda: self.export_reflectivity(self._selected_candidate()),
        )
        export.setEnabled(single or (not names and self._input_is_current()))
        menu.addSeparator()
        count = max(len(names), 1 if single else 0)
        delete = menu.addAction(
            "Delete" if count <= 1 else f"Delete {count} candidates", self._delete_candidate
        )
        delete.setEnabled(count > 0)
        return menu

    def _candidate_menu(self, position: QPoint) -> None:
        item = self.candidate_list.itemAt(position)
        if item is not None and not item.isSelected():
            self.candidate_list.setCurrentItem(item)
        self.candidate_menu().exec(self.candidate_list.viewport().mapToGlobal(position))

    def _candidate_clicked(self, current: QListWidgetItem | None, _previous: object) -> None:
        name = current.data(Qt.ItemDataRole.UserRole) if current is not None else None
        if name != self.controller.state.selected:
            self.controller.execute(select_candidate(name))

    def _candidate_item_changed(self, item: QListWidgetItem) -> None:
        old = item.data(Qt.ItemDataRole.UserRole)
        if old is None:
            return
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
