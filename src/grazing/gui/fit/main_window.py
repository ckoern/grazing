"""Main window of the fit application."""

import functools
from collections.abc import Callable
from dataclasses import replace

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QAction, QActionGroup, QColor, QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
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

from grazing.core.optim.events import ProgressEvent
from grazing.core.units import deg, kev, to_deg, to_kev
from grazing.fit.dataset import Dataset, Interval
from grazing.fit.project import (
    FitState,
    remove_dataset,
    replace_dataset,
    set_ranges,
)
from grazing.fit.templates import TEMPLATES
from grazing.gui.common.document_window import DocumentWindow
from grazing.gui.common.journal_view import JournalView
from grazing.gui.common.pages import StepPage
from grazing.gui.common.plot_tabs import PlotTabs
from grazing.gui.common.profile import LayerProfilePlot
from grazing.gui.common.style import (
    PLOT_PALETTE,
    set_variant,
)
from grazing.gui.common.wizard import ProjectWizard
from grazing.gui.common.workers import JobRunner, LatestRunner
from grazing.gui.fit.controller import FitController
from grazing.gui.fit.plots import ConvergencePlot, CorrelationPlot, FitReflectivityPlot
from grazing.gui.fit.preview import FitPreview, compute_preview
from grazing.gui.fit.steps import (
    DataPage,
    FitPage,
    MaterialsPage,
    ParametersPage,
    ResultsPage,
    StructurePage,
)

__all__ = ["PROJECT_FILTER", "WORKFLOW_STEPS", "MainWindow"]

WORKFLOW_STEPS = ("Data", "Structure", "Materials", "Parameters", "Fit", "Results")
"""Names of the workflow steps, in order."""

PROJECT_FILTER = "grazing fit projects (*.gfit);;All files (*)"


def _to_display(axis: str, bounds: Interval) -> tuple[float, float]:
    convert = to_deg if axis == "angle" else to_kev
    return float(convert(bounds[0])), float(convert(bounds[1]))


def _to_internal(axis: str, bounds: tuple[float, float]) -> Interval:
    convert = deg if axis == "angle" else kev
    lo, hi = sorted(bounds)
    return float(convert(lo)), float(convert(hi))


class MainWindow(DocumentWindow):
    """Main window of grazing-fit: datasets, workflow steps, plots, journal and history.

    Args:
        controller: The project controller. A new one is created if not given.
        settings: Settings store for window geometry, theme and recent files.
        confirm_close: Ask whether to save unsaved changes when closing or replacing the
            project.
    """

    project_kind = "fit project"
    file_filter = PROJECT_FILTER
    controller: FitController
    file_suffix = ".gfit"
    app_name = "grazing-fit"

    def __init__(
        self,
        controller: FitController | None = None,
        settings: QSettings | None = None,
        confirm_close: bool = True,
    ) -> None:
        super().__init__()
        self.controller = controller or FitController(parent=self)
        self.settings = settings or QSettings("grazing", "grazing-fit")
        self.confirm_close = confirm_close
        self.wizard_factory: Callable[[], ProjectWizard] = lambda: ProjectWizard(
            "fit project", ".gfit", PROJECT_FILTER, parent=self
        )
        self.selected_dataset: str | None = None
        self._selected_index = 0
        self.preview: FitPreview | None = None
        self._job_state: FitState | None = None
        self._live_values: dict[str, float] | None = None
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
            self.resize(1500, 920)
        state = self.settings.value("main_window/state")
        if state is not None:
            self.restoreState(state)

    # ------------------------------------------------------------------ layout
    def _build_central(self) -> None:
        self.reflectivity_plot = FitReflectivityPlot()
        self.correlation_plot = CorrelationPlot()
        self.layer_plot = LayerProfilePlot()
        self.convergence_plot = ConvergencePlot()
        self.plot_tabs = PlotTabs("grazing-fit")
        self.reflectivity_view = self._reflectivity_tab()
        self.plot_tabs.addTab(self.reflectivity_view, "Reflectivity")
        self.plot_tabs.addTab(self.correlation_plot, "Correlations")
        self.plot_tabs.addTab(self.layer_plot, "Layer profile")
        self.plot_tabs.addTab(self.convergence_plot, "Convergence")
        self.step_views: dict[str, QWidget] = {
            "Data": self.reflectivity_view,
            "Structure": self.layer_plot,
            "Materials": self.layer_plot,
            "Parameters": self.reflectivity_view,
            "Fit": self.reflectivity_view,
            "Results": self.correlation_plot,
        }
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._step_panel())
        plot_area = QWidget()
        plot_layout = QVBoxLayout(plot_area)
        plot_layout.setContentsMargins(4, 8, 8, 4)
        plot_layout.addWidget(self.plot_tabs, 1)
        splitter.addWidget(plot_area)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([660, 840])
        self.setCentralWidget(splitter)
        self.job_runner = JobRunner(parent=self)
        self.preview_runner = LatestRunner(parent=self)
        self.preview_runner.finished.connect(self._on_preview)
        self.preview_runner.failed.connect(self._show_error)
        self.live_runner = LatestRunner(delay_ms=300, parent=self)
        self.live_runner.finished.connect(self._on_live_preview)

    def _reflectivity_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        controls = QHBoxLayout()
        self.log_check = QCheckBox("log scale")
        self.log_check.setChecked(True)
        self.log_check.toggled.connect(self._log_toggled)
        controls.addWidget(self.log_check)
        controls.addStretch()
        hint = QLabel("Shift-drag: add fit range · Ctrl-drag: exclude range")
        hint.setProperty("role", "muted")
        controls.addWidget(hint)
        layout.addLayout(controls)
        layout.addWidget(self.reflectivity_plot, 1)
        return tab

    def _step_panel(self) -> QWidget:
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
        self.header_label = QLabel()
        self.header_label.setObjectName("design_header")
        self.header_label.setWordWrap(True)
        layout.addWidget(self.header_label)
        self.data_page = DataPage()
        self.structure_page = StructurePage()
        self.materials_page = MaterialsPage()
        self.parameters_page = ParametersPage()
        self.fit_page = FitPage()
        self.results_page = ResultsPage()
        self.pages: dict[str, StepPage] = {
            "Data": self.data_page,
            "Structure": self.structure_page,
            "Materials": self.materials_page,
            "Parameters": self.parameters_page,
            "Fit": self.fit_page,
            "Results": self.results_page,
        }
        self.step_tabs = QTabWidget()
        self.step_tabs.setObjectName("step_tabs")
        self.step_tabs.setDocumentMode(True)
        self.step_tabs.setUsesScrollButtons(True)
        for index, step in enumerate(WORKFLOW_STEPS):
            self.step_tabs.addTab(
                self._step_container(index, self.pages[step]), f"{index + 1} {step}"
            )
            self.step_tabs.setTabToolTip(index, f"Step {index + 1}: {step}")
        self.step_tabs.currentChanged.connect(lambda _: self.show_step_view())
        layout.addWidget(self.step_tabs, 1)
        return panel

    def _step_container(self, index: int, page: StepPage) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        if page.help_text:
            help_label = QLabel(page.help_text)
            help_label.setWordWrap(True)
            help_label.setTextFormat(Qt.TextFormat.RichText)
            help_label.setProperty("role", "help")
            help_label.setObjectName(f"help_{WORKFLOW_STEPS[index].lower()}")
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

    def _build_docks(self) -> None:
        self.dataset_list = QListWidget()
        self.dataset_list.setToolTip(
            "Measured datasets. Checked datasets take part in the fit; the selected one is "
            "edited on the Data page and its fit ranges are shown in the plot."
        )
        self.dataset_list.currentItemChanged.connect(self._dataset_clicked)
        self.dataset_list.itemChanged.connect(self._dataset_item_changed)
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.dataset_list, 1)
        buttons = QHBoxLayout()
        self.add_data_button = QPushButton("Add data…")
        set_variant(self.add_data_button, "primary")
        self.add_data_button.clicked.connect(self.add_data)
        self.remove_data_button = QPushButton("Remove")
        self.remove_data_button.clicked.connect(self._remove_dataset)
        buttons.addWidget(self.add_data_button)
        buttons.addWidget(self.remove_data_button)
        layout.addLayout(buttons)
        self.datasets_dock = self._dock("Datasets", panel, Qt.DockWidgetArea.LeftDockWidgetArea)
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
        self.resizeDocks([self.datasets_dock], [230], Qt.Orientation.Horizontal)
        self.resizeDocks([self.journal_dock], [130], Qt.Orientation.Vertical)

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
        self.rename_action.triggered.connect(self.rename_project)
        self.add_data_action = QAction("Add &data…", self)
        self.add_data_action.setShortcut(QKeySequence("Ctrl+D"))
        self.add_data_action.triggered.connect(self.add_data)
        self.quit_action = QAction("&Quit", self)
        self.quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        self.quit_action.triggered.connect(self.close)
        self.undo_action = self.controller.undo_stack.createUndoAction(self, "&Undo")
        self.undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        self.redo_action = self.controller.undo_stack.createRedoAction(self, "&Redo")
        self.redo_action.setShortcut(QKeySequence.StandardKey.Redo)
        self.run_action = QAction("&Run fit", self)
        self.run_action.setShortcut(QKeySequence("Ctrl+R"))
        self.run_action.triggered.connect(lambda: self.fit_page.run())
        self.local_action = QAction("Run &local refinement", self)
        self.local_action.setShortcut(QKeySequence("Ctrl+Shift+R"))
        self.local_action.triggered.connect(lambda: self.fit_page.run_local())
        self.auto_views_action = QAction("Switch plots &automatically", self, checkable=True)
        self.auto_views_action.setToolTip(
            "Show the plot that belongs to the current step, and the convergence during fits"
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
        file_menu.addAction(self.add_data_action)
        file_menu.addSeparator()
        file_menu.addAction(self.save_action)
        file_menu.addAction(self.save_as_action)
        file_menu.addAction(self.rename_action)
        file_menu.addSeparator()
        file_menu.addAction(self.quit_action)
        edit_menu = self.menuBar().addMenu("&Edit")
        edit_menu.addAction(self.undo_action)
        edit_menu.addAction(self.redo_action)
        fit_menu = self.menuBar().addMenu("F&it")
        fit_menu.addAction(self.run_action)
        fit_menu.addAction(self.local_action)
        fit_menu.addAction(self.stop_action)
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
        for dock in (self.datasets_dock, self.journal_dock, self.history_dock):
            view_menu.addAction(dock.toggleViewAction())
        toolbar = self.addToolBar("Main")
        toolbar.setObjectName("toolbar_main")
        toolbar.setMovable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._apply_icons()
        for action in (self.new_action, self.open_action, self.save_action):
            toolbar.addAction(action)
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
        for page in self.pages.values():
            page.commandRequested.connect(c.execute)
            page.errorOccurred.connect(self._show_error)
            page.jobRequested.connect(self.start_job)
        self.results_page.fitSelected.connect(self._show_fit)
        self.reflectivity_plot.rangeDrawn.connect(self._range_drawn)
        self.reflectivity_plot.rangesEdited.connect(self._ranges_edited)
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

    # ------------------------------------------------------------------ state
    def step_pages(self) -> list[StepPage]:
        """Return the workflow pages in order."""
        return [self.pages[step] for step in WORKFLOW_STEPS]

    def next_step(self) -> None:
        """Show the next workflow step."""
        self.step_tabs.setCurrentIndex(
            min(self.step_tabs.currentIndex() + 1, len(WORKFLOW_STEPS) - 1)
        )

    def previous_step(self) -> None:
        """Show the previous workflow step."""
        self.step_tabs.setCurrentIndex(max(self.step_tabs.currentIndex() - 1, 0))

    def _on_state_changed(self, state: FitState) -> None:
        names = [d.name for d in state.datasets]
        if self.selected_dataset not in names:
            index = min(self._selected_index, len(names) - 1)
            self.selected_dataset = names[index] if names else None
        if self.selected_dataset is not None:
            self._selected_index = names.index(self.selected_dataset)
        self._sync_dataset_list(state)
        for page in self.step_pages():
            page.set_state(state)
        self.data_page.set_selected(self.selected_dataset)
        if self.name_edit.text() != state.name and not self.name_edit.hasFocus():
            self.name_edit.setText(state.name)
        self.header_label.setText(self._header(state))
        self.remove_data_button.setEnabled(bool(names))
        self._update_title()
        self._request_preview()

    @staticmethod
    def _header(state: FitState) -> str:
        if state.mirror is None:
            structure = "no structure"
        else:
            coating = state.mirror.coating
            structure = (
                f"<b>{type(coating).__name__}</b> “{coating.name}” · substrate "
                f"{state.mirror.substrate.name}"
            )
        count = len(state.datasets)
        data = f"{count} dataset{'s' if count != 1 else ''}"
        return f"{structure} · {state.model.name} · {data} · {state.weighting} weighting"

    def _sync_dataset_list(self, state: FitState) -> None:
        self.dataset_list.blockSignals(True)
        self.dataset_list.clear()
        for index, dataset in enumerate(state.datasets):
            item = QListWidgetItem(self._dataset_text(dataset))
            item.setData(Qt.ItemDataRole.UserRole, dataset.name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked if dataset.enabled else Qt.CheckState.Unchecked
            )
            item.setData(
                Qt.ItemDataRole.DecorationRole, _swatch(PLOT_PALETTE[index % len(PLOT_PALETTE)])
            )
            item.setToolTip(dataset.curve.source or dataset.name)
            self.dataset_list.addItem(item)
            if dataset.name == self.selected_dataset:
                self.dataset_list.setCurrentItem(item)
        self.dataset_list.blockSignals(False)

    @staticmethod
    def _dataset_text(dataset: Dataset) -> str:
        curve = dataset.curve
        if curve.fixed is None:
            fixed = "energy not set" if curve.axis == "angle" else "angle not set"
        elif curve.axis == "angle":
            fixed = f"{to_kev(curve.fixed):.4g} keV"
        else:
            fixed = f"{to_deg(curve.fixed):.4g}°"
        return f"{dataset.name}\n{fixed} · {int(dataset.mask().sum())}/{curve.size} points"

    def _dataset_clicked(self, current: QListWidgetItem | None, _previous: object) -> None:
        if current is None:
            return
        self.selected_dataset = str(current.data(Qt.ItemDataRole.UserRole))
        self._selected_index = self.dataset_list.row(current)
        self.data_page.set_selected(self.selected_dataset)
        self._redraw()

    def _dataset_item_changed(self, item: QListWidgetItem) -> None:
        name = str(item.data(Qt.ItemDataRole.UserRole))
        enabled = item.checkState() == Qt.CheckState.Checked
        dataset = self.controller.state.dataset(name)
        if dataset.enabled != enabled:
            verb = "Include" if enabled else "Exclude"
            self.controller.execute(
                replace_dataset(replace(dataset, enabled=enabled), f"{verb} dataset {name!r}")
            )

    def add_data(self) -> None:
        """Open the import dialog and add the chosen datasets."""
        self.data_page.open_import_dialog()

    def _remove_dataset(self) -> None:
        if self.selected_dataset is not None:
            self.controller.execute(remove_dataset(self.selected_dataset))

    # ------------------------------------------------------------------ ranges
    def _selected(self) -> Dataset | None:
        if self.selected_dataset is None:
            return None
        try:
            return self.controller.state.dataset(self.selected_dataset)
        except KeyError:
            return None

    def _range_drawn(self, lo: float, hi: float, excluded: bool) -> None:
        dataset = self._selected()
        if dataset is None:
            return
        bounds = _to_internal(dataset.curve.axis, (lo, hi))
        if excluded:
            command = set_ranges(dataset.name, dataset.ranges, (*dataset.excluded, bounds))
        else:
            command = set_ranges(dataset.name, (*dataset.ranges, bounds))
        self.controller.execute(command)

    def _ranges_edited(self, ranges: object, excluded: object) -> None:
        dataset = self._selected()
        if dataset is None:
            return
        axis = dataset.curve.axis
        new_ranges = [_to_internal(axis, b) for b in ranges]  # type: ignore[attr-defined]
        new_excluded = [_to_internal(axis, b) for b in excluded]  # type: ignore[attr-defined]
        self.controller.execute(set_ranges(dataset.name, new_ranges, new_excluded))

    # ------------------------------------------------------------------ preview and plots
    def _request_preview(self) -> None:
        state = self.controller.state
        self.preview_runner.request(functools.partial(compute_preview, state))

    def _on_preview(self, preview: FitPreview) -> None:
        if self.job_runner.running:
            return
        self.preview = preview
        for page in self.step_pages():
            page.set_preview(preview)
        self.layer_plot.set_stack(preview.stack)
        self._update_summary()
        self._redraw()

    def _on_live_preview(self, preview: FitPreview) -> None:
        if self.job_runner.running:
            self.preview = preview
            self.layer_plot.set_stack(preview.stack)
            self._redraw()

    def _redraw(self) -> None:
        dataset = self._selected()
        ranges: list[tuple[float, float]] = []
        excluded: list[tuple[float, float]] = []
        if dataset is not None:
            axis = dataset.curve.axis
            ranges = [_to_display(axis, b) for b in dataset.ranges]
            excluded = [_to_display(axis, b) for b in dataset.excluded]
        self.reflectivity_plot.show_preview(self.preview, self.selected_dataset, ranges, excluded)

    def _update_summary(self) -> None:
        preview = self.preview
        state = self.controller.state
        total = sum(d.curve.size for d in state.datasets)
        if preview is None:
            self.summary_label.setText("")
            return
        parts = []
        if preview.error is None and preview.points:
            label = "χ²_red" if preview.weighting == "chi2" else "objective/dof"
            parts.append(f"{label} {preview.reduced_objective:.4g}")
        parts.append(f"{preview.points} of {total} points fitted")
        parts.append(f"{preview.free} free")
        if preview.error is not None:
            parts.append(f"⚠ {preview.error}")
        self.summary_label.setText(" · ".join(parts) + "  ")

    def _log_toggled(self, checked: bool) -> None:
        self.reflectivity_plot.set_log_scale(checked)
        self._redraw()

    def _show_fit(self, stored: object) -> None:
        result = getattr(stored, "result", None)
        self.correlation_plot.set_statistics(None if result is None else result.statistics)

    # ------------------------------------------------------------------ jobs
    def show_step_view(self) -> None:
        """Show the plot of the current workflow step, if plots switch automatically."""
        if not self.auto_views_action.isChecked():
            return
        view = self.step_views.get(WORKFLOW_STEPS[self.step_tabs.currentIndex()])
        if view is not None:
            self.plot_tabs.show_view(view)

    def _theme_changed(self) -> None:
        self._redraw()

    def _job_state_changed(self, running: bool) -> None:
        self.local_action.setEnabled(not running)
        if not running:
            self._request_preview()

    def _before_close(self) -> None:
        self.job_runner.cancel()

    def _auto_views_toggled(self, checked: bool) -> None:
        self.settings.setValue("plots/auto_switch", checked)
        if checked:
            self.show_step_view()

    def _job_progress(self, event: object) -> None:
        if not isinstance(event, ProgressEvent):
            return
        self.convergence_plot.add_point(event.task, event.evaluations, event.best_objective)
        values = dict(event.best_values)
        state = self._job_state
        if values and state is not None and values != self._live_values:
            self._live_values = values
            self.live_runner.request(functools.partial(compute_preview, state, values))
        if event.total:
            self.progress_bar.setRange(0, event.total)
            self.progress_bar.setValue(min(event.evaluations, event.total))
        self.statusBar().showMessage(
            f"{event.task}: {event.evaluations} evaluations, best objective "
            f"{event.best_objective:.5g}"
        )

    # ------------------------------------------------------------------ misc


def _swatch(color: str) -> QPixmap:
    pixmap = QPixmap(12, 12)
    pixmap.fill(QColor(color))
    return pixmap
