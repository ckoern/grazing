"""Workflow page for the figure of merit and the optimization pipeline."""

from collections.abc import Callable
from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from xstack.gui.forms import DataclassForm
from xstack.gui.steps.base import StepPage
from xstack.gui.workers import Job
from xstack.merit.base import FigureOfMerit, merits
from xstack.merit.composite import TimePenalized
from xstack.merit.optical import MeanReflectivity
from xstack.optim.events import CancellationToken
from xstack.optim.pipeline import Pipeline, PipelineResult
from xstack.optim.problem import DesignProblem
from xstack.project import ProjectState
from xstack.project.commands import set_merit

__all__ = ["OptimizePage"]

OPTICAL_MERITS = (
    "mean_reflectivity",
    "soft_min_reflectivity",
    "profile_deviation",
    "flatness",
    "peak_reflectivity",
)
"""Figures of merit offered as the optical part of the objective."""


class OptimizePage(StepPage):
    """Figure of merit (with optional coating-time penalty) and pipeline runs.

    Signals:
        pipelineFinished(DesignProblem, PipelineResult): A pipeline run finished.

    Args:
        parent: Parent widget.
    """

    title = "Optimize"
    pipelineFinished = Signal(object, object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state: ProjectState | None = None
        self._loading = False
        layout = QVBoxLayout(self)

        merit_box = QGroupBox("Figure of merit")
        merit_layout = QVBoxLayout(merit_box)
        self.merit_combo = QComboBox()
        for name in OPTICAL_MERITS:
            self.merit_combo.addItem(name.replace("_", " "), name)
        self.merit_combo.activated.connect(lambda _: self._merit_type_changed())
        merit_layout.addWidget(self.merit_combo)
        self.merit_form = DataclassForm()
        self.merit_form.valueChanged.connect(lambda _: self._emit_merit())
        self.merit_form.invalidInput.connect(self.errorOccurred)
        merit_layout.addWidget(self.merit_form)
        self.time_check = QCheckBox("Penalize coating time (needs a deposition process)")
        self.time_check.toggled.connect(lambda _: self._emit_merit())
        merit_layout.addWidget(self.time_check)
        self.time_form = DataclassForm()
        self.time_form.valueChanged.connect(lambda _: self._emit_merit())
        self.time_form.invalidInput.connect(self.errorOccurred)
        merit_layout.addWidget(self.time_form)
        self.custom_label = QLabel("The project uses a custom figure of merit.")
        self.custom_label.setVisible(False)
        merit_layout.addWidget(self.custom_label)
        layout.addWidget(merit_box)

        pipeline_box = QGroupBox("Pipeline")
        form = QFormLayout(pipeline_box)
        self.seed = QLineEdit()
        self.seed.setPlaceholderText("random")
        self.iterations = QSpinBox()
        self.iterations.setRange(1, 10000)
        self.iterations.setValue(30)
        self.population = QSpinBox()
        self.population.setRange(2, 200)
        self.population.setValue(10)
        self.coarse = QSpinBox()
        self.coarse.setRange(5, 2000)
        self.coarse.setValue(60)
        self.scan_points = QSpinBox()
        self.scan_points.setRange(2, 100)
        self.scan_points.setValue(9)
        form.addRow("Random seed", self.seed)
        form.addRow("Global generations", self.iterations)
        form.addRow("Population factor", self.population)
        form.addRow("Coarse grid points", self.coarse)
        form.addRow("Scan points per parameter", self.scan_points)
        self.custom_pipeline = QCheckBox("Use the pipeline stored in the project")
        form.addRow(self.custom_pipeline)
        self.stages_label = QLabel()
        self.stages_label.setWordWrap(True)
        form.addRow("Stages", self.stages_label)
        layout.addWidget(pipeline_box)

        buttons = QHBoxLayout()
        self.run_button = QPushButton("Run pipeline")
        self.run_button.clicked.connect(lambda: self.run())
        self.stage_combo = QComboBox()
        self.stage_button = QPushButton("Run stage")
        self.stage_button.clicked.connect(lambda: self.run(self.stage_combo.currentText()))
        buttons.addWidget(self.run_button)
        buttons.addWidget(self.stage_combo)
        buttons.addWidget(self.stage_button)
        layout.addLayout(buttons)
        for widget in (self.seed,):
            widget.editingFinished.connect(self._update_stages)
        for box in (self.iterations, self.population, self.coarse, self.scan_points):
            box.valueChanged.connect(lambda _: self._update_stages())
        self.custom_pipeline.toggled.connect(lambda _: self._update_stages())
        layout.addStretch()

    def set_state(self, state: ProjectState) -> None:
        """Show the figure of merit and pipeline of a project state.

        Args:
            state: The current project state.
        """
        previous = self._state
        self._state = state
        if previous is None or previous.merit != state.merit:
            self._load_merit(state.merit)
        self.custom_pipeline.setEnabled(state.pipeline is not None)
        if state.pipeline is None:
            self.custom_pipeline.setChecked(False)
        self._update_stages()

    def _load_merit(self, merit: FigureOfMerit | None) -> None:
        self._loading = True
        time = merit if isinstance(merit, TimePenalized) else None
        base = time.merit if time is not None else merit
        known = base is not None and base.type_name in OPTICAL_MERITS
        self.custom_label.setVisible(merit is not None and not known)
        if known and base is not None:
            self.merit_combo.setCurrentIndex(self.merit_combo.findData(base.type_name))
            self.merit_form.set_value(base)
        else:
            self.merit_form.set_value(MeanReflectivity())
        self.time_check.setChecked(time is not None)
        self.time_form.set_value(time or TimePenalized(self.merit_form.value()))
        self.time_form.setEnabled(time is not None)
        self._loading = False

    def _merit_type_changed(self) -> None:
        cls = merits.get(self.merit_combo.currentData())
        self.merit_form.set_value(cls())
        self._emit_merit()

    def current_merit(self) -> FigureOfMerit:
        """Return the figure of merit defined in the page."""
        base = self.merit_form.value()
        if not self.time_check.isChecked():
            return base
        template = self.time_form.value()
        return TimePenalized(base, template.weight, template.reference_time)

    def _emit_merit(self) -> None:
        self.time_form.setEnabled(self.time_check.isChecked())
        if self._loading or self._state is None:
            return
        merit = self.current_merit()
        if merit != self._state.merit:
            self.commandRequested.emit(set_merit(merit))

    def pipeline(self, problem: DesignProblem) -> Pipeline:
        """Return the pipeline defined in the page for a problem.

        Args:
            problem: The design problem.

        Returns:
            The project's pipeline if selected, otherwise the default pipeline with the page's
            settings.
        """
        if self.custom_pipeline.isChecked() and self._state and self._state.pipeline:
            return self._state.pipeline
        seed_text = self.seed.text().strip()
        return Pipeline.default(
            problem,
            scan_points=self.scan_points.value(),
            coarse_points=self.coarse.value(),
            global_iterations=self.iterations.value(),
            population=self.population.value(),
            seed=int(seed_text) if seed_text else None,
        )

    def _update_stages(self) -> None:
        names: list[str] = []
        try:
            if self._state is not None:
                names = [s.name for s in self.pipeline(self._state.problem()).stages]
        except ValueError:
            names = []
        self.stages_label.setText(" → ".join(names) if names else "(project incomplete)")
        current = self.stage_combo.currentText()
        self.stage_combo.clear()
        self.stage_combo.addItems(names)
        if current in names:
            self.stage_combo.setCurrentText(current)
        self.run_button.setEnabled(bool(names))
        self.stage_button.setEnabled(bool(names))

    def run(self, stage: str | None = None) -> None:
        """Request a pipeline run, or a run of a single stage.

        Args:
            stage: Name of the stage to run alone, or ``None`` for the whole pipeline.
        """
        if self._state is None:
            return
        try:
            problem = self._state.problem()
            pipeline = self.pipeline(problem)
        except ValueError as exc:
            self.errorOccurred.emit(f"Cannot optimize: {exc}")
            return
        if stage is not None:
            pipeline = Pipeline(tuple(s for s in pipeline.stages if s.name == stage))
        name = f"Stage {stage}" if stage else "Pipeline"

        def compute(progress: Callable[[Any], None], cancel: CancellationToken) -> PipelineResult:
            return pipeline.run(problem, events=progress, cancel=cancel)

        self.jobRequested.emit(
            Job(name, compute, lambda result: self.pipelineFinished.emit(problem, result))
        )
