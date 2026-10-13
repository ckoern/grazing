"""Workflow page for the figure of merit and the optimization pipeline."""

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from xstack.gui.forms import DataclassForm
from xstack.gui.steps.base import StepPage
from xstack.gui.workers import Job
from xstack.merit.base import FigureOfMerit, merits
from xstack.merit.composite import TimePenalized, WeightedSum
from xstack.merit.optical import MeanReflectivity
from xstack.optim.events import CancellationToken
from xstack.optim.optimizers import LmfitOptimizer
from xstack.optim.pipeline import Pipeline, PipelineResult
from xstack.optim.problem import DesignProblem, Evaluation
from xstack.optim.scan import GridScan
from xstack.optim.stages import BraggOutline, Stage
from xstack.project import ProjectState
from xstack.project.commands import evaluate_stages, set_merit

__all__ = ["LMFIT_METHODS", "MERIT_DESCRIPTIONS", "OPTICAL_MERITS", "OptimizePage"]

OPTICAL_MERITS = (
    "mean_reflectivity",
    "soft_min_reflectivity",
    "profile_deviation",
    "flatness",
    "peak_reflectivity",
)
"""Figures of merit offered as the optical part of the objective."""

MERIT_DESCRIPTIONS = {
    "mean_reflectivity": (
        "Mean reflectivity over the target band (spectrum weighted). The objective is one minus "
        "the mean, so a brighter mirror scores lower. The usual choice for maximum throughput."
    ),
    "soft_min_reflectivity": (
        "Smooth approximation of the lowest reflectivity in the band. Maximizing it lifts the "
        "weakest part of the band, which favors a uniformly high response over a high peak."
    ),
    "profile_deviation": (
        "Deviation of the reflectivity from the desired profile defined in the goal. Use it to "
        "match a target shape rather than to maximize reflectivity."
    ),
    "flatness": (
        "Relative variation of R over the band; on its own it does not reward high "
        "reflectivity — combine it with mean reflectivity."
    ),
    "peak_reflectivity": (
        "Highest reflectivity within the band. Ignores the rest of the band, so it suits "
        "narrow-band mirrors but may give a poor response elsewhere."
    ),
}
"""Short explanations of the optical figures of merit by type name."""

LMFIT_METHODS = (
    "differential_evolution",
    "least_squares",
    "leastsq",
    "nelder",
    "lbfgsb",
    "powell",
)
"""lmfit methods offered for the optimizer of a pipeline stage."""

_FLATNESS_HINT = (
    "Flatness alone does not reward high reflectivity. Enable 'Combine with second figure of "
    "merit' and add mean reflectivity."
)


@dataclass
class _StageRow:
    """Widgets of one row of the stage table."""

    name: str
    enabled: QCheckBox
    method: QComboBox | None
    evaluations: QSpinBox
    points: QSpinBox
    target: QComboBox | None = None


class OptimizePage(StepPage):
    """Figure of merit (mixable, with optional coating-time penalty) and pipeline runs.

    Signals:
        pipelineFinished(DesignProblem, PipelineResult, object): A pipeline run finished;
            the candidate name entered for the run (``None`` for an automatic name).

    Args:
        parent: Parent widget.
    """

    title = "Optimize"
    help_text: ClassVar[str] = (
        "<h3>Optimize</h3>"
        "<p><b>Objective.</b> The optimizer minimizes the <i>figure of merit</i>: lower is "
        "better. Constraint violations add penalties to it. You can combine two figures of "
        "merit and add a coating-time penalty.</p>"
        "<p><b>Pipeline stages.</b></p><ul>"
        "<li><b>outline</b>: analytic estimate from the Bragg law.</li>"
        "<li><b>scan</b>: grid scan over &Gamma; and the integer parameters.</li>"
        "<li><b>global</b>: differential evolution over all free parameters.</li>"
        "<li><b>refine</b>: local least-squares refinement.</li></ul>"
        "<p>Each stage can be disabled or configured in the stage table. <i>Run stage</i> runs "
        "a single stage from the current design. In a pipeline run, each stage starts from the "
        "best design so far, so the final design is never worse than the input design.</p>"
        "<p><b>Results.</b> Results are stored as candidates named &quot;run N&quot; (the "
        "final design of the pipeline or of a single stage run). "
        "The input design is never overwritten.</p>"
        "<p><b>Progress.</b> Watch the <i>Convergence</i> tab and the live reflectivity of the "
        "best design while the run is active.</p>"
    )
    pipelineFinished = Signal(object, object, object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state: ProjectState | None = None
        self._loading = False
        self._rows: dict[str, _StageRow] = {}
        layout = QVBoxLayout(self)
        layout.addWidget(self._build_merit_box())
        layout.addWidget(self._build_settings_box())
        layout.addWidget(self._build_stage_box())
        layout.addLayout(self._build_buttons())
        layout.addWidget(self._build_results_box())
        layout.addStretch()

    def _build_merit_box(self) -> QGroupBox:
        box = QGroupBox("Figure of merit")
        layout = QVBoxLayout(box)
        self.merit_combo = self._merit_combo(
            "Optical figure of merit that the optimizer minimizes (lower is better)."
        )
        self.merit_combo.activated.connect(lambda _: self._merit_type_changed())
        layout.addWidget(self.merit_combo)
        self.merit_description = QLabel()
        self.merit_description.setWordWrap(True)
        layout.addWidget(self.merit_description)
        self.flatness_hint = QLabel(_FLATNESS_HINT)
        self.flatness_hint.setWordWrap(True)
        self.flatness_hint.setVisible(False)
        layout.addWidget(self.flatness_hint)
        self.merit_form = DataclassForm()
        self.merit_form.valueChanged.connect(lambda _: self._emit_merit())
        self.merit_form.invalidInput.connect(self.errorOccurred)
        layout.addWidget(self.merit_form)

        self.mix_check = QCheckBox("Combine with second figure of merit")
        self.mix_check.setToolTip(
            "Minimize a weighted sum (1 - f)·primary + f·secondary of two optical figures of merit."
        )
        self.mix_check.toggled.connect(lambda _: self._emit_merit())
        layout.addWidget(self.mix_check)
        self.mix_widget = QWidget()
        mix_layout = QFormLayout(self.mix_widget)
        mix_layout.setContentsMargins(0, 0, 0, 0)
        self.secondary_combo = self._merit_combo("Second figure of merit of the weighted sum.")
        self.secondary_combo.setCurrentIndex(self.secondary_combo.findData("mean_reflectivity"))
        self.secondary_combo.activated.connect(lambda _: self._secondary_type_changed())
        self.fraction = QDoubleSpinBox()
        self.fraction.setRange(0.0, 1.0)
        self.fraction.setSingleStep(0.05)
        self.fraction.setDecimals(2)
        self.fraction.setValue(0.5)
        self.fraction.setToolTip(
            "Fraction f of the second figure of merit in the sum: 0 uses only the primary, "
            "1 only the second."
        )
        self.fraction.valueChanged.connect(lambda _: self._emit_merit())
        self.secondary_form = DataclassForm()
        self.secondary_form.set_value(MeanReflectivity())
        self.secondary_form.valueChanged.connect(lambda _: self._emit_merit())
        self.secondary_form.invalidInput.connect(self.errorOccurred)
        mix_layout.addRow("Second merit", self.secondary_combo)
        mix_layout.addRow(self.secondary_form)
        mix_layout.addRow("Fraction f", self.fraction)
        layout.addWidget(self.mix_widget)

        self.time_check = QCheckBox("Penalize coating time (needs a deposition process)")
        self.time_check.setToolTip(
            "Add the estimated coating time to the objective. Requires a deposition process."
        )
        self.time_check.toggled.connect(lambda _: self._emit_merit())
        layout.addWidget(self.time_check)
        self.time_form = DataclassForm()
        self.time_form.valueChanged.connect(lambda _: self._emit_merit())
        self.time_form.invalidInput.connect(self.errorOccurred)
        layout.addWidget(self.time_form)
        self.custom_label = QLabel("The project uses a custom figure of merit.")
        self.custom_label.setVisible(False)
        layout.addWidget(self.custom_label)
        self._refresh_merit_widgets()
        return box

    @staticmethod
    def _merit_combo(tooltip: str) -> QComboBox:
        combo = QComboBox()
        for name in OPTICAL_MERITS:
            combo.addItem(name.replace("_", " "), name)
            combo.setItemData(
                combo.count() - 1, MERIT_DESCRIPTIONS[name], Qt.ItemDataRole.ToolTipRole
            )
        combo.setToolTip(tooltip)
        return combo

    def _build_settings_box(self) -> QGroupBox:
        box = QGroupBox("Pipeline settings")
        form = QFormLayout(box)
        self.seed = QLineEdit()
        self.seed.setPlaceholderText("random")
        self.seed.setToolTip(
            "Integer seed of the global search. The same seed reproduces a run; leave empty for "
            "a different run each time."
        )
        self.iterations = QSpinBox()
        self.iterations.setRange(1, 10000)
        self.iterations.setValue(30)
        self.iterations.setToolTip(
            "Number of differential-evolution generations that the evaluation budget of the "
            "global stage allows for."
        )
        self.population = QSpinBox()
        self.population.setRange(2, 200)
        self.population.setValue(10)
        self.population.setToolTip(
            "Population size of the global search per free parameter. Larger values explore "
            "more but cost more evaluations."
        )
        self.coarse = QSpinBox()
        self.coarse.setRange(5, 2000)
        self.coarse.setValue(60)
        self.coarse.setToolTip(
            "Grid points per target in the scan and global stages. A coarse grid is faster; "
            "the refine stage uses the full grid."
        )
        self.scan_points = QSpinBox()
        self.scan_points.setRange(2, 100)
        self.scan_points.setValue(9)
        self.scan_points.setToolTip("Grid points per scanned parameter in the scan stage.")
        form.addRow("Random seed", self.seed)
        form.addRow("Global generations", self.iterations)
        form.addRow("Population factor", self.population)
        form.addRow("Coarse grid points", self.coarse)
        form.addRow("Scan points per parameter", self.scan_points)
        self.custom_pipeline = QCheckBox("Use the pipeline stored in the project")
        self.custom_pipeline.setToolTip(
            "Run the pipeline saved in the project instead of the stages configured below."
        )
        form.addRow(self.custom_pipeline)
        self.seed.editingFinished.connect(self._update_stages)
        for spin in (self.iterations, self.population, self.coarse, self.scan_points):
            spin.valueChanged.connect(lambda _: self._update_stages())
        self.custom_pipeline.toggled.connect(lambda _: self._update_stages())
        return box

    def _build_stage_box(self) -> QGroupBox:
        box = QGroupBox("Stages")
        layout = QVBoxLayout(box)
        self.stage_table = QTableWidget(0, 5)
        self.stage_table.setHorizontalHeaderLabels(
            ["Enabled", "Stage", "Optimizer", "Max evaluations", "Grid points"]
        )
        self._compact_table(self.stage_table)
        self.stage_table.setToolTip("Stages of the default pipeline and their settings.")
        layout.addWidget(self.stage_table)
        return box

    def _build_results_box(self) -> QGroupBox:
        box = QGroupBox("Last run")
        layout = QVBoxLayout(box)
        self.results_table = QTableWidget(0, 5)
        self.results_table.setHorizontalHeaderLabels(
            [
                "Stage",
                "Evaluations",
                "Best objective (stage grid)",
                "Figure of merit (full problem)",
                "Time (s)",
            ]
        )
        self._compact_table(self.results_table)
        self.results_table.setToolTip(
            "Per stage: objective evaluations, the best objective on the stage's own grid and "
            "merit, and the figure of merit of the stage result on the full problem."
        )
        layout.addWidget(self.results_table)
        return box

    @staticmethod
    def _compact_table(table: QTableWidget) -> None:
        table.setShowGrid(False)
        table.setFrameShape(QTableWidget.Shape.NoFrame)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setStretchLastSection(True)
        table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    @staticmethod
    def _fit_height(table: QTableWidget) -> None:
        height = table.horizontalHeader().height() + 2
        for row in range(table.rowCount()):
            height += table.rowHeight(row)
        table.setFixedHeight(height)

    def _build_buttons(self) -> QHBoxLayout:
        buttons = QHBoxLayout()
        self.run_name = QLineEdit()
        self.run_name.setPlaceholderText("run name (optional)")
        self.run_name.setToolTip(
            "Name of the candidate that stores the final design of the next run. Leave empty "
            "for an automatic name “run N”."
        )
        self.run_name.setMaximumWidth(180)
        buttons.addWidget(self.run_name)
        self.run_button = QPushButton("Run pipeline")
        self.run_button.setToolTip("Run all enabled stages in order, each from the previous one.")
        self.run_button.clicked.connect(lambda: self.run())
        self.stage_combo = QComboBox()
        self.stage_combo.setToolTip("Enabled stage to run alone.")
        self.stage_button = QPushButton("Run stage")
        self.stage_button.setToolTip("Run only the selected stage from the current design.")
        self.stage_button.clicked.connect(lambda: self.run(self.stage_combo.currentText()))
        buttons.addWidget(self.run_button)
        buttons.addWidget(self.stage_combo)
        buttons.addWidget(self.stage_button)
        return buttons

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
        inner = time.merit if time is not None else merit
        primary: FigureOfMerit | None = inner
        secondary: FigureOfMerit | None = None
        fraction = self.fraction.value()
        if isinstance(inner, WeightedSum):
            primary = secondary = None
            terms = inner.terms
            if (
                len(terms) == 2
                and terms[0][0].type_name in OPTICAL_MERITS
                and terms[1][0].type_name in OPTICAL_MERITS
                and abs(terms[0][1] + terms[1][1] - 1.0) < 1e-9
            ):
                primary, secondary, fraction = terms[0][0], terms[1][0], terms[1][1]
        known = primary is not None and primary.type_name in OPTICAL_MERITS
        self.custom_label.setVisible(merit is not None and not known)
        if known and primary is not None:
            self.merit_combo.setCurrentIndex(self.merit_combo.findData(primary.type_name))
            self.merit_form.set_value(primary)
        else:
            self.merit_combo.setCurrentIndex(self.merit_combo.findData("mean_reflectivity"))
            self.merit_form.set_value(MeanReflectivity())
        self.mix_check.setChecked(secondary is not None)
        if secondary is not None:
            self.secondary_combo.setCurrentIndex(self.secondary_combo.findData(secondary.type_name))
            self.secondary_form.set_value(secondary)
            self.fraction.setValue(fraction)
        self.time_check.setChecked(time is not None)
        self.time_form.set_value(time or TimePenalized(self.merit_form.value()))
        self._refresh_merit_widgets()
        self._loading = False

    def _refresh_merit_widgets(self) -> None:
        name = self.merit_combo.currentData()
        self.merit_description.setText(MERIT_DESCRIPTIONS.get(name, ""))
        self.mix_widget.setVisible(self.mix_check.isChecked())
        self.flatness_hint.setVisible(name == "flatness" and not self.mix_check.isChecked())
        self.time_form.setVisible(self.time_check.isChecked())
        self.time_form.setEnabled(self.time_check.isChecked())

    def _merit_type_changed(self) -> None:
        cls = merits.get(self.merit_combo.currentData())
        self.merit_form.set_value(cls())
        self._emit_merit()

    def _secondary_type_changed(self) -> None:
        cls = merits.get(self.secondary_combo.currentData())
        self.secondary_form.set_value(cls())
        self._emit_merit()

    def current_merit(self) -> FigureOfMerit:
        """Return the figure of merit defined in the page.

        Returns:
            The primary optical merit, a `WeightedSum` of primary and secondary merit if mixing
            is enabled, wrapped in `TimePenalized` if the time penalty is enabled.
        """
        merit: FigureOfMerit = self.merit_form.value()
        if self.mix_check.isChecked():
            f = self.fraction.value()
            merit = WeightedSum(((merit, 1.0 - f), (self.secondary_form.value(), f)))
        if not self.time_check.isChecked():
            return merit
        template = self.time_form.value()
        return TimePenalized(merit, template.weight, template.reference_time)

    def _emit_merit(self) -> None:
        self._refresh_merit_widgets()
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
            global settings and stage table: disabled stages are removed and the optimizer,
            evaluation limit and grid points of the stages follow the table.
        """
        if self.custom_pipeline.isChecked() and self._state and self._state.pipeline:
            return self._state.pipeline
        seed_text = self.seed.text().strip()
        default = Pipeline.default(
            problem,
            scan_points=self.scan_points.value(),
            coarse_points=self.coarse.value(),
            global_iterations=self.iterations.value(),
            population=self.population.value(),
            seed=int(seed_text) if seed_text else None,
        )
        stages: list[Stage] = []
        for stage in default.stages:
            row = self._rows.get(stage.name)
            if row is None:
                stages.append(stage)
            elif row.enabled.isChecked():
                stages.append(self._configured(stage, row))
        return Pipeline(tuple(stages))

    @staticmethod
    def _configured(stage: Stage, row: _StageRow) -> Stage:
        optimizer = stage.optimizer
        if isinstance(optimizer, LmfitOptimizer) and row.method is not None:
            method = row.method.currentData()
            changes: dict[str, Any] = {}
            if method != optimizer.method:
                changes.update(method=method, options=())
            if row.evaluations.value() > 0:
                changes["max_evaluations"] = row.evaluations.value()
            if changes:
                optimizer = dataclasses.replace(optimizer, **changes)
        if isinstance(optimizer, BraggOutline) and row.target is not None:
            optimizer = dataclasses.replace(optimizer, target=row.target.currentData())
        points = row.points.value()
        if points > 0 and not isinstance(optimizer, BraggOutline):
            return dataclasses.replace(stage, optimizer=optimizer, points=points)
        return dataclasses.replace(stage, optimizer=optimizer)

    def _default_stages(self) -> tuple[Stage, ...]:
        """Return the stages of the default pipeline of the current project, if complete."""
        if self._state is None:
            return ()
        try:
            return Pipeline.default(self._state.problem()).stages
        except ValueError:
            return ()

    def _update_stages(self) -> None:
        stages = self._default_stages()
        names = [s.name for s in stages]
        if names != list(self._rows):
            self._build_stage_table(stages)
        self._refresh_target_choice()
        self.stage_table.setEnabled(not self.custom_pipeline.isChecked())
        enabled = self._enabled_stage_names()
        current = self.stage_combo.currentText()
        self.stage_combo.clear()
        self.stage_combo.addItems(enabled)
        if current in enabled:
            self.stage_combo.setCurrentText(current)
        self.run_button.setEnabled(bool(names))
        self.stage_button.setEnabled(bool(enabled))

    def _enabled_stage_names(self) -> list[str]:
        if self.custom_pipeline.isChecked() and self._state and self._state.pipeline:
            return [s.name for s in self._state.pipeline.stages]
        return [name for name, row in self._rows.items() if row.enabled.isChecked()]

    def _build_stage_table(self, stages: tuple[Stage, ...]) -> None:
        self._rows = {}
        self.stage_table.setRowCount(len(stages))
        for index, stage in enumerate(stages):
            row = self._make_row(stage)
            self._rows[stage.name] = row
            self.stage_table.setCellWidget(index, 0, row.enabled)
            self.stage_table.setItem(index, 1, QTableWidgetItem(stage.name))
            optimizer: QWidget
            if row.method is not None:
                optimizer = row.method
            elif row.target is not None:
                optimizer = row.target
            else:
                optimizer = QLabel(
                    "Bragg law" if isinstance(stage.optimizer, BraggOutline) else "grid scan"
                )
            self.stage_table.setCellWidget(index, 2, optimizer)
            self.stage_table.setCellWidget(index, 3, row.evaluations)
            self.stage_table.setCellWidget(index, 4, row.points)
        self.stage_table.resizeRowsToContents()
        self._fit_height(self.stage_table)

    def _target_labels(self) -> list[str]:
        targets = self._state.targets if self._state is not None else None
        return [t.label(i) for i, t in enumerate(targets.targets)] if targets else []

    def _refresh_target_choice(self) -> None:
        """Fill the target selector of the Bragg-law stage; keep the selection if possible."""
        labels = self._target_labels()
        for row in self._rows.values():
            combo = row.target
            if combo is None:
                continue
            current = combo.currentData()
            items: list[tuple[str, int | None]] = [("Bragg law: all targets", None)]
            items += [(f"Bragg law: {label}", i) for i, label in enumerate(labels)]
            if [combo.itemText(i) for i in range(combo.count())] != [t for t, _ in items]:
                combo.blockSignals(True)
                combo.clear()
                for text, data in items:
                    combo.addItem(text, data)
                index = combo.findData(current)
                combo.setCurrentIndex(max(index, 0))
                combo.blockSignals(False)
            combo.setEnabled(len(labels) > 1)

    def _make_row(self, stage: Stage) -> _StageRow:
        enabled = QCheckBox()
        enabled.setChecked(True)
        enabled.setToolTip(f"Include the {stage.name} stage in the pipeline.")
        enabled.toggled.connect(lambda _: self._update_stages())
        method: QComboBox | None = None
        target: QComboBox | None = None
        evaluations = QSpinBox()
        evaluations.setRange(0, 10_000_000)
        evaluations.setSpecialValueText("default")
        evaluations.setToolTip("Maximum number of objective evaluations; 0 keeps the default.")
        points = QSpinBox()
        points.setRange(0, 100_000)
        points.setSpecialValueText("default")
        points.setToolTip(
            "Grid points per target in this stage; 0 keeps the default (see 'Coarse grid "
            "points' and the full grid of the project)."
        )
        if isinstance(stage.optimizer, LmfitOptimizer):
            method = QComboBox()
            methods: tuple[str, ...] = LMFIT_METHODS
            if stage.optimizer.method not in methods:
                methods = (stage.optimizer.method, *methods)
            for name in methods:
                method.addItem(name, name)
            method.setCurrentIndex(method.findData(stage.optimizer.method))
            method.setToolTip("lmfit minimization method of this stage.")
        else:
            evaluations.setEnabled(False)
            evaluations.setToolTip("This stage has no evaluation limit.")
            if isinstance(stage.optimizer, GridScan):
                points.setToolTip(
                    "Grid points per target during the scan; 0 keeps the default (coarse grid)."
                )
            else:
                points.setEnabled(False)
                points.setToolTip("The analytic estimate does not use a grid.")
        if isinstance(stage.optimizer, BraggOutline):
            target = QComboBox()
            target.addItem("Bragg law: all targets", None)
            target.setToolTip(
                "Target the Bragg-law estimate is based on. With several targets, 'all targets' "
                "pools them, which is only meaningful for similar energies and angles."
            )
        return _StageRow(stage.name, enabled, method, evaluations, points, target)

    def run(self, stage: str | None = None) -> None:
        """Request a pipeline run, or a run of a single stage.

        After the run, the job evaluates the stage results with the full problem for the results
        table and emits `pipelineFinished`; only the final design is stored as a candidate.

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
        if not pipeline.stages:
            self.errorOccurred.emit("Cannot optimize: no stage is enabled")
            return
        name = f"Stage {stage}" if stage else "Pipeline"
        candidate_name = self.run_name.text().strip() or None

        def compute(
            progress: Callable[[Any], None], cancel: CancellationToken
        ) -> tuple[PipelineResult, tuple[Evaluation, ...]]:
            result = pipeline.run(problem, events=progress, cancel=cancel)
            return result, evaluate_stages(problem, result)

        def finished(outcome: tuple[PipelineResult, tuple[Evaluation, ...]]) -> None:
            result, evaluations = outcome
            self._show_results(result, evaluations)
            self.pipelineFinished.emit(problem, result, candidate_name)

        self.jobRequested.emit(Job(name, compute, finished))

    def _show_results(self, result: PipelineResult, evaluations: tuple[Evaluation, ...]) -> None:
        table = self.results_table
        table.setRowCount(len(result.stages))
        for index, (outcome, full) in enumerate(zip(result.stages, evaluations, strict=True)):
            cells = [
                outcome.stage.name,
                str(outcome.result.evaluations),
                f"{outcome.result.best.objective:.5g}",
                f"{full.merit:.5g}",
                f"{outcome.result.elapsed:.2f}",
            ]
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if column > 0:
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    )
                table.setItem(index, column, item)
        table.resizeRowsToContents()
        self._fit_height(table)
