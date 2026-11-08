"""Workflow page for the residual weighting, the optimizer and running the fit."""

import dataclasses
from typing import Any, ClassVar

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from grazing.core.optim.optimizers import GLOBAL_METHODS, LmfitOptimizer, Optimizer
from grazing.fit.problem import Weighting
from grazing.fit.project import FitState, run_fit, set_optimizers, set_weighting, store_result
from grazing.fit.results import FitResult
from grazing.gui.common.style import set_variant
from grazing.gui.common.workers import Job
from grazing.gui.fit.steps.base import StepPage

__all__ = ["GLOBAL_EVALUATIONS", "LMFIT_METHODS", "STRATEGIES", "FitPage"]

LMFIT_METHODS = (
    "least_squares",
    "leastsq",
    "nelder",
    "lbfgsb",
    "powell",
    "differential_evolution",
    "dual_annealing",
)
"""Optimization methods offered in the stage table."""

STRATEGIES = ("local", "global", "custom")
"""Optimizer strategies: local refinement, global search then local refinement, custom stages."""

GLOBAL_EVALUATIONS = 4000
"""Evaluation limit of the global stage of the global strategy."""

_STRATEGY_TEXT = {
    "local": "Local refinement (least squares)",
    "global": "Global search, then local refinement",
    "custom": "Custom",
}

_WEIGHTING_TEXT: dict[Weighting, tuple[str, str]] = {
    "chi2": (
        "χ² (measured uncertainties)",
        "(R_model − R_meas)/σ · needs uncertainties in all datasets",
    ),
    "log": ("Logarithmic", "log₁₀ R_model − log₁₀ R_meas · equal weight to all decades"),
    "linear": ("Linear", "R_model − R_meas · dominated by the high-reflectivity region"),
}

type _Row = tuple[bool, str, int | None]
"""Stage table row: enabled, method, evaluation limit."""


class FitPage(StepPage):
    """Weighting of the residuals, optimizer stages, running fits and the last fit.

    Signals:
        fitFinished(FitResult): A fit started from this page finished (after its result was
            requested to be stored).

    Attributes:
        weighting_buttons: Radio buttons by weighting.
        strategy_combo: Optimizer strategy.
        stage_table: Table of the optimizer stages.
        seed: Random seed of the global search.
        population: Population factor of the global search.
        run_button: Runs the configured optimizers.
        local_button: Runs a local refinement only.
        last_label: Summary of the most recently stored fit.
    """

    fitFinished = Signal(object)

    title = "Fit"
    help_text: ClassVar[str] = (
        "Choose the <b>weighting</b> of the residuals and the optimizer. <b>Local</b> refinement "
        "is fast but needs good start values; <b>global then local</b> searches the bounds first. "
        "The plots update live while the fit runs; Stop keeps the best values so far."
    )

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state: FitState | None = None
        self._loading = False
        self._foreign = False
        layout = QVBoxLayout(self)
        layout.addWidget(self._build_weighting_box())
        layout.addWidget(self._build_optimizer_box())
        layout.addWidget(self._build_last_box())
        layout.addStretch()
        self._set_rows(self._preset_rows("local"))

    # ------------------------------------------------------------------ construction
    def _build_weighting_box(self) -> QGroupBox:
        box = QGroupBox("Weighting of the residuals")
        layout = QVBoxLayout(box)
        self._group = QButtonGroup(box)
        self.weighting_buttons: dict[Weighting, QRadioButton] = {}
        for weighting, (text, detail) in _WEIGHTING_TEXT.items():
            button = QRadioButton(text)
            button.setProperty("weighting", weighting)
            self._group.addButton(button)
            self.weighting_buttons[weighting] = button
            button.toggled.connect(self._weighting_toggled)
            layout.addWidget(button)
            note = QLabel(detail)
            note.setProperty("role", "muted")
            note.setContentsMargins(24, 0, 0, 4)
            layout.addWidget(note)
        return box

    def _build_optimizer_box(self) -> QGroupBox:
        box = QGroupBox("Optimizer")
        form = QFormLayout(box)
        self.strategy_combo = QComboBox()
        for key in STRATEGIES:
            self.strategy_combo.addItem(_STRATEGY_TEXT[key], key)
        self.strategy_combo.setToolTip("Preset stages of the fit, or custom stages.")
        self.strategy_combo.currentIndexChanged.connect(self._strategy_changed)
        form.addRow("Strategy", self.strategy_combo)

        self.stage_table = QTableWidget(0, 4)
        self.stage_table.setHorizontalHeaderLabels(
            ["Enabled", "Stage", "Method", "Max evaluations"]
        )
        self.stage_table.setShowGrid(False)
        self.stage_table.setFrameShape(QTableWidget.Shape.NoFrame)
        self.stage_table.verticalHeader().setVisible(False)
        self.stage_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.stage_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.stage_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self.stage_table.horizontalHeader().setStretchLastSection(True)
        self.stage_table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.stage_table.setToolTip(
            "Stages run in order, each from the best values of the previous one. "
            "An empty evaluation limit uses the method's default."
        )
        form.addRow("Stages", self.stage_table)

        self.seed = QSpinBox()
        self.seed.setRange(0, 2**31 - 1)
        self.seed.setToolTip("Seed of the random number generator of global methods.")
        self.population = QSpinBox()
        self.population.setRange(1, 1000)
        self.population.setValue(15)
        self.population.setToolTip(
            "Population per free parameter of differential evolution; larger is more thorough."
        )
        self.seed.valueChanged.connect(lambda _: self._settings_changed())
        self.population.valueChanged.connect(lambda _: self._settings_changed())
        settings = QHBoxLayout()
        settings.addWidget(self.seed)
        settings.addWidget(QLabel("Population factor"))
        settings.addWidget(self.population)
        settings.addStretch()
        form.addRow("Random seed", settings)

        self.run_button = QPushButton("Run fit")
        set_variant(self.run_button, "primary")
        self.run_button.setToolTip("Run the configured optimizer stages.")
        self.run_button.clicked.connect(lambda: self.run())
        self.local_button = QPushButton("Run local refinement only")
        self.local_button.setToolTip("Run a least-squares refinement from the current values.")
        self.local_button.clicked.connect(lambda: self.run_local())
        buttons = QHBoxLayout()
        buttons.addWidget(self.run_button)
        buttons.addWidget(self.local_button)
        buttons.addStretch()
        form.addRow("", buttons)
        return box

    def _build_last_box(self) -> QGroupBox:
        box = QGroupBox("Last fit")
        layout = QVBoxLayout(box)
        self.last_label = QLabel("No fit yet.")
        self.last_label.setWordWrap(True)
        self.last_label.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(self.last_label)
        return box

    # ------------------------------------------------------------------ stage table
    def _preset_rows(self, strategy: str) -> list[_Row]:
        return [
            (strategy == "global", "differential_evolution", GLOBAL_EVALUATIONS),
            (True, "least_squares", None),
        ]

    def _set_rows(self, rows: list[_Row]) -> None:
        was_loading = self._loading
        self._loading = True
        try:
            table = self.stage_table
            table.setRowCount(0)
            table.setRowCount(len(rows))
            for r, (enabled, method, limit) in enumerate(rows):
                check = QCheckBox()
                check.setChecked(enabled)
                check.toggled.connect(lambda _: self._table_edited())
                holder = QWidget()
                holder_layout = QHBoxLayout(holder)
                holder_layout.setContentsMargins(8, 0, 0, 0)
                holder_layout.addWidget(check)
                table.setCellWidget(r, 0, holder)
                stage = QLabel("global" if method in GLOBAL_METHODS else "local")
                stage.setContentsMargins(4, 0, 4, 0)
                table.setCellWidget(r, 1, stage)
                combo = QComboBox()
                combo.addItems(LMFIT_METHODS)
                if method not in LMFIT_METHODS:
                    combo.addItem(method)
                combo.setCurrentText(method)
                combo.currentTextChanged.connect(lambda _: self._table_edited())
                table.setCellWidget(r, 2, combo)
                edit = QLineEdit("" if limit is None else str(limit))
                edit.setPlaceholderText("auto")
                edit.editingFinished.connect(self._table_edited)
                table.setCellWidget(r, 3, edit)
            height = table.horizontalHeader().height() + 2
            for r in range(table.rowCount()):
                height += table.rowHeight(r)
            table.setFixedHeight(height)
        finally:
            self._loading = was_loading

    def _rows(self) -> list[_Row]:
        rows: list[_Row] = []
        for r in range(self.stage_table.rowCount()):
            check = self.stage_table.cellWidget(r, 0).findChild(QCheckBox)
            combo = self.stage_table.cellWidget(r, 2)
            edit = self.stage_table.cellWidget(r, 3)
            text = edit.text().strip()
            if text:
                try:
                    value = int(text)
                except ValueError:
                    raise ValueError(f"max evaluations {text!r} is not an integer") from None
                if value < 1:
                    raise ValueError("max evaluations must be positive")
                limit: int | None = value
            else:
                limit = None
            rows.append((check.isChecked(), combo.currentText(), limit))
        return rows

    def _options(self, method: str) -> tuple[tuple[str, Any], ...]:
        if method == "differential_evolution":
            return (
                ("seed", self.seed.value()),
                ("popsize", self.population.value()),
                ("polish", False),
            )
        if method == "dual_annealing":
            return (("seed", self.seed.value()),)
        return ()

    def _optimizers(self, rows: list[_Row]) -> tuple[Optimizer, ...]:
        return tuple(
            LmfitOptimizer(method, limit, self._options(method))
            for enabled, method, limit in rows
            if enabled
        )

    def _rows_of(self, optimizers: tuple[Optimizer, ...]) -> list[_Row]:
        return [(True, o.method, o.max_evaluations) for o in optimizers]  # type: ignore[attr-defined]

    def _preset(self, strategy: str) -> tuple[Optimizer, ...]:
        return self._optimizers(self._preset_rows(strategy))

    def optimizers(self) -> tuple[Optimizer, ...]:
        """Return the optimizer stages configured in the page.

        Returns:
            The enabled stages in order.

        Raises:
            ValueError: If an evaluation limit is invalid.
        """
        return self._optimizers(self._rows())

    def current_strategy(self) -> str:
        """Return the selected strategy: ``"local"``, ``"global"`` or ``"custom"``."""
        return str(self.strategy_combo.currentData())

    def _set_strategy(self, strategy: str) -> None:
        was_loading = self._loading
        self._loading = True
        try:
            self.strategy_combo.setCurrentIndex(STRATEGIES.index(strategy))
        finally:
            self._loading = was_loading

    # ------------------------------------------------------------------ events
    def _strategy_changed(self) -> None:
        if self._loading:
            return
        strategy = self.current_strategy()
        if strategy != "custom":
            self._set_rows(self._preset_rows(strategy))
        self._emit_optimizers()

    def _table_edited(self) -> None:
        if self._loading:
            return
        self._set_strategy("custom")
        self._update_stage_names()
        self._emit_optimizers()

    def _settings_changed(self) -> None:
        if self._loading:
            return
        self._emit_optimizers()

    def _update_stage_names(self) -> None:
        for r in range(self.stage_table.rowCount()):
            method = self.stage_table.cellWidget(r, 2).currentText()
            self.stage_table.cellWidget(r, 1).setText(
                "global" if method in GLOBAL_METHODS else "local"
            )

    def _emit_optimizers(self) -> None:
        if self._state is None or self._foreign:
            return
        try:
            optimizers = self.optimizers()
        except ValueError as exc:
            self.errorOccurred.emit(str(exc))
            return
        if not optimizers:
            self.errorOccurred.emit("at least one optimizer stage must be enabled")
            return
        if optimizers != self._state.optimizers:
            self.commandRequested.emit(set_optimizers(optimizers))

    def _weighting_toggled(self, checked: bool) -> None:
        if not checked or self._loading or self._state is None:
            return
        button = self._group.checkedButton()
        weighting = button.property("weighting")
        if weighting != self._state.weighting:
            self.commandRequested.emit(set_weighting(weighting))

    # ------------------------------------------------------------------ state
    def set_state(self, state: Any) -> None:
        """Show the weighting, optimizers and last fit of a fit state.

        Args:
            state: The current `FitState`.
        """
        self._state = state
        self._loading = True
        try:
            self.weighting_buttons[state.weighting].setChecked(True)
            self._update_chi2_availability(state)
            self._load_optimizers(state.optimizers)
        finally:
            self._loading = False
        self._show_last(state)

    def _update_chi2_availability(self, state: FitState) -> None:
        missing = [d.name for d in state.datasets if d.enabled and d.curve.arrays()[2] is None]
        button = self.weighting_buttons["chi2"]
        button.setEnabled(not missing)
        button.setToolTip(
            ""
            if not missing
            else "Needs uncertainties in all enabled datasets; missing in: " + ", ".join(missing)
        )

    def _load_optimizers(self, optimizers: tuple[Optimizer, ...]) -> None:
        self._foreign = not all(isinstance(o, LmfitOptimizer) for o in optimizers)
        self.stage_table.setEnabled(not self._foreign)
        if self._foreign:
            self._set_strategy("custom")
            return
        for o in optimizers:
            assert isinstance(o, LmfitOptimizer)
            if o.method == "differential_evolution":
                options = dict(o.options)
                self.seed.setValue(int(options.get("seed", self.seed.value())))
                self.population.setValue(int(options.get("popsize", self.population.value())))
                break
        for strategy in ("local", "global"):
            if optimizers == self._preset(strategy):
                self._set_strategy(strategy)
                self._set_rows(self._preset_rows(strategy))
                return
        self._set_strategy("custom")
        self._set_rows(self._rows_of(optimizers))

    def _show_last(self, state: FitState) -> None:
        if not state.results:
            self.last_label.setText("No fit yet.")
            return
        stored = state.results[-1]
        result = stored.result
        parts = [f"<b>{stored.name}</b>", f"{result.method}", f"{result.evaluations} evaluations"]
        if result.statistics is not None:
            label = "χ²_red" if result.statistics.weighting == "chi2" else "objective/dof"
            parts.insert(1, f"{label} = {result.statistics.reduced_objective:.4g}")
        parts.append(f"{result.elapsed:.1f} s")
        if result.cancelled:
            parts.append("cancelled")
        self.last_label.setText(" · ".join(parts))

    # ------------------------------------------------------------------ running
    def run(self) -> None:
        """Run the project's optimizer stages in the background."""
        self._start(None)

    def run_local(self) -> None:
        """Run a least-squares refinement only, without changing the project's optimizers."""
        self._start((LmfitOptimizer("least_squares"),))

    def _start(self, optimizers: tuple[Optimizer, ...] | None) -> None:
        state = self._state
        if state is None:
            self.errorOccurred.emit("no project")
            return
        try:
            problem = state.problem()
        except ValueError as exc:
            self.errorOccurred.emit(str(exc))
            return
        if not problem.parameters.free_names():
            self.errorOccurred.emit("no free parameters: enable Vary for some parameters first")
            return
        run_state = (
            state if optimizers is None else dataclasses.replace(state, optimizers=optimizers)
        )

        def function(progress: Any, cancel: Any) -> FitResult:
            return run_fit(run_state, progress, cancel)

        def finished(result: FitResult) -> None:
            self.commandRequested.emit(store_result(result))
            self.fitFinished.emit(result)

        self.jobRequested.emit(Job("Fit", function, finished))
