"""Workflow page for the trade-off between figure of merit and coating time."""

import math
from collections.abc import Callable
from typing import Any, ClassVar

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from xstack.gui.plots import MERIT_QUANTITY, quantity_value
from xstack.gui.preview import Preview
from xstack.gui.steps.base import StepPage
from xstack.gui.workers import Job
from xstack.optim.events import CancellationToken
from xstack.optim.pipeline import Pipeline
from xstack.optim.problem import DesignProblem
from xstack.optim.tradeoff import (
    TradeoffMode,
    TradeoffPoint,
    TradeoffResult,
    default_limits,
    time_tradeoff,
)
from xstack.project import ProjectState
from xstack.project.commands import store_results, store_tradeoff

__all__ = ["TradeoffPage", "limit_label"]

MODES: dict[str, tuple[TradeoffMode, str]] = {
    "Continuation (fast)": (
        "continuation",
        "Optimize the longest limit with the full pipeline. Every shorter limit starts from the "
        "design of the previous one, scans the free integer parameters (e.g. the number of "
        "bilayers) and refines locally. Fast, but may miss a different design family at short "
        "times; limits that end infeasible fall back to the full pipeline.",
    ),
    "Thorough (full pipeline per limit)": (
        "thorough",
        "Run the full optimization pipeline for every limit, starting from the design of the "
        "previous limit. Slower by about the number of limits, but better at finding "
        "different design families at short times.",
    ),
}
"""Trade-off modes by label: (mode, tooltip)."""

_COLUMNS = (
    "Limit (h)",
    "Coating time (h)",
    "Figure of merit",
    "",
    "Gain (FOM / h)",
    "Notes",
    "Free parameters",
)
_QUANTITY_COLUMN = 3


def limit_label(limit: float, unit: bool = False) -> str:
    """Return a time limit for display.

    Args:
        limit: Time limit in s; ``inf`` for no limit.
        unit: Append the unit ``h`` to finite limits.

    Returns:
        ``"no limit"`` or the limit in h.
    """
    if math.isinf(limit):
        return "no limit"
    return f"{limit / 3600.0:.3g}" + (" h" if unit else "")


class TradeoffPage(StepPage):
    """Sweep of the maximum coating time and the best design for each limit.

    Signals:
        quantityChanged(): The plotted quantity or the shown trade-off changed.
        pointSelected(object): The selected point changed; its index in
            `TradeoffResult.points`, or ``None`` if the selection was cleared.
        tradeoffFinished(object): A trade-off run finished and was handed over for storing; the
            `TradeoffResult`.

    Attributes:
        pipeline_provider: Function that returns the pipeline for full optimizations of a
            problem; ``None`` uses `Pipeline.default`.

    Args:
        parent: Parent widget.
    """

    title = "Trade-off"
    help_text: ClassVar[str] = (
        "<h3>Trade-off: figure of merit versus coating time</h3>"
        "<p>The sweep optimizes the design for a series of <i>maximum coating times</i> (an "
        "&epsilon;-constraint sweep). Every point is the best design found within its limit, "
        "so the points approximate the best achievable figure of merit for each coating time. "
        "It needs a deposition process (step Process).</p><ul>"
        "<li><b>Marginal gain</b>: improvement of the figure of merit per additional hour "
        "between neighbouring points (lower plot, table column). Where it becomes small, more "
        "coating time hardly pays.</li>"
        "<li><b>Knee</b>: the point farthest from the straight line between the shortest and "
        "the longest design, a good compromise between time and performance.</li>"
        "<li><b>Infeasible</b> (hollow marker): the limit is below the minimum achievable "
        "time, e.g. because the per-layer overhead of a fixed number of layers already exceeds "
        "it. Such points are not part of the front.</li>"
        "<li><b>Replaced</b> (diamond): the design of a shorter limit also satisfies this "
        "limit and is better than the one found for it, so it was taken over.</li></ul>"
        "<p>Select a table row or click a marker to see that design's reflectivity (dashed "
        "curve in the Reflectivity tab) and to store it as a candidate. The result is saved "
        "with the project.</p>"
    )
    quantityChanged = Signal()
    pointSelected = Signal(object)
    tradeoffFinished = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.pipeline_provider: Callable[[DesignProblem], Pipeline] | None = None
        self._state: ProjectState | None = None
        self._result: TradeoffResult | None = None
        self._edited_explicit = False
        layout = QVBoxLayout(self)
        layout.addWidget(self._build_settings_box())
        self.message = QLabel()
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        buttons = QHBoxLayout()
        self.run_button = QPushButton("Run trade-off")
        self.run_button.setToolTip(
            "Optimize the design for every coating-time limit. Stop with the Stop button of the "
            "toolbar; the points found so far are kept."
        )
        self.run_button.clicked.connect(lambda: self.run())
        buttons.addWidget(self.run_button)
        buttons.addStretch()
        layout.addLayout(buttons)
        layout.addWidget(self._build_results_box())
        layout.addStretch()
        self._update_mode_widgets()
        self._update_enabled()

    def _build_settings_box(self) -> QGroupBox:
        box = QGroupBox("Time limits")
        form = QFormLayout(box)
        self.auto_radio = QRadioButton("Automatic range (below the unconstrained optimum)")
        self.auto_radio.setToolTip(
            "First optimize without a time limit, then use limits between a fraction of the "
            "coating time of that design and just below it, spaced evenly on a log scale."
        )
        self.auto_radio.setChecked(True)
        self.explicit_radio = QRadioButton("Explicit range")
        self.explicit_radio.setToolTip(
            "Use limits between the given times, spaced evenly on a log scale. The design "
            "without a time limit is not computed."
        )
        self.auto_radio.toggled.connect(lambda _: self._update_mode_widgets())
        form.addRow(self.auto_radio)
        self.lowest = QDoubleSpinBox()
        self.lowest.setRange(1.0, 99.0)
        self.lowest.setDecimals(0)
        self.lowest.setValue(20.0)
        self.lowest.setSuffix(" %")
        self.lowest.setToolTip(
            "Shortest limit as a percentage of the coating time of the unconstrained design."
        )
        form.addRow("Shortest limit", self.lowest)
        form.addRow(self.explicit_radio)
        self.min_hours = self._hours_spin("Shortest limit in hours.", 1.0)
        self.max_hours = self._hours_spin("Longest limit in hours.", 5.0)
        form.addRow("Shortest limit (h)", self.min_hours)
        form.addRow("Longest limit (h)", self.max_hours)
        self.count = QSpinBox()
        self.count.setRange(1, 50)
        self.count.setValue(8)
        self.count.setToolTip(
            "Number of time limits. With the automatic range, the design without a limit comes "
            "in addition."
        )
        form.addRow("Number of limits", self.count)
        self.mode_combo = QComboBox()
        for label, (_, tooltip) in MODES.items():
            self.mode_combo.addItem(label)
            self.mode_combo.setItemData(
                self.mode_combo.count() - 1, tooltip, Qt.ItemDataRole.ToolTipRole
            )
        self.mode_combo.setToolTip("How the design for each limit is found; see the entries.")
        form.addRow("Mode", self.mode_combo)
        return box

    def _hours_spin(self, tooltip: str, value: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(0.001, 100000.0)
        spin.setDecimals(3)
        spin.setValue(value)
        spin.setSuffix(" h")
        spin.setToolTip(tooltip)
        spin.valueChanged.connect(lambda _: self._explicit_edited())
        return spin

    def _build_results_box(self) -> QGroupBox:
        box = QGroupBox("Result")
        layout = QVBoxLayout(box)
        row = QHBoxLayout()
        row.addWidget(QLabel("Plot"))
        self.quantity_combo = QComboBox()
        self.quantity_combo.addItem(MERIT_QUANTITY, MERIT_QUANTITY)
        self.quantity_combo.setToolTip(
            "Quantity plotted against the coating time: the figure of merit or one of the "
            "quantities that it rates. The reference designs are only shown for the figure of "
            "merit."
        )
        self.quantity_combo.activated.connect(lambda _: self._quantity_activated())
        row.addWidget(self.quantity_combo, 1)
        layout.addLayout(row)
        self.table = QTableWidget(0, len(_COLUMNS))
        self.table.setHorizontalHeaderLabels(list(_COLUMNS))
        self.table.setShowGrid(False)
        self.table.setFrameShape(QTableWidget.Shape.NoFrame)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setMinimumHeight(220)
        self.table.setToolTip(
            "One row per time limit. Select a row to show the design in the plots. The gain is "
            "the improvement of the figure of merit per additional hour compared with the "
            "previous feasible point."
        )
        self.table.itemSelectionChanged.connect(self._selection_changed)
        layout.addWidget(self.table)
        self.summary = QLabel("No trade-off yet.")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.store_button = QPushButton("Store as candidate")
        self.store_button.setToolTip(
            "Store the design of the selected row as a candidate and select it, so that all plots "
            "and exports can show it."
        )
        self.store_button.clicked.connect(lambda: self.store_selected())
        self.store_button.setEnabled(False)
        layout.addWidget(self.store_button)
        return box

    def _update_mode_widgets(self) -> None:
        automatic = self.auto_radio.isChecked()
        self.lowest.setEnabled(automatic)
        self.min_hours.setEnabled(not automatic)
        self.max_hours.setEnabled(not automatic)

    def _explicit_edited(self) -> None:
        self._edited_explicit = True

    def set_state(self, state: ProjectState) -> None:
        """Show the trade-off of a project state.

        Args:
            state: The current project state.
        """
        self._state = state
        self._update_enabled()
        if state.tradeoff is not self._result:
            self._show_result(state.tradeoff)

    def set_preview(self, preview: Preview) -> None:
        """Suggest explicit limits around the coating time of the current design.

        Args:
            preview: The latest preview; used until the limits are edited.
        """
        evaluation = preview.evaluation
        if self._edited_explicit or evaluation is None or not evaluation.coating_time:
            return
        hours = evaluation.coating_time / 3600.0
        for spin, value in ((self.max_hours, hours), (self.min_hours, hours * 0.2)):
            spin.blockSignals(True)
            spin.setValue(max(value, spin.minimum()))
            spin.blockSignals(False)

    def _update_enabled(self) -> None:
        state = self._state
        if state is None:
            reason = "No project."
        elif state.process is None:
            reason = (
                "The trade-off needs a deposition process to estimate the coating time. "
                "Define one in the Process step."
            )
        elif state.mirror is None or state.targets is None or state.merit is None:
            reason = "The design, targets and figure of merit must be defined first."
        else:
            reason = ""
        self.message.setText(reason)
        self.message.setVisible(bool(reason))
        self.run_button.setEnabled(not reason)

    def _show_result(self, result: TradeoffResult | None) -> None:
        self._result = result
        self.table.clearSelection()
        self._refresh_quantities()
        self._fill_table()
        self.quantityChanged.emit()

    def _refresh_quantities(self) -> None:
        labels: list[str] = []
        for point in self._result.points if self._result is not None else ():
            labels += [name for name, _ in point.quantities if name not in labels]
        current = self.quantity_combo.currentData()
        self.quantity_combo.clear()
        self.quantity_combo.addItem(MERIT_QUANTITY, MERIT_QUANTITY)
        for label in labels:
            self.quantity_combo.addItem(label, label)
        index = self.quantity_combo.findData(current)
        self.quantity_combo.setCurrentIndex(max(index, 0))

    def quantity(self) -> str:
        """Return the quantity to plot.

        Returns:
            `MERIT_QUANTITY` or the label of a quantity of the points.
        """
        return str(self.quantity_combo.currentData() or MERIT_QUANTITY)

    def result(self) -> TradeoffResult | None:
        """Return the trade-off shown.

        Returns:
            The trade-off of the project, or ``None``.
        """
        return self._result

    def _quantity_activated(self) -> None:
        self._fill_table()
        self.quantityChanged.emit()

    def _notes(self, point: TradeoffPoint, knee: TradeoffPoint | None) -> str:
        notes = []
        if not point.feasible:
            notes.append("infeasible")
        if point.replaced:
            notes.append("replaced by shorter-limit design")
        if knee is not None and point is knee:
            notes.append("knee")
        return ", ".join(notes)

    def _fill_table(self) -> None:
        selected = self.selected_index()
        table = self.table
        result = self._result
        quantity = self.quantity()
        header = table.horizontalHeaderItem(_QUANTITY_COLUMN)
        if header is not None:
            header.setText(quantity)
        table.setColumnHidden(_QUANTITY_COLUMN, quantity == MERIT_QUANTITY)
        table.blockSignals(True)
        table.setRowCount(0)
        if result is None:
            table.blockSignals(False)
            self.summary.setText("No trade-off yet.")
            self.store_button.setEnabled(False)
            return
        points = result.points
        feasible = result.feasible_points()
        gains = dict(zip((id(p) for p in feasible[1:]), result.marginal_gains(), strict=True))
        knee = result.knee()
        table.setRowCount(len(points))
        for row, point in enumerate(points):
            gain = gains.get(id(point))
            cells = [
                limit_label(point.limit),
                f"{point.coating_time / 3600.0:.3g}",
                f"{point.merit:.5g}",
                f"{quantity_value(point, quantity):.5g}",
                "" if gain is None else f"{gain:.3g}",
                self._notes(point, knee),
                self._parameter_text(point),
            ]
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if column < 5:
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    )
                if not point.feasible:
                    item.setForeground(self.palette().placeholderText())
                table.setItem(row, column, item)
        table.blockSignals(False)
        text = f"{len(points)} points, mode: {result.mode}."
        if knee is not None:
            text += f" Knee at {knee.coating_time / 3600.0:.3g} h."
        if result.cancelled:
            text += " The sweep was cancelled; later limits are missing."
        self.summary.setText(text)
        if selected is not None and selected < len(points):
            table.selectRow(selected)
        self.store_button.setEnabled(self.selected_index() is not None)

    @staticmethod
    def _parameter_text(point: TradeoffPoint) -> str:
        params = point.parameters
        return ", ".join(f"{name}={params[name].value:.4g}" for name in params.free_names())

    def selected_index(self) -> int | None:
        """Return the selected point.

        Returns:
            Index in `TradeoffResult.points`, or ``None`` if no row is selected.
        """
        rows = self.table.selectionModel().selectedRows() if self.table.rowCount() else []
        return rows[0].row() if rows else None

    def select_point(self, index: int | None) -> None:
        """Select a row of the table.

        Args:
            index: Index in `TradeoffResult.points`, or ``None`` to clear the selection.
        """
        if index is None:
            self.table.clearSelection()
        elif 0 <= index < self.table.rowCount():
            self.table.selectRow(index)

    def _selection_changed(self) -> None:
        index = self.selected_index()
        self.store_button.setEnabled(index is not None)
        self.pointSelected.emit(index)

    def store_selected(self) -> bool:
        """Store the design of the selected row as a candidate.

        The candidate is named after the limit, has the source ``"tradeoff"`` and is selected.

        Returns:
            Whether a command was requested.
        """
        index = self.selected_index()
        state, result = self._state, self._result
        if index is None or state is None or result is None or state.mirror is None:
            return False
        point = result.points[index]
        try:
            problem = state.problem().with_parameters(point.parameters)
            evaluation = problem.evaluate()
        except ValueError as exc:
            self.errorOccurred.emit(f"Cannot store the design: {exc}")
            return False
        name = f"trade-off {limit_label(point.limit, unit=True)}"
        command = store_results(state.problem(), [(name, point.parameters, evaluation, "tradeoff")])
        self.commandRequested.emit(command)
        return True

    def limits(self) -> tuple[float, ...] | None:
        """Return the time limits entered.

        Returns:
            The limits in s in increasing order for an explicit range, ``None`` for the
            automatic range.

        Raises:
            ValueError: If the explicit range is invalid.
        """
        if self.auto_radio.isChecked():
            return None
        low, high = self.min_hours.value(), self.max_hours.value()
        if self.count.value() > 1 and not low < high:
            raise ValueError("the shortest limit must be below the longest")
        return default_limits(high * 3600.0, self.count.value(), min(low / high, 1.0))

    def run(self) -> None:
        """Request the trade-off as a background job.

        When it finishes, the result is stored in the project with an undoable command.
        """
        state = self._state
        if state is None:
            return
        try:
            problem = state.problem()
            if problem.process is None:
                raise ValueError("the project has no deposition process")
            limits = self.limits()
            pipeline = self.pipeline_provider(problem) if self.pipeline_provider else None
        except ValueError as exc:
            self.errorOccurred.emit(f"Cannot run the trade-off: {exc}")
            return
        mode = MODES[self.mode_combo.currentText()][0]
        points, lowest = self.count.value(), self.lowest.value() / 100.0

        def compute(progress: Callable[[Any], None], cancel: CancellationToken) -> TradeoffResult:
            return time_tradeoff(
                problem,
                limits,
                points=points,
                lowest=lowest,
                mode=mode,
                pipeline=pipeline,
                events=progress,
                cancel=cancel,
            )

        def finished(result: TradeoffResult) -> None:
            self.commandRequested.emit(store_tradeoff(result))
            self.tradeoffFinished.emit(result)

        self.jobRequested.emit(Job("Trade-off", compute, finished))
