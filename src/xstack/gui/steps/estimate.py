"""Workflow page for analytic start values from the Bragg law."""

import math

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QRadioButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from xstack.gui.steps.base import StepPage
from xstack.optim.optimizers import OptimizationResult
from xstack.optim.problem import DesignProblem
from xstack.optim.stages import BraggOutline
from xstack.parameters import Parameter, ParameterSet
from xstack.project import ProjectState
from xstack.project.commands import set_parameters, store_results
from xstack.structure.coatings.graded import DepthGraded
from xstack.structure.coatings.periodic import Periodic

__all__ = ["EstimatePage"]

_KINDS = {"plain": "plain Bragg", "refraction": "refraction-corrected"}
_COLUMNS = ("Parameter", "Unit", "Input", "Plain Bragg", "With refraction")


def _format(value: float) -> str:
    return f"{value:.4g}"


class EstimatePage(StepPage):
    """Estimate start values of the coating from the Bragg law.

    Both the plain and the refraction-corrected estimate are computed and shown next to the
    current input values. One of them can be applied as start values of the input design or
    stored as a candidate.

    Attributes:
        estimates: Results of the last estimate by kind (``"plain"`` and ``"refraction"``);
            empty before an estimate has been made.

    Args:
        parent: Parent widget.
    """

    title = "Estimate"
    help_text = (
        "<b>Analytic start values.</b> This step derives start values for the coating from the "
        "Bragg law: for periodic multilayers the period for the centre of the band, &Gamma; "
        "from the Vinogradov&ndash;Zeldovich optimum and a range for <i>N</i> around the "
        "saturation estimate, for depth-graded multilayers <i>d</i><sub>max</sub> and "
        "<i>d</i><sub>min</sub> for the ends of the band. The plain Bragg law ignores "
        "refraction; the refraction-corrected version includes the peak shift caused by the "
        "mean refractive index, which matters at small angles and low energies. Only free "
        "parameters are estimated. Applying an estimate sets the <b>start values</b> of the "
        "input design (and optionally narrows the bounds) which the Scan and Optimize steps "
        "use. Other coating types are not estimated."
    )

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state: ProjectState | None = None
        self._problem: DesignProblem | None = None
        self.estimates: dict[str, OptimizationResult] = {}
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<b>Estimate</b> start values from the Bragg law"))

        self.unsupported_label = QLabel(
            "Estimates are available for periodic and depth-graded multilayers. "
            "The coating of the current design is of another type."
        )
        self.unsupported_label.setWordWrap(True)
        self.unsupported_label.setVisible(False)
        layout.addWidget(self.unsupported_label)

        self.estimate_button = QPushButton("Estimate")
        self.estimate_button.setToolTip(
            "Compute the plain and the refraction-corrected Bragg estimate for the free "
            "parameters of the current design."
        )
        self.estimate_button.clicked.connect(self.estimate)
        row = QHBoxLayout()
        row.addWidget(self.estimate_button)
        row.addStretch()
        layout.addLayout(row)

        self.table = QTableWidget(0, len(_COLUMNS))
        self.table.setHorizontalHeaderLabels(list(_COLUMNS))
        self.table.verticalHeader().setVisible(False)
        self.table.setShowGrid(False)
        self.table.setFrameShape(QTableWidget.Shape.NoFrame)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(True)
        self.table.setVisible(False)
        layout.addWidget(self.table)

        self.merit_labels = {kind: QLabel() for kind in _KINDS}
        for label in self.merit_labels.values():
            label.setWordWrap(True)
            layout.addWidget(label)

        self.plain_radio = QRadioButton("Plain Bragg")
        self.refraction_radio = QRadioButton("With refraction")
        self.refraction_radio.setChecked(True)
        self.plain_radio.setToolTip("Apply the estimate from the plain Bragg law.")
        self.refraction_radio.setToolTip(
            "Apply the estimate that includes the peak shift from the mean refractive index."
        )
        choice = QHBoxLayout()
        choice.addWidget(QLabel("Use estimate:"))
        choice.addWidget(self.plain_radio)
        choice.addWidget(self.refraction_radio)
        choice.addStretch()
        layout.addLayout(choice)

        self.narrow_check = QCheckBox("Also narrow the parameter bounds to the estimate")
        self.narrow_check.setToolTip(
            "Replace the bounds of the estimated parameters by the narrowed bounds of the "
            "estimate. Otherwise the bounds of the input design are kept."
        )
        layout.addWidget(self.narrow_check)

        self.apply_button = QPushButton("Use as start values")
        self.apply_button.setToolTip("Set the chosen estimate as start values of the input design.")
        self.apply_button.clicked.connect(self.apply)
        self.store_button = QPushButton("Store as candidate")
        self.store_button.setToolTip(
            "Store the estimated design as a candidate without changing the input design."
        )
        self.store_button.clicked.connect(self.store)
        buttons = QHBoxLayout()
        buttons.addWidget(self.apply_button)
        buttons.addWidget(self.store_button)
        buttons.addStretch()
        layout.addLayout(buttons)

        self.confirmation = QLabel()
        self.confirmation.setWordWrap(True)
        self.confirmation.setObjectName("confirmation")
        self.confirmation.setVisible(False)
        layout.addWidget(self.confirmation)
        layout.addStretch()
        self._refresh()

    def set_state(self, state: ProjectState) -> None:
        """Show the free parameters of the current design next to the last estimates.

        Estimates are discarded if the design has no supported coating.

        Args:
            state: The current project state.
        """
        self._state = state
        if not self._supported():
            self.estimates = {}
        self._refresh()

    def estimate(self) -> None:
        """Compute the plain and the refraction-corrected estimate.

        Emits `errorOccurred` if the project does not define a complete design problem or the
        estimate fails.
        """
        if self._state is None or not self._supported():
            return
        try:
            problem = self._state.problem()
            estimates = {
                "plain": BraggOutline(refraction=False).run(problem),
                "refraction": BraggOutline(refraction=True).run(problem),
            }
        except ValueError as exc:
            self.errorOccurred.emit(f"Cannot estimate: {exc}")
            return
        self._problem = problem
        self.estimates = estimates
        self.confirmation.setVisible(False)
        self._refresh()

    def apply(self) -> None:
        """Set the chosen estimate as start values of the input design.

        The bounds are narrowed to the estimate if the corresponding option is checked;
        otherwise they are widened only where the estimate lies outside them.
        """
        kind = self._chosen()
        if self._state is None or self._state.mirror is None or kind not in self.estimates:
            return
        params = self._state.mirror.parameters
        new = self._apply_to(params, self.estimates[kind].parameters)
        shown = ", ".join(
            f"{name} = {_format(new[name].value)}{' ' + new[name].unit if new[name].unit else ''}"
            for name in params.free_names()
        )
        text = f"Start values set from the {_KINDS[kind]} estimate: {shown}"
        self.confirmation.setText(text)
        self.confirmation.setVisible(True)
        self.commandRequested.emit(set_parameters(new, text))

    def store(self) -> None:
        """Store the chosen estimated design as a candidate."""
        kind = self._chosen()
        if self._problem is None or kind not in self.estimates:
            return
        result = self.estimates[kind]
        entry = (f"estimate ({kind})", result.parameters, result.best, "estimate")
        self.commandRequested.emit(store_results(self._problem, [entry]))

    def _chosen(self) -> str:
        return "plain" if self.plain_radio.isChecked() else "refraction"

    def _apply_to(self, params: ParameterSet, estimated: ParameterSet) -> ParameterSet:
        narrow = self.narrow_check.isChecked()
        for name in params.free_names():
            if name not in estimated:
                continue
            p, e = params[name], estimated[name]
            value = float(round(e.value)) if p.integer else e.value
            lo, hi = (e.min, e.max) if narrow else (p.min, p.max)
            params = params.replace(name, value=value, min=min(lo, value), max=max(hi, value))
        return params

    def _supported(self) -> bool:
        mirror = self._state.mirror if self._state is not None else None
        return mirror is not None and isinstance(mirror.coating, Periodic | DepthGraded)

    def _refresh(self) -> None:
        supported = self._supported() or self._state is None
        self.unsupported_label.setVisible(not supported)
        self.estimate_button.setEnabled(supported)
        has = bool(self.estimates)
        self.table.setVisible(has)
        for widget in (
            self.plain_radio,
            self.refraction_radio,
            self.narrow_check,
            self.apply_button,
            self.store_button,
        ):
            widget.setEnabled(has)
        for kind, label in self.merit_labels.items():
            label.setVisible(has)
            if has:
                best = self.estimates[kind].best
                feasibility = "feasible" if best.feasible else "infeasible"
                label.setText(
                    f"{_KINDS[kind].capitalize()} estimate: figure of merit "
                    f"{_format(best.merit)} ({feasibility})"
                )
        if has:
            self._fill_table()

    def _fill_table(self) -> None:
        params = (
            self._state.mirror.parameters
            if self._state is not None and self._state.mirror is not None
            else ParameterSet()
        )
        names = [
            n
            for n in params.free_names()
            if all(n in r.parameters for r in self.estimates.values())
        ]
        self.table.setRowCount(len(names))
        for row, name in enumerate(names):
            p = params[name]
            cells = [name, p.unit, _format(p.value)]
            tips: list[str | None] = [None, None, _bounds_tip(p)]
            for kind in _KINDS:
                e = self.estimates[kind].parameters[name]
                cells.append(_format(e.value))
                tips.append(_bounds_tip(e))
            for column, (text, tip) in enumerate(zip(cells, tips, strict=True)):
                item = QTableWidgetItem(text)
                if column >= 2:
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    )
                if tip is not None:
                    item.setToolTip(tip)
                self.table.setItem(row, column, item)
        self.table.resizeRowsToContents()
        height = self.table.horizontalHeader().height() + sum(
            self.table.rowHeight(r) for r in range(len(names))
        )
        self.table.setFixedHeight(height + 4)


def _bounds_tip(p: Parameter) -> str:
    lo = "-inf" if math.isinf(p.min) else _format(p.min)
    hi = "inf" if math.isinf(p.max) else _format(p.max)
    return f"bounds: {lo} – {hi}"
