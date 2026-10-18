"""Workflow page for the deposition process."""

from PySide6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QFrame,
    QHeaderView,
    QLabel,
    QLineEdit,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from xstack.design.manufacturing.process import DepositionProcess
from xstack.design.project import ProjectState
from xstack.design.project.commands import set_process
from xstack.gui.design.preview import Preview
from xstack.gui.design.steps.base import StepPage

__all__ = ["ProcessPage"]

_DEFAULT_RATE = 0.1


class ProcessPage(StepPage):
    """Deposition rates and overheads used for coating times.

    Args:
        parent: Parent widget.
    """

    title = "Process"
    help_text = (
        "<b>Deposition process.</b> This optional step describes your coating machine so that "
        "the time needed to deposit a design can be estimated. Enter the deposition rate of "
        "every material, an optional extra time per layer for each material, an overhead per "
        "layer (such as shutter or source switching) and a fixed overhead per run. The "
        "resulting coating time is shown with the design and can be used by the optimization "
        "goal and by the exported recipe. Leave the process disabled if the time does not "
        "matter."
    )

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._process: DepositionProcess | None = None
        self._materials: list[str] = []
        self._loading = False
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<b>Deposition process</b> (optional)"))
        self.enabled = QCheckBox("Estimate coating times with a deposition process")
        self.enabled.toggled.connect(lambda _: self._apply())
        self.enabled.setToolTip("Enable the coating-time estimate and the settings below.")
        layout.addWidget(self.enabled)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Material", "Rate (nm/s)", "Extra per layer (s)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setShowGrid(False)
        self.table.setFrameShape(QFrame.Shape.NoFrame)
        self.table.setAlternatingRowColors(True)
        self.table.setToolTip(
            "Deposition rate and extra time per layer for each material of the design."
        )
        self.table.itemChanged.connect(lambda _: self._apply())
        layout.addWidget(self.table)
        form = QFormLayout()
        self.per_layer = QLineEdit()
        self.fixed = QLineEdit()
        self.per_layer.editingFinished.connect(self._apply)
        self.fixed.editingFinished.connect(self._apply)
        self.per_layer.setToolTip(
            "Time in seconds added for every layer, for example for switching."
        )
        self.fixed.setToolTip("Time in hours added once per run, for example for pump-down.")
        form.addRow("Overhead per layer (s)", self.per_layer)
        form.addRow("Fixed overhead per run (h)", self.fixed)
        layout.addLayout(form)
        self.summary = QLabel()
        layout.addWidget(self.summary)
        layout.addStretch()

    def set_state(self, state: ProjectState) -> None:
        """Show the process of a project state.

        Args:
            state: The current project state.
        """
        materials = (
            sorted({m.name for m in state.mirror.coating.materials.values()})
            if state.mirror is not None
            else []
        )
        if state.process == self._process and materials == self._materials:
            return
        self._process = state.process
        self._materials = materials
        self._load()

    def set_preview(self, preview: Preview) -> None:
        """Show the coating time of the latest evaluation.

        Args:
            preview: The latest preview.
        """
        evaluation = preview.evaluation
        if evaluation is None or evaluation.coating_time is None:
            self.summary.setText("")
        else:
            self.summary.setText(f"Coating time: {evaluation.coating_time / 3600:.3g} h")

    def _load(self) -> None:
        self._loading = True
        process = self._process
        self.enabled.setChecked(process is not None)
        rates = dict(process.rates) if process is not None else {}
        extra = dict(process.overhead_per_material) if process is not None else {}
        self.table.setRowCount(len(self._materials))
        for row, name in enumerate(self._materials):
            item = QTableWidgetItem(name)
            item.setFlags(item.flags() & ~item.flags().ItemIsEditable)
            self.table.setItem(row, 0, item)
            rate_item = QTableWidgetItem(f"{rates.get(name, _DEFAULT_RATE):g}")
            rate_item.setToolTip("Deposition rate in nm/s (editable).")
            extra_item = QTableWidgetItem(f"{extra.get(name, 0.0):g}")
            extra_item.setToolTip(
                "Extra time in seconds for each layer of this material (editable)."
            )
            self.table.setItem(row, 1, rate_item)
            self.table.setItem(row, 2, extra_item)
        self.table.resizeRowsToContents()
        self.table.setFixedHeight(
            self.table.horizontalHeader().height()
            + sum(self.table.rowHeight(r) for r in range(len(self._materials)))
            + 4
        )
        self.per_layer.setText(f"{process.overhead_per_layer:g}" if process else "0")
        self.fixed.setText(f"{process.fixed_overhead / 3600:g}" if process else "0")
        for widget in (self.table, self.per_layer, self.fixed):
            widget.setEnabled(process is not None)
        self._loading = False

    def _collect(self) -> DepositionProcess:
        rates: dict[str, float] = {}
        extra: dict[str, float] = {}
        for row, name in enumerate(self._materials):
            rate_item, extra_item = self.table.item(row, 1), self.table.item(row, 2)
            rates[name] = float(rate_item.text()) if rate_item else _DEFAULT_RATE
            value = float(extra_item.text()) if extra_item else 0.0
            if value:
                extra[name] = value
        if self._process is not None:
            for name, rate in self._process.rates:
                rates.setdefault(name, rate)
        return DepositionProcess(
            rates,
            overhead_per_layer=float(self.per_layer.text() or 0),
            overhead_per_material=extra,
            fixed_overhead=float(self.fixed.text() or 0) * 3600,
        )

    def _apply(self) -> None:
        if self._loading:
            return
        if not self.enabled.isChecked():
            if self._process is not None:
                self.commandRequested.emit(set_process(None))
            return
        try:
            process = self._collect()
        except ValueError as exc:
            self.errorOccurred.emit(f"Invalid process setting: {exc}")
            self._load()
            return
        if process != self._process:
            self.commandRequested.emit(set_process(process))
