"""Workflow page for the design goal: targets and how they are combined."""

from dataclasses import replace

import numpy as np
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from xstack.gui.steps.base import StepPage
from xstack.project import ProjectState
from xstack.project.commands import set_targets
from xstack.targets import AngleScan, EnergyScan, Target, TargetSet

__all__ = ["GoalPage"]

_COLUMNS = ("Scan", "Fixed", "From", "To", "Units", "Points", "Weight", "Polarization")
_SCAN, _FIXED, _FROM, _TO, _UNITS, _POINTS, _WEIGHT, _POL = range(len(_COLUMNS))
_KINDS = ("Energy scan", "Angle scan")
_POLARIZATIONS = ("unpolarized", "s", "p")
_DEG = np.pi / 180.0


def _row_values(target: Target) -> tuple[str, float, float, float, str]:
    if isinstance(target, EnergyScan):
        return (
            "Energy scan",
            target.angle / _DEG,
            target.energy_min / 1e3,
            target.energy_max / 1e3,
            "deg | keV",
        )
    assert isinstance(target, AngleScan)
    return (
        "Angle scan",
        target.energy / 1e3,
        target.angle_min / _DEG,
        target.angle_max / _DEG,
        "keV | deg",
    )


class GoalPage(StepPage):
    """Table of targets (energy or angle scans) and the combine mode.

    Energies are shown in keV and angles in degrees. Settings of a target that are not in the
    table (profile, spectrum, resolution) are kept when the row is edited.

    Args:
        parent: Parent widget.
    """

    title = "Goal"

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._targets: TargetSet | None = None
        self._loading = False
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<b>Targets</b> — where the mirror shall reflect"))
        self.table = QTableWidget(0, len(_COLUMNS))
        self.table.setHorizontalHeaderLabels(list(_COLUMNS))
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.itemChanged.connect(lambda _: self._apply())
        layout.addWidget(self.table)
        buttons = QHBoxLayout()
        self.add_energy = QPushButton("Add energy scan")
        self.add_angle = QPushButton("Add angle scan")
        self.remove = QPushButton("Remove")
        for b in (self.add_energy, self.add_angle, self.remove):
            buttons.addWidget(b)
        layout.addLayout(buttons)
        self.add_energy.clicked.connect(
            lambda: self._add(EnergyScan(angle=_DEG, energy_min=7e3, energy_max=9e3))
        )
        self.add_angle.clicked.connect(
            lambda: self._add(AngleScan(energy=8048.0, angle_min=1e-3, angle_max=_DEG))
        )
        self.remove.clicked.connect(self._remove)
        combine = QHBoxLayout()
        combine.addWidget(QLabel("Combine targets:"))
        self.combine = QComboBox()
        self.combine.addItem("Average reflectivity", "average_reflectivity")
        self.combine.addItem("Average figure of merit", "average_merit")
        self.combine.activated.connect(lambda _: self._apply())
        combine.addWidget(self.combine)
        combine.addStretch()
        layout.addLayout(combine)
        layout.addStretch()

    def set_state(self, state: ProjectState) -> None:
        """Show the targets of a project state.

        Args:
            state: The current project state.
        """
        if state.targets == self._targets:
            return
        self._targets = state.targets
        self._load()

    def _load(self) -> None:
        self._loading = True
        targets = self._targets.targets if self._targets else ()
        self.table.clearContents()
        self.table.setRowCount(0)
        self.table.setRowCount(len(targets))
        for row, target in enumerate(targets):
            kind, fixed, lo, hi, units = _row_values(target)
            scan = QComboBox()
            scan.addItems(list(_KINDS))
            scan.setCurrentText(kind)
            scan.activated.connect(lambda _=0: self._apply())
            self.table.setCellWidget(row, _SCAN, scan)
            for column, value in ((_FIXED, fixed), (_FROM, lo), (_TO, hi)):
                self.table.setItem(row, column, QTableWidgetItem(f"{value:.6g}"))
            units_item = QTableWidgetItem(units)
            units_item.setFlags(units_item.flags() & ~units_item.flags().ItemIsEditable)
            self.table.setItem(row, _UNITS, units_item)
            self.table.setItem(row, _POINTS, QTableWidgetItem(str(target.points)))
            self.table.setItem(row, _WEIGHT, QTableWidgetItem(f"{target.weight:g}"))
            pol = QComboBox()
            pol.addItems(list(_POLARIZATIONS))
            pol.setCurrentText(target.polarization)
            pol.activated.connect(lambda _=0: self._apply())
            self.table.setCellWidget(row, _POL, pol)
        if self._targets is not None:
            self.combine.setCurrentIndex(max(0, self.combine.findData(self._targets.combine)))
        self._loading = False

    def _cell(self, row: int, column: int) -> float:
        item = self.table.item(row, column)
        if item is None:
            raise ValueError("missing value")
        return float(item.text())

    def _target(self, row: int, previous: Target | None) -> Target:
        scan = self.table.cellWidget(row, _SCAN)
        pol = self.table.cellWidget(row, _POL)
        if not isinstance(scan, QComboBox) or not isinstance(pol, QComboBox):
            raise ValueError("table row is incomplete")
        fixed, lo, hi = (self._cell(row, c) for c in (_FIXED, _FROM, _TO))
        common = {
            "points": int(self._cell(row, _POINTS)),
            "weight": self._cell(row, _WEIGHT),
            "polarization": pol.currentText(),
        }
        if scan.currentText() == "Energy scan":
            if isinstance(previous, EnergyScan):
                return replace(
                    previous, angle=fixed * _DEG, energy_min=lo * 1e3, energy_max=hi * 1e3, **common
                )
            return EnergyScan(
                angle=fixed * _DEG, energy_min=lo * 1e3, energy_max=hi * 1e3, **common
            )
        if isinstance(previous, AngleScan):
            return replace(
                previous, energy=fixed * 1e3, angle_min=lo * _DEG, angle_max=hi * _DEG, **common
            )
        return AngleScan(energy=fixed * 1e3, angle_min=lo * _DEG, angle_max=hi * _DEG, **common)

    def _collect(self) -> TargetSet:
        previous = self._targets.targets if self._targets else ()
        targets = [
            self._target(row, previous[row] if row < len(previous) else None)
            for row in range(self.table.rowCount())
        ]
        return TargetSet(targets, self.combine.currentData())

    def _apply(self) -> None:
        if self._loading:
            return
        try:
            targets = self._collect()
        except (ValueError, TypeError) as exc:
            self.errorOccurred.emit(f"Invalid target: {exc}")
            self._load()
            return
        if targets != self._targets:
            self.commandRequested.emit(set_targets(targets))

    def _add(self, target: Target) -> None:
        current = self._targets.targets if self._targets else ()
        combine = self._targets.combine if self._targets else "average_reflectivity"
        try:
            targets = TargetSet((*current, target), combine)
        except ValueError:
            targets = TargetSet((*current, target), "average_merit")
        self.commandRequested.emit(set_targets(targets))

    def _remove(self) -> None:
        if self._targets is None or len(self._targets) <= 1:
            self.errorOccurred.emit("At least one target is required")
            return
        row = self.table.currentRow()
        if row < 0:
            row = len(self._targets) - 1
        rest = tuple(t for i, t in enumerate(self._targets.targets) if i != row)
        self.commandRequested.emit(set_targets(TargetSet(rest, self._targets.combine)))
