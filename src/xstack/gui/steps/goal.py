"""Workflow page for the design goal: targets and how they are combined."""

from dataclasses import replace
from typing import ClassVar

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from xstack.gui.steps.base import StepPage
from xstack.project import ProjectState
from xstack.project.commands import set_targets
from xstack.targets import AngleScan, EnergyScan, Target, TargetSet

__all__ = ["GoalPage"]

_COLUMNS = ("Scan", "Fixed", "From", "To", "Points", "Weight", "Polarization", "Refl.")
_SCAN, _FIXED, _FROM, _TO, _POINTS, _WEIGHT, _POL, _REFL = range(len(_COLUMNS))
_COLUMN_WIDTHS = {
    _FIXED: 86,
    _FROM: 86,
    _TO: 86,
    _POINTS: 62,
    _WEIGHT: 62,
    _POL: 132,
    _REFL: 58,
}
"""Fixed widths in px of the compact columns; the scan-type column takes the rest."""
_FIXED_ANGLE = "Fixed angle"
_FIXED_ENERGY = "Fixed energy"
_KINDS = (_FIXED_ANGLE, _FIXED_ENERGY)
_POLARIZATIONS = ("unpolarized", "s", "p")
_PLACEHOLDERS = {
    _FIXED_ANGLE: ("angle (deg)", "min energy (keV)", "max energy (keV)"),
    _FIXED_ENERGY: ("energy (keV)", "min angle (deg)", "max angle (deg)"),
}
_COMBINE_HELP = {
    "average_reflectivity": (
        "Average reflectivity: the reflectivity curves of all targets are averaged with their "
        "weights and the figure of merit is evaluated once on the averaged curve (requires "
        "identical ranges and points, e.g. one energy range at several angles)"
    ),
    "average_merit": (
        "Average figure of merit: the figure of merit is evaluated for every target separately "
        "and the results are averaged with the weights (targets may differ)"
    ),
}
_DEG = np.pi / 180.0


def _row_values(target: Target) -> tuple[str, float, float, float]:
    if isinstance(target, EnergyScan):
        return (
            _FIXED_ANGLE,
            target.angle / _DEG,
            target.energy_min / 1e3,
            target.energy_max / 1e3,
        )
    assert isinstance(target, AngleScan)
    return (
        _FIXED_ENERGY,
        target.energy / 1e3,
        target.angle_min / _DEG,
        target.angle_max / _DEG,
    )


class GoalPage(StepPage):
    """Table of targets (fixed-angle or fixed-energy scans) and the combine mode.

    Energies are shown in keV and angles in degrees. Settings of a target that are not in the
    table (profile, spectrum, resolution, name) are kept when the row is edited. Edits are
    applied when a field is finished, a selection changes or a spin box value changes.

    Args:
        parent: Parent widget.

    Attributes:
        help_text: Rich-text explanation of the page.
        table: The table of targets.
        add_energy: Button adding a fixed-angle target (energy range).
        add_angle: Button adding a fixed-energy target (angle range).
        remove: Button removing the selected target.
        combine: Selection of the combine mode.
        combine_info: Label explaining the selected combine mode.
    """

    title = "Goal"
    help_text: ClassVar[str] = (
        "A <b>target</b> describes where the mirror shall reflect. It is either a "
        "<i>fixed-angle</i> target (an energy range at one grazing angle) or a "
        "<i>fixed-energy</i> target (an angle range at one photon energy); energies are in keV "
        "and angles in degrees. Each target has a <b>weight</b> relative to the others, a "
        "<b>polarization</b> (unpolarized, s or p) and a number of <b>reflections</b> n, so "
        "that its reflectivity is R<sup>n</sup>, e.g. 2 for a double-reflection telescope. "
        "The targets define the energies and angles on which the design is evaluated in the "
        "<b>scan</b> step, and the figure of merit and the <b>optimization</b> try to "
        "maximize the weighted reflectivity over all targets, combined as chosen below."
    )

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._targets: TargetSet | None = None
        self._loading = False
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<b>Targets</b> — where the mirror shall reflect"))
        self.table = QTableWidget(0, len(_COLUMNS))
        self.table.setHorizontalHeaderLabels(list(_COLUMNS))
        header = self.table.horizontalHeader()
        header.setDefaultAlignment(Qt.AlignmentFlag.AlignLeft)
        header.setStretchLastSection(False)
        header.setMinimumSectionSize(40)
        header.setSectionResizeMode(_SCAN, QHeaderView.ResizeMode.Stretch)
        for column, width in _COLUMN_WIDTHS.items():
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Fixed)
            self.table.setColumnWidth(column, width)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        header_item = self.table.horizontalHeaderItem(_REFL)
        if header_item is not None:
            header_item.setToolTip("Number of reflections n (the target reflectivity is Rⁿ)")
        self.table.verticalHeader().setVisible(False)
        self.table.setShowGrid(False)
        self.table.setFrameShape(QFrame.Shape.NoFrame)
        self.table.setAlternatingRowColors(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        layout.addWidget(self.table)
        buttons = QHBoxLayout()
        self.add_energy = QPushButton("Add fixed-angle target")
        self.add_energy.setToolTip("Add a target with an energy range at a fixed grazing angle")
        self.add_angle = QPushButton("Add fixed-energy target")
        self.add_angle.setToolTip("Add a target with an angle range at a fixed photon energy")
        self.remove = QPushButton("Remove")
        self.remove.setToolTip("Remove the selected target (the last one if none is selected)")
        for b in (self.add_energy, self.add_angle, self.remove):
            buttons.addWidget(b)
        buttons.addStretch()
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
        self.combine.setToolTip("How several targets are combined into one figure of merit")
        self.combine.addItem("Average reflectivity", "average_reflectivity")
        self.combine.addItem("Average figure of merit", "average_merit")
        for index, mode in enumerate(("average_reflectivity", "average_merit")):
            self.combine.setItemData(index, _COMBINE_HELP[mode], Qt.ItemDataRole.ToolTipRole)
        self.combine.activated.connect(lambda _: self._apply())
        self.combine.currentIndexChanged.connect(lambda _: self._update_combine_info())
        combine.addWidget(self.combine)
        combine.addStretch()
        layout.addLayout(combine)
        self.combine_info = QLabel()
        self.combine_info.setWordWrap(True)
        font = self.combine_info.font()
        font.setPointSizeF(font.pointSizeF() * 0.9)
        self.combine_info.setFont(font)
        layout.addWidget(self.combine_info)
        self._update_combine_info()
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

    def _update_combine_info(self) -> None:
        mode = self.combine.currentData()
        self.combine_info.setText(_COMBINE_HELP.get(mode, ""))

    def _line_edit(self, text: str, placeholder: str, tooltip: str) -> QLineEdit:
        edit = QLineEdit(text)
        edit.setCursorPosition(0)
        edit.setPlaceholderText(placeholder)
        edit.setToolTip(tooltip)
        edit.setFrame(False)
        edit.editingFinished.connect(self._apply)
        return edit

    def _set_placeholders(self, row: int, kind: str) -> None:
        for column, text in zip((_FIXED, _FROM, _TO), _PLACEHOLDERS[kind], strict=True):
            edit = self.table.cellWidget(row, column)
            if isinstance(edit, QLineEdit):
                edit.setPlaceholderText(text)
                edit.setToolTip(text[0].upper() + text[1:])

    def _scan_changed(self, row_widget: QComboBox) -> None:
        for row in range(self.table.rowCount()):
            if self.table.cellWidget(row, _SCAN) is row_widget:
                self._set_placeholders(row, row_widget.currentText())
        self._apply()

    def _load(self) -> None:
        self._loading = True
        targets = self._targets.targets if self._targets else ()
        self.table.clearContents()
        self.table.setRowCount(0)
        self.table.setRowCount(len(targets))
        for row, target in enumerate(targets):
            kind, fixed, lo, hi = _row_values(target)
            scan = QComboBox()
            scan.setSizeAdjustPolicy(
                QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
            )
            scan.setMinimumContentsLength(10)
            scan.addItems(list(_KINDS))
            scan.setCurrentText(kind)
            scan.setToolTip(
                "Fixed angle: scan the energy at one grazing angle. "
                "Fixed energy: scan the grazing angle at one photon energy."
            )
            scan.activated.connect(lambda _=0, w=scan: self._scan_changed(w))
            self.table.setCellWidget(row, _SCAN, scan)
            for column, value in ((_FIXED, fixed), (_FROM, lo), (_TO, hi)):
                self.table.setCellWidget(row, column, self._line_edit(f"{value:.5g}", "", ""))
            self._set_placeholders(row, kind)
            self.table.setCellWidget(
                row,
                _POINTS,
                self._line_edit(
                    str(target.points), "points", "Number of evaluation points (at least 2)"
                ),
            )
            self.table.setCellWidget(
                row,
                _WEIGHT,
                self._line_edit(
                    f"{target.weight:g}", "weight", "Relative weight of the target (positive)"
                ),
            )
            pol = QComboBox()
            pol.setSizeAdjustPolicy(
                QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
            )
            pol.setMinimumContentsLength(6)
            pol.addItems(list(_POLARIZATIONS))
            pol.setCurrentText(target.polarization)
            pol.setToolTip("Polarization of the incident radiation")
            pol.activated.connect(lambda _=0: self._apply())
            self.table.setCellWidget(row, _POL, pol)
            refl = QSpinBox()
            refl.setRange(1, 4)
            refl.setValue(target.reflections)
            refl.setToolTip("1 = single reflection R, 2 = double reflection R², …")
            refl.valueChanged.connect(lambda _=0: self._apply())
            self.table.setCellWidget(row, _REFL, refl)
        self.table.resizeRowsToContents()
        self._fit_height()
        if self._targets is not None:
            self.combine.setCurrentIndex(max(0, self.combine.findData(self._targets.combine)))
        self._loading = False

    def _fit_height(self) -> None:
        rows = max(1, self.table.rowCount())
        height = self.table.horizontalHeader().height() + 2 * self.table.frameWidth()
        height += sum(self.table.rowHeight(r) for r in range(self.table.rowCount()))
        if self.table.rowCount() == 0:
            height += self.table.verticalHeader().defaultSectionSize() * rows
        self.table.setFixedHeight(height)

    def _text(self, row: int, column: int) -> str:
        edit = self.table.cellWidget(row, column)
        if not isinstance(edit, QLineEdit):
            raise ValueError("missing value")
        return edit.text()

    def _number(
        self, row: int, column: int, scale: float = 1.0, shown: float | None = None
    ) -> float:
        """Return a cell as SI value; a text equal to the shown rounding returns `shown`."""
        text = self._text(row, column).strip()
        if shown is not None and text == f"{shown / scale:.5g}":
            return shown
        return float(text) * scale

    def _target(self, row: int, previous: Target | None) -> Target:
        scan = self.table.cellWidget(row, _SCAN)
        pol = self.table.cellWidget(row, _POL)
        refl = self.table.cellWidget(row, _REFL)
        if (
            not isinstance(scan, QComboBox)
            or not isinstance(pol, QComboBox)
            or not isinstance(refl, QSpinBox)
        ):
            raise ValueError("table row is incomplete")
        common = {
            "points": int(self._number(row, _POINTS)),
            "weight": self._number(row, _WEIGHT, 1.0, previous.weight if previous else None),
            "polarization": pol.currentText(),
            "reflections": refl.value(),
        }
        if scan.currentText() == _FIXED_ANGLE:
            old = previous if isinstance(previous, EnergyScan) else None
            angle = self._number(row, _FIXED, _DEG, old.angle if old else None)
            e_min = self._number(row, _FROM, 1e3, old.energy_min if old else None)
            e_max = self._number(row, _TO, 1e3, old.energy_max if old else None)
            if old is not None:
                return replace(old, angle=angle, energy_min=e_min, energy_max=e_max, **common)
            return EnergyScan(angle=angle, energy_min=e_min, energy_max=e_max, **common)
        older = previous if isinstance(previous, AngleScan) else None
        energy = self._number(row, _FIXED, 1e3, older.energy if older else None)
        a_min = self._number(row, _FROM, _DEG, older.angle_min if older else None)
        a_max = self._number(row, _TO, _DEG, older.angle_max if older else None)
        if older is not None:
            return replace(older, energy=energy, angle_min=a_min, angle_max=a_max, **common)
        return AngleScan(energy=energy, angle_min=a_min, angle_max=a_max, **common)

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
