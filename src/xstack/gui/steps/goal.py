"""Workflow page for the design goal: targets and how they are combined."""

from dataclasses import replace
from itertools import pairwise
from typing import ClassVar

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from xstack.gui.steps.base import StepPage
from xstack.project import ProjectState
from xstack.project.commands import set_targets
from xstack.targets import AngleScan, EnergyScan, Target, TargetSet

__all__ = ["GoalPage", "TargetEditor"]

_FIXED_ANGLE = "Fixed angle"
_FIXED_ENERGY = "Fixed energy"
_KINDS = (_FIXED_ANGLE, _FIXED_ENERGY)
_POLARIZATIONS = ("unpolarized", "s", "p")
_AXES = {
    _FIXED_ANGLE: (("Angle", "deg", "grazing angle"), ("Energy", "keV", "photon energy")),
    _FIXED_ENERGY: (("Energy", "keV", "photon energy"), ("Angle", "deg", "grazing angle")),
}
"""Label, unit and description of the fixed quantity and of the scanned range per scan type."""
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


def _set_text(edit: QLineEdit, text: str) -> None:
    """Set the text of a line edit unless it is already shown, keeping the cursor."""
    if edit.text() != text:
        edit.setText(text)
        edit.setCursorPosition(0)


class TargetEditor(QFrame):
    """Editor of one target, arranged on three lines.

    The first line holds the name, the scan type and the remove button, the second the fixed
    value and the scanned range, the third the points, weight, polarization and number of
    reflections. Energies are shown in keV and angles in degrees. Showing a target updates the
    existing fields, so that focus and tab order are kept.

    Args:
        parent: Parent widget.

    Signals:
        edited(): A field was finished or a selection changed.
        removeRequested(): The remove button was clicked.

    Attributes:
        name: Name of the target.
        scan: Scan type (fixed angle or fixed energy).
        fixed: The fixed angle (deg) or energy (keV).
        lo: Start of the scanned range (keV or deg).
        hi: End of the scanned range (keV or deg).
        points: Number of evaluation points.
        weight: Relative weight.
        polarization: Polarization of the incident radiation.
        reflections: Number of reflections n.
        remove: Button removing the target.
        fixed_label: Label of the fixed value.
        range_label: Label of the scanned range.
    """

    edited = Signal()
    removeRequested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._loading = False
        metrics = self.fontMetrics()
        label_width = max(metrics.horizontalAdvance(t) for t in ("Name", "Angle", "Energy")) + 8
        unit_width = max(metrics.horizontalAdvance(t) for t in ("deg", "keV")) + 4
        number_width = metrics.horizontalAdvance("0" * 8) + 16

        self.separator = QFrame()
        self.separator.setFrameShape(QFrame.Shape.HLine)
        self.separator.setFrameShadow(QFrame.Shadow.Sunken)

        self.name = self._line_edit("Name of the target in plots and exported files (optional)")
        self.scan = QComboBox()
        self.scan.addItems(list(_KINDS))
        self.scan.setToolTip(
            "Fixed angle: scan the energy at one grazing angle. "
            "Fixed energy: scan the grazing angle at one photon energy."
        )
        self.scan.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.scan.activated.connect(lambda _=0: self._scan_changed())
        self.remove = QPushButton("Remove")
        self.remove.setToolTip("Remove this target")
        self.remove.clicked.connect(self.removeRequested.emit)

        self.fixed = self._line_edit("", number_width)
        self.lo = self._line_edit("", number_width)
        self.hi = self._line_edit("", number_width)
        self.fixed_label = QLabel()
        self.fixed_label.setFixedWidth(label_width)
        self.fixed_unit = QLabel()
        self.fixed_unit.setFixedWidth(unit_width)
        self.range_label = QLabel()
        self.range_unit = QLabel()

        self.points = self._line_edit("Number of evaluation points (at least 2)", number_width)
        self.points.setPlaceholderText("points")
        self.weight = self._line_edit("Relative weight of the target (positive)", number_width)
        self.weight.setPlaceholderText("weight")
        self.polarization = QComboBox()
        self.polarization.addItems(list(_POLARIZATIONS))
        self.polarization.setToolTip("Polarization of the incident radiation")
        self.polarization.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.polarization.activated.connect(lambda _=0: self._emit_edited())
        self.reflections = QSpinBox()
        self.reflections.setRange(1, 4)
        self.reflections.setToolTip("1 = single reflection R, 2 = double reflection R², …")
        self.reflections.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.reflections.valueChanged.connect(lambda _=0: self._emit_edited())

        name_label = QLabel("Name")
        name_label.setFixedWidth(label_width)
        first = QHBoxLayout()
        first.addWidget(name_label)
        first.addWidget(self.name, 1)
        first.addWidget(self.scan)
        first.addWidget(self.remove)
        second = QHBoxLayout()
        second.addWidget(self.fixed_label)
        second.addWidget(self.fixed)
        second.addWidget(self.fixed_unit)
        second.addSpacing(12)
        second.addWidget(self.range_label)
        second.addWidget(self.lo)
        second.addWidget(QLabel("–"))
        second.addWidget(self.hi)
        second.addWidget(self.range_unit)
        second.addStretch()
        third = QHBoxLayout()
        points_label = QLabel("Points")
        points_label.setFixedWidth(label_width)
        third.addWidget(points_label)
        third.addWidget(self.points)
        third.addSpacing(12)
        for text, widget, tip in (
            ("Weight", self.weight, None),
            ("Polarization", self.polarization, None),
            ("Reflections", self.reflections, "Number of reflections n (target reflectivity Rⁿ)"),
        ):
            label = QLabel(text)
            if tip:
                label.setToolTip(tip)
            label.setBuddy(widget)
            third.addWidget(label)
            third.addWidget(widget)
            third.addSpacing(12)
        third.addStretch()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 4)
        layout.setSpacing(4)
        layout.addWidget(self.separator)
        for line in (first, second, third):
            layout.addLayout(line)

    def fields(self) -> list[QWidget]:
        """Return the editable fields in tab order.

        Returns:
            The fields from the name to the remove button.
        """
        return [
            self.name,
            self.scan,
            self.fixed,
            self.lo,
            self.hi,
            self.points,
            self.weight,
            self.polarization,
            self.reflections,
            self.remove,
        ]

    def show_target(self, target: Target, index: int) -> None:
        """Show a target without emitting `edited`.

        Args:
            target: The target.
            index: Position of the target in the target set (0-based).
        """
        self._loading = True
        kind, fixed, lo, hi = _row_values(target)
        self.separator.setVisible(index > 0)
        _set_text(self.name, target.name)
        self.name.setPlaceholderText(target.label(index))
        self.scan.setCurrentText(kind)
        self._update_labels()
        for edit, value in ((self.fixed, fixed), (self.lo, lo), (self.hi, hi)):
            _set_text(edit, f"{value:.5g}")
        _set_text(self.points, str(target.points))
        _set_text(self.weight, f"{target.weight:g}")
        self.polarization.setCurrentText(target.polarization)
        self.reflections.setValue(target.reflections)
        self._loading = False

    def target(self, previous: Target | None) -> Target:
        """Return the target shown in the fields.

        Args:
            previous: The target shown before; its settings that have no field (profile,
                spectrum, resolution) are kept, and unchanged rounded values keep their exact
                value.

        Returns:
            The edited target.

        Raises:
            ValueError: A field is not a valid number or the target is invalid.
        """
        common = {
            "name": self.name.text().strip(),
            "points": int(self._number(self.points)),
            "weight": self._number(self.weight, 1.0, previous.weight if previous else None),
            "polarization": self.polarization.currentText(),
            "reflections": self.reflections.value(),
        }
        if self.scan.currentText() == _FIXED_ANGLE:
            old = previous if isinstance(previous, EnergyScan) else None
            angle = self._number(self.fixed, _DEG, old.angle if old else None)
            e_min = self._number(self.lo, 1e3, old.energy_min if old else None)
            e_max = self._number(self.hi, 1e3, old.energy_max if old else None)
            if old is not None:
                return replace(old, angle=angle, energy_min=e_min, energy_max=e_max, **common)
            return EnergyScan(angle=angle, energy_min=e_min, energy_max=e_max, **common)
        older = previous if isinstance(previous, AngleScan) else None
        energy = self._number(self.fixed, 1e3, older.energy if older else None)
        a_min = self._number(self.lo, _DEG, older.angle_min if older else None)
        a_max = self._number(self.hi, _DEG, older.angle_max if older else None)
        if older is not None:
            return replace(older, energy=energy, angle_min=a_min, angle_max=a_max, **common)
        return AngleScan(energy=energy, angle_min=a_min, angle_max=a_max, **common)

    @staticmethod
    def _number(edit: QLineEdit, scale: float = 1.0, shown: float | None = None) -> float:
        """Return a field as SI value; a text equal to the shown rounding returns `shown`."""
        text = edit.text().strip()
        if shown is not None and text == f"{shown / scale:.5g}":
            return shown
        return float(text) * scale

    def _line_edit(self, tooltip: str, width: int | None = None) -> QLineEdit:
        edit = QLineEdit()
        edit.setToolTip(tooltip)
        if width is not None:
            edit.setFixedWidth(width)
        edit.editingFinished.connect(self._emit_edited)
        return edit

    def _update_labels(self) -> None:
        (fixed, fixed_unit, fixed_what), (scanned, unit, what) = _AXES[self.scan.currentText()]
        self.fixed_label.setText(fixed)
        self.fixed_unit.setText(fixed_unit)
        self.range_label.setText(f"{scanned} from")
        self.range_unit.setText(unit)
        self.fixed.setPlaceholderText(f"{fixed.lower()} ({fixed_unit})")
        self.fixed.setToolTip(f"Fixed {fixed_what} in {fixed_unit}")
        self.lo.setPlaceholderText(f"min {scanned.lower()} ({unit})")
        self.lo.setToolTip(f"Start of the {what} range in {unit}")
        self.hi.setPlaceholderText(f"max {scanned.lower()} ({unit})")
        self.hi.setToolTip(f"End of the {what} range in {unit}")

    def _scan_changed(self) -> None:
        self._update_labels()
        self._emit_edited()

    def _emit_edited(self) -> None:
        if not self._loading:
            self.edited.emit()


class GoalPage(StepPage):
    """List of targets (fixed-angle or fixed-energy scans) and the combine mode.

    Every target is edited in a `TargetEditor`. Settings of a target that have no field
    (profile, spectrum, resolution) are kept when it is edited. Edits are applied when a field
    is finished, a selection changes or a spin box value changes.

    Args:
        parent: Parent widget.

    Attributes:
        help_text: Rich-text explanation of the page.
        rows: The editors of the targets, in order.
        add_energy: Button adding a fixed-angle target (energy range).
        add_angle: Button adding a fixed-energy target (angle range).
        combine: Selection of the combine mode.
        combine_info: Label explaining the selected combine mode.
    """

    title = "Goal"
    help_text: ClassVar[str] = (
        "A <b>target</b> describes where the mirror shall reflect. It is either a "
        "<i>fixed-angle</i> target (an energy range at one grazing angle) or a "
        "<i>fixed-energy</i> target (an angle range at one photon energy); energies are in keV "
        "and angles in degrees. An optional <b>name</b> labels the target in plots and "
        "exported files (unnamed targets are called “target 1”, “target 2”, …). "
        "Each target has a <b>weight</b> relative to the others, a "
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
        self.rows: list[TargetEditor] = []
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<b>Targets</b> — where the mirror shall reflect"))
        self._rows_layout = QVBoxLayout()
        self._rows_layout.setSpacing(6)
        layout.addLayout(self._rows_layout)
        buttons = QHBoxLayout()
        self.add_energy = QPushButton("Add fixed-angle target")
        self.add_energy.setToolTip("Add a target with an energy range at a fixed grazing angle")
        self.add_angle = QPushButton("Add fixed-energy target")
        self.add_angle.setToolTip("Add a target with an angle range at a fixed photon energy")
        for b in (self.add_energy, self.add_angle):
            buttons.addWidget(b)
        buttons.addStretch()
        layout.addLayout(buttons)
        self.add_energy.clicked.connect(
            lambda: self._add(EnergyScan(angle=_DEG, energy_min=7e3, energy_max=9e3))
        )
        self.add_angle.clicked.connect(
            lambda: self._add(AngleScan(energy=8048.0, angle_min=1e-3, angle_max=_DEG))
        )
        combine = QHBoxLayout()
        combine.addWidget(QLabel("Combine targets:"))
        self.combine = QComboBox()
        self.combine.setToolTip("How several targets are combined into one figure of merit")
        self.combine.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
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

    def _load(self) -> None:
        self._loading = True
        targets = self._targets.targets if self._targets else ()
        resized = len(self.rows) != len(targets)
        while len(self.rows) > len(targets):
            row = self.rows.pop()
            self._rows_layout.removeWidget(row)
            row.deleteLater()
        while len(self.rows) < len(targets):
            row = TargetEditor()
            row.edited.connect(self._apply)
            row.removeRequested.connect(lambda r=row: self._remove(r))
            self._rows_layout.addWidget(row)
            self.rows.append(row)
        for index, (row, target) in enumerate(zip(self.rows, targets, strict=True)):
            row.show_target(target, index)
            row.remove.setEnabled(len(targets) > 1)
        if resized:
            self._update_tab_order()
        if self._targets is not None:
            self.combine.setCurrentIndex(max(0, self.combine.findData(self._targets.combine)))
        self._loading = False

    def _update_tab_order(self) -> None:
        chain = [w for row in self.rows for w in row.fields()]
        chain += [self.add_energy, self.add_angle, self.combine]
        for first, second in pairwise(chain):
            QWidget.setTabOrder(first, second)

    def _collect(self) -> TargetSet:
        previous = self._targets.targets if self._targets else ()
        targets = [
            row.target(previous[index] if index < len(previous) else None)
            for index, row in enumerate(self.rows)
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

    def _remove(self, row: TargetEditor) -> None:
        if self._targets is None or len(self._targets) <= 1:
            self.errorOccurred.emit("At least one target is required")
            return
        index = self.rows.index(row)
        rest = tuple(t for i, t in enumerate(self._targets.targets) if i != index)
        self.commandRequested.emit(set_targets(TargetSet(rest, self._targets.combine)))
