"""Workflow page for the measured data: dataset settings and fit ranges."""

from collections.abc import Callable
from dataclasses import replace
from typing import Any, cast

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from grazing.core.measurement import MeasuredCurve
from grazing.core.units import deg, kev, to_deg, to_kev
from grazing.fit.dataset import Dataset, Interval
from grazing.fit.project import (
    FitState,
    add_dataset,
    remove_dataset,
    rename_dataset,
    replace_dataset,
    set_ranges,
)
from grazing.gui.common.pages import StepPage
from grazing.gui.common.style import set_variant
from grazing.gui.fit.widgets.import_dialog import ImportDialog

__all__ = ["DataPage"]

_EXCLUDED_FRACTION = 0.1


class DataPage(StepPage):
    """Settings of the selected dataset, its fit ranges, and import of new data.

    The dataset shown is chosen with `set_selected` (the application's dataset list). Every edit
    is emitted as a command; the page changes only when the new state arrives.

    Attributes:
        dialog_factory: Creates the import dialog from the names in use and the parent widget.
            Replace it to bypass the dialog, e.g. in tests; the result needs ``exec()`` and
            ``datasets()``.

    Args:
        parent: Parent widget.
    """

    title = "Data"
    help_text = (
        "Load <b>measured curves</b> (CSV, TSV, column files; grazing exports are recognized). "
        "Describe the file layout once; the preview shows how the columns are read. Each curve "
        "becomes a <b>dataset</b> with its own instrument. Select the <b>fit ranges</b>: only "
        "points inside a range and outside all excluded ranges are fitted. In the reflectivity "
        "plot, <b>Shift-drag</b> adds a fit range and <b>Ctrl-drag</b> an excluded range; drag "
        "the shaded edges to adjust them."
    )

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.dialog_factory: Callable[[tuple[str, ...], QWidget], Any] = lambda used, parent: (
            ImportDialog(used, parent)
        )
        self._state = FitState()
        self._selected: str | None = None
        self._loading = False

        layout = QVBoxLayout(self)
        self.hint = QLabel(
            "No dataset selected. Add measured data with “Add data…” or select a dataset "
            "in the list."
        )
        self.hint.setWordWrap(True)
        self.hint.setProperty("muted", True)
        layout.addWidget(self.hint)

        self.dataset_group = QGroupBox("Dataset")
        form = QFormLayout(self.dataset_group)
        self.name_edit = QLineEdit()
        self.name_edit.setToolTip("Identifier of the dataset; also the prefix of its parameters.")
        self.name_edit.editingFinished.connect(self._rename)
        self.enabled_check = QCheckBox("include in fit")
        self.enabled_check.toggled.connect(self._set_enabled)
        form.addRow("Name", _row(self.name_edit, self.enabled_check))
        self.summary_label = QLabel()
        self.summary_label.setWordWrap(True)
        form.addRow("Measurement", self.summary_label)
        self.fixed_label = QLabel("Photon energy")
        self.fixed_spin = QDoubleSpinBox()
        self.fixed_spin.setDecimals(4)
        self.fixed_spin.setKeyboardTracking(False)
        self.fixed_spin.setSpecialValueText("not set")
        self.fixed_spin.valueChanged.connect(self._set_fixed)
        self.polarization_combo = QComboBox()
        self.polarization_combo.addItems(["unpolarized", "s", "p"])
        self.polarization_combo.currentTextChanged.connect(self._set_polarization)
        form.addRow(
            self.fixed_label, _row(self.fixed_spin, QLabel("Polarization"), self.polarization_combo)
        )
        self.weight_spin = QDoubleSpinBox()
        self.weight_spin.setDecimals(3)
        self.weight_spin.setRange(0.001, 1000.0)
        self.weight_spin.setKeyboardTracking(False)
        self.weight_spin.setToolTip("Residuals are multiplied by √weight in a simultaneous fit.")
        self.weight_spin.valueChanged.connect(self._set_weight)
        note = QLabel("residuals × √weight in a simultaneous fit")
        note.setProperty("muted", True)
        form.addRow("Weight", _row(self.weight_spin, note))
        layout.addWidget(self.dataset_group)

        self.ranges_group = QGroupBox("Fit ranges")
        box = QVBoxLayout(self.ranges_group)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Type", "From", "To", "Points"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setShowGrid(False)
        self.table.setFrameShape(QFrame.Shape.NoFrame)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setMinimumHeight(110)
        self.table.itemChanged.connect(self._range_edited)
        box.addWidget(self.table)
        buttons = QHBoxLayout()
        self.add_range_button = QPushButton("Add fit range")
        self.add_excluded_button = QPushButton("Add excluded range")
        self.remove_range_button = QPushButton("Remove")
        self.fit_all_button = QPushButton("Fit all points")
        self.add_range_button.clicked.connect(lambda: self._add_range(False))
        self.add_excluded_button.clicked.connect(lambda: self._add_range(True))
        self.remove_range_button.clicked.connect(self._remove_range)
        self.fit_all_button.clicked.connect(self._fit_all)
        for button in (
            self.add_range_button,
            self.add_excluded_button,
            self.remove_range_button,
            self.fit_all_button,
        ):
            buttons.addWidget(button)
        buttons.addStretch()
        box.addLayout(buttons)
        self.count_label = QLabel()
        self.count_label.setWordWrap(True)
        self.count_label.setTextFormat(Qt.TextFormat.RichText)
        self.count_label.setProperty("muted", True)
        box.addWidget(self.count_label)
        layout.addWidget(self.ranges_group)

        actions = QHBoxLayout()
        self.add_button = QPushButton("Add data…")
        set_variant(self.add_button, "primary")
        self.add_button.clicked.connect(self.open_import_dialog)
        self.remove_button = QPushButton("Remove dataset")
        self.remove_button.clicked.connect(self._remove_dataset)
        actions.addWidget(self.add_button)
        actions.addWidget(self.remove_button)
        actions.addStretch()
        layout.addLayout(actions)
        layout.addStretch()
        self._refresh()

    # --- public API ---------------------------------------------------------------------

    def set_state(self, state: FitState) -> None:
        """Show a fit state.

        Args:
            state: The current state of the fit project.
        """
        self._state = state
        self._refresh()

    def set_preview(self, preview: Any) -> None:
        """Ignore the preview; the page shows the project state only.

        Args:
            preview: The latest preview.
        """

    def set_selected(self, name: str | None) -> None:
        """Choose the dataset shown on the page.

        Args:
            name: Dataset name, or ``None`` for none.
        """
        self._selected = name
        self._refresh()

    def open_import_dialog(self) -> list[Dataset]:
        """Ask for measured data and emit a command adding each dataset.

        Returns:
            The datasets added; empty if the dialog was cancelled.
        """
        used = tuple(d.name for d in self._state.datasets)
        dialog = self.dialog_factory(used, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return []
        datasets = cast("list[Dataset]", dialog.datasets())
        for dataset in datasets:
            self.commandRequested.emit(add_dataset(dataset))
        return datasets

    # --- internals ----------------------------------------------------------------------

    @property
    def _dataset(self) -> Dataset | None:
        for d in self._state.datasets:
            if d.name == self._selected:
                return d
        return None

    def _refresh(self) -> None:
        data = self._dataset
        self._loading = True
        try:
            for widget in (self.dataset_group, self.ranges_group, self.remove_button):
                widget.setVisible(data is not None)
            self.hint.setVisible(data is None)
            if data is None:
                self.table.setRowCount(0)
                return
            curve = data.curve
            energy_axis = curve.axis == "energy"
            self.dataset_group.setTitle(f"Dataset “{data.name}”")
            self.name_edit.setText(data.name)
            self.enabled_check.setChecked(data.enabled)
            lo, hi = _display(curve.axis, np.array(curve.scan_range))
            what = "energy scan" if energy_axis else "angle scan"
            unit = "keV" if energy_axis else "deg"
            self.summary_label.setText(
                f"{curve.source or curve.name or 'measured curve'} · {what} "
                f"{lo:.4g}–{hi:.4g} {unit} · {curve.size} points · uncertainties: "
                f"{'yes' if curve.uncertainty is not None else 'no'}"
            )
            self.fixed_label.setText("Grazing angle" if energy_axis else "Photon energy")
            self.fixed_spin.setSuffix(" deg" if energy_axis else " keV")
            self.fixed_spin.setRange(0.0, 90.0 if energy_axis else 1000.0)
            fixed = curve.fixed
            self.fixed_spin.setValue(
                0.0 if fixed is None else float(to_deg(fixed) if energy_axis else to_kev(fixed))
            )
            self.polarization_combo.setCurrentText(curve.polarization)
            self.weight_spin.setValue(data.weight)
            self._fill_table(data)
            fitted = int(data.mask().sum())
            self.count_label.setText(
                f"<b>{fitted} of {curve.size}</b> points are fitted: inside a fit range (or all "
                "points if there is none) and outside every excluded range."
            )
            self.fit_all_button.setEnabled(bool(data.ranges or data.excluded))
        finally:
            self._loading = False

    def _fill_table(self, data: Dataset) -> None:
        selected = self.table.currentRow()
        axis = data.curve.axis
        x = np.array(data.curve.x)
        rows = [("fit", r) for r in data.ranges] + [("excluded", r) for r in data.excluded]
        suffix = "keV" if axis == "energy" else "deg"
        self.table.setHorizontalHeaderLabels(
            ["Type", f"From ({suffix})", f"To ({suffix})", "Points"]
        )
        self.table.setRowCount(len(rows))
        for i, (kind, (lo, hi)) in enumerate(rows):
            dlo, dhi = _display(axis, np.array([lo, hi]))
            inside = int(((x >= lo) & (x <= hi)).sum())
            cells = [kind, f"{dlo:.4f}", f"{dhi:.4f}", str(inside)]
            for j, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if j in (0, 3):
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(i, j, item)
        if 0 <= selected < len(rows):
            self.table.selectRow(selected)

    def _emit_replace(self, dataset: Dataset, description: str) -> None:
        self.commandRequested.emit(replace_dataset(dataset, description))

    def _rename(self) -> None:
        data = self._dataset
        if data is None or self._loading:
            return
        new = self.name_edit.text().strip()
        if new == data.name:
            return
        error = None
        if not new.isidentifier():
            error = f"{new!r} is not a valid dataset name (letters, digits and _)"
        elif any(d.name == new for d in self._state.datasets):
            error = f"a dataset named {new!r} already exists"
        if error:
            self.errorOccurred.emit(error)
            self.name_edit.setText(data.name)
            return
        self.commandRequested.emit(rename_dataset(data.name, new))

    def _set_enabled(self, checked: bool) -> None:
        data = self._dataset
        if data is None or self._loading:
            return
        state = "Include" if checked else "Exclude"
        self._emit_replace(replace(data, enabled=checked), f"{state} dataset {data.name!r}")

    def _set_fixed(self, value: float) -> None:
        data = self._dataset
        if data is None or self._loading or value <= 0.0:
            return
        curve = data.curve
        internal = deg(value) if curve.axis == "energy" else kev(value)
        what = "angle" if curve.axis == "energy" else "photon energy"
        self._replace_curve(data, curve.with_fixed(internal), f"Set {what} of {data.name!r}")

    def _set_polarization(self, text: str) -> None:
        data = self._dataset
        if data is None or self._loading:
            return
        curve = replace(data.curve, polarization=cast(Any, text))
        self._replace_curve(data, curve, f"Set polarization of {data.name!r}")

    def _replace_curve(self, data: Dataset, curve: MeasuredCurve, description: str) -> None:
        try:
            new = replace(data, curve=curve)
        except ValueError as error:
            self.errorOccurred.emit(str(error))
            self._refresh()
            return
        self._emit_replace(new, description)

    def _set_weight(self, value: float) -> None:
        data = self._dataset
        if data is None or self._loading:
            return
        self._emit_replace(replace(data, weight=value), f"Set weight of {data.name!r}")

    # --- ranges -------------------------------------------------------------------------

    def _emit_ranges(self, data: Dataset, ranges: list[Interval], excluded: list[Interval]) -> None:
        try:
            data.with_ranges(ranges, excluded)
        except ValueError as error:
            self.errorOccurred.emit(str(error))
            self._refresh()
            return
        self.commandRequested.emit(set_ranges(data.name, ranges, excluded))

    def _range_edited(self, item: QTableWidgetItem) -> None:
        data = self._dataset
        if data is None or self._loading or item.column() not in (1, 2):
            return
        row, column = item.row(), item.column()
        try:
            value = float(item.text().replace(",", "."))
        except ValueError:
            self.errorOccurred.emit(f"{item.text()!r} is not a number")
            self._refresh()
            return
        internal = float(deg(value) if data.curve.axis == "angle" else kev(value))
        ranges, excluded = list(data.ranges), list(data.excluded)
        target = ranges if row < len(ranges) else excluded
        index = row if row < len(ranges) else row - len(ranges)
        lo, hi = target[index]
        target[index] = (internal, hi) if column == 1 else (lo, internal)
        self._emit_ranges(data, ranges, excluded)

    def _add_range(self, excluded_kind: bool) -> None:
        data = self._dataset
        if data is None:
            return
        lo, hi = data.curve.scan_range
        ranges, excluded = list(data.ranges), list(data.excluded)
        if excluded_kind:
            width = _EXCLUDED_FRACTION * (hi - lo)
            start = max(excluded[-1][1] if excluded else lo + 0.45 * (hi - lo), lo)
            start = min(start, hi - width)
            excluded.append((start, start + width))
        else:
            last = ranges[-1][1] if ranges else None
            ranges.append((last, hi) if last is not None and last < hi else (lo, hi))
        self._emit_ranges(data, ranges, excluded)

    def _remove_range(self) -> None:
        data = self._dataset
        row = self.table.currentRow()
        if data is None or row < 0:
            return
        ranges, excluded = list(data.ranges), list(data.excluded)
        if row < len(ranges):
            del ranges[row]
        elif row - len(ranges) < len(excluded):
            del excluded[row - len(ranges)]
        else:
            return
        self._emit_ranges(data, ranges, excluded)

    def _fit_all(self) -> None:
        data = self._dataset
        if data is not None:
            self.commandRequested.emit(set_ranges(data.name, (), ()))

    def _remove_dataset(self) -> None:
        data = self._dataset
        if data is not None:
            self.commandRequested.emit(remove_dataset(data.name))


def _display(axis: str, x: Any) -> Any:
    return to_deg(x) if axis == "angle" else to_kev(x)


def _row(*widgets: QWidget) -> QWidget:
    holder = QWidget()
    box = QHBoxLayout(holder)
    box.setContentsMargins(0, 0, 0, 0)
    for i, w in enumerate(widgets):
        box.addWidget(w, 1 if i == 0 else 0)
    return holder
