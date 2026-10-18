"""Results step: saved fits, their parameters, correlations and exports."""

import csv
import math
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QGuiApplication, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from xstack.core.parameters import Parameter
from xstack.fit.project import FitState, StoredFit, apply_fit, remove_fit, rename_fit
from xstack.fit.report import display_value, format_result
from xstack.gui.common.files import ask_save_path
from xstack.gui.fit.preview import FitPreview
from xstack.gui.fit.steps.base import StepPage

__all__ = ["ResultsPage"]

EXPORT_COLUMNS = (
    "dataset",
    "axis",
    "x",
    "x_unit",
    "measured",
    "uncertainty",
    "model",
    "fitted",
)
_COLUMNS = ("Parameter", "Value", "Uncertainty", "Relative", "Unit", "Current", "Note")
_HINT = "Run a fit on the Fit page; every fit is saved here."
_BOUND_TOLERANCE = 1e-9


def _at_bound(parameter: Parameter) -> bool:
    for bound in (parameter.min, parameter.max):
        if math.isfinite(bound) and abs(parameter.value - bound) <= _BOUND_TOLERANCE * max(
            abs(bound), 1.0
        ):
            return True
    return False


def _created(text: str) -> str:
    return text[:16].replace("T", " ")


def _goodness_label(fit: StoredFit) -> str:
    stats = fit.result.statistics
    if stats is None:
        return "no statistics"
    label = "χ²_red" if stats.weighting == "chi2" else "objective/dof"
    return f"{label} {stats.reduced_objective:.3g}"


class ResultsPage(StepPage):
    """Manage the saved fits and show the selected one.

    Every fit run is stored automatically; this page lists the stored fits, applies, renames
    and deletes them, shows parameters with uncertainties and correlations, and exports a
    report or the fitted curves.

    Signals:
        fitSelected(object): The selected `StoredFit`, or None if there is none.
    """

    fitSelected = Signal(object)

    title = "Results"
    help_text = (
        "Every fit is <b>saved</b> automatically. Select a fit to see its best-fit values with "
        "<b>standard uncertainties</b>, the goodness of fit and strongly correlated parameters. "
        "<b>Apply</b> restores the structure and parameter values of a fit, so that you can "
        "compare it with others in the plots. Export the report or the measured and fitted "
        "curves."
    )

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state: FitState | None = None
        self._preview: FitPreview | None = None
        self._names: tuple[str, ...] = ()
        self._selected: str | None = None

        layout = QVBoxLayout(self)
        self.hint = QLabel(_HINT)
        self.hint.setObjectName("muted")
        layout.addWidget(self.hint)

        group = QGroupBox("Saved fits")
        box = QVBoxLayout(group)
        self.fit_list = QListWidget()
        self.fit_list.setObjectName("saved_fits")
        self.fit_list.setMaximumHeight(130)
        self.fit_list.currentRowChanged.connect(self._row_changed)
        self.fit_list.itemDoubleClicked.connect(lambda _item: self.rename_selected())
        QShortcut(QKeySequence(Qt.Key.Key_F2), self.fit_list, self.rename_selected)
        box.addWidget(self.fit_list)
        buttons = QHBoxLayout()
        self.apply_button = QPushButton("Apply")
        self.apply_button.setToolTip("Restore the structure and parameter values of this fit")
        self.rename_button = QPushButton("Rename…")
        self.delete_button = QPushButton("Delete")
        self.apply_button.clicked.connect(self.apply_selected)
        self.rename_button.clicked.connect(self.rename_selected)
        self.delete_button.clicked.connect(self.delete_selected)
        for button in (self.apply_button, self.rename_button, self.delete_button):
            buttons.addWidget(button)
        buttons.addStretch()
        box.addLayout(buttons)
        layout.addWidget(group)

        self.summary = QLabel()
        self.summary.setObjectName("design_header")
        self.summary.setWordWrap(True)
        self.summary.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(self.summary)

        self.table = QTableWidget(0, len(_COLUMNS))
        self.table.setHorizontalHeaderLabels(list(_COLUMNS))
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table, 1)

        self.correlation_group = QGroupBox("Strong correlations (|r| ≥ 0.8)")
        corr_box = QVBoxLayout(self.correlation_group)
        self.correlation_list = QListWidget()
        self.correlation_list.setObjectName("correlations")
        self.correlation_list.setMaximumHeight(90)
        corr_box.addWidget(self.correlation_list)
        layout.addWidget(self.correlation_group)

        export = QHBoxLayout()
        self.copy_button = QPushButton("Copy report")
        self.report_button = QPushButton("Export report…")
        self.curves_button = QPushButton("Export curves (CSV)…")
        self.copy_button.clicked.connect(self.copy_report)
        self.report_button.clicked.connect(lambda: self.export_report())
        self.curves_button.clicked.connect(lambda: self.export_curves())
        for button in (self.curves_button, self.copy_button, self.report_button):
            export.addWidget(button)
        export.addStretch()
        layout.addLayout(export)
        self._refresh_enabled()

    # ------------------------------------------------------------------ API
    def selected_fit(self) -> StoredFit | None:
        """Return the selected saved fit.

        Returns:
            The fit, or None if none is selected.
        """
        if self._state is None or self._selected is None:
            return None
        return next((f for f in self._state.results if f.name == self._selected), None)

    def set_state(self, state: FitState) -> None:
        """Show the saved fits of a fit state.

        The selection is kept by name; a new fit is selected.

        Args:
            state: The current fit state.
        """
        self._state = state
        names = tuple(f.name for f in state.results)
        target: str | None
        added = [n for n in names if n not in self._names]
        if added:
            target = added[-1]
        elif self._selected in names:
            target = self._selected
        else:
            target = names[-1] if names else None
        self._names = names
        self.fit_list.blockSignals(True)
        self.fit_list.clear()
        for fit in state.results:
            item = QListWidgetItem(self._entry_text(fit))
            item.setToolTip(self._tooltip(fit))
            self.fit_list.addItem(item)
        self.fit_list.blockSignals(False)
        self._selected = target
        if target is not None:
            self.fit_list.blockSignals(True)
            self.fit_list.setCurrentRow(names.index(target))
            self.fit_list.blockSignals(False)
        self._show_fit()
        self.fitSelected.emit(self.selected_fit())

    def set_preview(self, preview: FitPreview) -> None:
        """Remember the evaluation of the current state for the curve export.

        Args:
            preview: The latest preview.
        """
        self._preview = preview

    # ------------------------------------------------------------------ actions
    def apply_selected(self) -> None:
        """Request restoring the selected fit."""
        if (fit := self.selected_fit()) is not None:
            self.commandRequested.emit(apply_fit(fit.name))

    def delete_selected(self) -> None:
        """Request deleting the selected fit."""
        if (fit := self.selected_fit()) is not None:
            self.commandRequested.emit(remove_fit(fit.name))

    def rename_selected(self) -> None:
        """Ask for a new name and request renaming the selected fit."""
        fit = self.selected_fit()
        if fit is None:
            return
        name = self._ask_name(fit.name)
        if name is not None and name != fit.name:
            self.commandRequested.emit(rename_fit(fit.name, name))

    def _ask_name(self, current: str) -> str | None:
        """Ask for the new name of a saved fit.

        Args:
            current: The current name.

        Returns:
            The entered name, or None if cancelled.
        """
        text, ok = QInputDialog.getText(self, "Rename fit", "Name:", text=current)
        return text if ok else None

    def copy_report(self) -> None:
        """Copy the text report of the selected fit to the clipboard."""
        if (fit := self.selected_fit()) is not None:
            clipboard = QApplication.clipboard() or QGuiApplication.clipboard()
            clipboard.setText(format_result(fit.result))

    def export_report(self, path: str | Path | None = None) -> bool:
        """Write the text report of the selected fit.

        Args:
            path: Target file, or None to ask with a dialog.

        Returns:
            Whether a file was written.
        """
        fit = self.selected_fit()
        if fit is None:
            return False
        if path is None:
            path = ask_save_path(self, "Export report", "Text files (*.txt)", ".txt")
            if path is None:
                return False
        try:
            Path(path).write_text(format_result(fit.result) + "\n", encoding="utf-8")
        except OSError as exc:
            self.errorOccurred.emit(f"cannot write {path}: {exc}")
            return False
        return True

    def export_curves(self, path: str | Path | None = None) -> bool:
        """Write the measured and modeled curves of the current preview as CSV.

        Args:
            path: Target file, or None to ask with a dialog.

        Returns:
            Whether a file was written.
        """
        preview = self._preview
        if preview is None or not preview.curves:
            self.errorOccurred.emit("there are no curves to export")
            return False
        if path is None:
            path = ask_save_path(self, "Export curves", "CSV files (*.csv)", ".csv")
            if path is None:
                return False
        try:
            with Path(path).open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(EXPORT_COLUMNS)
                for c in preview.curves:
                    unit = "deg" if c.axis == "angle" else "keV"
                    for i in range(len(c.x)):
                        writer.writerow(
                            [
                                c.name,
                                c.axis,
                                repr(float(c.x[i])),
                                unit,
                                repr(float(c.measured[i])),
                                "" if c.uncertainty is None else repr(float(c.uncertainty[i])),
                                "" if c.model is None else repr(float(c.model[i])),
                                int(c.fitted[i]),
                            ]
                        )
        except OSError as exc:
            self.errorOccurred.emit(f"cannot write {path}: {exc}")
            return False
        return True

    # ------------------------------------------------------------------ internals
    def _row_changed(self, row: int) -> None:
        if self._state is None:
            return
        self._selected = self._names[row] if 0 <= row < len(self._names) else None
        self._show_fit()
        self.fitSelected.emit(self.selected_fit())

    @staticmethod
    def _entry_text(fit: StoredFit) -> str:
        r = fit.result
        return f"{fit.name} — {_goodness_label(fit)} · {r.method} · {_created(r.created)}"

    @staticmethod
    def _tooltip(fit: StoredFit) -> str:
        r = fit.result
        lines = [
            f"{r.method}, {r.evaluations} evaluations, {r.elapsed:.1f} s",
            f"created {_created(r.created)}",
        ]
        if fit.notes:
            lines.append(fit.notes)
        return "\n".join(lines)

    def _refresh_enabled(self) -> None:
        has = self.selected_fit() is not None
        for button in (
            self.apply_button,
            self.rename_button,
            self.delete_button,
            self.copy_button,
            self.report_button,
        ):
            button.setEnabled(has)
        self.curves_button.setEnabled(has)
        self.hint.setVisible(not self._names)

    def _show_fit(self) -> None:
        fit = self.selected_fit()
        self._refresh_enabled()
        self.table.setRowCount(0)
        self.correlation_list.clear()
        if fit is None:
            self.summary.setText("")
            self.summary.setVisible(False)
            return
        self.summary.setVisible(True)
        self.summary.setText(self._summary(fit))
        stats = fit.result.statistics
        current: dict[str, Parameter] = {}
        if self._state is not None:
            try:
                current = dict(self._state.problem().parameters.items())
            except ValueError:
                current = {}
        if stats is not None:
            for name, value, error in zip(stats.names, stats.values, stats.errors, strict=True):
                self._add_row(fit, name, value, error, current.get(name))
            pairs = stats.strong_correlations(0.8)
            for a, b, c in pairs:
                self.correlation_list.addItem(f"{a}  ↔  {b}:   r = {c:+.3f}")
            if not pairs:
                self.correlation_list.addItem("none")
        self.table.resizeColumnsToContents()

    def _summary(self, fit: StoredFit) -> str:
        r = fit.result
        stats = r.statistics
        parts = []
        if stats is not None:
            label = "χ²_red" if stats.weighting == "chi2" else "objective/dof"
            parts.append(f"<b>{label} = {stats.reduced_objective:.3f}</b>")
            parts.append(f"{stats.points} points")
            parts.append(f"{len(stats.names)} free parameters")
        parts.append(r.method)
        parts.append(f"{r.evaluations} evaluations")
        parts.append(f"{r.elapsed:.1f} s")
        parts.append(_created(r.created))
        text = " · ".join(parts)
        if r.cancelled:
            text += "<br>The fit was cancelled."
        if stats is None:
            text += f"<br>No uncertainties: {r.note}" if r.note else "<br>No uncertainties."
        elif not stats.determined:
            text += "<br>Warning: the data do not determine all free parameters."
        return text

    def _add_row(
        self, fit: StoredFit, name: str, value: float, error: float, current: Parameter | None
    ) -> None:
        stored = fit.result.parameters.get(name)
        probe = stored if stored is not None else current
        if probe is None:
            shown, unit, err, rel = value, "", math.nan, math.nan
        else:
            shown, unit = display_value(probe, value)
            err = abs(display_value(probe, error)[0]) if math.isfinite(error) else math.nan
            rel = 100 * err / abs(shown) if shown and math.isfinite(err) else math.nan
        notes = []
        if stored is not None and _at_bound(stored):
            notes.append("at bound")
        if not math.isfinite(error):
            notes.append("undetermined")
        cells = [
            name,
            f"{shown:.6g}",
            f"± {err:.2g}" if math.isfinite(err) else "",
            f"{rel:.2g} %" if math.isfinite(rel) else "",
            unit,
            f"{display_value(current)[0]:.6g}" if current is not None else "",
            ", ".join(notes),
        ]
        row = self.table.rowCount()
        self.table.insertRow(row)
        for column, text in enumerate(cells):
            item = QTableWidgetItem(text)
            if 1 <= column <= 3 or column == 5:
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.table.setItem(row, column, item)
