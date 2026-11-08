"""Dialog that imports a measured curve from a data file."""

import csv
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
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

from xstack.core.io.data import read_curves
from xstack.core.io.delimited import UNITS, CsvLayout, Quantity
from xstack.core.io.reflectivity import REFLECTIVITY_COLUMNS
from xstack.core.measurement import MeasuredCurve
from xstack.core.physics.fresnel import Polarization
from xstack.core.units import deg, kev, to_deg, to_kev, wavelength_to_energy
from xstack.fit.dataset import Dataset
from xstack.gui.common.style import set_variant

__all__ = [
    "ENERGY_PRESETS",
    "ImportDialog",
    "energy_hint",
    "unique_identifier",
]

ENERGY_PRESETS: dict[str, float] = {"Cu Kα": 8.048, "Mo Kα": 17.479, "Ag Kα": 22.163}
"""Photon energies in keV offered for angle scans."""

_FILE_FILTER = "Data files (*.csv *.tsv *.txt *.dat);;All files (*)"
_QUANTITIES: tuple[tuple[str, Quantity], ...] = (
    ("grazing angle θ", "angle"),
    ("scattering angle 2θ", "two_theta"),
    ("momentum transfer q", "q"),
    ("energy", "energy"),
)
_NONE = "(none)"
_PREVIEW_ROWS = 6
_DELIMITER_NAMES = {",": "comma", ";": "semicolon", "\t": "tab", "": "whitespace"}


def unique_identifier(text: str, used: Sequence[str] = ()) -> str:
    """Turn text into a dataset name that is an identifier and not yet used.

    Characters other than letters, digits and underscore become underscores, a name starting
    with a digit gets the prefix ``d_``, and a used name gets the suffix ``_2``, ``_3``, ...

    Args:
        text: Proposed name, e.g. a file stem.
        used: Names that must not be returned.

    Returns:
        The identifier.
    """
    name = re.sub(r"[^0-9A-Za-z_]", "_", text) or "data"
    if name[0].isdigit():
        name = f"d_{name}"
    candidate, index = name, 2
    while candidate in used:
        candidate = f"{name}_{index}"
        index += 1
    return candidate


@dataclass(frozen=True)
class _Raw:
    """The rows of a delimited file as text."""

    header: tuple[str, ...] | None
    rows: tuple[tuple[str, ...], ...]
    delimiter: str
    energy: float | None = None

    @property
    def columns(self) -> int:
        return max(len(r) for r in self.rows)


def _is_number(text: str) -> bool:
    try:
        float(text.strip())
    except ValueError:
        return False
    return True


_ENERGY = re.compile(r"energy\D{0,30}?(\d+(?:\.\d+)?)\s*(kev|ev)\b", re.IGNORECASE)
_WAVELENGTH = re.compile(
    r"(?:wavelength|lambda|λ)\D{0,30}?(\d+(?:\.\d+)?)\s*(å|a|angstrom|nm)\b", re.IGNORECASE
)


def energy_hint(comments: Sequence[str]) -> float | None:
    """Return the photon energy stated in comment lines of a data file.

    Recognizes an energy in keV or eV after the word "energy", or a wavelength in Å or nm after
    "wavelength", "lambda" or "λ".

    Args:
        comments: The comment lines.

    Returns:
        The photon energy in eV, or ``None`` if no line states one.

    Examples:
        >>> energy_hint(["# Photon energy: 17.479 keV"])
        17479.0
        >>> round(energy_hint(["# wavelength = 1.5406 A"]))
        8048
    """
    for line in comments:
        match = _ENERGY.search(line)
        if match:
            value = float(match.group(1))
            return value * 1000.0 if match.group(2).lower() == "kev" else value
        match = _WAVELENGTH.search(line)
        if match and float(match.group(1)) > 0:
            value = float(match.group(1))
            nm = value if match.group(2).lower() == "nm" else value / 10.0
            return float(wavelength_to_energy(nm))
    return None


def _scan(path: Path) -> _Raw:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        text = path.read_text(encoding="latin-1")
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    if not lines:
        raise ValueError("the file contains no data")
    delimiter = next((d for d in (",", ";", "\t") if d in lines[-1]), "")
    if delimiter:
        rows = [
            tuple(f.strip() for f in next(csv.reader([ln], delimiter=delimiter))) for ln in lines
        ]
    else:
        rows = [tuple(ln.split()) for ln in lines]
    header = None
    if not all(_is_number(f) for f in rows[0]):
        header = rows[0]
        rows = rows[1:]
    if not rows:
        raise ValueError("the file contains no data rows")
    comments = [ln for ln in text.splitlines() if ln.lstrip().startswith("#")]
    return _Raw(header, tuple(rows), delimiter, energy_hint(comments))


class ImportDialog(QDialog):
    """Dialog "Add measured data": choose a file, describe its layout and preview the result.

    The file is read with the real parser whenever an entry changes, so the preview shows
    exactly what will be added. Files written by xstack's reflectivity export are recognized;
    their layout is fixed and every curve in them becomes a dataset.

    Args:
        used_names: Names of the existing datasets, which new names must avoid.
        parent: Parent widget.
    """

    def __init__(self, used_names: Sequence[str] = (), parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add measured data")
        self.resize(720, 600)
        self.setAcceptDrops(True)
        self._used = tuple(used_names)
        self._raw: _Raw | None = None
        self._export = False
        self._curves: tuple[MeasuredCurve, ...] = ()
        self._names_edited = False
        self._loading = False

        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Path of a CSV, TSV or column file")
        self.path_edit.editingFinished.connect(lambda: self.set_path(self.path_edit.text()))
        self.browse_button = QPushButton("Browse…")
        self.browse_button.clicked.connect(self._browse)
        form.addRow("File", _row(self.path_edit, self.browse_button))
        self.format_label = QLabel("No file selected")
        form.addRow("Format", self.format_label)
        self.note_label = QLabel(
            "This file was written by xstack: its layout is known and every curve in it is "
            "added as a dataset."
        )
        self.note_label.setWordWrap(True)
        self.note_label.setProperty("muted", True)
        form.addRow("", self.note_label)

        self.x_combo = QComboBox()
        self.r_combo = QComboBox()
        self.sigma_combo = QComboBox()
        for combo in (self.x_combo, self.r_combo, self.sigma_combo):
            combo.currentIndexChanged.connect(self._refresh)
        self.x_combo.setToolTip("Column of the scanned variable.")
        self.r_label = QLabel("R")
        columns = QHBoxLayout()
        for label, combo in (
            (QLabel("x"), self.x_combo),
            (self.r_label, self.r_combo),
            (QLabel("σ"), self.sigma_combo),
        ):
            columns.addWidget(label)
            columns.addWidget(combo)
        columns.addStretch()
        form.addRow("Columns", columns)

        self.quantity_combo = QComboBox()
        for label, quantity in _QUANTITIES:
            self.quantity_combo.addItem(label, quantity)
        self.unit_combo = QComboBox()
        self.quantity_combo.currentIndexChanged.connect(self._quantity_changed)
        self.unit_combo.currentIndexChanged.connect(self._refresh)
        form.addRow("x is", _row(self.quantity_combo, self.unit_combo))

        self.energy_spin = QDoubleSpinBox()
        self.energy_spin.setDecimals(4)
        self.energy_spin.setRange(0.01, 1000.0)
        self.energy_spin.setSuffix(" keV")
        self.energy_spin.setKeyboardTracking(False)
        self.energy_spin.valueChanged.connect(self._energy_changed)
        self.preset_combo = QComboBox()
        self.preset_combo.addItems([*ENERGY_PRESETS, "custom"])
        self.preset_combo.currentIndexChanged.connect(self._preset_changed)
        self.energy_widget = _row(self.energy_spin, self.preset_combo)
        self.energy_label = QLabel("Photon energy")
        form.addRow(self.energy_label, self.energy_widget)
        self.angle_spin = QDoubleSpinBox()
        self.angle_spin.setDecimals(4)
        self.angle_spin.setRange(0.0001, 90.0)
        self.angle_spin.setValue(1.0)
        self.angle_spin.setSuffix(" deg")
        self.angle_spin.setKeyboardTracking(False)
        self.angle_spin.valueChanged.connect(self._refresh)
        self.angle_label = QLabel("Grazing angle")
        form.addRow(self.angle_label, self.angle_spin)

        self.polarization_combo = QComboBox()
        self.polarization_combo.addItems(["unpolarized", "s", "p"])
        self.polarization_combo.currentIndexChanged.connect(self._refresh)
        self.percent_check = QCheckBox("reflectivity in %")
        self.percent_check.toggled.connect(self._refresh)
        form.addRow("Polarization", _row(self.polarization_combo, self.percent_check))

        self.values_combo = QComboBox()
        self.values_combo.addItem("reflectivity", False)
        self.values_combo.addItem("detector counts", True)
        self.values_combo.currentIndexChanged.connect(self._values_changed)
        form.addRow("Values", self.values_combo)
        self.normalize_combo = QComboBox()
        self.normalize_combo.addItem("largest count", False)
        self.normalize_combo.addItem("full-scale count", True)
        self.normalize_combo.setToolTip(
            "Count that corresponds to reflectivity 1: the largest count of the curve or a "
            "given count such as the direct-beam count."
        )
        self.normalize_combo.currentIndexChanged.connect(self._values_changed)
        self.full_scale_spin = QDoubleSpinBox()
        self.full_scale_spin.setDecimals(0)
        self.full_scale_spin.setRange(1.0, 1e15)
        self.full_scale_spin.setValue(1e6)
        self.full_scale_spin.setKeyboardTracking(False)
        self.full_scale_spin.setToolTip("Count that corresponds to reflectivity 1.")
        self.full_scale_spin.valueChanged.connect(self._refresh)
        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setDecimals(6)
        self.scale_spin.setRange(1e-6, 1e6)
        self.scale_spin.setValue(1.0)
        self.scale_spin.setPrefix("factor ")
        self.scale_spin.setKeyboardTracking(False)
        self.scale_spin.setToolTip("Factor applied to the normalized reflectivity.")
        self.scale_spin.valueChanged.connect(self._refresh)
        self.normalize_widget = _row(self.normalize_combo, self.full_scale_spin, self.scale_spin)
        self.normalize_label = QLabel("Normalize to")
        form.addRow(self.normalize_label, self.normalize_widget)

        self.name_edit = QLineEdit()
        self.name_edit.textEdited.connect(self._name_edited)
        self.name_edit.setToolTip("Identifier of the dataset; also the prefix of its parameters.")
        form.addRow("Dataset name", self.name_edit)
        layout.addLayout(form)

        layout.addWidget(QLabel("<b>Preview</b> (file columns → values used)"))
        self.preview_table = QTableWidget(0, 0)
        self.preview_table.verticalHeader().setVisible(False)
        self.preview_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.preview_table.setAlternatingRowColors(True)
        self.preview_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.preview_table, 1)
        self.count_label = QLabel()
        self.count_label.setProperty("muted", True)
        layout.addWidget(self.count_label)
        self.error_label = QLabel()
        self.error_label.setWordWrap(True)
        self.error_label.setStyleSheet("color: #c0392b;")
        layout.addWidget(self.error_label)

        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        self.ok_button = QPushButton("Add dataset")
        set_variant(self.ok_button, "primary")
        self.buttons.addButton(self.ok_button, QDialogButtonBox.ButtonRole.AcceptRole)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        self._loading = True
        self.energy_spin.setValue(ENERGY_PRESETS["Cu Kα"])
        self._fill_units()
        self._loading = False
        self._update_visibility()
        self._refresh()

    # --- public API ---------------------------------------------------------------------

    def set_path(self, path: str | Path) -> None:
        """Choose the file to import and read it.

        Layout controls are reset to the file's columns; errors are shown in the dialog.

        Args:
            path: The data file.
        """
        text = str(path).strip()
        self.path_edit.setText(text)
        self._raw = None
        self._export = False
        self._names_edited = False
        self._loading = True
        for combo in (self.x_combo, self.r_combo, self.sigma_combo):
            combo.clear()
        if not text:
            self._loading = False
            self._refresh()
            return
        try:
            self._raw = _scan(Path(text))
        except (ValueError, OSError) as error:
            self._loading = False
            self._refresh(error=str(error))
            return
        self._export = self._raw.header == REFLECTIVITY_COLUMNS
        self._fill_columns(self._raw)
        if self._raw.energy is not None:
            self.energy_spin.setValue(to_kev(self._raw.energy))
        self.name_edit.setText(unique_identifier(Path(text).stem, self._used))
        self._loading = False
        self._refresh()

    def layout_options(self) -> CsvLayout | None:
        """Return the file layout described by the dialog.

        Returns:
            The layout, or ``None`` if no file is chosen, the file is an xstack export (read
            without a layout) or the entries are inconsistent.
        """
        if self._raw is None or self._export:
            return None
        try:
            return self._build_layout()
        except ValueError:
            return None

    def datasets(self) -> list[Dataset]:
        """Return the datasets that "Add dataset" adds.

        Returns:
            One dataset per curve read; empty if the file or entries are invalid.
        """
        return [Dataset.create(n, c) for n, c in self._named_curves()]

    # --- internals ----------------------------------------------------------------------

    def _build_layout(self) -> CsvLayout:
        quantity = cast(Quantity, self.quantity_combo.currentData())
        sigma = self.sigma_combo.currentData()
        energy = kev(self.energy_spin.value()) if quantity != "energy" else None
        angle = deg(self.angle_spin.value()) if quantity == "energy" else None
        counts = bool(self.values_combo.currentData())
        if counts:
            scale = self.scale_spin.value()
        else:
            scale = 0.01 if self.percent_check.isChecked() else 1.0
        full_scale = None
        if counts and self.normalize_combo.currentData():
            full_scale = self.full_scale_spin.value()
        return CsvLayout(
            x_column=int(self.x_combo.currentData()),
            reflectivity_column=int(self.r_combo.currentData()),
            uncertainty_column=None if sigma is None else int(sigma),
            quantity=quantity,
            unit=self.unit_combo.currentText() or None,
            reflectivity_scale=scale,
            counts=counts,
            full_scale=full_scale,
            energy=energy,
            angle=angle,
            polarization=cast(Polarization, self.polarization_combo.currentText()),
            name=self.name_edit.text() or "",
        )

    def _named_curves(self) -> list[tuple[str, MeasuredCurve]]:
        out: list[tuple[str, MeasuredCurve]] = []
        used = list(self._used)
        if self._raw is None or not self._curves:
            return out
        for curve in self._curves:
            if self._export:
                name = unique_identifier(curve.name, used)
            else:
                name = self.name_edit.text().strip()
                if not name.isidentifier() or name in used:
                    return []
            if curve.fixed is None and not self._export:
                value = self.angle_spin.value()
                curve = curve.with_fixed(deg(value) if curve.axis == "energy" else kev(value))
            used.append(name)
            out.append((name, curve))
        return out

    def _fill_columns(self, raw: _Raw) -> None:
        names = [
            f"{h}" if raw.header else f"column {i + 1}"
            for i, h in enumerate(raw.header or [""] * raw.columns)
        ]
        for combo in (self.x_combo, self.r_combo, self.sigma_combo):
            combo.clear()
        self.sigma_combo.addItem(_NONE, None)
        for i, name in enumerate(names):
            for combo in (self.x_combo, self.r_combo, self.sigma_combo):
                combo.addItem(name, i)
        self.x_combo.setCurrentIndex(0)
        self.r_combo.setCurrentIndex(min(1, len(names) - 1))
        lowered = [n.casefold() for n in names]
        sigma = next(
            (
                i
                for i, n in enumerate(lowered)
                if i > 1 and re.search(r"err|sigma|σ|unc|^d", n) is not None
            ),
            None,
        )
        self.sigma_combo.setCurrentIndex(0 if sigma is None else sigma + 1)
        if raw.header:
            self._guess_quantity(lowered[0])

    def _guess_quantity(self, header: str) -> None:
        if re.search(r"2\s*(theta|θ)|two", header):
            quantity: Quantity = "two_theta"
        elif re.match(r"q\b|q_|momentum", header):
            quantity = "q"
        elif re.search(r"energy|\bev\b|kev", header):
            quantity = "energy"
        else:
            quantity = "angle"
        self.quantity_combo.setCurrentIndex(self.quantity_combo.findData(quantity))
        self._fill_units()
        for unit in UNITS[quantity]:
            if re.search(rf"(?<![a-z]){re.escape(unit.casefold())}(?![a-z])", header):
                self.unit_combo.setCurrentText(unit)
        self._update_visibility()

    def _fill_units(self) -> None:
        was = self._loading
        self._loading = True
        self.unit_combo.clear()
        self.unit_combo.addItems(list(UNITS[cast(Quantity, self.quantity_combo.currentData())]))
        self._loading = was

    def _quantity_changed(self) -> None:
        if self._loading:
            return
        self._fill_units()
        self._update_visibility()
        self._refresh()

    def _values_changed(self) -> None:
        if self._loading:
            return
        self._update_visibility()
        self._refresh()

    def _update_visibility(self) -> None:
        counts = bool(self.values_combo.currentData())
        self.percent_check.setHidden(counts)
        self.normalize_label.setHidden(not counts)
        self.normalize_widget.setHidden(not counts)
        self.full_scale_spin.setHidden(not self.normalize_combo.currentData())
        self.r_label.setText("N" if counts else "R")
        self.r_combo.setToolTip(
            "Column of the detector counts." if counts else "Column of the measured reflectivity."
        )
        self.sigma_combo.setToolTip(
            "Column of the count uncertainty; without one the uncertainty is √N, at least one "
            "count."
            if counts
            else "Column of the reflectivity uncertainty, if the file has one."
        )
        energy_scan = self.quantity_combo.currentData() == "energy"
        self.energy_label.setHidden(energy_scan)
        self.energy_widget.setHidden(energy_scan)
        self.angle_label.setHidden(not energy_scan)
        self.angle_spin.setHidden(not energy_scan)
        self.note_label.setHidden(not self._export)

    def _preset_changed(self) -> None:
        if self._loading:
            return
        name = self.preset_combo.currentText()
        if name in ENERGY_PRESETS:
            self.energy_spin.setValue(ENERGY_PRESETS[name])
        else:
            self._refresh()

    def _energy_changed(self) -> None:
        if self._loading:
            return
        match = next(
            (n for n, e in ENERGY_PRESETS.items() if abs(e - self.energy_spin.value()) < 5e-5),
            "custom",
        )
        self._loading = True
        self.preset_combo.setCurrentText(match)
        self._loading = False
        self._refresh()

    def _name_edited(self) -> None:
        self._names_edited = True
        self._refresh()

    def _browse(self) -> None:
        start = str(Path(self.path_edit.text()).parent) if self.path_edit.text() else ""
        path, _ = QFileDialog.getOpenFileName(self, "Add measured data", start, _FILE_FILTER)
        if path:
            self.set_path(path)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        """Accept dragged files."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        """Use a dropped file."""
        urls = event.mimeData().urls()
        if urls and urls[0].isLocalFile():
            self.set_path(urls[0].toLocalFile())

    def _refresh(self, *_: object, error: str | None = None) -> None:
        if self._loading:
            return
        self._curves = ()
        for widget in (
            self.x_combo,
            self.r_combo,
            self.sigma_combo,
            self.quantity_combo,
            self.unit_combo,
            self.energy_spin,
            self.preset_combo,
            self.angle_spin,
            self.polarization_combo,
            self.percent_check,
            self.values_combo,
            self.normalize_combo,
            self.full_scale_spin,
            self.scale_spin,
        ):
            widget.setEnabled(self._raw is not None and not self._export)
        self.name_edit.setEnabled(self._raw is not None and not self._export)
        self._update_visibility()
        self.preview_table.clear()
        self.preview_table.setRowCount(0)
        self.preview_table.setColumnCount(0)
        self.count_label.setText("")
        self.format_label.setText("No file selected" if self._raw is None else self._format_text())
        message = error or ""
        if self._raw is not None and not message:
            message = self._read()
        if self._raw is not None and not message and not self._export:
            name = self.name_edit.text().strip()
            if not name.isidentifier():
                message = f"{name!r} is not a valid dataset name (letters, digits and _)"
            elif name in self._used:
                message = f"a dataset named {name!r} already exists"
        self.error_label.setText(message)
        self.ok_button.setEnabled(bool(self._curves) and not message)

    def _format_text(self) -> str:
        raw = self._raw
        assert raw is not None
        if self._export:
            return "xstack reflectivity export"
        header = "header row" if raw.header else "no header"
        text = (
            f"delimited text · {_DELIMITER_NAMES[raw.delimiter]} · {header} · {raw.columns} columns"
        )
        if raw.energy is not None:
            text += f" · energy {to_kev(raw.energy):.4g} keV from the file header"
        return text

    def _read(self) -> str:
        raw = self._raw
        assert raw is not None
        try:
            options = None if self._export else self._build_layout()
            self._curves = read_curves(self.path_edit.text().strip(), options)
        except (ValueError, OSError) as error:
            return str(error)
        self._fill_preview(raw)
        return ""

    def _fill_preview(self, raw: _Raw) -> None:
        table = self.preview_table
        total = len(raw.rows)
        read = sum(c.size for c in self._curves)
        if self._export:
            headers = list(raw.header or ())
            table.setColumnCount(len(headers))
            table.setHorizontalHeaderLabels(headers)
            table.setRowCount(min(_PREVIEW_ROWS, total))
            for r, row in enumerate(raw.rows[:_PREVIEW_ROWS]):
                for c, text in enumerate(row):
                    table.setItem(r, c, QTableWidgetItem(text))
            names = ", ".join(c.name for c in self._curves)
            self.count_label.setText(f"{len(self._curves)} curves ({names}), {total} rows read")
            return
        x = int(self.x_combo.currentData())
        r_col = int(self.r_combo.currentData())
        sigma = self.sigma_combo.currentData()
        curve = self._curves[0]
        energy_axis = curve.axis == "energy"
        converted = ["→ E (keV)" if energy_axis else "→ θ (deg)", "→ R", "→ σ"]
        file_cols = [x, r_col] + ([] if sigma is None else [int(sigma)])
        labels = [self.x_combo.itemText(x), self.r_combo.itemText(r_col)]
        if sigma is not None:
            labels.append(self.sigma_combo.itemText(int(sigma) + 1))
        table.setColumnCount(len(file_cols) + 3)
        table.setHorizontalHeaderLabels(labels + converted)
        n = min(_PREVIEW_ROWS, curve.size)
        table.setRowCount(n)
        xs = (to_kev if energy_axis else to_deg)(np.array(curve.x))
        for i in range(n):
            for j, col in enumerate(file_cols):
                text = raw.rows[i][col] if i < len(raw.rows) and col < len(raw.rows[i]) else ""
                table.setItem(i, j, QTableWidgetItem(text))
            base = len(file_cols)
            table.setItem(i, base, QTableWidgetItem(f"{xs[i]:.4f}"))
            table.setItem(i, base + 1, QTableWidgetItem(f"{curve.reflectivity[i]:.4g}"))
            u = "—" if curve.uncertainty is None else f"{curve.uncertainty[i]:.3g}"
            table.setItem(i, base + 2, QTableWidgetItem(u))
        self.count_label.setText(f"{read} rows read, {total - read} skipped")


def _row(*widgets: QWidget) -> QWidget:
    holder = QWidget()
    box = QHBoxLayout(holder)
    box.setContentsMargins(0, 0, 0, 0)
    for i, w in enumerate(widgets):
        box.addWidget(w, 1 if i == 0 else 0)
    return holder
