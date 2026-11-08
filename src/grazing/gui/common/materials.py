"""Page for the materials of the coating roles and the substrate, shared by the applications."""

from dataclasses import replace
from typing import Any

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QFrame,
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

from grazing.core.materials import (
    DEFAULT_PROVIDER,
    Material,
    MaterialLibrary,
    TabulatedProvider,
    default_library,
    provider_library,
    providers,
)
from grazing.core.parameters import Parameter
from grazing.core.structure.coating import SUBSTRATE, Coating, Mirror, density_parameter_name
from grazing.core.structure.coatings.bilayer import INTERLAYER_ROLES
from grazing.core.structure.coatings.composite import Composite
from grazing.gui.common.coating import COATING_TYPES
from grazing.gui.common.pages import StepPage, StructureCommands
from grazing.gui.common.style import COLORS
from grazing.gui.common.style import PLOT_PALETTE as PALETTE

__all__ = ["PROVIDER_LABELS", "MaterialDialog", "MaterialsPage", "provider_choices"]

_SUBSTRATE = SUBSTRATE
_NONE = "(none)"

INTERLAYER_START = 0.5
"""Thickness in nm given to an interlayer that gets a material while its thickness is 0."""

PROVIDER_LABELS = {
    "henke_xraylib": "Henke (CXRO) up to 30 keV, xraylib above",
    "henke": "Henke (CXRO), 10 eV to 30 keV",
    "chantler": "Chantler (NIST FFAST), 10.7 eV to 433 keV",
    "xraylib": "xraylib, 100 eV to 1 MeV",
}
"""Descriptions of the optical-constants providers offered for a project."""

_PLOT_RANGE = (100.0, 1e5)


def _thickness_name(interlayer_role: str) -> str:
    """Name of the thickness parameter of a (possibly block-prefixed) interlayer role."""
    prefix, dot, local = interlayer_role.rpartition(".")
    return f"{prefix}{dot}t_{local}"


def provider_choices() -> list[tuple[str, str]]:
    """Return the optical-constants providers a project can use, with descriptions.

    The providers of `PROVIDER_LABELS` come first, then their photoabsorption-only variants,
    then other registered providers. Tables of single materials (`TabulatedProvider`) are
    left out, since they don't apply to all materials.

    Returns:
        Pairs of provider name and description.
    """
    choices = [(name, label) for name, label in PROVIDER_LABELS.items() if name in providers]
    choices += [
        (f"{name}_photo", f"{label}, photoabsorption only")
        for name, label in PROVIDER_LABELS.items()
        if f"{name}_photo" in providers
    ]
    listed = {name for name, _ in choices}
    choices += [
        (name, name)
        for name in providers.names()
        if name not in listed and not isinstance(providers.get(name), TabulatedProvider)
    ]
    return choices


class MaterialDialog(QDialog):
    """Dialog to define a new material (name, formula, density).

    Args:
        parent: Parent widget.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("New material")
        layout = QFormLayout(self)
        self.name = QLineEdit()
        self.formula = QLineEdit()
        self.density = QLineEdit()
        self.name.setToolTip("Display name; defaults to the formula.")
        self.formula.setToolTip("Chemical formula, for example Si, MoSi2 or B4C.")
        self.density.setToolTip("Mass density of the film in g/cm³.")
        layout.addRow("Name", self.name)
        layout.addRow("Chemical formula", self.formula)
        layout.addRow("Density (g/cm³)", self.density)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def material(self) -> Material:
        """Return the entered material.

        Raises:
            ValueError: If the entries are invalid.
        """
        name = self.name.text().strip() or self.formula.text().strip()
        return Material(name, self.formula.text().strip(), float(self.density.text()))


class MaterialsPage(StepPage):
    """Materials of the coating roles and the substrate, and their optical constants.

    Args:
        commands: Factories of the application's commands that edit the mirror.
        library: Material library offering the selectable materials.
        parent: Parent widget.
    """

    title = "Materials"
    help_text = (
        "<b>Materials.</b> Each role of the coating (for example the absorber and the spacer "
        "of a multilayer) and the substrate is assigned a material here. The optical constants "
        "delta and beta of these materials, evaluated at the energies of your goal, determine "
        "the reflectivity of every later step. Choose a material from the library or define "
        "your own from a chemical formula and density; the density may be edited if your "
        "deposited film differs from the bulk value. <b>Optical constants</b> selects the "
        "tables the optical constants of all materials are computed from, for the whole "
        "project. The plot below shows the resulting optical constants over a wide energy "
        "range."
    )

    def __init__(
        self,
        commands: StructureCommands,
        library: MaterialLibrary | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.commands = commands
        self.library = library or default_library()
        self._mirror: Mirror | None = None
        self._provider = DEFAULT_PROVIDER
        self._loading = False
        self._rows: list[str | None] = []
        layout = QVBoxLayout(self)
        provider_row = QHBoxLayout()
        provider_row.addWidget(QLabel("<b>Optical constants</b>"))
        self.provider_combo = QComboBox()
        for name, label in provider_choices():
            self.provider_combo.addItem(label, name)
        self.provider_combo.setToolTip(
            "Tables of atomic scattering factors from which δ and β of all materials are "
            "computed, for the whole project. Henke (CXRO) data are measured-based and the best "
            "choice below about 1 keV; Chantler (NIST FFAST) and xraylib cover higher energies. "
            "f2 includes Compton scattering unless the choice says photoabsorption only."
        )
        self.provider_combo.activated.connect(self._provider_chosen)
        provider_row.addWidget(self.provider_combo, 1)
        layout.addLayout(provider_row)
        layout.addWidget(QLabel("<b>Materials</b> of the coating roles and the substrate"))
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(
            ["Role", "Material", "Density (g/cm³)", "Density parameter"]
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setShowGrid(False)
        self.table.setFrameShape(QFrame.Shape.NoFrame)
        self.table.setAlternatingRowColors(True)
        self.table.setToolTip(
            "One row per coating role and the substrate. Choose the material and edit its "
            "density. Density parameter makes the density a parameter (rho_<role>) that can be "
            "varied like the coating parameters."
        )
        self.table.itemChanged.connect(self._density_edited)
        layout.addWidget(self.table)
        self.new_material = QPushButton("New material…")
        self.new_material.setToolTip("Define a material from a chemical formula and a density.")
        self.new_material.clicked.connect(self._define_material)
        layout.addWidget(self.new_material)
        layout.addWidget(QLabel("Optical constants (δ solid, β dashed)"))
        self.plot = pg.PlotWidget()
        self.plot.setLogMode(x=True, y=True)
        self.plot.setLabel("bottom", "Energy (keV)")
        self.plot.addLegend(offset=(-10, 10))
        self.plot.setMinimumHeight(220)
        self.plot.setToolTip("Optical constants delta (solid) and beta (dashed) versus energy.")
        layout.addWidget(self.plot)

    def set_state(self, state: Any) -> None:
        """Show the materials of the mirror of a document state.

        Args:
            state: A document state with a ``mirror`` attribute.
        """
        provider = getattr(state, "optical_constants", DEFAULT_PROVIDER)
        if state.mirror == self._mirror and provider == self._provider:
            return
        self._mirror = state.mirror
        self._provider = provider
        self._show_provider()
        self._load()

    def _show_provider(self) -> None:
        index = self.provider_combo.findData(self._provider)
        if index < 0:
            self.provider_combo.addItem(self._provider, self._provider)
            index = self.provider_combo.count() - 1
        self.provider_combo.setCurrentIndex(index)

    def _provider_chosen(self, index: int) -> None:
        name = self.provider_combo.itemData(index)
        if name != self._provider:
            self.commandRequested.emit(self.commands.set_optical_constants(name))

    def _entries(self) -> list[tuple[str | None, Material | None, str]]:
        """Table rows as role, material and heading text; heading rows have no role."""
        if self._mirror is None:
            return []
        coating = self._mirror.coating
        entries: list[tuple[str | None, Material | None, str]] = []
        if isinstance(coating, Composite):
            for i, name in enumerate(coating.block_names(), start=1):
                block = coating.block(name)
                label = COATING_TYPES.get(block.type_name, block.type_name)
                entries.append((None, None, f"{i}  {name} \u2014 {label}"))
                entries += [
                    (f"{name}.{role}", material, "") for role, material in self._block_roles(block)
                ]
        else:
            entries += [(role, material, "") for role, material in self._block_roles(coating)]
        return [*entries, (_SUBSTRATE, self._mirror.substrate, "")]

    @staticmethod
    def _block_roles(coating: Coating) -> list[tuple[str, Material | None]]:
        """Roles of a coating with their materials; enabled interlayers also without material."""
        roles: list[tuple[str, Material | None]] = list(coating.materials.items())
        if getattr(coating, "use_interlayers", False):
            roles += [
                (role, None)
                for role in INTERLAYER_ROLES
                if role in coating.optional_roles and role not in coating.materials
            ]
        return roles

    def _heading_row(self, row: int, text: str) -> None:
        item = QTableWidgetItem(text)
        item.setFlags(Qt.ItemFlag.ItemIsEnabled)
        font = QFont(item.font())
        font.setBold(True)
        item.setFont(font)
        item.setBackground(QBrush(QColor(COLORS["accent_soft"])))
        self.table.setItem(row, 0, item)
        self.table.setSpan(row, 0, 1, self.table.columnCount())

    def _load(self) -> None:
        self._loading = True
        entries = self._entries()
        self._rows = [role for role, _, _ in entries]
        self.table.clearContents()
        self.table.clearSpans()
        self.table.setRowCount(0)
        self.table.setRowCount(len(entries))
        names = list(self.library.names())
        for row, (role, material, heading) in enumerate(entries):
            if role is None:
                self._heading_row(row, heading)
                continue
            local = role.rpartition(".")[2]
            role_item = QTableWidgetItem(role)
            role_item.setFlags(role_item.flags() & ~role_item.flags().ItemIsEditable)
            self.table.setItem(row, 0, role_item)
            combo = QComboBox()
            if local in INTERLAYER_ROLES:
                combo.addItem(_NONE)
            if material is not None and material.name not in names:
                combo.addItem(material.name)
            combo.addItems(names)
            combo.setCurrentText(_NONE if material is None else material.name)
            combo.activated.connect(
                lambda _=0, r=role, c=combo: self._material_chosen(r, c.currentText())
            )
            combo.setToolTip(
                f"Material of the role '{role}'."
                if local not in INTERLAYER_ROLES
                else f"Material of the {local.replace('_', ' ')}, or {_NONE} for no interlayer. "
                f"Its thickness is the parameter {_thickness_name(role)}."
            )
            self.table.setCellWidget(row, 1, combo)
            if material is None:
                empty = QTableWidgetItem("")
                empty.setFlags(Qt.ItemFlag.NoItemFlags)
                self.table.setItem(row, 2, empty)
                continue
            parameter = self._density_parameter(role)
            density = material.density if parameter is None else parameter.value
            density_item = QTableWidgetItem(f"{density:g}")
            density_item.setToolTip(
                "Mass density in g/cm³ (editable)."
                if parameter is None
                else f"Value of the density parameter {parameter.name} in g/cm³ (editable); "
                "vary it and set its bounds in the parameter table."
            )
            self.table.setItem(row, 2, density_item)
            check = QCheckBox()
            check.setChecked(parameter is not None)
            check.setToolTip(
                f"Make the density of '{role}' a parameter ({density_parameter_name(role)}) that "
                "can be varied, e.g. fitted to a measurement."
            )
            check.toggled.connect(lambda on, r=role: self._toggle_density_parameter(r, on))
            holder = QWidget()
            box = QHBoxLayout(holder)
            box.setContentsMargins(0, 0, 0, 0)
            box.addWidget(check)
            box.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setCellWidget(row, 3, holder)
        self._fit_table_height()
        self._loading = False
        self._plot_constants([(r, m) for r, m, _ in entries if r is not None])

    def _fit_table_height(self) -> None:
        self.table.resizeRowsToContents()
        rows = self.table.rowCount()
        height = self.table.horizontalHeader().height() + sum(
            self.table.rowHeight(r) for r in range(rows)
        )
        self.table.setFixedHeight(height + 4)

    def _plot_constants(self, roles: list[tuple[str, Material | None]]) -> None:
        self.plot.clear()
        try:
            library = provider_library(self._provider)
        except KeyError:
            return
        shown: set[Material] = set()
        for i, (_, material) in enumerate(roles):
            if material is None or material in shown or material.is_vacuum:
                continue
            shown.add(material)
            try:
                low, high = providers.get(library.provider_for(material)).energy_range
                low, high = max(low, _PLOT_RANGE[0]), min(high, _PLOT_RANGE[1])
                energy = np.geomspace(low, high, 300)
                db = library.delta_beta(material, energy)
            except (ValueError, KeyError):
                continue
            color = PALETTE[i % len(PALETTE)]
            self.plot.plot(energy / 1e3, db.real, pen=pg.mkPen(color, width=2), name=material.name)
            self.plot.plot(
                energy / 1e3,
                db.imag,
                pen=pg.mkPen(color, width=2, style=pg.QtCore.Qt.PenStyle.DashLine),
            )

    def _replace_material(self, role: str, material: Material) -> None:
        if self._mirror is None:
            return
        if role == _SUBSTRATE:
            self.commandRequested.emit(
                self.commands.set_mirror(replace(self._mirror, substrate=material))
            )
        else:
            coating = self._mirror.coating.with_material(role, material)
            self.commandRequested.emit(self.commands.set_coating(coating))

    def _material_chosen(self, role: str, name: str) -> None:
        if self._mirror is None:
            return
        if role.rpartition(".")[2] in INTERLAYER_ROLES:
            self._choose_interlayer(role, None if name == _NONE else self.library.get(name))
        elif name in self.library:
            self._replace_material(role, self.library.get(name))

    def _choose_interlayer(self, role: str, material: Material | None) -> None:
        """Set or remove an interlayer material, keeping its thickness consistent.

        Without a material the thickness is set to 0; a new material with zero thickness starts
        at `INTERLAYER_START` nm, so that the interlayer appears in the stack.
        """
        assert self._mirror is not None
        coating = self._mirror.coating
        thickness = _thickness_name(role)
        if material is None:
            coating = coating.without_material(role).with_parameter(thickness, value=0.0)
        else:
            coating = coating.with_material(role, material)
            if (
                coating.parameters[thickness].value == 0.0
                and not coating.parameters[thickness].expr
            ):
                coating = coating.with_parameter(thickness, value=INTERLAYER_START)
        self.commandRequested.emit(self.commands.set_coating(coating))

    def _density_edited(self, item: QTableWidgetItem) -> None:
        if self._loading or item.column() != 2:
            return
        if item.row() >= len(self._rows):
            return
        role = self._rows[item.row()]
        material = self._entries()[item.row()][1]
        if role is None or material is None:
            return
        parameter = self._density_parameter(role)
        try:
            value = float(item.text())
            updated = material.with_density(value)
        except ValueError as exc:
            self.errorOccurred.emit(f"Invalid density: {exc}")
            self._load()
            return
        if parameter is not None:
            self.commandRequested.emit(self.commands.set_parameter(parameter.name, value=value))
        else:
            self._replace_material(role, updated)

    def _density_parameter(self, role: str) -> Parameter | None:
        if self._mirror is None:
            return None
        name = density_parameter_name(role)
        return self._mirror.densities.get(name)

    def _toggle_density_parameter(self, role: str, on: bool) -> None:
        if self._loading or self._mirror is None:
            return
        if on:
            mirror = self._mirror.with_density_parameters([role])
        else:
            mirror = self._mirror.without_density_parameters([role])
        self.commandRequested.emit(self.commands.set_mirror(mirror))

    def _define_material(self) -> None:
        dialog = MaterialDialog(self)
        if not dialog.exec():
            return
        try:
            material = dialog.material()
            self.library.add(material)
        except ValueError as exc:
            self.errorOccurred.emit(f"Cannot add material: {exc}")
            return
        self._load()
