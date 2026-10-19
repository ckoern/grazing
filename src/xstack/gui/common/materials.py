"""Page for the materials of the coating roles and the substrate, shared by the applications."""

from dataclasses import replace
from typing import Any

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
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

from xstack.core.materials import Material, MaterialLibrary, default_library
from xstack.core.parameters import Parameter
from xstack.core.structure.coating import SUBSTRATE, Mirror, density_parameter_name
from xstack.core.structure.coatings.bilayer import INTERLAYER_ROLES
from xstack.gui.common.pages import StepPage, StructureCommands
from xstack.gui.common.style import PLOT_PALETTE as PALETTE

__all__ = ["MaterialDialog", "MaterialsPage"]

_SUBSTRATE = SUBSTRATE
_NONE = "(none)"

INTERLAYER_START = 0.5
"""Thickness in nm given to an interlayer that gets a material while its thickness is 0."""


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
        "deposited film differs from the bulk value. The plot below shows the resulting "
        "optical constants over a wide energy range."
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
        self._loading = False
        layout = QVBoxLayout(self)
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
        if state.mirror == self._mirror:
            return
        self._mirror = state.mirror
        self._load()

    def _roles(self) -> list[tuple[str, Material | None]]:
        """Roles with their materials; interlayers of bilayer coatings also without material."""
        if self._mirror is None:
            return []
        coating = self._mirror.coating
        roles: list[tuple[str, Material | None]] = list(coating.materials.items())
        roles += [
            (role, None)
            for role in INTERLAYER_ROLES
            if role in coating.optional_roles and role not in coating.materials
        ]
        return [*roles, (_SUBSTRATE, self._mirror.substrate)]

    def _load(self) -> None:
        self._loading = True
        roles = self._roles()
        self.table.clearContents()
        self.table.setRowCount(0)
        self.table.setRowCount(len(roles))
        names = list(self.library.names())
        for row, (role, material) in enumerate(roles):
            role_item = QTableWidgetItem(role)
            role_item.setFlags(role_item.flags() & ~role_item.flags().ItemIsEditable)
            self.table.setItem(row, 0, role_item)
            combo = QComboBox()
            if role in INTERLAYER_ROLES:
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
                if role not in INTERLAYER_ROLES
                else f"Material of the {role.replace('_', ' ')}, or {_NONE} for no interlayer. "
                f"Its thickness is the parameter t_{role}."
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
        self._plot_constants(roles)

    def _fit_table_height(self) -> None:
        self.table.resizeRowsToContents()
        rows = self.table.rowCount()
        height = self.table.horizontalHeader().height() + sum(
            self.table.rowHeight(r) for r in range(rows)
        )
        self.table.setFixedHeight(height + 4)

    def _plot_constants(self, roles: list[tuple[str, Material | None]]) -> None:
        self.plot.clear()
        energy = np.logspace(np.log10(100.0), np.log10(1e5), 300)
        shown: set[Material] = set()
        for i, (_, material) in enumerate(roles):
            if material is None or material in shown or material.is_vacuum:
                continue
            shown.add(material)
            try:
                db = self.library.delta_beta(material, energy)
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
        if role in INTERLAYER_ROLES:
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
        thickness = f"t_{role}"
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
        role, material = self._roles()[item.row()]
        if material is None:
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
