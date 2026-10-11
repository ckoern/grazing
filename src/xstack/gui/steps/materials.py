"""Workflow page for the materials of the design."""

from dataclasses import replace

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from xstack.gui.plots import PALETTE
from xstack.gui.steps.base import StepPage
from xstack.materials import Material, MaterialLibrary, default_library
from xstack.project import ProjectState
from xstack.project.commands import set_coating, set_mirror
from xstack.structure.coating import Mirror

__all__ = ["MaterialDialog", "MaterialsPage"]

_SUBSTRATE = "substrate"


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
        library: Material library offering the selectable materials.
        parent: Parent widget.
    """

    title = "Materials"

    def __init__(
        self, library: MaterialLibrary | None = None, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.library = library or default_library()
        self._mirror: Mirror | None = None
        self._loading = False
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<b>Materials</b> of the coating roles and the substrate"))
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Role", "Material", "Density (g/cm³)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.itemChanged.connect(self._density_edited)
        layout.addWidget(self.table)
        self.new_material = QPushButton("New material…")
        self.new_material.clicked.connect(self._define_material)
        layout.addWidget(self.new_material)
        layout.addWidget(QLabel("Optical constants (δ solid, β dashed)"))
        self.plot = pg.PlotWidget()
        self.plot.setLogMode(x=True, y=True)
        self.plot.setLabel("bottom", "Energy (keV)")
        self.plot.addLegend(offset=(-10, 10))
        self.plot.setMinimumHeight(220)
        layout.addWidget(self.plot)

    def set_state(self, state: ProjectState) -> None:
        """Show the materials of the current design.

        Args:
            state: The current project state.
        """
        if state.mirror == self._mirror:
            return
        self._mirror = state.mirror
        self._load()

    def _roles(self) -> list[tuple[str, Material]]:
        if self._mirror is None:
            return []
        return [*self._mirror.coating.materials.items(), (_SUBSTRATE, self._mirror.substrate)]

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
            combo.addItems(names if material.name in names else [material.name, *names])
            combo.setCurrentText(material.name)
            combo.activated.connect(
                lambda _=0, r=role, c=combo: self._material_chosen(r, c.currentText())
            )
            self.table.setCellWidget(row, 1, combo)
            self.table.setItem(row, 2, QTableWidgetItem(f"{material.density:g}"))
        self._loading = False
        self._plot_constants(roles)

    def _plot_constants(self, roles: list[tuple[str, Material]]) -> None:
        self.plot.clear()
        energy = np.logspace(np.log10(100.0), np.log10(1e5), 300)
        shown: set[Material] = set()
        for i, (_, material) in enumerate(roles):
            if material in shown or material.is_vacuum:
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
            self.commandRequested.emit(set_mirror(replace(self._mirror, substrate=material)))
        else:
            coating = self._mirror.coating.with_material(role, material)
            self.commandRequested.emit(set_coating(coating))

    def _material_chosen(self, role: str, name: str) -> None:
        if name in self.library:
            self._replace_material(role, self.library.get(name))

    def _density_edited(self, item: QTableWidgetItem) -> None:
        if self._loading or item.column() != 2:
            return
        role, material = self._roles()[item.row()]
        try:
            updated = material.with_density(float(item.text()))
        except ValueError as exc:
            self.errorOccurred.emit(f"Invalid density: {exc}")
            self._load()
            return
        self._replace_material(role, updated)

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
