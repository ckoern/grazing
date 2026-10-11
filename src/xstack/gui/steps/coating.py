"""Workflow page for the coating type, its options and parameters."""

import dataclasses
from collections.abc import Sequence
from typing import ClassVar

from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from xstack.gui.steps.base import StepPage
from xstack.gui.widgets.parameter_table import ParameterTable
from xstack.materials import Material, default_library, get
from xstack.parameters import ParameterSet
from xstack.project import ProjectState
from xstack.project.commands import set_coating, set_mirror
from xstack.structure.coating import Coating, Mirror
from xstack.structure.coatings.graded import DepthGraded, grading_laws
from xstack.structure.coatings.periodic import Periodic
from xstack.structure.coatings.simple import NLayer, SingleLayer

__all__ = ["CoatingPage", "convert_coating"]

COATING_TYPES = {
    "periodic": "Periodic multilayer",
    "depth_graded": "Depth-graded multilayer",
    "single_layer": "Single layer",
    "n_layer": "Layer stack (N layers)",
}
"""Coating types that can be selected in the GUI, by registry name."""


def _carry_over(new: Coating, old: Coating) -> Coating:
    """Return `new` with the settings of parameters that also exist in `old`."""
    params = ParameterSet(old.parameters.get(name, p) for name, p in new.parameters.items())
    try:
        return new.with_parameters(params)
    except ValueError:
        return new


def _bilayer_stack(old: Coating) -> tuple[list[Material], list[float], list[float]]:
    """Return the layers (top first) of one period of a bilayer coating."""
    roles = old.materials
    params = old.parameters
    d = params["d"].value if "d" in params else params["d_max"].value if "d_max" in params else 4.0
    gamma = (
        params["gamma"].value
        if "gamma" in params
        else params["gamma_top"].value
        if "gamma_top" in params
        else 0.4
    )
    thickness = {"H": gamma * d, "L": (1.0 - gamma) * d}
    sigma = {r: params[f"sigma_{r}"].value if f"sigma_{r}" in params else 0.0 for r in "HL"}
    order = ["L", "H"] if getattr(old, "top", "H") == "L" else ["H", "L"]
    return (
        [roles[r] for r in order],
        [thickness[r] for r in order],
        [sigma[r] for r in order],
    )


def _to_n_layer(old: Coating, materials: Sequence[Material] | None) -> NLayer:
    """Create an N-layer coating from `old` or, if given, from the new `materials`."""
    if isinstance(old, NLayer):
        mats = list(materials) if materials is not None else [m for _, m in old.material_roles]
        return NLayer.create(mats, [5.0] * len(mats))
    if isinstance(old, Periodic | DepthGraded):
        mats, thicknesses, sigmas = _bilayer_stack(old)
    else:
        params = old.parameters
        mats = [next(iter(old.materials.values()), get("W"))]
        thicknesses = [params["t"].value if "t" in params else 5.0]
        sigmas = [params["sigma"].value if "sigma" in params else 0.0]
    if materials is not None:
        mats = list(materials)
        thicknesses = (thicknesses + [5.0] * len(mats))[: len(mats)]
        sigmas = (sigmas + [0.0] * len(mats))[: len(mats)]
    return NLayer.create(mats, thicknesses, sigmas)


def convert_coating(old: Coating, type_name: str, **options: object) -> Coating:
    """Create a coating of another type or with other options from an existing one.

    Materials and the settings of parameters with the same name are taken over. Converting a
    multilayer to a layer stack yields the two layers of one period (thicknesses Γ·d and
    (1 - Γ)·d); converting a layer stack to a multilayer uses the materials of its first two
    layers as absorber and spacer.

    Args:
        old: The current coating.
        type_name: Registry name of the new coating type.
        **options: Options of the new type (``top``, ``law``, ``cap``, ``interlayers``, and
            for ``"n_layer"`` the ``materials`` of all layers from the top).

    Returns:
        The new coating.

    Raises:
        ValueError: If the type is not supported.
    """
    roles = old.materials
    first = next(iter(roles.values()), get("W"))
    h = roles.get("H", roles.get("M1", roles.get("M", first)))
    low = roles.get("L", roles.get("M2", get("Si")))
    interlayers = {
        r: m for r, m in roles.items() if r in ("interlayer_H_on_L", "interlayer_L_on_H")
    }
    if "interlayers" in options:
        interlayers = options.pop("interlayers")  # type: ignore[assignment]
    new: Coating
    if type_name == "periodic":
        new = Periodic.create(h, low, interlayers=interlayers, **options)  # type: ignore[arg-type]
    elif type_name == "depth_graded":
        new = DepthGraded.create(h, low, interlayers=interlayers, **options)  # type: ignore[arg-type]
    elif type_name == "single_layer":
        new = SingleLayer.create(h)
    elif type_name == "n_layer":
        if isinstance(old, NLayer) and "materials" not in options:
            return old
        new = _to_n_layer(old, options.get("materials"))  # type: ignore[arg-type]
    else:
        raise ValueError(f"coating type {type_name!r} cannot be created here")
    return _carry_over(new, old)


def _restack(old: NLayer, materials: Sequence[Material], sources: Sequence[int | None]) -> NLayer:
    """Return a layer stack with new materials whose layers take over old parameter settings.

    Args:
        old: The current layer stack.
        materials: Materials of the new stack from the top.
        sources: For each new layer the 1-based index of the old layer whose thickness and
            roughness settings it takes over, or None for a new layer.
    """
    new = NLayer.create(materials, [5.0] * len(materials), name=old.name)
    params = []
    for i, source in enumerate(sources, start=1):
        for base in ("t", "sigma"):
            template = new.parameters[f"{base}{i}"]
            if source is None or f"{base}{source}" not in old.parameters:
                params.append(template)
                continue
            carried = old.parameters[f"{base}{source}"]
            params.append(
                dataclasses.replace(
                    carried,
                    name=template.name,
                    description=template.description,
                    expr=carried.expr if source == i else None,
                )
            )
    try:
        return new.with_parameters(ParameterSet(params))
    except ValueError:
        return new


class CoatingPage(StepPage):
    """Coating type, type-specific options, substrate roughness and the parameter table.

    Args:
        parent: Parent widget.
    """

    title = "Coating"
    help_text: ClassVar[str] = (
        "Choose the <b>coating type</b>: a periodic multilayer, a depth-graded multilayer "
        "whose period changes with depth, a single layer, or a stack of N layers of different "
        "materials. This page defines the <b>input (start) design</b>; results of the "
        "optimization are stored as candidates and never overwrite it. In the parameter table, "
        "a parameter with <b>Vary</b> checked is explored by the optimizer between its "
        "<b>Min</b> and <b>Max</b>; otherwise it stays fixed at its value. An "
        "<b>expression</b> computes a parameter from other parameters, for example "
        "<tt>2*d_min</tt>, and then it is neither fixed nor varied."
    )

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._mirror: Mirror | None = None
        self._loading = False
        layout = QVBoxLayout(self)
        self.header = QLabel()
        layout.addWidget(self.header)
        form = QFormLayout()
        self.type_combo = QComboBox()
        self.type_combo.setToolTip("Kind of coating that is deposited on the substrate.")
        for name, label in COATING_TYPES.items():
            self.type_combo.addItem(label, name)
        self.type_combo.activated.connect(lambda _: self._change_type())
        form.addRow("Coating type", self.type_combo)
        self.top_combo = QComboBox()
        self.top_combo.setToolTip("Material of each bilayer that faces the ambient medium.")
        self.top_combo.addItem("Absorber (H)", "H")
        self.top_combo.addItem("Spacer (L)", "L")
        self.top_combo.activated.connect(lambda _: self._change_options())
        form.addRow("Top layer of each bilayer", self.top_combo)
        self.law_combo = QComboBox()
        self.law_combo.setToolTip("Law by which the bilayer period changes with depth.")
        self.law_combo.addItems(list(grading_laws.names()))
        self.law_combo.activated.connect(lambda _: self._change_options())
        form.addRow("Grading law", self.law_combo)
        self.cap_check = QCheckBox("Cap bilayer with independent period and Γ")
        self.cap_check.setToolTip(
            "Add a topmost bilayer whose period and Γ are independent of the grading."
        )
        self.cap_check.toggled.connect(lambda _: self._change_options())
        form.addRow("", self.cap_check)
        self.roughness_spin = QDoubleSpinBox()
        self.roughness_spin.setRange(0.0, 100.0)
        self.roughness_spin.setDecimals(3)
        self.roughness_spin.setSingleStep(0.1)
        self.roughness_spin.setSuffix(" nm")
        self.roughness_spin.setToolTip(
            "Root-mean-square roughness σ of the interface between coating and substrate, in nm."
        )
        self.roughness_spin.editingFinished.connect(self._change_substrate_roughness)
        form.addRow("Substrate roughness", self.roughness_spin)
        layout.addLayout(form)

        self.layers_box = QGroupBox("Layers (from the top)")
        box_layout = QHBoxLayout(self.layers_box)
        self.layer_table = QTableWidget(0, 1)
        self.layer_table.setHorizontalHeaderLabels(["Material"])
        self.layer_table.horizontalHeader().setStretchLastSection(True)
        self.layer_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.layer_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.layer_table.setToolTip("Material of each layer, listed from the top.")
        self.layer_table.setMaximumHeight(160)
        box_layout.addWidget(self.layer_table)
        buttons = QVBoxLayout()
        self.add_layer_button = QPushButton("Add layer")
        self.add_layer_button.setToolTip(
            "Append a layer at the bottom, or insert below the selection."
        )
        self.add_layer_button.clicked.connect(self._add_layer)
        self.remove_layer_button = QPushButton("Remove layer")
        self.remove_layer_button.setToolTip("Remove the selected layer, or the bottom layer.")
        self.remove_layer_button.clicked.connect(self._remove_layer)
        buttons.addWidget(self.add_layer_button)
        buttons.addWidget(self.remove_layer_button)
        buttons.addStretch()
        box_layout.addLayout(buttons)
        layout.addWidget(self.layers_box)

        layout.addWidget(QLabel("<b>Parameters</b> (check Vary to optimize within Min–Max)"))
        self.parameter_table = ParameterTable()
        self.parameter_table.setToolTip(
            "Coating parameters. Vary lets the optimizer change a parameter within Min and Max."
        )
        self.parameter_table.commandRequested.connect(self.commandRequested)
        layout.addWidget(self.parameter_table)

    @property
    def _coating(self) -> Coating | None:
        return self._mirror.coating if self._mirror is not None else None

    def set_state(self, state: ProjectState) -> None:
        """Show the design of the current project state.

        Args:
            state: The current project state.
        """
        mirror = state.mirror
        if mirror == self._mirror:
            return
        self._mirror = mirror
        coating = mirror.coating if mirror is not None else None
        self._loading = True
        self.roughness_spin.setEnabled(mirror is not None)
        if mirror is not None:
            self.roughness_spin.setValue(mirror.substrate_roughness)
        if coating is None:
            self.header.setText("No design defined.")
            self.parameter_table.set_parameters(ParameterSet())
            self.layers_box.setVisible(False)
        else:
            self.header.setText(f"<b>{type(coating).__name__}</b> “{coating.name}”")
            index = self.type_combo.findData(coating.type_name)
            self.type_combo.setEnabled(index >= 0)
            if index >= 0:
                self.type_combo.setCurrentIndex(index)
            bilayer = isinstance(coating, Periodic | DepthGraded)
            self.top_combo.setEnabled(bilayer)
            if isinstance(coating, Periodic | DepthGraded):
                self.top_combo.setCurrentIndex(self.top_combo.findData(coating.top))
            graded = isinstance(coating, DepthGraded)
            self.law_combo.setEnabled(graded)
            self.cap_check.setEnabled(graded)
            if isinstance(coating, DepthGraded):
                self.law_combo.setCurrentText(coating.law)
                self.cap_check.setChecked(coating.cap)
            self.layers_box.setVisible(isinstance(coating, NLayer))
            if isinstance(coating, NLayer):
                self._show_layers(coating)
            self.parameter_table.set_parameters(coating.parameters)
        self._loading = False

    def _show_layers(self, coating: NLayer) -> None:
        materials = [m for _, m in coating.material_roles]
        names = [m.name for m in materials]
        if names == self._layer_names():
            return
        selected = self._selected_layer()
        library = default_library().names()
        self.layer_table.setRowCount(len(materials))
        for row, name in enumerate(names):
            combo = QComboBox()
            combo.setToolTip(f"Material of layer {row + 1}.")
            combo.addItems(list(library))
            if name not in library:
                combo.addItem(name)
            combo.setCurrentText(name)
            combo.activated.connect(lambda _, r=row: self._change_layer_material(r))
            self.layer_table.setCellWidget(row, 0, combo)
        if selected is not None and selected < len(materials):
            self.layer_table.selectRow(selected)

    def _layer_combo(self, row: int) -> QComboBox:
        widget = self.layer_table.cellWidget(row, 0)
        assert isinstance(widget, QComboBox)
        return widget

    def _layer_names(self) -> list[str]:
        return [self._layer_combo(r).currentText() for r in range(self.layer_table.rowCount())]

    def _selected_layer(self) -> int | None:
        rows = self.layer_table.selectionModel().selectedRows()
        return rows[0].row() if rows else None

    def _layer_materials(self, current: NLayer) -> list[Material]:
        """Materials of the table rows; unchanged rows keep the coating's own material."""
        result = []
        for row, (_, old) in enumerate(current.material_roles):
            name = self._layer_combo(row).currentText()
            result.append(old if old.name == name else default_library().get(name))
        return result

    def _emit_layers(self, materials: list[Material], sources: list[int | None]) -> None:
        coating = self._coating
        if self._loading or not isinstance(coating, NLayer):
            return
        if not materials:
            self.errorOccurred.emit("A layer stack needs at least one layer.")
            return
        new = _restack(coating, materials, sources)
        if new != coating:
            self.commandRequested.emit(set_coating(new))

    def _change_layer_material(self, row: int) -> None:
        coating = self._coating
        if isinstance(coating, NLayer):
            materials = self._layer_materials(coating)
            self._emit_layers(materials, list(range(1, len(materials) + 1)))

    def _add_layer(self) -> None:
        coating = self._coating
        if not isinstance(coating, NLayer):
            return
        materials = [m for _, m in coating.material_roles]
        selected = self._selected_layer()
        position = len(materials) if selected is None else selected + 1
        sources: list[int | None] = list(range(1, len(materials) + 1))
        materials.insert(position, materials[position - 1])
        sources.insert(position, None)
        self._emit_layers(materials, sources)

    def _remove_layer(self) -> None:
        coating = self._coating
        if not isinstance(coating, NLayer):
            return
        materials = [m for _, m in coating.material_roles]
        selected = self._selected_layer()
        position = len(materials) - 1 if selected is None else selected
        sources: list[int | None] = list(range(1, len(materials) + 1))
        del materials[position]
        del sources[position]
        self._emit_layers(materials, sources)

    def _change_substrate_roughness(self) -> None:
        mirror = self._mirror
        if self._loading or mirror is None:
            return
        value = self.roughness_spin.value()
        if value != mirror.substrate_roughness:
            self.commandRequested.emit(
                set_mirror(dataclasses.replace(mirror, substrate_roughness=value))
            )

    def _options(self, type_name: str) -> dict[str, object]:
        options: dict[str, object] = {}
        if type_name in ("periodic", "depth_graded"):
            options["top"] = self.top_combo.currentData()
        if type_name == "depth_graded":
            options["law"] = self.law_combo.currentText()
            options["cap"] = self.cap_check.isChecked()
        return options

    def _emit(self, type_name: str) -> None:
        coating = self._coating
        if self._loading or coating is None:
            return
        try:
            new = convert_coating(coating, type_name, **self._options(type_name))
        except (ValueError, KeyError) as exc:
            self.errorOccurred.emit(f"Cannot change coating: {exc}")
            return
        if new != coating:
            self.commandRequested.emit(set_coating(new))

    def _change_type(self) -> None:
        self._emit(self.type_combo.currentData())

    def _change_options(self) -> None:
        if self._coating is not None:
            self._emit(self._coating.type_name)
