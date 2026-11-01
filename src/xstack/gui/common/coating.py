"""Page for the coating type, its options and parameters, shared by the applications."""

import dataclasses
import re
from collections.abc import Callable, Collection, Sequence
from typing import Any, ClassVar

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QTableWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from xstack.core.materials import Material, default_library, get
from xstack.core.parameters import ParameterSet
from xstack.core.structure.coating import Coating, Mirror
from xstack.core.structure.coatings.composite import Composite
from xstack.core.structure.coatings.graded import DepthGraded, grading_laws
from xstack.core.structure.coatings.periodic import Periodic
from xstack.core.structure.coatings.simple import NLayer, SingleLayer
from xstack.gui.common.pages import StepPage, StructureCommands
from xstack.gui.common.parameter_table import ParameterTable
from xstack.gui.common.style import COLORS
from xstack.gui.common.style import PLOT_PALETTE as PALETTE

__all__ = [
    "BLOCK_TYPES",
    "COATING_TYPES",
    "BlockSketch",
    "CoatingPage",
    "convert_coating",
    "has_constant_gamma",
    "set_constant_gamma",
    "unique_block_name",
]

BLOCK_TYPES = {
    "periodic": "Periodic multilayer",
    "depth_graded": "Depth-graded multilayer",
    "single_layer": "Single layer",
    "n_layer": "Layer stack (N layers)",
}
"""Coating types that can be a block of a combined coating, by registry name."""

COATING_TYPES = {**BLOCK_TYPES, "composite": "Combined coating (blocks)"}
"""Coating types that can be selected in the GUI, by registry name."""

_BLOCK_NAMES = {
    "periodic": "periodic",
    "depth_graded": "graded",
    "single_layer": "layer",
    "n_layer": "stack",
}


def unique_block_name(type_name: str, used: Collection[str]) -> str:
    """Return a name for a new block of a combined coating.

    Args:
        type_name: Registry name of the block's coating type.
        used: Names of the existing blocks.

    Returns:
        A short name after the type (``"periodic"``, ``"graded"``, ``"layer"``, ``"stack"``),
        with a number appended if it is already used.
    """
    base = _BLOCK_NAMES.get(type_name, "block")
    name, number = base, 2
    while name in used:
        name, number = f"{base}{number}", number + 1
    return name


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
    layers as absorber and spacer. Converting to ``"composite"`` makes the coating the only
    block of a combined coating; converting a combined coating to another type converts its
    first block.

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
    if type_name == "composite":
        if isinstance(old, Composite):
            return old
        return Composite.create([old.renamed(unique_block_name(old.type_name, ()))])
    if isinstance(old, Composite):
        old = old.block(old.block_names()[0])
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
    new = _carry_over(new, old)
    if isinstance(new, Periodic | DepthGraded) and isinstance(old, Periodic | DepthGraded):
        new = new.with_interlayers(old.use_interlayers)
    return new


def visible_parameters(parameters: ParameterSet, inactive: Collection[str]) -> ParameterSet:
    """Return the parameters that an application shows: all but the inactive ones.

    Args:
        parameters: The parameters.
        inactive: Names of parameters without effect, e.g. from `Coating.inactive_parameters`.

    Returns:
        The parameters without the inactive ones.
    """
    return ParameterSet(p for name, p in parameters.items() if name not in inactive)


def has_constant_gamma(coating: DepthGraded) -> bool:
    """Return whether all graded bilayers of a coating share one Γ.

    Args:
        coating: The depth-graded coating.

    Returns:
        True if ``gamma_bottom`` is tied to ``gamma_top`` by an expression.
    """
    expr = coating.parameters["gamma_bottom"].expr
    return expr is not None and expr.replace(" ", "") == "gamma_top"


def set_constant_gamma(
    coating: DepthGraded, constant: bool, set_parameter: Callable[..., Any]
) -> Any:
    """Return a command that switches between one Γ and a linear Γ gradient.

    With a constant Γ, ``gamma_bottom`` follows ``gamma_top`` through an expression. Without,
    ``gamma_bottom`` starts at the current value, bounds and vary setting of ``gamma_top``.

    Args:
        coating: The current depth-graded coating.
        constant: Use one Γ for all graded bilayers.
        set_parameter: Factory of the application's command changing one parameter.

    Returns:
        The command.
    """
    if constant:
        return set_parameter("gamma_bottom", expr="gamma_top", vary=False)
    top = coating.parameters["gamma_top"]
    value = coating.parameters.evaluate()["gamma_top"]
    return set_parameter(
        "gamma_bottom", expr=None, value=value, vary=top.vary, min=top.min, max=top.max
    )


def _restack(old: NLayer, materials: Sequence[Material], sources: Sequence[int | None]) -> NLayer:
    """Return a layer stack with new materials whose layers take over old parameter settings.

    Args:
        old: The current layer stack.
        materials: Materials of the new stack from the top.
        sources: For each new layer the 1-based index of the old layer whose thickness and
            roughness settings it takes over, or None for a new layer.
    """
    new = NLayer.create(materials, [5.0] * len(materials), name=old.name)
    valid = [s is not None and 2 * s <= len(old.parameters) for s in sources]
    renamed: dict[str, str] = {}
    for i, (source, ok) in enumerate(zip(sources, valid, strict=True), start=1):
        if source is not None and ok:
            renamed[old.thickness_name(source)] = new.thickness_name(i)
            renamed[old.roughness_name(source)] = new.roughness_name(i)
    old_names = set(old.parameters)
    word = re.compile(r"[A-Za-z_]\w*")

    def translate(expr: str | None) -> str | None:
        """Rewrite an expression to the new names; None if it refers to a removed layer."""
        if expr is None or any(w in old_names and w not in renamed for w in word.findall(expr)):
            return None
        return word.sub(lambda m: renamed.get(m.group(0), m.group(0)), expr)

    params = []
    for i, (source, ok) in enumerate(zip(sources, valid, strict=True), start=1):
        pairs = (
            (new.thickness_name(i), old.thickness_name),
            (new.roughness_name(i), old.roughness_name),
        )
        for new_name, old_name_of in pairs:
            template = new.parameters[new_name]
            if source is None or not ok:
                params.append(template)
                continue
            carried = old.parameters[old_name_of(source)]
            params.append(
                dataclasses.replace(
                    carried,
                    name=template.name,
                    description=template.description,
                    expr=translate(carried.expr),
                )
            )
    try:
        return new.with_parameters(ParameterSet(params))
    except ValueError:
        return new


class BlockSketch(QWidget):
    """Sketch of the blocks of a combined coating: one band per block, height by thickness.

    Signals:
        blockClicked(str): A band was clicked; the name of its block.
    """

    blockClicked = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(200, 150)
        self.setToolTip("The blocks from the top with their thickness. Click a block to edit it.")
        self._blocks: list[tuple[str, int, float]] = []
        self._selected: str | None = None

    @property
    def blocks(self) -> list[tuple[str, int, float]]:
        """Name, number of layers and thickness in nm of each shown block."""
        return list(self._blocks)

    def set_blocks(self, blocks: Sequence[tuple[str, int, float]], selected: str | None) -> None:
        """Show blocks.

        Args:
            blocks: Name, number of layers and thickness in nm of each block from the top.
            selected: Name of the highlighted block.
        """
        self._blocks, self._selected = list(blocks), selected
        self.update()

    def _geometry(self) -> list[tuple[QRectF, str, int, float]]:
        """Band rectangle, name, number of layers and thickness of each block."""
        if not self._blocks:
            return []
        top, bottom, width = 4.0, self.height() - 4.0, 64.0
        gap, minimum = 3.0, 22.0
        total = sum(max(t, 0.0) for _, _, t in self._blocks) or 1.0
        free = bottom - top - (gap + minimum) * len(self._blocks)
        bands, y = [], top
        for name, count, thickness in self._blocks:
            height = minimum + max(free, 0.0) * max(thickness, 0.0) / total
            bands.append((QRectF(4.0, y, width, height), name, count, thickness))
            y += height + gap
        return bands

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Select the block whose band or label is under the cursor."""
        for rect, name, _, _ in self._geometry():
            if rect.adjusted(0, 0, self.width(), 0).contains(event.position()):
                self.blockClicked.emit(name)
                return

    def paintEvent(self, _event: QPaintEvent) -> None:
        """Draw the bands."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        for index, (rect, name, count, thickness) in enumerate(self._geometry()):
            color = QColor(PALETTE[index % len(PALETTE)])
            color.setAlpha(220 if name == self._selected else 110)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(color)
            painter.drawRoundedRect(rect, 4, 4)
            if name == self._selected:
                painter.setPen(QPen(QColor(COLORS["text"]), 2))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRoundedRect(rect, 4, 4)
            painter.setPen(QColor(COLORS["text"]))
            label = QRectF(
                rect.right() + 8, rect.top(), self.width() - rect.right() - 8, rect.height()
            )
            painter.drawText(
                label,
                int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                f"{name}\n{count} layers · {thickness:.3g} nm",
            )


class CoatingPage(StepPage):
    """Coating type, type-specific options, substrate roughness and the parameter table.

    For a combined coating (`Composite`), the page lists the blocks from the top with buttons
    to add, duplicate, remove and move them, and the type, options and layers shown below
    belong to the selected block.

    Args:
        commands: Factories of the application's commands that edit the mirror.
        parent: Parent widget.
    """

    title = "Coating"
    help_text: ClassVar[str] = (
        "Choose the <b>coating type</b>: a periodic multilayer, a depth-graded multilayer "
        "whose period changes with depth, a single layer, a stack of N layers of different "
        "materials, or a <b>combined coating</b> of several such blocks stacked from the top "
        "(e.g. a depth-graded section on a periodic one). The parameters of a block carry its "
        "name as prefix, e.g. <tt>graded.d_min</tt>. In the parameter table, a parameter with "
        "<b>Vary</b> checked is varied between its <b>Min</b> and <b>Max</b>; otherwise it "
        "stays fixed at its value. An <b>expression</b> computes a parameter from other "
        "parameters, for example <tt>2*d_min</tt>, and then it is neither fixed nor varied."
    )

    def __init__(self, commands: StructureCommands, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.commands = commands
        self._mirror: Mirror | None = None
        self._loading = False
        self._block: str | None = None
        layout = QVBoxLayout(self)
        self.header = QLabel()
        layout.addWidget(self.header)
        type_form = QFormLayout()
        self.type_combo = QComboBox()
        self.type_combo.setToolTip("Kind of coating that is deposited on the substrate.")
        for name, label in COATING_TYPES.items():
            self.type_combo.addItem(label, name)
        self.type_combo.activated.connect(lambda _: self._change_type())
        type_form.addRow("Coating type", self.type_combo)
        layout.addLayout(type_form)
        layout.addWidget(self._blocks_box())

        self.options_box = QGroupBox()
        options = QVBoxLayout(self.options_box)
        form = QFormLayout()
        self.block_name_edit = QLineEdit()
        self.block_name_edit.setToolTip(
            "Name of the block: prefix of its parameters and material roles; letters, digits "
            "and underscores."
        )
        self.block_name_edit.editingFinished.connect(self._rename_block)
        form.addRow("Block name", self.block_name_edit)
        self.block_type_combo = QComboBox()
        self.block_type_combo.setToolTip("Kind of coating of the selected block.")
        for name, label in BLOCK_TYPES.items():
            self.block_type_combo.addItem(label, name)
        self.block_type_combo.activated.connect(lambda _: self._change_block_type())
        form.addRow("Block type", self.block_type_combo)
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
        self.constant_gamma_check = QCheckBox("Same Γ for all graded bilayers")
        self.constant_gamma_check.setToolTip(
            "Use one Γ for all graded bilayers: gamma_bottom follows gamma_top. Uncheck to let "
            "Γ change linearly from gamma_top to gamma_bottom."
        )
        self.constant_gamma_check.toggled.connect(self._change_constant_gamma)
        form.addRow("", self.constant_gamma_check)
        self.interlayer_check = QCheckBox("Interlayers at the interfaces")
        self.interlayer_check.setToolTip(
            "Model interlayers that form where one material is deposited on the other (e.g. a "
            "silicide). Their materials are chosen on the Materials page and their thicknesses "
            "are the parameters t_interlayer_H_on_L and t_interlayer_L_on_H."
        )
        self.interlayer_check.toggled.connect(self._change_interlayers)
        form.addRow("", self.interlayer_check)
        self._options_form = form
        options.addLayout(form)

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
        options.addWidget(self.layers_box)
        layout.addWidget(self.options_box)

        roughness_form = QFormLayout()
        self.roughness_spin = QDoubleSpinBox()
        self.roughness_spin.setRange(0.0, 100.0)
        self.roughness_spin.setDecimals(3)
        self.roughness_spin.setSingleStep(0.1)
        self.roughness_spin.setSuffix(" nm")
        self.roughness_spin.setToolTip(
            "Root-mean-square roughness σ of the interface between coating and substrate, in nm."
        )
        self.roughness_spin.editingFinished.connect(self._change_substrate_roughness)
        roughness_form.addRow("Substrate roughness", self.roughness_spin)
        layout.addLayout(roughness_form)

        parameter_row = QHBoxLayout()
        self.parameters_label = QLabel("<b>Parameters</b> (check Vary to optimize within Min–Max)")
        parameter_row.addWidget(self.parameters_label)
        parameter_row.addStretch()
        self.scope_label = QLabel("Show")
        parameter_row.addWidget(self.scope_label)
        self.scope_combo = QComboBox()
        self.scope_combo.addItem("All blocks", "all")
        self.scope_combo.addItem("Selected block", "block")
        self.scope_combo.setToolTip("Show the parameters of all blocks or of the selected block.")
        self.scope_combo.activated.connect(lambda _: self._show_parameters())
        parameter_row.addWidget(self.scope_combo)
        layout.addLayout(parameter_row)
        self.parameter_table = ParameterTable(commands.set_parameter)
        self.parameter_table.setToolTip(
            "Coating parameters. Vary lets the optimizer change a parameter within Min and Max."
        )
        self.parameter_table.commandRequested.connect(self.commandRequested)
        layout.addWidget(self.parameter_table)

    def _blocks_box(self) -> QGroupBox:
        self.blocks_box = QGroupBox("Blocks (from the top)")
        row = QHBoxLayout(self.blocks_box)
        self.block_list = QListWidget()
        self.block_list.setToolTip("Blocks of the combined coating, the first one on top.")
        self.block_list.currentRowChanged.connect(self._block_row_changed)
        row.addWidget(self.block_list, 3)
        buttons = QVBoxLayout()
        self.add_block_button = QToolButton()
        self.add_block_button.setText("Add block")
        self.add_block_button.setToolTip("Insert a block of the chosen type below the selection.")
        self.add_block_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.add_block_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        menu = QMenu(self.add_block_button)
        for name, label in BLOCK_TYPES.items():
            menu.addAction(label, lambda n=name: self.add_block(n))
        self.add_block_button.setMenu(menu)
        buttons.addWidget(self.add_block_button)
        self.duplicate_block_button = QPushButton("Duplicate")
        self.duplicate_block_button.setToolTip("Insert a copy of the selected block below it.")
        self.duplicate_block_button.clicked.connect(self.duplicate_block)
        self.remove_block_button = QPushButton("Remove")
        self.remove_block_button.setToolTip("Remove the selected block.")
        self.remove_block_button.clicked.connect(self.remove_block)
        self.up_block_button = QPushButton("▲ Up")
        self.up_block_button.setToolTip("Move the selected block towards the top.")
        self.up_block_button.clicked.connect(lambda: self.move_block(-1))
        self.down_block_button = QPushButton("▼ Down")
        self.down_block_button.setToolTip("Move the selected block towards the substrate.")
        self.down_block_button.clicked.connect(lambda: self.move_block(1))
        for button in (
            self.duplicate_block_button,
            self.remove_block_button,
            self.up_block_button,
            self.down_block_button,
        ):
            buttons.addWidget(button)
        buttons.addStretch()
        row.addLayout(buttons)
        self.sketch = BlockSketch()
        self.sketch.blockClicked.connect(self.select_block)
        row.addWidget(self.sketch, 2)
        return self.blocks_box

    @property
    def _coating(self) -> Coating | None:
        return self._mirror.coating if self._mirror is not None else None

    @property
    def _composite(self) -> Composite | None:
        coating = self._coating
        return coating if isinstance(coating, Composite) else None

    @property
    def _target(self) -> Coating | None:
        """The coating whose type and options the page edits: the selected block or the coating."""
        composite = self._composite
        if composite is not None and self._block is not None:
            return composite.block(self._block)
        return self._coating

    @property
    def selected_block(self) -> str | None:
        """Name of the selected block of a combined coating, or ``None``."""
        return self._block if self._composite is not None else None

    def set_state(self, state: Any) -> None:
        """Show the mirror of a document state.

        Args:
            state: A document state with a ``mirror`` attribute.
        """
        mirror = state.mirror
        if mirror == self._mirror:
            return
        self._mirror = mirror
        self._show()

    def _show(self) -> None:
        mirror = self._mirror
        coating = mirror.coating if mirror is not None else None
        composite = self._composite
        if composite is not None and self._block not in composite.block_names():
            self._block = composite.block_names()[0]
        self._loading = True
        self.roughness_spin.setEnabled(mirror is not None)
        if mirror is not None:
            self.roughness_spin.setValue(mirror.substrate_roughness)
        self.blocks_box.setVisible(composite is not None)
        scoped = composite is not None and not self.parameter_table.isHidden()
        self.scope_label.setVisible(scoped)
        self.scope_combo.setVisible(scoped)
        if composite is not None:
            self._show_blocks(composite)
        target = self._target
        if coating is None or target is None:
            self.header.setText("No design defined.")
            self.parameter_table.set_parameters(ParameterSet())
            self.options_box.setVisible(False)
        else:
            self.header.setText(f"<b>{type(coating).__name__}</b> “{coating.name}”")
            index = self.type_combo.findData(coating.type_name)
            self.type_combo.setEnabled(index >= 0)
            if index >= 0:
                self.type_combo.setCurrentIndex(index)
            self.options_box.setVisible(True)
            self._show_options(target, composite is not None)
            self._show_parameters()
        self._loading = False

    def _show_blocks(self, composite: Composite) -> None:
        names = composite.block_names()
        self.block_list.blockSignals(True)
        self.block_list.clear()
        for number, block in enumerate(composite.blocks, start=1):
            label = BLOCK_TYPES.get(block.type_name, block.type_name)
            materials = ", ".join(m.name for m in block.materials.values())
            item = QListWidgetItem(f"{number}. {block.name} — {label} · {materials}")
            item.setData(Qt.ItemDataRole.UserRole, block.name)
            self.block_list.addItem(item)
        assert self._block is not None
        row = names.index(self._block)
        self.block_list.setCurrentRow(row)
        self.block_list.blockSignals(False)
        self.remove_block_button.setEnabled(len(names) > 1)
        self.up_block_button.setEnabled(row > 0)
        self.down_block_button.setEnabled(row < len(names) - 1)
        try:
            layers = composite.layers_by_block()
            sketch = [
                (n, len(ls), sum(layer.thickness for layer in ls)) for n, ls in layers.items()
            ]
        except (ValueError, KeyError):
            sketch = [(n, 0, 0.0) for n in names]
        self.sketch.set_blocks(sketch, self._block)

    def _show_options(self, target: Coating, in_block: bool) -> None:
        self.options_box.setTitle(f"Block “{target.name}”" if in_block else "")
        self._options_form.setRowVisible(self.block_name_edit, in_block)
        self._options_form.setRowVisible(self.block_type_combo, in_block)
        if in_block:
            self.block_name_edit.setText(target.name)
            self.block_type_combo.setCurrentIndex(self.block_type_combo.findData(target.type_name))
        bilayer = isinstance(target, Periodic | DepthGraded)
        self.top_combo.setEnabled(bilayer)
        if isinstance(target, Periodic | DepthGraded):
            self.top_combo.setCurrentIndex(self.top_combo.findData(target.top))
        self.interlayer_check.setEnabled(bilayer)
        self.interlayer_check.setChecked(
            isinstance(target, Periodic | DepthGraded) and target.use_interlayers
        )
        graded = isinstance(target, DepthGraded)
        self.law_combo.setEnabled(graded)
        self.cap_check.setEnabled(graded)
        self.constant_gamma_check.setEnabled(graded)
        self.cap_check.setChecked(isinstance(target, DepthGraded) and target.cap)
        self.constant_gamma_check.setChecked(
            isinstance(target, DepthGraded) and has_constant_gamma(target)
        )
        if isinstance(target, DepthGraded):
            self.law_combo.setCurrentText(target.law)
        self.layers_box.setVisible(isinstance(target, NLayer))
        if isinstance(target, NLayer):
            self._show_layers(target)

    def _show_parameters(self) -> None:
        mirror, coating = self._mirror, self._coating
        if mirror is None or coating is None:
            return
        parameters = mirror.parameters
        try:
            evaluated = parameters.evaluate()
        except ValueError:
            evaluated = {}
        shown = visible_parameters(parameters, coating.inactive_parameters())
        block = self.selected_block
        if block is not None and self.scope_combo.currentData() == "block":
            shown = ParameterSet(p for name, p in shown.items() if name.startswith(block + "."))
        self.parameter_table.set_parameters(shown, evaluated)

    def _commit(self, new: Coating) -> None:
        """Request the coating with `new` as the edited coating or as the selected block."""
        composite, target = self._composite, self._target
        if self._loading or target is None or new == target:
            return
        if composite is not None and self._block is not None:
            self.commandRequested.emit(
                self.commands.set_coating(composite.with_block(self._block, new))
            )
        else:
            self.commandRequested.emit(self.commands.set_coating(new))

    def _edit_composite(self, edit: Callable[[Composite], Composite], block: str | None) -> None:
        """Request a change of the block list; `block` is selected afterwards."""
        composite = self._composite
        if self._loading or composite is None:
            return
        try:
            new = edit(composite)
        except (ValueError, KeyError) as exc:
            self.errorOccurred.emit(f"Cannot change the blocks: {exc}")
            return
        if new != composite:
            self._block = block
            self.commandRequested.emit(self.commands.set_coating(new))

    def select_block(self, name: str) -> None:
        """Show the type, options and layers of a block of the combined coating.

        Args:
            name: Block name.
        """
        composite = self._composite
        if composite is None or name not in composite.block_names() or name == self._block:
            return
        self._block = name
        self._show()

    def _block_row_changed(self, row: int) -> None:
        item = self.block_list.item(row)
        if item is not None:
            self.select_block(item.data(Qt.ItemDataRole.UserRole))

    def add_block(self, type_name: str) -> None:
        """Insert a new block below the selected one.

        The new block takes over the materials of the selected block where its type allows.

        Args:
            type_name: Registry name of the block's coating type.
        """
        composite, target = self._composite, self._target
        if composite is None or target is None or self._block is None:
            return
        name = unique_block_name(type_name, composite.block_names())
        try:
            block = convert_coating(target, type_name).renamed(name)
        except (ValueError, KeyError) as exc:
            self.errorOccurred.emit(f"Cannot add a block: {exc}")
            return
        index = composite.block_names().index(self._block) + 1
        self._edit_composite(lambda c: c.insert_block(index, block), name)

    def duplicate_block(self) -> None:
        """Insert a copy of the selected block below it."""
        composite, target = self._composite, self._target
        if composite is None or target is None or self._block is None:
            return
        name = unique_block_name(target.type_name, composite.block_names())
        index = composite.block_names().index(self._block) + 1
        self._edit_composite(lambda c: c.insert_block(index, target.renamed(name)), name)

    def remove_block(self) -> None:
        """Remove the selected block; the block above it (or the new first block) is selected."""
        composite, block = self._composite, self._block
        if composite is None or block is None:
            return
        names = composite.block_names()
        index = names.index(block)
        remaining = [n for n in names if n != block]
        selected = remaining[max(index - 1, 0)] if remaining else None
        self._edit_composite(lambda c: c.remove_block(block), selected)

    def move_block(self, step: int) -> None:
        """Move the selected block up (``-1``) or down (``+1``).

        Args:
            step: Number of positions towards the substrate.
        """
        composite, block = self._composite, self._block
        if composite is None or block is None:
            return
        index = composite.block_names().index(block) + step
        if 0 <= index < len(composite.blocks):
            self._edit_composite(lambda c: c.move_block(block, index), block)

    def _rename_block(self) -> None:
        block = self._block
        name = self.block_name_edit.text().strip()
        if self._loading or block is None or name == block:
            return
        self._edit_composite(lambda c: c.rename_block(block, name), name)
        if self._block != name:
            self.block_name_edit.setText(block)

    def _change_block_type(self) -> None:
        self._emit(self.block_type_combo.currentData())

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
        target = self._target
        if self._loading or not isinstance(target, NLayer):
            return
        if not materials:
            self.errorOccurred.emit("A layer stack needs at least one layer.")
            return
        self._commit(_restack(target, materials, sources))

    def _change_layer_material(self, row: int) -> None:
        target = self._target
        if isinstance(target, NLayer):
            materials = self._layer_materials(target)
            self._emit_layers(materials, list(range(1, len(materials) + 1)))

    def _add_layer(self) -> None:
        target = self._target
        if not isinstance(target, NLayer):
            return
        materials = [m for _, m in target.material_roles]
        selected = self._selected_layer()
        position = len(materials) if selected is None else selected + 1
        sources: list[int | None] = list(range(1, len(materials) + 1))
        materials.insert(position, materials[position - 1])
        sources.insert(position, None)
        self._emit_layers(materials, sources)

    def _remove_layer(self) -> None:
        target = self._target
        if not isinstance(target, NLayer):
            return
        materials = [m for _, m in target.material_roles]
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
                self.commands.set_mirror(dataclasses.replace(mirror, substrate_roughness=value))
            )

    def _change_interlayers(self, checked: bool) -> None:
        target = self._target
        if self._loading or not isinstance(target, Periodic | DepthGraded):
            return
        if checked != target.use_interlayers:
            self._commit(target.with_interlayers(checked))

    def _change_constant_gamma(self, checked: bool) -> None:
        target, composite, block = self._target, self._composite, self._block
        if self._loading or not isinstance(target, DepthGraded):
            return
        if checked == has_constant_gamma(target):
            return
        if composite is not None and block is not None:

            def set_parameter(name: str, **changes: Any) -> Any:
                changed = target.with_parameter(name, **changes)
                return self.commands.set_coating(composite.with_block(block, changed))

            self.commandRequested.emit(set_constant_gamma(target, checked, set_parameter))
        else:
            self.commandRequested.emit(
                set_constant_gamma(target, checked, self.commands.set_parameter)
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
        """Convert the edited coating (or selected block) to a type with the shown options."""
        target = self._target
        if self._loading or target is None:
            return
        try:
            new = convert_coating(target, type_name, **self._options(type_name))
        except (ValueError, KeyError) as exc:
            self.errorOccurred.emit(f"Cannot change coating: {exc}")
            return
        self._commit(new)

    def _change_type(self) -> None:
        coating, target = self._coating, self._target
        type_name = self.type_combo.currentData()
        if self._loading or coating is None or target is None or type_name == coating.type_name:
            return
        source = target if isinstance(coating, Composite) else coating
        try:
            new = convert_coating(source, type_name, **self._options(type_name))
        except (ValueError, KeyError) as exc:
            self.errorOccurred.emit(f"Cannot change coating: {exc}")
            return
        if isinstance(new, Composite):
            self._block = new.block_names()[0]
        self.commandRequested.emit(self.commands.set_coating(new))

    def _change_options(self) -> None:
        target = self._target
        if target is not None:
            self._emit(target.type_name)
