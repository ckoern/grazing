"""Wizard that asks for the basic choices of a new project.

`ProjectWizard` collects the project name, the file to save it to and a starting structure
(coating kind, materials, top layer, substrate). The applications use the resulting
`ProjectSetup` to pre-fill a new project; everything stays editable afterwards.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QStackedWidget,
    QTableWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QWizard,
    QWizardPage,
)

from xstack.core.materials import MaterialLibrary, default_library
from xstack.core.structure.presets import (
    COATING_KINDS,
    MATERIAL_PAIRS,
    CoatingKind,
    StructureChoice,
    build_mirror,
)
from xstack.gui.common.files import with_suffix
from xstack.gui.common.style import set_variant

__all__ = ["ProjectSetup", "ProjectWizard"]

_CUSTOM = "Custom"

_DESCRIPTIONS: dict[CoatingKind, str] = {
    "single_layer": "One film on a substrate, e.g. a mirror coating for grazing incidence.",
    "layers": "A stack of layers of different materials, each with its own thickness.",
    "periodic": "Repeated absorber/spacer bilayers of equal period, for a narrow band.",
    "graded_linear": "Bilayer period changes linearly with depth, for a broad band.",
    "graded_power_law": "Bilayer period follows a power law in depth (e.g. for hard X-ray optics).",
}
_PAGE_INDEX: dict[CoatingKind, int] = {
    "single_layer": 0,
    "layers": 1,
    "periodic": 2,
    "graded_linear": 2,
    "graded_power_law": 2,
}


@dataclass(frozen=True)
class ProjectSetup:
    """Choices made in the project wizard.

    Attributes:
        name: Project name.
        path: File to save the project to, or None to not save yet.
        structure: Choices for the starting structure.
    """

    name: str
    path: Path | None
    structure: StructureChoice


def _help(text: str) -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setProperty("role", "help")
    return label


class _ProjectPage(QWizardPage):
    """Name and save location."""

    def __init__(self, wizard: "ProjectWizard") -> None:
        super().__init__(wizard)
        self._wizard = wizard
        self.setTitle("Project")
        self.setSubTitle(f"Name the new {wizard.document} and choose where to keep it.")
        self.name_edit = QLineEdit("Untitled")
        self.name_edit.setObjectName("wizard_name")
        self.path_edit = QLineEdit()
        self.path_edit.setObjectName("wizard_path")
        self.path_edit.setPlaceholderText("Not chosen yet")
        browse = QPushButton("Browse…")
        browse.setObjectName("wizard_browse")
        browse.clicked.connect(self._browse)
        self.save_now = QCheckBox("Save the project now")
        self.save_now.setObjectName("wizard_save_now")
        self.message = QLabel()
        self.message.setObjectName("wizard_message")
        self.message.setWordWrap(True)

        row = QHBoxLayout()
        row.addWidget(self.path_edit, 1)
        row.addWidget(browse)
        form = QFormLayout()
        form.addRow("Name", self.name_edit)
        form.addRow("Save to", row)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.save_now)
        layout.addWidget(self.message)
        layout.addWidget(
            _help("The project can also be saved later; the name can be changed at any time.")
        )
        layout.addStretch(1)

        self.name_edit.textChanged.connect(self._changed)
        self.path_edit.textChanged.connect(self._path_changed)
        self.save_now.toggled.connect(self._changed)

    def _browse(self) -> None:
        chosen, _ = QFileDialog.getSaveFileName(
            self,
            f"Save {self._wizard.document}",
            self.path_edit.text(),
            self._wizard.file_filter,
        )
        if chosen:
            self.path_edit.setText(with_suffix(chosen, self._wizard.suffix))

    def _path_changed(self, text: str) -> None:
        self.save_now.setChecked(bool(text.strip()))
        self._changed()

    def _changed(self) -> None:
        self.message.setText(self._problem())
        self.completeChanged.emit()

    def _problem(self) -> str:
        if not self.name_edit.text().strip():
            return "Enter a project name."
        if self.save_now.isChecked():
            text = self.path_edit.text().strip()
            if not text:
                return "Choose a file to save to, or untick “Save the project now”."
            if not Path(text).expanduser().parent.is_dir():
                return "The folder of the chosen file does not exist."
        return ""

    def isComplete(self) -> bool:
        return not self._problem()


def _material_combo(library: MaterialLibrary, current: str) -> QComboBox:
    combo = QComboBox()
    combo.addItems(library.names())
    combo.setCurrentText(current)
    return combo


class _CoatingPage(QWizardPage):
    """Coating kind, materials and substrate."""

    def __init__(self, wizard: "ProjectWizard", library: MaterialLibrary) -> None:
        super().__init__(wizard)
        self._library = library
        self._syncing = False
        self._kind: CoatingKind = "periodic"
        self.setTitle("Coating")
        self.setSubTitle("Choose the kind of coating and its materials.")
        layout = QVBoxLayout(self)

        kinds = QGroupBox("Coating type")
        grid = QGridLayout(kinds)
        grid.setColumnStretch(1, 1)
        self.kind_buttons: dict[CoatingKind, QRadioButton] = {}
        for row, (kind, label) in enumerate(COATING_KINDS.items()):
            button = QRadioButton(label)
            button.setObjectName(f"wizard_kind_{kind}")
            self.kind_buttons[kind] = button
            description = QLabel(_DESCRIPTIONS[kind])
            description.setWordWrap(True)
            description.setStyleSheet("color: palette(placeholder-text);")
            grid.addWidget(button, row, 0)
            grid.addWidget(description, row, 1)
            button.toggled.connect(self._kind_toggled)
        layout.addWidget(kinds)

        materials = QGroupBox("Materials")
        materials_layout = QVBoxLayout(materials)
        self.stack = QStackedWidget()
        self.stack.addWidget(self._single_panel())
        self.stack.addWidget(self._layers_panel())
        self.stack.addWidget(self._multilayer_panel())
        self.stack.setCurrentIndex(2)
        materials_layout.addWidget(self.stack)
        self.substrate = _material_combo(library, "Si")
        self.substrate.setObjectName("wizard_substrate")
        form = QFormLayout()
        form.addRow("Substrate", self.substrate)
        materials_layout.addLayout(form)
        layout.addWidget(materials)
        layout.addStretch()
        self.kind_buttons["periodic"].setChecked(True)

    def _single_panel(self) -> QWidget:
        panel = QWidget()
        form = QFormLayout(panel)
        self.single_material = _material_combo(self._library, "Pt")
        self.single_material.setObjectName("wizard_single_material")
        form.addRow("Material", self.single_material)
        return panel

    def _layers_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.addWidget(QLabel("Layer materials, from the top:"))
        self.layer_table = QTableWidget(0, 1)
        self.layer_table.setObjectName("wizard_layers")
        self.layer_table.horizontalHeader().setVisible(False)
        self.layer_table.horizontalHeader().setStretchLastSection(True)
        self.layer_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.layer_table.setMaximumHeight(130)
        add = QPushButton("Add layer")
        add.setObjectName("wizard_add_layer")
        remove = QPushButton("Remove layer")
        remove.setObjectName("wizard_remove_layer")
        add.clicked.connect(lambda: self.add_layer())
        remove.clicked.connect(self.remove_layer)
        row = QHBoxLayout()
        row.addWidget(add)
        row.addWidget(remove)
        row.addStretch(1)
        layout.addWidget(self.layer_table)
        layout.addLayout(row)
        self.set_layers(("Mo", "Si"))
        return panel

    def _multilayer_panel(self) -> QWidget:
        panel = QWidget()
        form = QFormLayout(panel)
        self.pair = QComboBox()
        self.pair.setObjectName("wizard_pair")
        self.pair.addItems([*MATERIAL_PAIRS, _CUSTOM])
        self.absorber = _material_combo(self._library, "W")
        self.absorber.setObjectName("wizard_absorber")
        self.spacer = _material_combo(self._library, "Si")
        self.spacer.setObjectName("wizard_spacer")
        self.top_h = QRadioButton("Absorber (H) on top")
        self.top_h.setObjectName("wizard_top_h")
        self.top_l = QRadioButton("Spacer (L) on top")
        self.top_l.setObjectName("wizard_top_l")
        self.top_h.setChecked(True)
        top_row = QHBoxLayout()
        top_row.addWidget(self.top_h)
        top_row.addWidget(self.top_l)
        top_row.addStretch(1)
        form.addRow("Material combination", self.pair)
        form.addRow("Absorber (H)", self.absorber)
        form.addRow("Spacer (L)", self.spacer)
        form.addRow("Top layer", top_row)
        self.pair.setCurrentText("W/Si")
        self.pair.currentTextChanged.connect(self._pair_chosen)
        self.absorber.currentTextChanged.connect(self._materials_edited)
        self.spacer.currentTextChanged.connect(self._materials_edited)
        return panel

    # -- behavior ---------------------------------------------------------------------------

    def _pair_chosen(self, label: str) -> None:
        if label in MATERIAL_PAIRS and not self._syncing:
            self._syncing = True
            self.absorber.setCurrentText(MATERIAL_PAIRS[label][0])
            self.spacer.setCurrentText(MATERIAL_PAIRS[label][1])
            self._syncing = False

    def _materials_edited(self) -> None:
        if self._syncing:
            return
        pair = (self.absorber.currentText(), self.spacer.currentText())
        label = next((k for k, v in MATERIAL_PAIRS.items() if v == pair), _CUSTOM)
        self._syncing = True
        self.pair.setCurrentText(label)
        self._syncing = False

    def _kind_toggled(self, checked: bool) -> None:
        if not checked:
            return
        old = self._kind
        self._kind = next(k for k, b in self.kind_buttons.items() if b.isChecked())
        self.stack.setCurrentIndex(_PAGE_INDEX[self._kind])
        # Follow the default substrate unless the user chose another one.
        if self.substrate.currentText() == self.default_substrate(old):
            self.substrate.setCurrentText(self.default_substrate(self._kind))

    @staticmethod
    def default_substrate(kind: CoatingKind) -> str:
        return "SiO2" if kind == "single_layer" else "Si"

    def kind(self) -> CoatingKind:
        return self._kind

    def add_layer(self, material: str = "Si") -> None:
        row = self.layer_table.rowCount()
        self.layer_table.insertRow(row)
        self.layer_table.setCellWidget(row, 0, _material_combo(self._library, material))

    def remove_layer(self) -> None:
        count = self.layer_table.rowCount()
        if count <= 1:
            return
        row = self.layer_table.currentRow()
        self.layer_table.removeRow(row if row >= 0 else count - 1)

    def set_layers(self, names: tuple[str, ...]) -> None:
        self.layer_table.setRowCount(0)
        for name in names:
            self.add_layer(name)

    def layers(self) -> tuple[str, ...]:
        combos = (self.layer_table.cellWidget(r, 0) for r in range(self.layer_table.rowCount()))
        return tuple(cast(QComboBox, c).currentText() for c in combos)

    def materials(self) -> tuple[str, ...]:
        if self._kind == "single_layer":
            return (self.single_material.currentText(),)
        if self._kind == "layers":
            return self.layers()
        return (self.absorber.currentText(), self.spacer.currentText())

    def top(self) -> Literal["H", "L"]:
        return "L" if self.top_l.isChecked() else "H"


class _SummaryPage(QWizardPage):
    """Read-only summary of the choices."""

    def __init__(self, wizard: "ProjectWizard") -> None:
        super().__init__(wizard)
        self._wizard = wizard
        self._valid = False
        self.setTitle("Summary")
        self.setSubTitle("Check your choices and finish.")
        self.text = QTextEdit()
        self.text.setObjectName("wizard_summary")
        self.text.setReadOnly(True)
        layout = QVBoxLayout(self)
        layout.addWidget(self.text, 1)
        layout.addWidget(_help("Everything can be changed later in the project."))

    def initializePage(self) -> None:
        setup = self._wizard.setup()
        choice = setup.structure
        lines = [
            f"Name: {setup.name}",
            f"File: {setup.path if setup.path else 'not saved yet'}",
            f"Coating: {COATING_KINDS[choice.kind]}",
        ]
        if choice.kind == "single_layer":
            lines.append(f"Material: {choice.materials[0]}")
        elif choice.kind == "layers":
            lines.append("Layers from the top: " + ", ".join(choice.materials))
        else:
            lines.append(f"Absorber (H): {choice.materials[0]}")
            lines.append(f"Spacer (L): {choice.materials[1]}")
            top = choice.materials[0 if choice.top == "H" else 1]
            lines.append(
                f"Top layer: {'Absorber (H)' if choice.top == 'H' else 'Spacer (L)'} ({top})"
            )
        lines.append(f"Substrate: {choice.substrate}")
        try:
            build_mirror(choice, self._wizard.library)
            self._valid = True
        except Exception as error:
            self._valid = False
            lines.append("")
            lines.append(f"Error: {error}")
        self.text.setPlainText("\n".join(lines))
        self.completeChanged.emit()

    def isComplete(self) -> bool:
        return self._valid


class ProjectWizard(QWizard):
    """Wizard for the name, location and starting structure of a new project.

    Args:
        document: Kind of project for texts, e.g. ``"design project"``.
        suffix: File suffix including the dot, e.g. ``".xstack"``.
        file_filter: Name filter of the save dialog, e.g. ``"xstack projects (*.xstack)"``.
        library: Material library offering the materials. Defaults to the default library.
        parent: Parent widget.
    """

    def __init__(
        self,
        document: str,
        suffix: str,
        file_filter: str,
        library: MaterialLibrary | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.document = document
        self.suffix = suffix
        self.file_filter = file_filter
        self.library = library or default_library()
        self.setWindowTitle(f"New {document}")
        self.setWizardStyle(QWizard.WizardStyle.ModernStyle)
        self.setOption(QWizard.WizardOption.NoBackButtonOnStartPage, True)
        self._project = _ProjectPage(self)
        self._coating = _CoatingPage(self, self.library)
        self._summary = _SummaryPage(self)
        self.addPage(self._project)
        self.addPage(self._coating)
        self.addPage(self._summary)
        self.setButtonText(QWizard.WizardButton.FinishButton, "Create")
        finish = self.button(QWizard.WizardButton.FinishButton)
        set_variant(finish, "primary")
        self.setMinimumSize(680, 680)

    # -- results ----------------------------------------------------------------------------

    def setup(self) -> ProjectSetup:
        """Return the current choices.

        Returns:
            Project name, file (None if the project is not to be saved yet) and structure.

        Raises:
            ValueError: If the materials do not fit the coating kind (e.g. no layers).
        """
        page = self._project
        text = page.path_edit.text().strip()
        path = None
        if page.save_now.isChecked() and text:
            path = Path(with_suffix(text, self.suffix)).expanduser()
        structure = StructureChoice(
            self._coating.kind(),
            self._coating.materials(),
            self._coating.top(),
            self._coating.substrate.currentText(),
        )
        return ProjectSetup(page.name_edit.text().strip(), path, structure)

    # -- programmatic setters ---------------------------------------------------------------

    def set_name(self, name: str) -> None:
        """Set the project name.

        Args:
            name: Project name.
        """
        self._project.name_edit.setText(name)

    def set_path(self, path: str | Path | None) -> None:
        """Set the file to save to and tick or untick "Save the project now".

        Args:
            path: File path, or None to clear it and not save yet.
        """
        self._project.path_edit.setText("" if path is None else str(path))

    def set_kind(self, kind: CoatingKind) -> None:
        """Select the coating kind.

        Args:
            kind: Coating kind.

        Raises:
            KeyError: If the kind is unknown.
        """
        self._coating.kind_buttons[kind].setChecked(True)

    def set_pair(self, label: str) -> None:
        """Select a material combination of a multilayer; sets absorber and spacer.

        Args:
            label: Key of `MATERIAL_PAIRS`, e.g. ``"Pt/C"``.

        Raises:
            KeyError: If the label is not a known combination.
        """
        if label not in MATERIAL_PAIRS:
            raise KeyError(label)
        self._coating.pair.setCurrentText(label)

    def set_materials(self, names: tuple[str, ...]) -> None:
        """Set the materials of the current coating kind.

        Args:
            names: One material for a single layer, the layers from the top for individual
                layers, absorber (H) and spacer (L) for multilayers.

        Raises:
            ValueError: If the number of materials does not fit the kind or a material is not
                in the library.
        """
        page = self._coating
        kind = page.kind()
        unknown = [n for n in names if n not in self.library]
        if unknown:
            raise ValueError(f"unknown materials: {unknown}")
        if kind == "single_layer" and len(names) == 1:
            page.single_material.setCurrentText(names[0])
        elif kind == "layers" and names:
            page.set_layers(tuple(names))
        elif kind not in ("single_layer", "layers") and len(names) == 2:
            page.absorber.setCurrentText(names[0])
            page.spacer.setCurrentText(names[1])
        else:
            raise ValueError(f"{len(names)} materials do not fit the kind {kind!r}")

    def set_top(self, top: Literal["H", "L"]) -> None:
        """Select the top layer of a multilayer.

        Args:
            top: ``"H"`` for the absorber, ``"L"`` for the spacer.
        """
        (self._coating.top_h if top == "H" else self._coating.top_l).setChecked(True)

    def set_substrate(self, name: str) -> None:
        """Set the substrate material.

        Args:
            name: Material name.

        Raises:
            ValueError: If the material is not in the library.
        """
        if name not in self.library:
            raise ValueError(f"unknown material {name!r}")
        self._coating.substrate.setCurrentText(name)
