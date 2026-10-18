"""Workflow page for the structure model of the fit."""

from pathlib import Path
from typing import ClassVar

from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from xstack.core.io.models import ImportedModel, read_models
from xstack.core.physics.fresnel import interface_models
from xstack.core.physics.models import KinematicModel, ParrattModel, ReflectivityModel
from xstack.fit.project import FitState, set_mirror, set_model
from xstack.fit.templates import TEMPLATES
from xstack.gui.common.coating import CoatingPage
from xstack.gui.common.style import set_variant
from xstack.gui.fit.steps.base import FIT_COMMANDS

__all__ = ["REFLECTIVITY_MODELS", "StructurePage", "model_key", "reflectivity_model"]

DESIGN_FILTER = "xstack design projects (*.xstack);;All files (*)"

REFLECTIVITY_MODELS = {
    "parratt": "Parratt (exact)",
    "kinematic": "Kinematic (fast)",
    "kinematic_no_refraction": "Kinematic without refraction",
}
"""Reflectivity models offered for fits, by key."""

_INTERFACE_LABELS = {
    "nevot_croce": "Névot–Croce",
    "debye_waller": "Debye–Waller",
    "sharp": "Sharp interfaces",
}


def model_key(model: ReflectivityModel) -> tuple[str, str]:
    """Return the key in `REFLECTIVITY_MODELS` and the interface model of a model.

    Args:
        model: The reflectivity model.

    Returns:
        Model key and interface model name.
    """
    interface = str(getattr(model, "interface_model", "nevot_croce"))
    if isinstance(model, KinematicModel):
        return ("kinematic" if model.refraction else "kinematic_no_refraction"), interface
    return "parratt", interface


def reflectivity_model(key: str, interface: str) -> ReflectivityModel:
    """Create a reflectivity model from a key of `REFLECTIVITY_MODELS`.

    Args:
        key: The model key.
        interface: Name of a registered interface roughness model.

    Returns:
        The model.
    """
    if key == "parratt":
        return ParrattModel(interface_model=interface)
    return KinematicModel(refraction=key == "kinematic", interface_model=interface)


class StructurePage(CoatingPage):
    """Structure model of the fit: start structure, coating, substrate and reflectivity model.

    The structure is defined here directly, started from a template, or imported from an
    ``.xstack`` design project. Its parameters are edited on the Parameters page.

    Args:
        parent: Parent widget.
    """

    title = "Structure"
    help_text: ClassVar[str] = (
        "Define the <b>structure model</b> of the measured sample: choose the coating type and "
        "its options here and the materials on the next page. Start from a <b>template</b>, or "
        "import the input design or a candidate from an <b>xstack design project</b> "
        "(.xstack). The <b>reflectivity model</b> decides between the exact Parratt "
        "calculation and the fast kinematic approximation. The structure parameters are set up "
        "for the fit on the Parameters page."
    )

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(FIT_COMMANDS, parent)
        self.parameters_label.setVisible(False)
        self.parameter_table.setVisible(False)
        self._models: tuple[ImportedModel, ...] = ()
        self._model: ReflectivityModel | None = None
        layout = self.layout()
        assert isinstance(layout, QVBoxLayout)

        start = QGroupBox("Start structure")
        form = QFormLayout(start)
        self.source_label = QLabel()
        self.source_label.setWordWrap(True)
        form.addRow("Current", self.source_label)
        templates = QHBoxLayout()
        self.template_combo = QComboBox()
        for name in TEMPLATES:
            if name != "empty":
                self.template_combo.addItem(name.replace("_", " ").capitalize(), name)
        self.template_button = QPushButton("Use template")
        self.template_button.setToolTip("Replace the structure by the template's structure.")
        self.template_button.clicked.connect(self._use_template)
        templates.addWidget(self.template_combo, 1)
        templates.addWidget(self.template_button)
        form.addRow("Template", templates)
        design = QHBoxLayout()
        self.design_path = QLineEdit()
        self.design_path.setPlaceholderText("design project (.xstack)")
        self.design_path.editingFinished.connect(lambda: self.load_design(self.design_path.text()))
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        design.addWidget(self.design_path, 1)
        design.addWidget(browse)
        form.addRow("Design file", design)
        imported = QHBoxLayout()
        self.design_combo = QComboBox()
        self.design_combo.setToolTip("Input design or stored candidate of the design project.")
        self.import_button = QPushButton("Use as structure")
        set_variant(self.import_button, "primary")
        self.import_button.setToolTip(
            "Replace the structure by the chosen design; measured data and fits are kept."
        )
        self.import_button.clicked.connect(self._use_design)
        imported.addWidget(self.design_combo, 1)
        imported.addWidget(self.import_button)
        form.addRow("Design", imported)
        layout.insertWidget(0, start)

        model_box = QGroupBox("Reflectivity model")
        model_form = QFormLayout(model_box)
        self.model_combo = QComboBox()
        for key, label in REFLECTIVITY_MODELS.items():
            self.model_combo.addItem(label, key)
        self.interface_combo = QComboBox()
        for name in interface_models.names():
            self.interface_combo.addItem(_INTERFACE_LABELS.get(name, name), name)
        self.interface_combo.setToolTip("Model of the interface roughness.")
        self.model_combo.activated.connect(lambda _: self._change_model())
        self.interface_combo.activated.connect(lambda _: self._change_model())
        model_form.addRow("Model", self.model_combo)
        model_form.addRow("Roughness", self.interface_combo)
        layout.insertWidget(layout.indexOf(self.layers_box) + 1, model_box)
        self._update_design_controls()

    def set_state(self, state: FitState) -> None:
        """Show the structure of a fit state.

        Args:
            state: The current fit state.
        """
        super().set_state(state)
        if state.mirror is None:
            source = "No structure defined. Use a template or import a design."
        else:
            source = state.model_source or "defined in this project"
        self.source_label.setText(source)
        if state.model != self._model:
            self._model = state.model
            key, interface = model_key(state.model)
            self.model_combo.setCurrentIndex(self.model_combo.findData(key))
            self.interface_combo.setCurrentIndex(self.interface_combo.findData(interface))

    def load_design(self, path: str | Path) -> bool:
        """Read the designs of an ``.xstack`` file into the design chooser.

        Args:
            path: The design project.

        Returns:
            Whether the file was read.
        """
        text = str(path).strip()
        self._models = ()
        self.design_combo.clear()
        if text:
            self.design_path.setText(text)
            try:
                self._models = read_models(text)
            except (OSError, ValueError) as exc:
                self.errorOccurred.emit(f"Cannot read {Path(text).name}: {exc}")
        for model in self._models:
            coating = model.mirror.coating
            self.design_combo.addItem(
                f"{model.name} — {type(coating).__name__} “{coating.name}”, "
                f"substrate {model.mirror.substrate.name}"
            )
        self._update_design_controls()
        return bool(self._models)

    def _update_design_controls(self) -> None:
        available = bool(self._models)
        self.design_combo.setEnabled(available)
        self.import_button.setEnabled(available)

    def _browse(self) -> None:
        chosen, _ = QFileDialog.getOpenFileName(self, "Import design", "", DESIGN_FILTER)
        if chosen:
            self.load_design(chosen)

    def _use_design(self) -> None:
        index = self.design_combo.currentIndex()
        if not 0 <= index < len(self._models):
            return
        model = self._models[index]
        source = f"{Path(self.design_path.text()).name}: {model.name}"
        self.commandRequested.emit(set_mirror(model.mirror, model.model, source))

    def _use_template(self) -> None:
        name = self.template_combo.currentData()
        mirror = TEMPLATES[name]().mirror
        if mirror is not None:
            label = self.template_combo.currentText()
            self.commandRequested.emit(set_mirror(mirror, source=f"template: {label}"))

    def _change_model(self) -> None:
        model = reflectivity_model(
            self.model_combo.currentData(), self.interface_combo.currentData()
        )
        if model != self._model:
            self.commandRequested.emit(set_model(model))
