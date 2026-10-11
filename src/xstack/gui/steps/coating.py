"""Workflow page for the coating type, its options and parameters."""

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from xstack.gui.steps.base import StepPage
from xstack.gui.widgets.parameter_table import ParameterTable
from xstack.materials import get
from xstack.parameters import ParameterSet
from xstack.project import ProjectState
from xstack.project.commands import set_coating
from xstack.structure.coating import Coating
from xstack.structure.coatings.graded import DepthGraded, grading_laws
from xstack.structure.coatings.periodic import Periodic
from xstack.structure.coatings.simple import SingleLayer

__all__ = ["CoatingPage", "convert_coating"]

COATING_TYPES = {
    "periodic": "Periodic multilayer",
    "depth_graded": "Depth-graded multilayer",
    "single_layer": "Single layer",
}
"""Coating types that can be selected in the GUI, by registry name."""


def _carry_over(new: Coating, old: Coating) -> Coating:
    """Return `new` with the settings of parameters that also exist in `old`."""
    params = ParameterSet(old.parameters.get(name, p) for name, p in new.parameters.items())
    try:
        return new.with_parameters(params)
    except ValueError:
        return new


def convert_coating(old: Coating, type_name: str, **options: object) -> Coating:
    """Create a coating of another type or with other options from an existing one.

    Materials and the settings of parameters with the same name are taken over.

    Args:
        old: The current coating.
        type_name: Registry name of the new coating type.
        **options: Options of the new type (``top``, ``law``, ``cap``, ``interlayers``).

    Returns:
        The new coating.

    Raises:
        ValueError: If the type is not supported.
    """
    roles = old.materials
    first = next(iter(roles.values()), get("W"))
    h = roles.get("H", roles.get("M", first))
    low = roles.get("L", get("Si"))
    interlayers = {r: m for r, m in roles.items() if r in ("HonL", "LonH")}
    if "interlayers" in options:
        interlayers = options.pop("interlayers")  # type: ignore[assignment]
    new: Coating
    if type_name == "periodic":
        new = Periodic.create(h, low, interlayers=interlayers, **options)  # type: ignore[arg-type]
    elif type_name == "depth_graded":
        new = DepthGraded.create(h, low, interlayers=interlayers, **options)  # type: ignore[arg-type]
    elif type_name == "single_layer":
        new = SingleLayer.create(h)
    else:
        raise ValueError(f"coating type {type_name!r} cannot be created here")
    return _carry_over(new, old)


class CoatingPage(StepPage):
    """Coating type, type-specific options and the parameter table.

    Args:
        parent: Parent widget.
    """

    title = "Coating"

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._coating: Coating | None = None
        self._loading = False
        layout = QVBoxLayout(self)
        self.header = QLabel()
        layout.addWidget(self.header)
        form = QFormLayout()
        self.type_combo = QComboBox()
        for name, label in COATING_TYPES.items():
            self.type_combo.addItem(label, name)
        self.type_combo.activated.connect(lambda _: self._change_type())
        form.addRow("Coating type", self.type_combo)
        self.top_combo = QComboBox()
        self.top_combo.addItem("Absorber (H)", "H")
        self.top_combo.addItem("Spacer (L)", "L")
        self.top_combo.activated.connect(lambda _: self._change_options())
        form.addRow("Top layer of each bilayer", self.top_combo)
        self.law_combo = QComboBox()
        self.law_combo.addItems(list(grading_laws.names()))
        self.law_combo.activated.connect(lambda _: self._change_options())
        form.addRow("Grading law", self.law_combo)
        self.cap_check = QCheckBox("Cap bilayer with independent period and Γ")
        self.cap_check.toggled.connect(lambda _: self._change_options())
        form.addRow("", self.cap_check)
        layout.addLayout(form)
        layout.addWidget(QLabel("<b>Parameters</b> (check Vary to optimize within Min–Max)"))
        self.parameter_table = ParameterTable()
        self.parameter_table.commandRequested.connect(self.commandRequested)
        layout.addWidget(self.parameter_table)

    def set_state(self, state: ProjectState) -> None:
        """Show the coating of the current design.

        Args:
            state: The current project state.
        """
        coating = state.mirror.coating if state.mirror is not None else None
        if coating == self._coating:
            return
        self._coating = coating
        self._loading = True
        if coating is None:
            self.header.setText("No design defined.")
            self.parameter_table.set_parameters(ParameterSet())
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
            self.parameter_table.set_parameters(coating.parameters)
        self._loading = False

    def _options(self, type_name: str) -> dict[str, object]:
        options: dict[str, object] = {}
        if type_name in ("periodic", "depth_graded"):
            options["top"] = self.top_combo.currentData()
        if type_name == "depth_graded":
            options["law"] = self.law_combo.currentText()
            options["cap"] = self.cap_check.isChecked()
        return options

    def _emit(self, type_name: str) -> None:
        if self._loading or self._coating is None:
            return
        try:
            coating = convert_coating(self._coating, type_name, **self._options(type_name))
        except (ValueError, KeyError) as exc:
            self.errorOccurred.emit(f"Cannot change coating: {exc}")
            return
        if coating != self._coating:
            self.commandRequested.emit(set_coating(coating))

    def _change_type(self) -> None:
        self._emit(self.type_combo.currentData())

    def _change_options(self) -> None:
        if self._coating is not None:
            self._emit(self._coating.type_name)
