"""Tests for `grazing.gui.common.wizard`."""

from pathlib import Path

import pytest
from PySide6.QtWidgets import QWizard
from pytestqt.qtbot import QtBot

from grazing.core.materials import MaterialLibrary, default_library
from grazing.core.structure.presets import MATERIAL_PAIRS, StructureChoice, build_mirror
from grazing.gui.common.wizard import ProjectWizard

pytestmark = pytest.mark.gui


@pytest.fixture
def wizard(qtbot: QtBot) -> ProjectWizard:
    w = ProjectWizard("design project", ".grazing", "grazing projects (*.grazing)")
    qtbot.addWidget(w)
    w.show()
    return w


def finish_enabled(w: ProjectWizard) -> bool:
    return bool(w.button(QWizard.WizardButton.FinishButton).isEnabled())


def test_defaults(wizard: ProjectWizard) -> None:
    setup = wizard.setup()
    assert wizard.windowTitle() == "New design project"
    assert setup.name == "Untitled"
    assert setup.path is None
    assert setup.structure == StructureChoice("periodic", ("W", "Si"), "H", "Si")
    build_mirror(setup.structure)


@pytest.mark.parametrize(
    ("kind", "materials", "n"),
    [
        ("single_layer", ("Pt",), 1),
        ("layers", ("Mo", "Si"), 2),
        ("periodic", ("W", "Si"), 2),
        ("graded_linear", ("W", "Si"), 2),
        ("graded_power_law", ("W", "Si"), 2),
    ],
)
def test_kinds(wizard: ProjectWizard, kind: str, materials: tuple[str, ...], n: int) -> None:
    wizard.set_kind(kind)  # type: ignore[arg-type]
    choice = wizard.setup().structure
    assert choice.kind == kind
    assert choice.materials == materials
    assert len(choice.materials) == n
    build_mirror(choice)


def test_pair_and_custom(wizard: ProjectWizard) -> None:
    wizard.set_pair("Pt/C")
    assert wizard.setup().structure.materials == MATERIAL_PAIRS["Pt/C"]
    page = wizard._coating
    assert page.pair.currentText() == "Pt/C"
    wizard.set_materials(("Ni", "Si"))
    assert page.pair.currentText() == "Custom"
    assert wizard.setup().structure.materials == ("Ni", "Si")
    wizard.set_materials(("Ir", "B4C"))
    assert page.pair.currentText() == "Ir/B4C"
    with pytest.raises(KeyError):
        wizard.set_pair("Foo/Bar")


def test_individual_layers(wizard: ProjectWizard) -> None:
    wizard.set_kind("layers")
    page = wizard._coating
    page.add_layer("C")
    assert wizard.setup().structure.materials == ("Mo", "Si", "C")
    page.layer_table.setCurrentCell(0, 0)
    page.remove_layer()
    assert wizard.setup().structure.materials == ("Si", "C")
    page.remove_layer()
    page.remove_layer()
    assert len(wizard.setup().structure.materials) == 1
    wizard.set_materials(("Ru", "B4C", "Cr"))
    assert wizard.setup().structure.materials == ("Ru", "B4C", "Cr")


def test_top_and_substrate(wizard: ProjectWizard) -> None:
    wizard.set_top("L")
    assert wizard.setup().structure.top == "L"
    wizard.set_top("H")
    assert wizard.setup().structure.top == "H"
    wizard.set_substrate("SiO2")
    assert wizard.setup().structure.substrate == "SiO2"
    with pytest.raises(ValueError, match="unknown"):
        wizard.set_substrate("Unobtainium")


def test_substrate_defaults_per_kind(wizard: ProjectWizard) -> None:
    wizard.set_kind("single_layer")
    assert wizard.setup().structure.substrate == "SiO2"
    wizard.set_kind("periodic")
    assert wizard.setup().structure.substrate == "Si"
    wizard.set_substrate("Ni")
    wizard.set_kind("single_layer")
    assert wizard.setup().structure.substrate == "Ni"


def test_set_materials_errors(wizard: ProjectWizard) -> None:
    with pytest.raises(ValueError, match="unknown"):
        wizard.set_materials(("W", "Unobtainium"))
    with pytest.raises(ValueError, match="do not fit"):
        wizard.set_materials(("W",))


def test_name_validation(wizard: ProjectWizard) -> None:
    page = wizard._project
    assert page.isComplete()
    wizard.set_name("  ")
    assert not page.isComplete()
    assert "name" in page.message.text()
    wizard.set_name("Mirror")
    assert page.isComplete()
    assert wizard.setup().name == "Mirror"


def test_location_validation(wizard: ProjectWizard, tmp_path: Path) -> None:
    page = wizard._project
    page.save_now.setChecked(True)
    assert not page.isComplete()
    wizard.set_path(tmp_path / "missing" / "m")
    assert not page.isComplete()
    assert "does not exist" in page.message.text()
    wizard.set_path(tmp_path / "m")
    assert page.isComplete()
    assert page.message.text() == ""


def test_path_suffix_and_save_now(wizard: ProjectWizard, tmp_path: Path) -> None:
    wizard.set_path(tmp_path / "m")
    assert wizard._project.save_now.isChecked()
    assert wizard.setup().path == tmp_path / "m.grazing"
    wizard.set_path(tmp_path / "n.GRAZING")
    assert wizard.setup().path == tmp_path / "n.GRAZING"
    wizard._project.save_now.setChecked(False)
    assert wizard.setup().path is None
    assert wizard._project.isComplete()
    wizard.set_path(None)
    assert wizard.setup().path is None


def test_navigation_and_summary(wizard: ProjectWizard, tmp_path: Path, qtbot: QtBot) -> None:
    wizard.set_name("My mirror")
    wizard.set_path(tmp_path / "m")
    wizard.set_pair("Pt/C")
    wizard.set_top("L")
    wizard.next()
    wizard.next()
    text = wizard._summary.text.toPlainText()
    assert "My mirror" in text
    assert str(tmp_path / "m.grazing") in text
    assert "Periodic multilayer" in text
    assert "Absorber (H): Pt" in text
    assert "Spacer (L): C" in text
    assert "Spacer (L)" in text
    assert "Substrate: Si" in text
    assert wizard.currentPage() is wizard._summary
    assert finish_enabled(wizard)
    wizard.back()
    wizard.set_path(None)
    wizard.next()
    assert "not saved yet" in wizard._summary.text.toPlainText()


def test_summary_error_disables_finish(qtbot: QtBot) -> None:
    library = MaterialLibrary.from_dict(default_library().to_dict())
    wizard = ProjectWizard("design project", ".grazing", "grazing (*.grazing)", library)
    qtbot.addWidget(wizard)
    wizard.show()
    library.remove("W")
    wizard.next()
    wizard.next()
    assert "Error" in wizard._summary.text.toPlainText()
    assert not finish_enabled(wizard)


def test_next_blocked_by_invalid_page(wizard: ProjectWizard) -> None:
    wizard.set_name("")
    assert not wizard.button(QWizard.WizardButton.NextButton).isEnabled()
