"""New projects created with the project wizard."""

from pathlib import Path

import pytest
from PySide6.QtCore import QSettings
from pytestqt.qtbot import QtBot

from grazing.core.structure.coatings.graded import DepthGraded
from grazing.core.structure.coatings.simple import NLayer
from grazing.core.structure.presets import StructureChoice
from grazing.design.project import Project as Doc
from grazing.gui.common.wizard import ProjectSetup, ProjectWizard
from grazing.gui.design.main_window import MainWindow

pytestmark = pytest.mark.gui


class StubWizard:
    def __init__(self, setup: ProjectSetup | None) -> None:
        self._setup = setup

    def exec(self) -> int:
        return 1 if self._setup is not None else 0

    def setup(self) -> ProjectSetup:
        assert self._setup is not None
        return self._setup


def make(qtbot: QtBot, settings: QSettings, setup: ProjectSetup | None) -> MainWindow:
    window = MainWindow(settings=settings, confirm_close=False)
    qtbot.addWidget(window)
    window.wizard_factory = lambda: StubWizard(setup)  # type: ignore[assignment,return-value]
    return window


def test_wizard_creates_and_saves_project(
    qtbot: QtBot, settings: QSettings, tmp_path: Path
) -> None:
    path = tmp_path / "graded.grazing"
    setup = ProjectSetup(
        "Graded Pt/C", path, StructureChoice("graded_linear", ("Pt", "C"), top="L")
    )
    window = make(qtbot, settings, setup)
    assert window.new_project_wizard()
    state = window.controller.state
    assert state.name == "Graded Pt/C"
    assert state.mirror is not None
    assert isinstance(state.mirror.coating, DepthGraded)
    assert state.mirror.coating.law == "linear"
    assert state.mirror.coating.top == "L"
    assert window.controller.path == path
    assert Doc.load(path).state.name == "Graded Pt/C"
    assert not window.controller.is_dirty


def test_wizard_without_location_keeps_project_unsaved(qtbot: QtBot, settings: QSettings) -> None:
    setup = ProjectSetup("Layers", None, StructureChoice("layers", ("Cr", "Pt", "C")))
    window = make(qtbot, settings, setup)
    assert window.new_project_wizard()
    mirror = window.controller.state.mirror
    assert mirror is not None
    assert isinstance(mirror.coating, NLayer)
    assert window.controller.path is None


def test_cancelled_wizard_keeps_project(qtbot: QtBot, settings: QSettings) -> None:
    window = make(qtbot, settings, None)
    before = window.controller.state
    assert not window.new_project_wizard()
    assert window.controller.state is before


def test_new_action_opens_wizard(qtbot: QtBot, settings: QSettings) -> None:
    setup = ProjectSetup("From action", None, StructureChoice("single_layer", ("Ir",)))
    window = make(qtbot, settings, setup)
    window.new_action.trigger()
    assert window.controller.state.name == "From action"


def test_default_factory_builds_project_wizard(qtbot: QtBot, settings: QSettings) -> None:
    window = MainWindow(settings=settings, confirm_close=False)
    qtbot.addWidget(window)
    wizard = window.wizard_factory()
    assert isinstance(wizard, ProjectWizard)
    assert wizard.suffix == ".grazing"


@pytest.mark.parametrize(
    ("kind", "materials", "scan"),
    [("single_layer", ("Pt",), "angle"), ("periodic", ("W", "Si"), "energy")],
)
def test_design_targets_come_from_matching_template(
    qtbot: QtBot, settings: QSettings, kind: str, materials: tuple[str, ...], scan: str
) -> None:
    setup = ProjectSetup("x", None, StructureChoice(kind, materials))  # type: ignore[arg-type]
    window = make(qtbot, settings, setup)
    window.new_project_wizard()
    targets = window.controller.state.targets
    assert targets is not None
    assert targets.targets[0].axis == scan


def test_name_field_shows_new_project_name_while_focused(qtbot: QtBot, settings: QSettings) -> None:
    setup = ProjectSetup("Fresh name", None, StructureChoice("single_layer", ("Pt",)))
    window = make(qtbot, settings, setup)
    window.show()
    window.name_edit.setFocus()
    qtbot.waitUntil(window.name_edit.hasFocus, timeout=2000)
    window.new_project_wizard()
    assert window.name_edit.text() == "Fresh name"
