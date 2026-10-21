"""The optical-constants selector of the shared Materials page."""

import pytest
from pytestqt.qtbot import QtBot

from tests.gui.fit.helpers import make_state
from xstack.core.materials import DEFAULT_PROVIDER, TabulatedProvider, providers
from xstack.design.project import Project, ProjectState
from xstack.design.project.templates import new_state
from xstack.fit.project import FitProject
from xstack.gui.common.materials import PROVIDER_LABELS, provider_choices
from xstack.gui.design.steps import MaterialsPage
from xstack.gui.fit.steps.materials import MaterialsPage as FitMaterialsPage

pytestmark = pytest.mark.gui


def design_page(qtbot: QtBot, state: ProjectState) -> tuple[MaterialsPage, list[object]]:
    page = MaterialsPage()
    qtbot.addWidget(page)
    commands: list[object] = []
    page.commandRequested.connect(commands.append)
    page.set_state(state)
    return page, commands


def choose(page: MaterialsPage | FitMaterialsPage, provider: str) -> None:
    index = page.provider_combo.findData(provider)
    assert index >= 0
    page.provider_combo.setCurrentIndex(index)
    page.provider_combo.activated.emit(index)


def plotted_items(page: MaterialsPage | FitMaterialsPage) -> int:
    return len(page.plot.getPlotItem().listDataItems())


def test_choices() -> None:
    names = [name for name, _ in provider_choices()]
    assert names[: len(PROVIDER_LABELS)] == list(PROVIDER_LABELS)
    assert names[0] == DEFAULT_PROVIDER
    assert "chantler" in names
    assert "chantler_photo" in names
    assert len(names) == len(set(names))


def test_tabulated_providers_not_offered() -> None:
    providers.register("test_table", TabulatedProvider("test_table", [1, 2], [1, 1], [1, 1]))
    try:
        assert "test_table" not in [name for name, _ in provider_choices()]
    finally:
        providers.unregister("test_table")


def test_shows_project_provider(qtbot: QtBot) -> None:
    state = new_state("periodic")
    page, _ = design_page(qtbot, state)
    assert page.provider_combo.currentData() == DEFAULT_PROVIDER
    page.set_state(state.evolve(optical_constants="chantler"))
    assert page.provider_combo.currentData() == "chantler"


def test_choosing_emits_command(qtbot: QtBot) -> None:
    state = new_state("periodic")
    page, commands = design_page(qtbot, state)
    choose(page, "chantler")
    assert len(commands) == 1
    project = Project(state)
    project.execute(commands[0])  # type: ignore[arg-type]
    assert project.state.optical_constants == "chantler"
    page.set_state(project.state)
    choose(page, DEFAULT_PROVIDER)
    assert len(commands) == 2


def test_unchanged_choice_emits_nothing(qtbot: QtBot) -> None:
    page, commands = design_page(qtbot, new_state("periodic"))
    choose(page, DEFAULT_PROVIDER)
    assert commands == []


def test_unregistered_provider_shown_without_plot(qtbot: QtBot) -> None:
    page, _ = design_page(qtbot, new_state("periodic").evolve(optical_constants="other"))
    assert page.provider_combo.currentData() == "other"
    assert plotted_items(page) == 0


@pytest.mark.parametrize("provider", ["henke", "chantler", "xraylib"])
def test_plot_within_provider_range(qtbot: QtBot, provider: str) -> None:
    """Every material is plotted, also with providers that end below 100 keV."""
    state = new_state("periodic").evolve(optical_constants=provider)
    assert state.mirror is not None
    materials = {*state.mirror.coating.materials.values(), state.mirror.substrate}
    page, _ = design_page(qtbot, state)
    assert plotted_items(page) == 2 * len(materials)


def test_fit_page(qtbot: QtBot) -> None:
    page = FitMaterialsPage()
    qtbot.addWidget(page)
    commands: list[object] = []
    page.commandRequested.connect(commands.append)
    state = make_state()
    page.set_state(state)
    choose(page, "chantler")
    project = FitProject(state)
    project.execute(commands[-1])  # type: ignore[arg-type]
    assert project.state.optical_constants == "chantler"
