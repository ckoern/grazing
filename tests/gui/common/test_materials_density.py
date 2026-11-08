"""Density parameters on the shared Materials page."""

import pytest
from PySide6.QtWidgets import QCheckBox
from pytestqt.qtbot import QtBot

from grazing.fit.project import FitProject, FitState
from grazing.gui.fit.steps import MaterialsPage, ParametersPage
from tests.gui.fit.helpers import make_state

pytestmark = pytest.mark.gui


def page_with(qtbot: QtBot, state: FitState) -> tuple[MaterialsPage, list[object]]:
    page = MaterialsPage()
    qtbot.addWidget(page)
    commands: list[object] = []
    page.commandRequested.connect(commands.append)
    page.set_state(state)
    return page, commands


def apply(state: FitState, command: object) -> FitState:
    project = FitProject(state)
    project.execute(command)  # type: ignore[arg-type]
    return project.state


def checkbox(page: MaterialsPage, row: int) -> QCheckBox:
    holder = page.table.cellWidget(row, 3)
    assert holder is not None
    box = holder.findChild(QCheckBox)
    assert box is not None
    return box


def test_toggle_adds_and_removes_density_parameter(qtbot: QtBot) -> None:
    state = make_state()
    page, commands = page_with(qtbot, state)
    assert not checkbox(page, 0).isChecked()
    checkbox(page, 0).setChecked(True)
    state = apply(state, commands[-1])
    assert state.mirror is not None
    assert list(state.mirror.densities) == ["rho_M"]
    page.set_state(state)
    assert checkbox(page, 0).isChecked()
    checkbox(page, 0).setChecked(False)
    state = apply(state, commands[-1])
    assert state.mirror is not None
    assert len(state.mirror.densities) == 0


def test_density_cell_edits_the_parameter(qtbot: QtBot) -> None:
    state = make_state()
    assert state.mirror is not None
    state = state.evolve(mirror=state.mirror.with_density_parameters(["M"]))
    page, commands = page_with(qtbot, state)
    page.table.item(0, 2).setText("19.5")
    state = apply(state, commands[-1])
    assert state.mirror is not None
    assert state.mirror.densities["rho_M"].value == 19.5
    assert state.mirror.coating.materials["M"].density != 19.5
    page.set_state(state)
    assert page.table.item(0, 2).text() == "19.5"


def test_density_parameter_appears_on_parameters_page(qtbot: QtBot) -> None:
    state = make_state()
    assert state.mirror is not None
    state = state.evolve(mirror=state.mirror.with_density_parameters(["M", "substrate"]))
    page = ParametersPage()
    qtbot.addWidget(page)
    page.set_state(state)
    names = list(page.parameter_table.table_model.parameters())
    assert "rho_M" in names
    assert "rho_substrate" in names
    row = names.index("rho_M")
    model = page.parameter_table.table_model
    assert model.data(model.index(row, 2)) == "g/cm³"


def test_design_coating_page_shows_density_parameters(qtbot: QtBot) -> None:
    from grazing.design.project.commands import set_parameter
    from grazing.design.project.templates import new_state
    from grazing.gui.design.steps import CoatingPage

    state = new_state("periodic")
    assert state.mirror is not None
    state = state.evolve(mirror=state.mirror.with_density_parameters(["H"]))
    page = CoatingPage()
    qtbot.addWidget(page)
    page.set_state(state)
    assert "rho_H" in list(page.parameter_table.table_model.parameters())
    changed = set_parameter("rho_H", vary=True).transform(state)
    assert changed.mirror is not None
    assert changed.mirror.densities["rho_H"].vary
