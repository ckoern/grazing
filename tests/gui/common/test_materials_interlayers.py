"""Interlayer materials on the shared Materials page and in the layer profile."""

import pytest
from PySide6.QtWidgets import QComboBox
from pytestqt.qtbot import QtBot

from xstack.core.structure import Periodic
from xstack.design.project import Project, ProjectState
from xstack.design.project.templates import new_state
from xstack.gui.common.materials import INTERLAYER_START
from xstack.gui.common.profile import LayerProfilePlot
from xstack.gui.design.steps import MaterialsPage

pytestmark = pytest.mark.gui


def page_with(qtbot: QtBot, state: ProjectState) -> tuple[MaterialsPage, list[object]]:
    page = MaterialsPage()
    qtbot.addWidget(page)
    commands: list[object] = []
    page.commandRequested.connect(commands.append)
    page.set_state(state)
    return page, commands


def rows(page: MaterialsPage) -> dict[str, int]:
    return {page.table.item(r, 0).text(): r for r in range(page.table.rowCount())}


def choose(page: MaterialsPage, role: str, name: str) -> None:
    combo = page.table.cellWidget(rows(page)[role], 1)
    assert isinstance(combo, QComboBox)
    combo.setCurrentText(name)
    combo.activated.emit(combo.currentIndex())


def apply(state: ProjectState, command: object) -> ProjectState:
    project = Project(state)
    project.execute(command)  # type: ignore[arg-type]
    return project.state


def with_interlayers(state: ProjectState) -> ProjectState:
    assert state.mirror is not None
    coating = state.mirror.coating
    assert isinstance(coating, Periodic)
    return state.evolve(mirror=state.mirror.with_coating(coating.with_interlayers(True)))


def test_interlayer_rows_hidden_by_default(qtbot: QtBot) -> None:
    page, _ = page_with(qtbot, new_state("periodic"))
    assert list(rows(page)) == ["H", "L", "substrate"]


def test_interlayer_roles_listed_without_material(qtbot: QtBot) -> None:
    page, _ = page_with(qtbot, with_interlayers(new_state("periodic")))
    assert list(rows(page)) == ["H", "L", "interlayer_H_on_L", "interlayer_L_on_H", "substrate"]
    combo = page.table.cellWidget(rows(page)["interlayer_H_on_L"], 1)
    assert isinstance(combo, QComboBox)
    assert combo.currentText() == "(none)"


def test_single_layer_has_no_interlayer_rows(qtbot: QtBot) -> None:
    page, _ = page_with(qtbot, new_state("single_layer"))
    assert list(rows(page)) == ["M", "substrate"]


def test_choosing_an_interlayer_adds_it_to_the_stack(qtbot: QtBot) -> None:
    state = with_interlayers(new_state("periodic"))
    page, commands = page_with(qtbot, state)
    choose(page, "interlayer_H_on_L", "WSi2")
    state = apply(state, commands[-1])
    assert state.mirror is not None
    coating = state.mirror.coating
    assert coating.materials["interlayer_H_on_L"].name == "WSi2"
    assert coating.parameters["t_interlayer_H_on_L"].value == INTERLAYER_START
    tags = [layer.tag for layer in state.mirror.stack().layers]
    assert any(tag.endswith("interlayer_H_on_L") for tag in tags)


def test_choosing_none_removes_interlayer_and_thickness(qtbot: QtBot) -> None:
    state = with_interlayers(new_state("periodic"))
    page, commands = page_with(qtbot, state)
    choose(page, "interlayer_L_on_H", "WSi2")
    state = apply(state, commands[-1])
    page.set_state(state)
    choose(page, "interlayer_L_on_H", "(none)")
    state = apply(state, commands[-1])
    assert state.mirror is not None
    assert "interlayer_L_on_H" not in state.mirror.coating.materials
    assert state.mirror.coating.parameters["t_interlayer_L_on_H"].value == 0.0
    state.mirror.stack()


def test_layer_profile_shows_interlayers(qtbot: QtBot) -> None:
    state = with_interlayers(new_state("periodic"))
    page, commands = page_with(qtbot, state)
    choose(page, "interlayer_H_on_L", "WSi2")
    state = apply(state, commands[-1])
    assert state.mirror is not None
    plot = LayerProfilePlot()
    qtbot.addWidget(plot)
    plot.set_stack(state.mirror.stack())
    first = plot.graphics.getItem(0, 0)
    legend = [label.text for _, label in first.legend.items]
    assert "WSi2" in legend
