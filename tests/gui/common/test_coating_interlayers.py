"""The interlayer option on the shared Coating page and the fit Parameters page."""

import pytest
from pytestqt.qtbot import QtBot

from xstack.core.materials import get
from xstack.core.structure import DepthGraded, Periodic
from xstack.design.project import Command, ProjectState
from xstack.design.project.templates import new_state
from xstack.fit.templates import new_state as new_fit_state
from xstack.gui.common.coating import convert_coating, visible_parameters
from xstack.gui.design.steps import CoatingPage
from xstack.gui.fit.steps.parameters import ParametersPage

pytestmark = pytest.mark.gui

INTERLAYER_PARAMETERS = {"t_interlayer_H_on_L", "t_interlayer_L_on_H"}


def coating_page(qtbot: QtBot, state: ProjectState) -> tuple[CoatingPage, list[Command]]:
    page = CoatingPage()
    qtbot.addWidget(page)
    page.set_state(state)
    commands: list[Command] = []
    page.commandRequested.connect(commands.append)
    return page, commands


def shown(page: CoatingPage | ParametersPage) -> set[str]:
    return set(page.parameter_table.table_model.parameters())


@pytest.mark.parametrize("kind", ["periodic", "depth_graded"])
def test_off_by_default_and_parameters_hidden(qtbot: QtBot, kind: str) -> None:
    page, _ = coating_page(qtbot, new_state(kind))
    assert page.interlayer_check.isEnabled()
    assert not page.interlayer_check.isChecked()
    assert not shown(page) & INTERLAYER_PARAMETERS
    assert "sigma_H" in shown(page)


def test_enabling_shows_parameters(qtbot: QtBot) -> None:
    state = new_state("periodic")
    page, commands = coating_page(qtbot, state)
    page.interlayer_check.setChecked(True)
    assert len(commands) == 1
    new = commands[0].transform(state)
    assert new.mirror is not None
    coating = new.mirror.coating
    assert isinstance(coating, Periodic)
    assert coating.use_interlayers
    page.set_state(new)
    assert page.interlayer_check.isChecked()
    assert shown(page) >= INTERLAYER_PARAMETERS


def test_disabling_zeroes_thicknesses_and_removes_materials(qtbot: QtBot) -> None:
    coating = Periodic.create(
        get("W"),
        get("Si"),
        t_interlayer_H_on_L=0.4,
        interlayers={"interlayer_H_on_L": get("WSi2")},
    )
    state = new_state("periodic")
    assert state.mirror is not None
    state = state.evolve(mirror=state.mirror.with_coating(coating))
    page, commands = coating_page(qtbot, state)
    assert page.interlayer_check.isChecked()
    page.interlayer_check.setChecked(False)
    new = commands[-1].transform(state)
    assert new.mirror is not None
    off = new.mirror.coating
    assert isinstance(off, Periodic)
    assert not off.use_interlayers
    assert "interlayer_H_on_L" not in off.materials
    assert off.parameters["t_interlayer_H_on_L"].value == 0.0
    assert not off.parameters["t_interlayer_H_on_L"].vary
    assert not any("interlayer" in layer.tag for layer in new.mirror.stack().layers)


def test_disabled_for_other_coatings(qtbot: QtBot) -> None:
    page, _ = coating_page(qtbot, new_state("single_layer"))
    assert not page.interlayer_check.isEnabled()
    assert not page.interlayer_check.isChecked()


def test_convert_keeps_option() -> None:
    on = Periodic.create(get("W"), get("Si")).with_interlayers(True)
    converted = convert_coating(on, "depth_graded")
    assert isinstance(converted, DepthGraded)
    assert converted.use_interlayers
    assert not convert_coating(on.with_interlayers(False), "depth_graded").use_interlayers  # type: ignore[attr-defined]


def test_visible_parameters() -> None:
    parameters = Periodic.create(get("W"), get("Si")).parameters
    visible = visible_parameters(parameters, INTERLAYER_PARAMETERS)
    assert list(visible) == [name for name in parameters if name not in INTERLAYER_PARAMETERS]


def test_fit_parameters_page_hides_inactive(qtbot: QtBot) -> None:
    page = ParametersPage()
    qtbot.addWidget(page)
    state = new_fit_state("periodic")
    assert state.mirror is not None
    page.set_state(state)
    assert not shown(page) & INTERLAYER_PARAMETERS
    coating = state.mirror.coating
    assert isinstance(coating, Periodic)
    page.set_state(state.evolve(mirror=state.mirror.with_coating(coating.with_interlayers(True))))
    assert shown(page) >= INTERLAYER_PARAMETERS
