"""Tests for the parameters page of the fit application."""

import math
from typing import Any

import pytest
from PySide6.QtCore import Qt
from pytestqt.qtbot import QtBot

from tests.gui.fit.helpers import make_state
from xstack.core.parameters import ParameterSet
from xstack.fit.project import FitProject, FitState, set_parameter
from xstack.gui.fit.steps.parameters import ParametersPage


@pytest.fixture
def page(qtbot: QtBot) -> ParametersPage:
    p = ParametersPage()
    qtbot.addWidget(p)
    return p


def column(page: ParametersPage, name: str) -> int:
    return list(page.parameter_table.table_model.parameters()).index(name)


def test_shows_problem_parameters(page: ParametersPage) -> None:
    state = make_state(("a", "b"))
    page.set_state(state)
    names = list(page.parameter_table.table_model.parameters())
    assert names == list(state.problem().parameters)
    assert "a.offset" in names
    assert "b.scale" in names
    assert page.free_label.text() == "2 free parameters"


def test_mirror_parameters_without_data(page: ParametersPage) -> None:
    state = make_state(()).evolve(datasets=())
    page.set_state(state)
    assert list(page.parameter_table.table_model.parameters()) == list(state.mirror.parameters)  # type: ignore[union-attr]
    page.set_state(state.evolve(mirror=None))
    assert len(page.parameter_table.table_model.parameters()) == 0


def test_display_units_and_edit(page: ParametersPage) -> None:
    state = make_state()
    state = FitProject(state).state
    page.set_state(state.evolve(datasets=state.datasets))
    model = page.parameter_table.table_model
    row = column(page, "a.offset")
    # set a known offset in rad
    page.set_state(_with(state, "a.offset", value=math.radians(0.5), min=-0.1, max=0.1))
    unit = model.index(row, 2).data()
    value = model.index(row, 1).data()
    assert unit == "deg"
    assert float(value) == pytest.approx(0.5)
    commands: list[Any] = []
    page.commandRequested.connect(commands.append)
    assert model.setData(model.index(row, 1), "1.0")
    assert len(commands) == 1
    new = commands[0].transform(page._state)
    assert new.dataset("a").instrument.parameters["offset"].value == pytest.approx(
        math.radians(1.0)
    )


def _with(state: FitState, name: str, **changes: Any) -> FitState:
    return set_parameter(name, **changes).transform(state)


def test_share_disabled_with_one_dataset(page: ParametersPage) -> None:
    page.set_state(make_state(("a",)))
    assert not page.share_box.isEnabled()
    page.set_state(make_state(("a", "b")))
    assert page.share_box.isEnabled()
    assert [page.reference_combo.itemText(i) for i in range(2)] == ["a", "b"]


def test_share_is_one_undoable_command(page: ParametersPage) -> None:
    state = _with(make_state(("a", "b", "c")), "b.offset", vary=True, min=-0.1, max=0.1)
    page.set_state(state)
    commands: list[Any] = []
    page.commandRequested.connect(commands.append)
    page.parameter_combo.setCurrentIndex(page.parameter_combo.findData("offset"))
    page.reference_combo.setCurrentText("a")
    page.share_button.click()
    assert len(commands) == 1
    project = FitProject(state)
    project.execute(commands[0])
    for name in ("b", "c"):
        p = project.state.dataset(name).instrument.parameters["offset"]
        assert p.expr == "a.offset"
        assert not p.vary
    assert project.state.dataset("a").instrument.parameters["offset"].expr is None
    project.undo()
    assert project.state == state


def test_make_independent_keeps_values(page: ParametersPage) -> None:
    state = _with(make_state(("a", "b")), "a.offset", value=0.002, min=-0.1, max=0.1)
    shared = FitProject(state)
    page.set_state(state)
    commands: list[Any] = []
    page.commandRequested.connect(commands.append)
    page.parameter_combo.setCurrentIndex(page.parameter_combo.findData("offset"))
    page.reference_combo.setCurrentText("a")
    page.share_button.click()
    shared.execute(commands.pop())
    page.set_state(shared.state)
    page.independent_button.click()
    assert len(commands) == 1
    shared.execute(commands[0])
    b = shared.state.dataset("b").instrument.parameters["offset"]
    assert b.expr is None
    assert b.value == pytest.approx(0.002)
    shared.undo()
    assert shared.state.dataset("b").instrument.parameters["offset"].expr == "a.offset"


def test_make_independent_without_expressions_emits_nothing(page: ParametersPage) -> None:
    page.set_state(make_state(("a", "b")))
    commands: list[Any] = []
    page.commandRequested.connect(commands.append)
    page.independent_button.click()
    assert commands == []


def test_table_edit_emits_command(page: ParametersPage) -> None:
    page.set_state(make_state())
    commands: list[Any] = []
    page.commandRequested.connect(commands.append)
    model = page.parameter_table.table_model
    row = column(page, "t")
    index = model.index(row, 3)
    assert model.setData(index, Qt.CheckState.Unchecked.value, Qt.ItemDataRole.CheckStateRole)
    assert len(commands) == 1
    assert isinstance(page.parameter_table.table_model.parameters(), ParameterSet)


def test_expression_rows_show_evaluated_value(page: ParametersPage) -> None:
    state = make_state(("a", "b"))
    state = _with(state, "a.offset", value=0.001)
    state = _with(state, "b.offset", expr="a.offset", vary=False)
    page.set_state(state)
    model = page.parameter_table.table_model
    shown = {n: model.data(model.index(column(page, n), 1)) for n in ("a.offset", "b.offset")}
    assert shown["b.offset"] == shown["a.offset"]
    assert float(shown["a.offset"]) == pytest.approx(math.degrees(0.001), rel=1e-4)
