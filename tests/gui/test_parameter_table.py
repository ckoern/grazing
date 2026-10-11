"""Tests for `xstack.gui.widgets.parameter_table`."""

import pytest
from PySide6.QtCore import QModelIndex, Qt
from pytestqt.qtbot import QtBot

from xstack.gui.controller import ProjectController
from xstack.gui.widgets.parameter_table import ParameterTable, ParameterTableModel
from xstack.parameters import ParameterSet
from xstack.project import Command
from xstack.project.templates import new_state

pytestmark = pytest.mark.gui

COLS = ["Parameter", "Value", "Unit", "Vary", "Min", "Max", "Expression"]
NAME, VALUE, UNIT, VARY, MIN, MAX, EXPR = range(7)


def periodic_params() -> ParameterSet:
    state = new_state("periodic")
    assert state.mirror is not None
    return state.mirror.coating.parameters


@pytest.fixture
def model() -> ParameterTableModel:
    m = ParameterTableModel()
    m.set_parameters(periodic_params())
    return m


def row(model: ParameterTableModel, name: str) -> int:
    return list(model.parameters()).index(name)


def cell(model: ParameterTableModel, name: str, column: int) -> QModelIndex:
    return model.index(row(model, name), column)


def edit(
    model: ParameterTableModel,
    name: str,
    column: int,
    value: object,
    role: int = Qt.ItemDataRole.EditRole,
) -> tuple[bool, list[Command]]:
    """Call setData and return its result with the emitted commands."""
    emitted: list[Command] = []
    model.commandRequested.connect(emitted.append)
    ok = model.setData(cell(model, name, column), value, role)
    return ok, emitted


def apply(command: Command) -> ProjectController:
    c = ProjectController()
    assert c.execute(command)
    return c


def test_shape_and_header(model: ParameterTableModel) -> None:
    assert model.rowCount() == len(periodic_params())
    assert model.columnCount() == 7
    headers = [model.headerData(i, Qt.Orientation.Horizontal) for i in range(7)]
    assert headers == COLS


def test_display_values(model: ParameterTableModel) -> None:
    n = row(model, "N")
    assert model.index(n, NAME).data() == "N"
    assert model.index(n, VALUE).data() == "40"
    assert model.index(n, MIN).data() == "10"
    assert model.index(n, MAX).data() == "200"
    assert model.index(row(model, "d"), VALUE).data() == "4"
    assert model.index(n, VARY).data(Qt.ItemDataRole.CheckStateRole) == Qt.CheckState.Checked


def test_infinite_bounds_shown() -> None:
    m = ParameterTableModel()
    m.set_parameters(periodic_params().replace("d", min=float("-inf"), max=float("inf")))
    r = row(m, "d")
    assert m.index(r, MIN).data() == "-∞"
    assert m.index(r, MAX).data() == "∞"


def test_flags(model: ParameterTableModel) -> None:
    for column in (VALUE, MIN, MAX, EXPR):
        assert model.flags(model.index(0, column)) & Qt.ItemFlag.ItemIsEditable
    assert not model.flags(model.index(0, NAME)) & Qt.ItemFlag.ItemIsEditable
    assert not model.flags(model.index(0, UNIT)) & Qt.ItemFlag.ItemIsEditable
    assert model.flags(model.index(0, VARY)) & Qt.ItemFlag.ItemIsUserCheckable


def test_set_value(model: ParameterTableModel) -> None:
    ok, emitted = edit(model, "N", VALUE, "55")
    assert ok
    assert len(emitted) == 1
    c = apply(emitted[0])
    assert c.state.mirror.coating.parameters["N"].value == 55  # type: ignore[union-attr]


def test_set_min_max(model: ParameterTableModel) -> None:
    _, emitted = edit(model, "N", MIN, "20")
    _, emitted_max = edit(model, "N", MAX, "150")
    c = ProjectController()
    c.execute(emitted[0])
    c.execute(emitted_max[-1])
    p = c.state.mirror.coating.parameters["N"]  # type: ignore[union-attr]
    assert (p.min, p.max) == (20, 150)


def test_set_vary(model: ParameterTableModel) -> None:
    ok, emitted = edit(model, "N", VARY, Qt.CheckState.Unchecked, Qt.ItemDataRole.CheckStateRole)
    assert ok
    c = apply(emitted[0])
    assert not c.state.mirror.coating.parameters["N"].vary  # type: ignore[union-attr]


def test_set_expression(model: ParameterTableModel) -> None:
    ok, emitted = edit(model, "d", EXPR, "gamma * 10")
    assert ok
    c = apply(emitted[0])
    assert c.state.mirror.coating.parameters["d"].expr == "gamma * 10"  # type: ignore[union-attr]


def test_clear_expression(model: ParameterTableModel) -> None:
    ok, emitted = edit(model, "d", EXPR, "  ")
    assert ok
    assert len(emitted) == 1


@pytest.mark.parametrize("column", [VALUE, MIN, MAX])
def test_invalid_text(model: ParameterTableModel, column: int) -> None:
    for text in ("abc", ""):
        ok, emitted = edit(model, "N", column, text)
        assert not ok
        assert emitted == []


def test_non_editable_column(model: ParameterTableModel) -> None:
    ok, emitted = edit(model, "N", NAME, "x")
    assert not ok
    assert emitted == []


def test_same_names_emit_data_changed(model: ParameterTableModel, qtbot: QtBot) -> None:
    resets: list[int] = []
    model.modelReset.connect(lambda: resets.append(1))
    with qtbot.waitSignal(model.dataChanged):
        model.set_parameters(periodic_params().replace("N", value=60))
    assert resets == []
    assert model.index(row(model, "N"), VALUE).data() == "60"


def test_identical_parameters_emit_nothing(model: ParameterTableModel) -> None:
    changes: list[int] = []
    model.dataChanged.connect(lambda *_: changes.append(1))
    model.modelReset.connect(lambda: changes.append(2))
    model.set_parameters(periodic_params())
    assert changes == []


def test_different_names_reset(model: ParameterTableModel, qtbot: QtBot) -> None:
    state = new_state("single_layer")
    assert state.mirror is not None
    with qtbot.waitSignal(model.modelReset):
        model.set_parameters(state.mirror.coating.parameters)
    assert model.rowCount() == len(state.mirror.coating.parameters)


def test_table_widget_forwards_commands(qtbot: QtBot) -> None:
    table = ParameterTable()
    qtbot.addWidget(table)
    table.set_parameters(periodic_params())
    with qtbot.waitSignal(table.commandRequested):
        table.table_model.setData(table.table_model.index(0, VALUE), "12")
