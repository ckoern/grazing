"""Tests for `xstack.gui.design.widgets.design_comparison`."""

import pytest
from PySide6.QtCore import Qt
from pytestqt.qtbot import QtBot

from xstack.design.project import Command
from xstack.design.project.project import Candidate, ProjectState
from xstack.design.project.templates import new_state
from xstack.gui.design.widgets.design_comparison import DesignComparisonView

pytestmark = pytest.mark.gui

NAME, UNIT, INPUT, SELECTED, CHANGE = range(5)


def make_state(**values: float) -> ProjectState:
    """Periodic input design plus a selected candidate with modified values."""
    state = new_state("periodic")
    assert state.mirror is not None
    coating = state.mirror.coating.with_parameter("N", max=500.0)
    coating = coating.with_parameter("d", min=0.5, max=20.0)
    state = state.evolve(mirror=state.mirror.with_coating(coating))
    assert state.mirror is not None
    cand_coating = coating.with_values(**values)
    candidate = Candidate(
        "cand1",
        state.mirror.with_coating(cand_coating),
        source="refine",
        objective=0.25,
        merit=0.75,
        coating_time=7200.0,
    )
    return state.evolve(candidates=(candidate,), selected="cand1")


@pytest.fixture
def view(qtbot: QtBot) -> DesignComparisonView:
    v = DesignComparisonView()
    qtbot.addWidget(v)
    return v


def row_of(view: DesignComparisonView, name: str) -> int:
    table = view.table()
    for r in range(table.rowCount()):
        if table.item(r, NAME).text().split()[0] == name:
            return r
    raise AssertionError(name)


def text(view: DesignComparisonView, name: str, column: int) -> str:
    return str(view.table().item(row_of(view, name), column).text())


def click(view: DesignComparisonView, button: str) -> Command:
    commands: list[Command] = []
    view.commandRequested.connect(commands.append)
    getattr(view, button)().click()
    assert len(commands) == 1
    return commands[0]


def test_no_selection(view: DesignComparisonView) -> None:
    view.set_state(new_state("periodic"))
    assert view.header_text().startswith("Select a candidate")
    assert view.table().rowCount() == 0
    assert not view.promote_button().isEnabled()
    assert not view.copy_button().isEnabled()


def test_table_content(view: DesignComparisonView) -> None:
    view.set_state(make_state(N=60, d=4.4))
    assert view.header_text() == "cand1 (refine) vs. input design"
    table = view.table()
    assert table.rowCount() == len(make_state().candidates[0].mirror.coating.parameters)
    assert [table.horizontalHeaderItem(c).text() for c in range(5)] == [
        "Parameter",
        "Unit",
        "Input",
        "Selected",
        "Change",
    ]
    assert text(view, "N", INPUT) == "40"
    assert text(view, "N", SELECTED) == "60"
    assert text(view, "N", CHANGE) == "+50 %"
    assert text(view, "d", INPUT) == "4"
    assert text(view, "d", SELECTED) == "4.4"
    assert text(view, "d", CHANGE) == "+10 %"
    assert text(view, "gamma", CHANGE) == "—"
    assert text(view, "d", UNIT) == "nm"


def test_changed_values_are_bold(view: DesignComparisonView) -> None:
    view.set_state(make_state(N=60))
    table = view.table()
    assert table.item(row_of(view, "N"), SELECTED).font().bold()
    assert not table.item(row_of(view, "d"), SELECTED).font().bold()


def test_free_parameters_marked(view: DesignComparisonView) -> None:
    state = make_state(N=60)
    assert state.mirror is not None
    coating = state.mirror.coating.with_parameter("gamma", vary=False)
    state = state.evolve(mirror=state.mirror.with_coating(coating))
    view.set_state(state)
    item = view.table().item(row_of(view, "N"), NAME)
    assert "●" in item.text()
    assert "optimized (free) parameter" in item.toolTip()
    assert "●" not in view.table().item(row_of(view, "gamma"), NAME).text()


def test_description_tooltip(view: DesignComparisonView) -> None:
    state = make_state()
    view.set_state(state)
    assert state.mirror is not None
    description = state.mirror.coating.parameters["d"].description
    assert description in view.table().item(row_of(view, "d"), NAME).toolTip()


def test_different_coating_type(view: DesignComparisonView) -> None:
    state = make_state()
    other = new_state("single_layer").mirror
    assert other is not None
    candidate = Candidate("single", other, source="manual")
    view.set_state(state.evolve(candidates=(candidate,), selected="single"))
    assert "differs" in view.info_text()
    assert view.table().rowCount() == len(other.coating.parameters)
    assert text(view, "t", INPUT) == "—"
    view.table().selectRow(0)
    assert not view.copy_button().isEnabled()
    assert view.promote_button().isEnabled()


def test_zero_input_value_shows_difference(view: DesignComparisonView) -> None:
    state = make_state(N=60)
    assert state.mirror is not None
    coating = state.mirror.coating.with_parameter("N", min=0.0, value=0.0)
    state = state.evolve(mirror=state.mirror.with_coating(coating))
    view.set_state(state)
    change = text(view, "N", CHANGE)
    assert change.startswith("n/a")
    assert "60" in change


def test_summary(view: DesignComparisonView) -> None:
    state = make_state(N=60)
    view.set_state(state)
    summary = view.summary_text()
    assert "0.75" in summary
    assert "0.25" in summary
    assert "2 h" in summary
    assert "120 layers" in summary
    assert "80 layers" in summary
    assert "nm" in summary


def test_summary_without_evaluation(view: DesignComparisonView) -> None:
    state = make_state()
    assert state.mirror is not None
    candidate = Candidate("bare", state.mirror)
    view.set_state(state.evolve(candidates=(candidate,), selected="bare"))
    assert "Figure of merit: —" in view.summary_text()


def test_promote_button(view: DesignComparisonView, qtbot: QtBot) -> None:
    state = make_state(N=60)
    view.set_state(state)
    command = click(view, "promote_button")
    new = command.transform(state)
    assert new.mirror == state.candidate("cand1").mirror


def test_copy_button(view: DesignComparisonView, qtbot: QtBot) -> None:
    state = make_state(N=60, d=4.4)
    view.set_state(state)
    assert not view.copy_button().isEnabled()
    view.table().selectRow(row_of(view, "d"))
    assert view.copy_button().isEnabled()
    command = click(view, "copy_button")
    new = command.transform(state)
    assert new.mirror is not None
    assert state.mirror is not None
    params = new.mirror.coating.parameters
    assert params["d"].value == pytest.approx(4.4)
    assert params["N"].value == 40
    assert params["d"].min == state.mirror.coating.parameters["d"].min
    assert params["d"].max == state.mirror.coating.parameters["d"].max


def test_copy_widens_bounds(view: DesignComparisonView, qtbot: QtBot) -> None:
    state = make_state(d=4.4)
    assert state.mirror is not None
    narrow = state.mirror.coating.with_parameter("d", min=3.0, max=4.2, value=4.0)
    state = state.evolve(mirror=state.mirror.with_coating(narrow))
    view.set_state(state)
    view.table().selectRow(row_of(view, "d"))
    command = click(view, "copy_button")
    new = command.transform(state)
    assert new.mirror is not None
    d = new.mirror.coating.parameters["d"]
    assert d.value == pytest.approx(4.4)
    assert d.max == pytest.approx(4.4)
    assert d.min == 3.0


def test_selection_cleared_on_new_state(view: DesignComparisonView) -> None:
    state = make_state(d=4.4)
    view.set_state(state)
    view.table().selectRow(0)
    assert view.copy_button().isEnabled()
    view.set_state(state.evolve(selected=None))
    assert not view.copy_button().isEnabled()
    assert view.table().rowCount() == 0
    assert view.table().editTriggers() == view.table().EditTrigger.NoEditTriggers
    assert Qt.AlignmentFlag.AlignLeft in view.table().horizontalHeader().defaultAlignment()
