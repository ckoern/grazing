"""Tests of the candidates list selection, context menu and parameter dialog of the main window."""

import pytest
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QAbstractItemView, QMenu

from xstack.gui.main_window import MainWindow
from xstack.gui.widgets.candidate_dialog import CandidateParametersDialog

pytestmark = pytest.mark.gui


def set_d(window: MainWindow, value: float) -> None:
    model = window.parameter_table.table_model
    names = list(model.parameters())
    assert model.setData(model.index(names.index("d"), 1), str(value))


def action_texts(menu: QMenu) -> dict[str, bool]:
    return {a.text(): a.isEnabled() for a in menu.actions() if not a.isSeparator()}


def find_action(menu: QMenu, text: str) -> QAction:
    for action in menu.actions():
        if action.text() == text:
            return action
    raise AssertionError(f"no action {text!r} in {action_texts(menu)}")


@pytest.fixture
def stored(window: MainWindow) -> MainWindow:
    for name in ("a", "b", "c"):
        window.store_candidate(name)
    return window


def test_candidate_list_allows_extended_selection(stored: MainWindow) -> None:
    assert (
        stored.candidate_list.selectionMode() == QAbstractItemView.SelectionMode.ExtendedSelection
    )
    stored.controller.undo_stack.setClean()


def test_selected_candidates_and_bulk_delete(stored: MainWindow) -> None:
    window = stored
    count = window.controller.undo_stack.count()
    window.candidate_list.item(0).setSelected(True)
    window.candidate_list.item(2).setSelected(True)
    assert window.selected_candidates() == ["a", "c"]

    window.delete_button.click()
    assert [c.name for c in window.controller.state.candidates] == ["b"]
    assert window.controller.undo_stack.count() == count + 1

    window.controller.undo()
    assert [c.name for c in window.controller.state.candidates] == ["a", "b", "c"]
    window.controller.undo_stack.setClean()


def test_menu_for_single_current_candidate(stored: MainWindow) -> None:
    window = stored
    window.candidate_list.setCurrentRow(1)
    window.candidate_list.item(1).setSelected(True)
    menu = window.candidate_menu()
    states = action_texts(menu)
    assert states == {
        "Use as start design": True,
        "Rename": True,
        "Show parameters…": True,
        "Delete": True,
    }
    menu.deleteLater()
    window.controller.undo_stack.setClean()


def test_menu_for_two_selected_candidates(stored: MainWindow) -> None:
    window = stored
    window.candidate_list.item(0).setSelected(True)
    window.candidate_list.item(2).setSelected(True)
    menu = window.candidate_menu()
    states = action_texts(menu)
    assert states["Use as start design"] is False
    assert states["Rename"] is False
    assert states["Show parameters…"] is False
    assert states["Delete 2 candidates"] is True
    assert "Delete" not in states
    menu.deleteLater()
    window.controller.undo_stack.setClean()


def test_use_as_start_design_copies_candidate(stored: MainWindow) -> None:
    window = stored
    set_d(window, 4.5)
    state = window.controller.state
    assert state.mirror != state.candidates[0].mirror
    window.candidate_list.setCurrentRow(0)
    menu = window.candidate_menu()
    find_action(menu, "Use as start design").trigger()
    state = window.controller.state
    assert state.mirror == state.candidate("a").mirror
    menu.deleteLater()
    window.controller.undo_stack.setClean()


def test_show_parameters_opens_dialog(stored: MainWindow) -> None:
    window = stored
    dialog = window.show_candidate_parameters("a")
    assert isinstance(dialog, CandidateParametersDialog)
    assert dialog.isVisible()
    assert "a" in dialog.windowTitle()
    dialog.close()
    assert window.show_candidate_parameters("missing") is None
    window.controller.undo_stack.setClean()
