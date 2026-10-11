"""Tests for `xstack.gui.main_window` and `xstack.gui.app`."""

from pathlib import Path

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QDockWidget

from xstack.gui.main_window import MainWindow
from xstack.project import Candidate
from xstack.project.commands import add_candidate, set_parameter

pytestmark = pytest.mark.gui


def n_value(window: MainWindow) -> float:
    assert window.controller.state.mirror is not None
    return window.controller.state.mirror.coating.parameters["N"].value


def edit_n(window: MainWindow, text: str) -> None:
    model = window.parameter_table.table_model
    names = list(model.parameters())
    assert model.setData(model.index(names.index("N"), 1), text)


def test_title_unsaved(window: MainWindow) -> None:
    title = window.windowTitle()
    assert window.controller.state.name in title
    assert "unsaved" in title
    assert not window.isWindowModified()


def test_modified_and_save(window: MainWindow, tmp_path: Path) -> None:
    edit_n(window, "55")
    assert window.isWindowModified()
    path = tmp_path / "design.xstack"
    assert window.save_project_as(path)
    assert not window.isWindowModified()
    assert "design.xstack" in window.windowTitle()
    assert window.save_project()


def test_table_edit_updates_state_and_history(window: MainWindow) -> None:
    edit_n(window, "55")
    assert n_value(window) == 55
    assert window.controller.undo_stack.count() == 1
    assert window.undo_view.model().rowCount() == 2  # empty label + one step
    model = window.parameter_table.table_model
    assert model.index(list(model.parameters()).index("N"), 1).data() == "55"


def test_undo_redo_actions(window: MainWindow) -> None:
    edit_n(window, "55")
    assert window.undo_action.isEnabled()
    window.undo_action.trigger()
    assert n_value(window) == 40
    assert window.redo_action.isEnabled()
    window.redo_action.trigger()
    assert n_value(window) == 55
    model = window.parameter_table.table_model
    assert model.index(list(model.parameters()).index("N"), 1).data() == "55"


def test_new_depth_graded(window: MainWindow) -> None:
    window.new_project("depth_graded")
    assert "DepthGraded" in window.design_label.text()
    assert window.parameter_table.table_model.rowCount() > 3
    assert "unsaved" in window.windowTitle()


def test_open_adds_recent(window: MainWindow, tmp_path: Path, settings: QSettings) -> None:
    path = tmp_path / "a.xstack"
    edit_n(window, "55")
    window.save_project_as(path)
    window.new_project("single_layer")
    assert window.open_project(path)
    assert n_value(window) == 55
    assert str(path) in settings.value("recent_files")
    assert [a.text() for a in window.recent_menu.actions()] == [str(path)]
    assert window.recent_menu.isEnabled()


def test_open_invalid_file(window: MainWindow, tmp_path: Path, settings: QSettings) -> None:
    bad = tmp_path / "bad.xstack"
    bad.write_text("garbage")
    assert not window.open_project(bad)
    assert not settings.value("recent_files")


def test_recent_entry_opens_file(window: MainWindow, tmp_path: Path) -> None:
    path = tmp_path / "a.xstack"
    edit_n(window, "55")
    window.save_project_as(path)
    window.new_project("periodic")
    assert n_value(window) == 40
    window.recent_menu.actions()[0].trigger()
    assert n_value(window) == 55


def test_candidate_list(window: MainWindow) -> None:
    assert window.candidate_list.count() == 0
    mirror = window.controller.state.mirror
    assert mirror is not None
    window.controller.execute(add_candidate(Candidate("first", mirror)))
    assert window.candidate_list.count() == 1
    assert window.candidate_list.item(0).text() == "first"


def test_journal_grows(window: MainWindow) -> None:
    before = window.journal_view.count()
    window.controller.execute(set_parameter("N", value=60))
    assert window.journal_view.count() == before + 1
    window.controller.undo()
    assert window.journal_view.count() == before + 2


def test_error_shown_in_status_bar(window: MainWindow) -> None:
    window.controller.execute(set_parameter("nope", value=1))
    assert "nope" in window.statusBar().currentMessage()


def test_close_event_stores_layout(window: MainWindow, settings: QSettings) -> None:
    assert settings.value("main_window/geometry") is None
    window.show()
    assert window.close()
    assert settings.value("main_window/geometry") is not None
    assert settings.value("main_window/state") is not None


def test_close_restores_layout(window: MainWindow, settings: QSettings) -> None:
    window.resize(900, 700)
    window.close()
    second = MainWindow(settings=settings, confirm_close=False)
    try:
        assert second.restoreGeometry(settings.value("main_window/geometry"))
    finally:
        second.close()


def test_docks_and_view_menu(window: MainWindow) -> None:
    docks = [
        window.workflow_dock,
        window.properties_dock,
        window.journal_dock,
        window.history_dock,
    ]
    assert all(isinstance(d, QDockWidget) for d in docks)
    view = next(a.menu() for a in window.menuBar().actions() if "View" in a.text())
    toggles = {a.text() for a in view.actions()}
    assert {d.windowTitle() for d in docks} <= toggles
    window.show()
    dock = window.properties_dock
    assert dock.isVisible()
    dock.toggleViewAction().trigger()
    assert not dock.isVisible()
    dock.toggleViewAction().trigger()
    assert dock.isVisible()


def test_workflow_list(window: MainWindow) -> None:
    assert window.workflow_list.count() == 8


def test_new_actions_per_template(window: MainWindow) -> None:
    window.new_actions["single_layer"].trigger()
    assert "SingleLayer" in window.design_label.text()


def test_app_main_importable() -> None:
    from xstack.gui import app

    assert callable(app.main)
