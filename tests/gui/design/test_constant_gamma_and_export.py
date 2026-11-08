"""Tests of the constant-Γ option of the coating page and of the reflectivity export."""

import csv
from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from pytestqt.qtbot import QtBot

from grazing.core.io.reflectivity import REFLECTIVITY_COLUMNS
from grazing.core.structure.coatings.graded import DepthGraded
from grazing.core.structure.conversion import has_constant_gamma
from grazing.core.targets import Target
from grazing.design.project import ProjectState
from grazing.design.project.commands import set_parameter
from grazing.design.project.templates import new_state
from grazing.gui.design.main_window import INPUT_DESIGN_LABEL, MainWindow
from grazing.gui.design.preview import display_targets, lowest_energy
from grazing.gui.design.steps.coating import CoatingPage, set_constant_gamma

pytestmark = pytest.mark.gui


def graded(state: ProjectState) -> DepthGraded:
    assert state.mirror is not None
    assert isinstance(state.mirror.coating, DepthGraded)
    return state.mirror.coating


def tied_state() -> ProjectState:
    return set_parameter("gamma_bottom", expr="gamma_top", vary=False).transform(
        new_state("depth_graded")
    )


def test_set_constant_gamma_untie_copies_top_settings() -> None:
    state = tied_state()
    top = graded(state).parameters["gamma_top"]
    new = graded(set_constant_gamma(graded(state), False).transform(state))
    bottom = new.parameters["gamma_bottom"]
    assert bottom.expr is None
    assert bottom.value == pytest.approx(top.value)
    assert bottom.vary == top.vary
    assert bottom.min == top.min
    assert bottom.max == top.max
    assert not has_constant_gamma(new)


def test_set_constant_gamma_tie() -> None:
    state = new_state("depth_graded")
    new = graded(set_constant_gamma(graded(state), True).transform(state))
    assert new.parameters["gamma_bottom"].expr == "gamma_top"
    assert has_constant_gamma(new)


def test_checkbox_reflects_state(window: MainWindow) -> None:
    page = window.coating_page
    window.new_project("depth_graded")
    assert page.constant_gamma_check.isEnabled()
    assert not page.constant_gamma_check.isChecked()
    window.controller.execute(set_constant_gamma(graded(window.controller.state), True))
    assert page.constant_gamma_check.isChecked()


def test_checkbox_toggle_and_undo(window: MainWindow) -> None:
    window.new_project("depth_graded")
    page = window.coating_page
    window.controller.execute(set_constant_gamma(graded(window.controller.state), True))
    assert page.constant_gamma_check.isChecked()
    # A different parameter edit keeps the following toggle a separate undo step.
    window.controller.execute(set_parameter("c", value=0.3))

    page.constant_gamma_check.setChecked(False)
    coating = graded(window.controller.state)
    assert coating.parameters["gamma_bottom"].expr is None
    assert not has_constant_gamma(coating)
    assert not page.constant_gamma_check.isChecked()

    window.controller.undo()
    assert has_constant_gamma(graded(window.controller.state))
    assert page.constant_gamma_check.isChecked()


def test_checkbox_toggle_on_ties(window: MainWindow) -> None:
    window.new_project("depth_graded")
    page = window.coating_page
    page.constant_gamma_check.setChecked(True)
    assert has_constant_gamma(graded(window.controller.state))
    window.controller.undo()
    assert not has_constant_gamma(graded(window.controller.state))
    assert not page.constant_gamma_check.isChecked()


def test_checkbox_disabled_for_periodic(qtbot: QtBot) -> None:
    page = CoatingPage()
    qtbot.addWidget(page)
    page.set_state(new_state("periodic"))
    assert not page.constant_gamma_check.isEnabled()
    page.set_state(new_state("depth_graded"))
    assert page.constant_gamma_check.isEnabled()
    page.set_state(new_state("periodic"))
    assert not page.constant_gamma_check.isEnabled()


def test_checkbox_after_loading_project(window: MainWindow, tmp_path: Path) -> None:
    window.new_project("depth_graded")
    window.controller.execute(set_constant_gamma(graded(window.controller.state), True))
    path = tmp_path / "tied.grazing"
    assert window.save_project_as(path)

    window.new_project("depth_graded")
    assert not window.coating_page.constant_gamma_check.isChecked()
    assert window.open_project(path)
    assert window.coating_page.constant_gamma_check.isChecked()
    assert window.coating_page.constant_gamma_check.isEnabled()

    window.new_project("periodic")
    assert not window.coating_page.constant_gamma_check.isEnabled()


def shown_targets(window: MainWindow) -> list[Target]:
    state = window.controller.state
    assert state.mirror is not None
    assert state.targets is not None
    floor = lowest_energy(frozenset(state.mirror.stack().media))
    return list(display_targets(state.targets.targets, window.display_ranges, floor))


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        assert tuple(reader.fieldnames or ()) == REFLECTIVITY_COLUMNS
        return list(reader)


def set_range(window: MainWindow, axis: str, low: str, high: str) -> None:
    low_edit, high_edit = window.range_edits[axis]
    low_edit.setText(low)
    high_edit.setText(high)
    high_edit.editingFinished.emit()


def test_input_row_is_first_and_current(window: MainWindow) -> None:
    window.store_candidate("a")
    item = window.candidate_list.item(0)
    assert item.text() == INPUT_DESIGN_LABEL
    assert item.data(Qt.ItemDataRole.UserRole) is None
    assert not item.flags() & Qt.ItemFlag.ItemIsUserCheckable
    assert not item.flags() & Qt.ItemFlag.ItemIsEditable
    assert window.candidate_list.item(1).text() == "a"
    assert window.controller.state.selected is None
    assert window.candidate_list.currentRow() == 0
    assert window.selected_candidates() == []
    window.controller.undo_stack.setClean()


def test_clicking_input_row_deselects_candidate(window: MainWindow) -> None:
    window.store_candidate("a")
    window.candidate_list.setCurrentRow(1)
    assert window.selected_candidates() == ["a"]
    window.candidate_list.setCurrentRow(0)
    assert window.controller.state.selected is None
    assert window.candidate_list.currentRow() == 0
    window.controller.undo_stack.setClean()


def test_export_input_design(window: MainWindow, tmp_path: Path) -> None:
    path = tmp_path / "input.csv"
    assert window.export_reflectivity(None, path)
    rows = read_rows(path)
    assert len(rows) == sum(t.points for t in shown_targets(window))
    assert "Exported" in window.statusBar().currentMessage()
    window.controller.undo_stack.setClean()


def test_export_candidate(window: MainWindow, tmp_path: Path) -> None:
    window.store_candidate("name")
    path = tmp_path / "candidate.csv"
    assert window.export_reflectivity("name", path)
    rows = read_rows(path)
    assert len(rows) == sum(t.points for t in shown_targets(window))
    window.controller.undo_stack.setClean()


def test_export_follows_display_range(window: MainWindow, tmp_path: Path) -> None:
    set_range(window, "energy", "3", "9")
    path = tmp_path / "ranged.csv"
    assert window.export_reflectivity(None, path)
    rows = [r for r in read_rows(path) if r["axis"] == "energy"]
    assert rows
    energies = [float(r["energy_keV"]) for r in rows]
    assert min(energies) == pytest.approx(3.0)
    assert max(energies) == pytest.approx(9.0)
    window.controller.undo_stack.setClean()


def test_export_failures_return_false(window: MainWindow, tmp_path: Path) -> None:
    assert not window.export_reflectivity("missing", tmp_path / "a.csv")
    assert not (tmp_path / "a.csv").exists()
    assert not window.export_reflectivity(None, tmp_path / "no" / "such" / "dir.csv")
    assert "Cannot export" in window.statusBar().currentMessage()
    window.controller.undo_stack.setClean()


def test_menu_export_action_states(window: MainWindow) -> None:
    for name in ("a", "b"):
        window.store_candidate(name)
    text = "Export reflectivity…"

    window.candidate_list.setCurrentRow(0)
    menu = window.candidate_menu()
    assert {a.text(): a.isEnabled() for a in menu.actions()}[text]
    menu.deleteLater()

    window.candidate_list.setCurrentRow(1)
    window.candidate_list.item(1).setSelected(True)
    menu = window.candidate_menu()
    assert {a.text(): a.isEnabled() for a in menu.actions()}[text]
    menu.deleteLater()

    window.candidate_list.item(2).setSelected(True)
    menu = window.candidate_menu()
    assert not {a.text(): a.isEnabled() for a in menu.actions()}[text]
    menu.deleteLater()
    window.controller.undo_stack.setClean()
