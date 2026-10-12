"""Tests for `xstack.gui.steps.GoalPage`."""

import math

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox, QLineEdit, QSpinBox
from pytestqt.qtbot import QtBot

from xstack.gui.steps import GoalPage
from xstack.project import Command, ProjectState
from xstack.project.templates import new_state
from xstack.targets import AngleScan, EnergyScan, TargetSet

pytestmark = pytest.mark.gui

DEG = math.pi / 180.0


class Recorder:
    """Collects the commands and errors emitted by the page."""

    def __init__(self, page: GoalPage) -> None:
        self.commands: list[Command] = []
        self.errors: list[str] = []
        page.commandRequested.connect(self.commands.append)
        page.errorOccurred.connect(self.errors.append)

    def applied(self, state: ProjectState) -> ProjectState:
        assert len(self.commands) == 1
        return self.commands[-1].transform(state)


def make_page(qtbot: QtBot, state: ProjectState) -> tuple[GoalPage, Recorder]:
    page = GoalPage()
    qtbot.addWidget(page)
    page.set_state(state)
    return page, Recorder(page)


def combo_at(page: GoalPage, row: int, column: int) -> QComboBox:
    widget = page.table.cellWidget(row, column)
    assert isinstance(widget, QComboBox)
    return widget


def edit_at(page: GoalPage, row: int, column: int) -> QLineEdit:
    widget = page.table.cellWidget(row, column)
    assert isinstance(widget, QLineEdit)
    return widget


def finish_edit(edit: QLineEdit, text: str) -> None:
    edit.setText(text)
    edit.editingFinished.emit()


def choose(combo: QComboBox, text: str) -> None:
    combo.setCurrentText(text)
    combo.activated.emit(combo.currentIndex())


def energy_target() -> EnergyScan:
    return EnergyScan(
        angle=1.1 * DEG, energy_min=7900.0, energy_max=8200.0, resolution=2.5, name="mine"
    )


def test_help_text_present() -> None:
    assert len(GoalPage.help_text) > 100
    assert "reflection" in GoalPage.help_text


def test_shows_values_and_scan_names(qtbot: QtBot) -> None:
    page, _ = make_page(qtbot, new_state("periodic"))
    assert page.table.rowCount() == 1
    assert combo_at(page, 0, 1).currentText() == "Fixed angle"
    assert [edit_at(page, 0, c).text() for c in (2, 3, 4)] == ["1.1", "7.9", "8.2"]
    angle, _ = make_page(qtbot, new_state("single_layer"))
    assert combo_at(angle, 0, 1).currentText() == "Fixed energy"
    assert edit_at(angle, 0, 2).text() == "8.048"


def test_no_units_column_and_compact_style(qtbot: QtBot) -> None:
    page, _ = make_page(qtbot, new_state("periodic"))
    headers = [page.table.horizontalHeaderItem(c).text() for c in range(page.table.columnCount())]
    assert "Units" not in headers
    assert "Refl." in headers
    assert not page.table.showGrid()
    assert not page.table.verticalHeader().isVisible()
    assert not page.table.alternatingRowColors()
    assert page.add_energy.text() == "Add fixed-angle target"
    assert page.add_angle.text() == "Add fixed-energy target"


def test_placeholders_per_scan_type(qtbot: QtBot) -> None:
    page, _ = make_page(qtbot, new_state("periodic"))
    assert [edit_at(page, 0, c).placeholderText() for c in (2, 3, 4)] == [
        "angle (deg)",
        "min energy (keV)",
        "max energy (keV)",
    ]
    assert edit_at(page, 0, 5).placeholderText() == "points"
    assert edit_at(page, 0, 6).placeholderText() == "weight"
    angle, _ = make_page(qtbot, new_state("single_layer"))
    assert [edit_at(angle, 0, c).placeholderText() for c in (2, 3, 4)] == [
        "energy (keV)",
        "min angle (deg)",
        "max angle (deg)",
    ]


def test_placeholders_follow_scan_type_change(qtbot: QtBot) -> None:
    page, _ = make_page(qtbot, new_state("periodic"))
    choose(combo_at(page, 0, 1), "Fixed energy")
    assert edit_at(page, 0, 2).placeholderText() == "energy (keV)"


def test_tooltips_present(qtbot: QtBot) -> None:
    page, _ = make_page(qtbot, new_state("periodic"))
    for column in range(page.table.columnCount()):
        assert page.table.cellWidget(0, column).toolTip()
    for widget in (page.add_energy, page.add_angle, page.remove, page.combine):
        assert widget.toolTip()
    assert "averaged" in page.combine.itemData(0, 3)
    assert "separately" in page.combine.itemData(1, 3)


def test_edit_to_converts_and_preserves(qtbot: QtBot) -> None:
    state = new_state("periodic").evolve(targets=TargetSet(energy_target()))
    page, rec = make_page(qtbot, state)
    finish_edit(edit_at(page, 0, 4), "8.5")
    new = rec.applied(state).targets
    assert new is not None
    target = new.targets[0]
    assert isinstance(target, EnergyScan)
    assert target.energy_max == pytest.approx(8500.0)
    assert target.energy_min == pytest.approx(7900.0)
    assert target.angle == pytest.approx(1.1 * DEG)
    assert target.resolution == 2.5
    assert target.name == "mine"


def test_unchanged_edit_emits_nothing(qtbot: QtBot) -> None:
    state = new_state("periodic").evolve(targets=TargetSet(energy_target()))
    page, rec = make_page(qtbot, state)
    for column in (0, 2, 3, 4, 5, 6):
        edit_at(page, 0, column).editingFinished.emit()
    assert not rec.commands


def test_reflections_column(qtbot: QtBot) -> None:
    state = new_state("periodic").evolve(targets=TargetSet(energy_target()))
    page, rec = make_page(qtbot, state)
    spin = page.table.cellWidget(0, 8)
    assert isinstance(spin, QSpinBox)
    assert (spin.minimum(), spin.maximum(), spin.value()) == (1, 4, 1)
    assert "R²" in spin.toolTip()
    spin.setValue(2)
    new = rec.applied(state).targets
    assert new is not None
    assert new.targets[0].reflections == 2
    assert new.targets[0].name == "mine"


def test_scan_type_to_angle(qtbot: QtBot) -> None:
    state = new_state("periodic")
    page, rec = make_page(qtbot, state)
    choose(combo_at(page, 0, 1), "Fixed energy")
    targets = rec.applied(state).targets
    assert targets is not None
    target = targets.targets[0]
    assert isinstance(target, AngleScan)
    assert target.energy == pytest.approx(1100.0)
    assert target.angle_max == pytest.approx(8.2 * DEG)


def test_invalid_text(qtbot: QtBot) -> None:
    page, rec = make_page(qtbot, new_state("periodic"))
    finish_edit(edit_at(page, 0, 4), "abc")
    assert not rec.commands
    assert len(rec.errors) == 1
    assert edit_at(page, 0, 4).text() == "8.2"


def test_add_scans_and_remove(qtbot: QtBot) -> None:
    state = new_state("periodic")
    page, rec = make_page(qtbot, state)
    page.add_energy.click()
    added = rec.applied(state)
    assert added.targets is not None
    assert len(added.targets) == 2
    assert isinstance(added.targets.targets[1], EnergyScan)
    assert added.targets.combine == "average_merit"

    rec.commands.clear()
    page.add_angle.click()
    angle_state = rec.applied(state)
    assert angle_state.targets is not None
    assert isinstance(angle_state.targets.targets[1], AngleScan)

    page.set_state(added)
    rec.commands.clear()
    page.table.setCurrentCell(1, 2)
    page.remove.click()
    removed = rec.applied(added).targets
    assert removed is not None
    assert state.targets is not None
    assert removed.targets == (state.targets.targets[0],)


def test_table_height_follows_rows(qtbot: QtBot) -> None:
    state = new_state("periodic")
    page, rec = make_page(qtbot, state)
    one = page.table.height()
    page.add_energy.click()
    page.set_state(rec.applied(state))
    assert page.table.height() > one


def test_remove_last_target_refused(qtbot: QtBot) -> None:
    page, rec = make_page(qtbot, new_state("periodic"))
    page.remove.click()
    assert not rec.commands
    assert len(rec.errors) == 1


def test_combine_combo_and_explanation(qtbot: QtBot) -> None:
    state = new_state("periodic")
    page, rec = make_page(qtbot, state)
    assert page.combine.currentData() == "average_reflectivity"
    first = page.combine_info.text()
    assert first.startswith("Average reflectivity")
    page.combine.setCurrentIndex(1)
    page.combine.activated.emit(1)
    assert page.combine_info.text().startswith("Average figure of merit")
    assert page.combine_info.text() != first
    targets = rec.applied(state).targets
    assert targets is not None
    assert targets.combine == "average_merit"


def test_table_fits_without_horizontal_scrolling(qtbot: QtBot) -> None:
    from xstack.gui.main_window import MainWindow

    window = MainWindow(confirm_close=False)
    qtbot.addWidget(window)
    window.resize(1400, 900)
    window.show()
    window.step_tabs.setCurrentIndex(0)
    window.goal_page.add_angle.click()
    qtbot.wait(50)
    table = window.goal_page.table
    total = sum(table.columnWidth(c) for c in range(table.columnCount()))
    assert total <= table.viewport().width()
    assert table.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    window.controller.undo_stack.setClean()


def test_name_column_shows_name_and_placeholder(qtbot: QtBot) -> None:
    state = new_state("periodic").evolve(
        targets=TargetSet(
            (energy_target(), EnergyScan(angle=0.01, energy_min=1e3, energy_max=2e3)),
            combine="average_merit",
        )
    )
    page, _ = make_page(qtbot, state)
    assert page.table.horizontalHeaderItem(0).text() == "Name"
    assert edit_at(page, 0, 0).text() == "mine"
    assert edit_at(page, 0, 0).placeholderText() == "mine"
    assert edit_at(page, 1, 0).text() == ""
    assert edit_at(page, 1, 0).placeholderText() == "target 2"


def test_name_edit_sets_name_and_preserves_settings(qtbot: QtBot) -> None:
    state = new_state("periodic").evolve(targets=TargetSet(energy_target()))
    page, rec = make_page(qtbot, state)
    finish_edit(edit_at(page, 0, 0), "  Cu K  ")
    new = rec.applied(state).targets
    assert new is not None
    target = new.targets[0]
    assert isinstance(target, EnergyScan)
    assert target.name == "Cu K"
    assert target.resolution == 2.5
    assert target.energy_min == pytest.approx(7900.0)
    assert target.energy_max == pytest.approx(8200.0)
    assert target.angle == pytest.approx(1.1 * DEG)


def test_clearing_name(qtbot: QtBot) -> None:
    state = new_state("periodic").evolve(targets=TargetSet(energy_target()))
    page, rec = make_page(qtbot, state)
    finish_edit(edit_at(page, 0, 0), "")
    new = rec.applied(state).targets
    assert new is not None
    assert new.targets[0].name == ""
