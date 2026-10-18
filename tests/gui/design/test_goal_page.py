"""Tests for `xstack.gui.design.steps.GoalPage`."""

import math
from typing import TYPE_CHECKING

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QComboBox, QLineEdit
from pytestqt.qtbot import QtBot

from xstack.core.targets import AngleScan, EnergyScan, TargetSet
from xstack.design.project import Command, ProjectState
from xstack.design.project.templates import new_state
from xstack.gui.design.steps import GoalPage
from xstack.gui.design.steps.goal import TargetEditor

if TYPE_CHECKING:
    from collections.abc import Iterator

    from xstack.gui.design.main_window import MainWindow

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


def row_at(page: GoalPage, index: int) -> TargetEditor:
    return page.rows[index]


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
    assert len(page.rows) == 1
    row = row_at(page, 0)
    assert row.scan.currentText() == "Fixed angle"
    assert [e.text() for e in (row.fixed, row.lo, row.hi)] == ["1.1", "7.9", "8.2"]
    angle, _ = make_page(qtbot, new_state("single_layer"))
    assert row_at(angle, 0).scan.currentText() == "Fixed energy"
    assert row_at(angle, 0).fixed.text() == "8.048"


def test_labels_and_units_per_scan_type(qtbot: QtBot) -> None:
    page, _ = make_page(qtbot, new_state("periodic"))
    row = row_at(page, 0)
    assert (row.fixed_label.text(), row.fixed_unit.text()) == ("Angle", "deg")
    assert (row.range_label.text(), row.range_unit.text()) == ("Energy from", "keV")
    assert [e.placeholderText() for e in (row.fixed, row.lo, row.hi)] == [
        "angle (deg)",
        "min energy (keV)",
        "max energy (keV)",
    ]
    assert row.points.placeholderText() == "points"
    assert row.weight.placeholderText() == "weight"
    angle, _ = make_page(qtbot, new_state("single_layer"))
    other = row_at(angle, 0)
    assert (other.fixed_label.text(), other.fixed_unit.text()) == ("Energy", "keV")
    assert (other.range_label.text(), other.range_unit.text()) == ("Angle from", "deg")


def test_labels_follow_scan_type_change(qtbot: QtBot) -> None:
    page, _ = make_page(qtbot, new_state("periodic"))
    row = row_at(page, 0)
    choose(row.scan, "Fixed energy")
    assert row.fixed_label.text() == "Energy"
    assert row.fixed.placeholderText() == "energy (keV)"


def test_tooltips_present(qtbot: QtBot) -> None:
    page, _ = make_page(qtbot, new_state("periodic"))
    for widget in row_at(page, 0).fields():
        assert widget.toolTip()
    for widget in (page.add_energy, page.add_angle, page.combine):
        assert widget.toolTip()
    assert "averaged" in page.combine.itemData(0, 3)
    assert "separately" in page.combine.itemData(1, 3)


def test_edit_to_converts_and_preserves(qtbot: QtBot) -> None:
    state = new_state("periodic").evolve(targets=TargetSet(energy_target()))
    page, rec = make_page(qtbot, state)
    finish_edit(row_at(page, 0).hi, "8.5")
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
    row = row_at(page, 0)
    for edit in (row.name, row.fixed, row.lo, row.hi, row.points, row.weight):
        edit.editingFinished.emit()
    assert not rec.commands


def test_reflections_field(qtbot: QtBot) -> None:
    state = new_state("periodic").evolve(targets=TargetSet(energy_target()))
    page, rec = make_page(qtbot, state)
    spin = row_at(page, 0).reflections
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
    choose(row_at(page, 0).scan, "Fixed energy")
    targets = rec.applied(state).targets
    assert targets is not None
    target = targets.targets[0]
    assert isinstance(target, AngleScan)
    assert target.energy == pytest.approx(1100.0)
    assert target.angle_max == pytest.approx(8.2 * DEG)


def test_invalid_text(qtbot: QtBot) -> None:
    page, rec = make_page(qtbot, new_state("periodic"))
    finish_edit(row_at(page, 0).hi, "abc")
    assert not rec.commands
    assert len(rec.errors) == 1
    assert row_at(page, 0).hi.text() == "8.2"


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
    assert len(page.rows) == 2
    rec.commands.clear()
    row_at(page, 1).remove.click()
    removed = rec.applied(added).targets
    assert removed is not None
    assert state.targets is not None
    assert removed.targets == (state.targets.targets[0],)


def test_remove_first_target(qtbot: QtBot) -> None:
    second = EnergyScan(angle=0.01, energy_min=1e3, energy_max=2e3)
    state = new_state("periodic").evolve(
        targets=TargetSet((energy_target(), second), combine="average_merit")
    )
    page, rec = make_page(qtbot, state)
    row_at(page, 0).remove.click()
    removed = rec.applied(state).targets
    assert removed is not None
    assert removed.targets == (second,)
    page.set_state(state.evolve(targets=removed))
    assert len(page.rows) == 1
    assert row_at(page, 0).lo.text() == "1"


def test_remove_disabled_for_last_target(qtbot: QtBot) -> None:
    page, rec = make_page(qtbot, new_state("periodic"))
    assert not row_at(page, 0).remove.isEnabled()
    row_at(page, 0).removeRequested.emit()
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


def test_name_field_shows_name_and_placeholder(qtbot: QtBot) -> None:
    state = new_state("periodic").evolve(
        targets=TargetSet(
            (energy_target(), EnergyScan(angle=0.01, energy_min=1e3, energy_max=2e3)),
            combine="average_merit",
        )
    )
    page, _ = make_page(qtbot, state)
    assert row_at(page, 0).name.text() == "mine"
    assert row_at(page, 0).name.placeholderText() == "mine"
    assert row_at(page, 1).name.text() == ""
    assert row_at(page, 1).name.placeholderText() == "target 2"


def test_name_edit_sets_name_and_preserves_settings(qtbot: QtBot) -> None:
    state = new_state("periodic").evolve(targets=TargetSet(energy_target()))
    page, rec = make_page(qtbot, state)
    finish_edit(row_at(page, 0).name, "  Cu K  ")
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
    finish_edit(row_at(page, 0).name, "")
    new = rec.applied(state).targets
    assert new is not None
    assert new.targets[0].name == ""


def test_new_state_updates_fields_in_place(qtbot: QtBot) -> None:
    state = new_state("periodic").evolve(targets=TargetSet(energy_target()))
    page, rec = make_page(qtbot, state)
    row = row_at(page, 0)
    fields = row.fields()
    finish_edit(row.hi, "8.5")
    page.set_state(rec.applied(state))
    assert page.rows == [row]
    assert row.fields() == fields
    assert row.hi.text() == "8.5"


@pytest.fixture
def window(qtbot: QtBot) -> "Iterator[MainWindow]":
    from xstack.gui.design.main_window import MainWindow

    window = MainWindow(confirm_close=False)
    qtbot.addWidget(window)
    window.resize(1400, 900)
    window.show()
    window.step_tabs.setCurrentIndex(0)
    yield window
    window.controller.undo_stack.setClean()


def test_tab_after_edit_moves_to_next_field(qtbot: QtBot, window: "MainWindow") -> None:
    page = window.goal_page
    page.add_angle.click()
    row = row_at(page, 0)
    row.hi.setFocus()
    qtbot.waitUntil(row.hi.hasFocus)
    row.hi.selectAll()
    QTest.keyClicks(row.hi, "9")
    QTest.keyClick(row.hi, Qt.Key.Key_Tab)
    targets = window.controller.state.targets
    assert targets is not None
    assert targets.targets[0].scan_range[1] == pytest.approx(9000.0)
    assert QApplication.focusWidget() is row.points
    QTest.keyClick(row.points, Qt.Key.Key_Tab)
    assert QApplication.focusWidget() is row.weight


def test_tab_order_runs_through_targets(qtbot: QtBot, window: "MainWindow") -> None:
    page = window.goal_page
    page.add_angle.click()
    first, second = page.rows
    assert first.remove.nextInFocusChain() is second.name
    assert second.remove.nextInFocusChain() is page.add_energy


def test_rows_fit_page_width(qtbot: QtBot, window: "MainWindow") -> None:
    page = window.goal_page
    page.add_angle.click()
    qtbot.wait(50)
    for row in page.rows:
        assert row.minimumSizeHint().width() <= page.width()
        assert row.remove.geometry().right() <= row.width()
