"""Tests for `xstack.gui.fit.steps.data`."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PySide6.QtWidgets import QDialog
from pytestqt.qtbot import QtBot

from xstack.core.io import files
from xstack.core.measurement import MeasuredCurve
from xstack.core.units import deg, kev
from xstack.fit.dataset import Dataset
from xstack.fit.project import Command, FitProject, FitState
from xstack.gui.fit.steps.data import DataPage
from xstack.gui.fit.widgets import ImportDialog


@pytest.fixture(autouse=True)
def no_plugins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(files, "load_plugins", lambda: ())


def angle_dataset(name: str = "cu_ka", **kwargs: Any) -> Dataset:
    x = deg(np.linspace(0.1, 1.0, 10))
    curve = MeasuredCurve(
        "angle",
        x,
        np.linspace(1.0, 0.1, 10),
        np.full(10, 0.01),
        energy=kev(8.048),
        source="cu.csv",
    )
    return Dataset.create(name, curve).with_ranges(
        kwargs.get("ranges", ()), kwargs.get("excluded", ())
    )


def energy_dataset(name: str = "scan") -> Dataset:
    x = kev(np.linspace(5.0, 14.0, 10))
    curve = MeasuredCurve("energy", x, np.linspace(1.0, 0.1, 10), angle=deg(1.0))
    return Dataset.create(name, curve)


class Harness:
    """A page connected to a project: commands are applied and the state fed back."""

    def __init__(self, page: DataPage, state: FitState) -> None:
        self.page = page
        self.project = FitProject(state)
        self.commands: list[Command] = []
        self.errors: list[str] = []
        page.commandRequested.connect(self.commands.append)
        page.errorOccurred.connect(self.errors.append)
        page.set_state(state)

    def apply(self) -> FitState:
        for command in self.commands:
            self.project.execute(command)
        self.commands.clear()
        self.page.set_state(self.project.state)
        return self.project.state


@pytest.fixture
def harness(qtbot: QtBot) -> Harness:
    page = DataPage()
    qtbot.addWidget(page)
    state = FitState(datasets=(angle_dataset(), energy_dataset()))
    h = Harness(page, state)
    page.set_selected("cu_ka")
    return h


def cell(page: DataPage, row: int, column: int) -> str:
    item = page.table.item(row, column)
    assert item is not None
    return str(item.text())


def edit_cell(page: DataPage, row: int, column: int, text: str) -> None:
    item = page.table.item(row, column)
    assert item is not None
    item.setText(text)


def test_no_selection_shows_hint(qtbot: QtBot) -> None:
    page = DataPage()
    qtbot.addWidget(page)
    page.set_state(FitState(datasets=(angle_dataset(),)))
    assert not page.hint.isHidden()
    assert page.dataset_group.isHidden()
    assert page.ranges_group.isHidden()
    assert page.remove_button.isHidden()
    assert not page.add_button.isHidden()
    assert page.title == "Data"
    assert "Shift-drag" in page.help_text


def test_unknown_selection_shows_hint(harness: Harness) -> None:
    harness.page.set_selected("missing")
    assert not harness.page.hint.isHidden()
    assert harness.page.dataset_group.isHidden()


def test_angle_dataset_shown(harness: Harness) -> None:
    page = harness.page
    assert not page.dataset_group.isHidden()
    assert page.hint.isHidden()
    assert page.dataset_group.title() == "Dataset “cu_ka”"
    assert page.name_edit.text() == "cu_ka"
    assert page.enabled_check.isChecked()
    assert page.fixed_label.text() == "Photon energy"
    assert page.fixed_spin.value() == pytest.approx(8.048)
    assert page.fixed_spin.suffix() == " keV"
    assert "cu.csv" in page.summary_label.text()
    assert "10 points" in page.summary_label.text()
    assert "uncertainties: yes" in page.summary_label.text()
    assert "<b>10 of 10</b>" in page.count_label.text()
    assert not harness.commands


def test_energy_dataset_shown(harness: Harness) -> None:
    page = harness.page
    page.set_selected("scan")
    assert page.fixed_label.text() == "Grazing angle"
    assert page.fixed_spin.value() == pytest.approx(1.0)
    assert page.fixed_spin.suffix() == " deg"
    assert "energy scan" in page.summary_label.text()
    assert "uncertainties: no" in page.summary_label.text()
    assert "From (keV)" in page.table.horizontalHeaderItem(1).text()
    assert not harness.commands


def test_set_energy(harness: Harness) -> None:
    harness.page.fixed_spin.setValue(17.479)
    state = harness.apply()
    assert state.dataset("cu_ka").curve.energy == pytest.approx(kev(17.479))
    assert state.dataset("scan") == energy_dataset()


def test_set_angle_of_energy_scan(harness: Harness) -> None:
    harness.page.set_selected("scan")
    harness.page.fixed_spin.setValue(2.0)
    state = harness.apply()
    assert state.dataset("scan").curve.angle == pytest.approx(deg(2.0))


def test_polarization(harness: Harness) -> None:
    harness.page.polarization_combo.setCurrentText("p")
    state = harness.apply()
    assert state.dataset("cu_ka").curve.polarization == "p"
    assert harness.page.polarization_combo.currentText() == "p"


def test_weight(harness: Harness) -> None:
    harness.page.weight_spin.setValue(2.5)
    state = harness.apply()
    assert state.dataset("cu_ka").weight == 2.5


def test_include_checkbox(harness: Harness) -> None:
    harness.page.enabled_check.setChecked(False)
    state = harness.apply()
    assert not state.dataset("cu_ka").enabled
    assert state.dataset("scan").enabled


def test_refresh_does_not_emit(harness: Harness) -> None:
    harness.page.set_state(harness.project.state.evolve(name="other"))
    harness.page.set_selected("scan")
    assert not harness.commands


def test_rename(harness: Harness) -> None:
    harness.page.name_edit.setText("cu_new")
    harness.page.name_edit.editingFinished.emit()
    state = harness.apply()
    assert [d.name for d in state.datasets] == ["cu_new", "scan"]
    assert state.datasets[0].curve == angle_dataset().curve
    assert not harness.errors


@pytest.mark.parametrize("text", ["1abc", "has space", "", "scan"])
def test_rename_rejected(harness: Harness, text: str) -> None:
    harness.page.name_edit.setText(text)
    harness.page.name_edit.editingFinished.emit()
    assert not harness.commands
    assert len(harness.errors) == 1
    assert harness.page.name_edit.text() == "cu_ka"


def test_rename_to_same_name_is_noop(harness: Harness) -> None:
    harness.page.name_edit.editingFinished.emit()
    assert not harness.commands
    assert not harness.errors


def test_range_table_in_display_units(qtbot: QtBot) -> None:
    page = DataPage()
    qtbot.addWidget(page)
    data = angle_dataset(ranges=[(deg(0.2), deg(0.5))], excluded=[(deg(0.3), deg(0.35))])
    page.set_state(FitState(datasets=(data,)))
    page.set_selected("cu_ka")
    assert page.table.rowCount() == 2
    assert [cell(page, 0, c) for c in range(4)] == ["fit", "0.2000", "0.5000", "4"]
    assert [cell(page, 1, c) for c in range(4)] == ["excluded", "0.3000", "0.3500", "1"]
    assert "<b>3 of 10</b>" in page.count_label.text()
    assert page.table.horizontalHeaderItem(1).text() == "From (deg)"


def test_energy_ranges_in_kev(harness: Harness) -> None:
    page = harness.page
    page.set_selected("scan")
    page.add_range_button.click()
    state = harness.apply()
    assert state.dataset("scan").ranges == (state.dataset("scan").curve.scan_range,)
    assert [cell(page, 0, c) for c in range(4)] == ["fit", "5.0000", "14.0000", "10"]
    edit_cell(page, 0, 1, "6.5")
    state = harness.apply()
    assert state.dataset("scan").ranges[0][0] == pytest.approx(kev(6.5))
    assert state.dataset("scan").ranges[0][1] == pytest.approx(kev(14.0))


def test_add_fit_range_full_then_gap(harness: Harness) -> None:
    page = harness.page
    page.add_range_button.click()
    state = harness.apply()
    lo, hi = state.dataset("cu_ka").curve.scan_range
    assert state.dataset("cu_ka").ranges == ((lo, hi),)
    harness.project.execute(
        Command(
            "shrink",
            lambda s: s.evolve(
                datasets=(s.datasets[0].with_ranges([(lo, deg(0.5))]), *s.datasets[1:])
            ),
        )
    )
    page.set_state(harness.project.state)
    page.add_range_button.click()
    state = harness.apply()
    assert state.dataset("cu_ka").ranges == ((lo, deg(0.5)), (deg(0.5), hi))


def test_add_excluded_range(harness: Harness) -> None:
    harness.page.add_excluded_button.click()
    state = harness.apply()
    data = state.dataset("cu_ka")
    lo, hi = data.curve.scan_range
    assert len(data.excluded) == 1
    ex_lo, ex_hi = data.excluded[0]
    assert lo < ex_lo < ex_hi < hi
    assert data.ranges == ()
    harness.page.add_excluded_button.click()
    state = harness.apply()
    assert len(state.dataset("cu_ka").excluded) == 2
    assert harness.page.table.rowCount() == 2


def test_edit_range_cells(qtbot: QtBot) -> None:
    page = DataPage()
    qtbot.addWidget(page)
    data = angle_dataset(ranges=[(deg(0.2), deg(0.5))], excluded=[(deg(0.3), deg(0.35))])
    h = Harness(page, FitState(datasets=(data,)))
    page.set_selected("cu_ka")
    edit_cell(page, 0, 2, "0.8")
    state = h.apply()
    assert state.dataset("cu_ka").ranges[0] == pytest.approx((deg(0.2), deg(0.8)))
    assert state.dataset("cu_ka").excluded == data.excluded
    edit_cell(page, 1, 1, "0.4")
    state = h.apply()
    assert h.errors  # lower bound above upper bound
    assert state.dataset("cu_ka").excluded == data.excluded
    assert cell(page, 1, 1) == "0.3000"


def test_edit_range_invalid_number(harness: Harness) -> None:
    harness.page.add_range_button.click()
    harness.apply()
    edit_cell(harness.page, 0, 1, "abc")
    assert not harness.commands
    assert harness.errors
    assert cell(harness.page, 0, 1) != "abc"


def test_remove_range(qtbot: QtBot) -> None:
    page = DataPage()
    qtbot.addWidget(page)
    data = angle_dataset(ranges=[(deg(0.2), deg(0.5))], excluded=[(deg(0.3), deg(0.35))])
    h = Harness(page, FitState(datasets=(data,)))
    page.set_selected("cu_ka")
    page.remove_range_button.click()
    assert not h.commands  # nothing selected
    page.table.selectRow(1)
    page.remove_range_button.click()
    state = h.apply()
    assert state.dataset("cu_ka").excluded == ()
    assert len(state.dataset("cu_ka").ranges) == 1
    page.table.selectRow(0)
    page.remove_range_button.click()
    state = h.apply()
    assert state.dataset("cu_ka").ranges == ()
    assert page.table.rowCount() == 0


def test_fit_all_points(qtbot: QtBot) -> None:
    page = DataPage()
    qtbot.addWidget(page)
    data = angle_dataset(ranges=[(deg(0.2), deg(0.5))], excluded=[(deg(0.3), deg(0.35))])
    h = Harness(page, FitState(datasets=(data,)))
    page.set_selected("cu_ka")
    assert page.fit_all_button.isEnabled()
    page.fit_all_button.click()
    state = h.apply()
    assert state.dataset("cu_ka").ranges == ()
    assert state.dataset("cu_ka").excluded == ()
    assert "<b>10 of 10</b>" in page.count_label.text()
    assert not page.fit_all_button.isEnabled()


def test_remove_dataset(harness: Harness) -> None:
    harness.page.remove_button.click()
    state = harness.apply()
    assert [d.name for d in state.datasets] == ["scan"]
    assert harness.page.dataset_group.isHidden()


class FakeDialog:
    def __init__(self, datasets: list[Dataset], result: QDialog.DialogCode) -> None:
        self._datasets = datasets
        self._result = result
        self.args: tuple[Any, ...] = ()

    def exec(self) -> QDialog.DialogCode:
        return self._result

    def datasets(self) -> list[Dataset]:
        return self._datasets


def test_add_data_emits_commands(harness: Harness) -> None:
    new = [angle_dataset("a"), angle_dataset("b")]
    used: list[tuple[str, ...]] = []

    def factory(names: tuple[str, ...], parent: Any) -> FakeDialog:
        used.append(names)
        return FakeDialog(new, QDialog.DialogCode.Accepted)

    harness.page.dialog_factory = factory
    harness.page.add_button.click()
    assert used == [("cu_ka", "scan")]
    state = harness.apply()
    assert [d.name for d in state.datasets] == ["cu_ka", "scan", "a", "b"]


def test_open_import_dialog_returns_datasets(harness: Harness) -> None:
    new = [angle_dataset("a")]
    harness.page.dialog_factory = lambda names, parent: FakeDialog(new, QDialog.DialogCode.Accepted)
    assert harness.page.open_import_dialog() == new
    assert len(harness.commands) == 1


def test_cancelled_dialog(harness: Harness) -> None:
    harness.page.dialog_factory = lambda names, parent: FakeDialog(
        [angle_dataset("a")], QDialog.DialogCode.Rejected
    )
    assert harness.page.open_import_dialog() == []
    assert not harness.commands


def test_add_data_without_selection(qtbot: QtBot) -> None:
    page = DataPage()
    qtbot.addWidget(page)
    commands: list[Command] = []
    page.commandRequested.connect(commands.append)
    page.dialog_factory = lambda names, parent: FakeDialog(
        [angle_dataset("a")], QDialog.DialogCode.Accepted
    )
    page.add_button.click()
    assert len(commands) == 1
    state = FitProject(FitState()).state
    project = FitProject(state)
    project.execute(commands[0])
    assert project.state.dataset("a").name == "a"


def test_default_factory_builds_import_dialog(qtbot: QtBot, tmp_path: Path) -> None:
    page = DataPage()
    qtbot.addWidget(page)
    page.set_state(FitState(datasets=(angle_dataset("scan"),)))
    dialog = page.dialog_factory(("scan",), page)
    qtbot.addWidget(dialog)
    assert isinstance(dialog, ImportDialog)
    path = tmp_path / "scan.csv"
    path.write_text("0.1,0.9\n0.2,0.5\n")
    dialog.set_path(path)
    assert dialog.name_edit.text() == "scan_2"
