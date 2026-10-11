"""Tests for the workflow pages in `xstack.gui.steps`."""

import math
from collections.abc import Callable

import pytest
from PySide6.QtWidgets import QComboBox, QLineEdit
from pytestqt.qtbot import QtBot

from xstack.gui.preview import compute_preview
from xstack.gui.steps import (
    CoatingPage,
    ConstraintsPage,
    GoalPage,
    MaterialsPage,
    ProcessPage,
    StepPage,
    convert_coating,
)
from xstack.manufacturing.constraints import MaxTotalThickness, MinLayerThickness, constraints
from xstack.manufacturing.process import DepositionProcess
from xstack.materials import get
from xstack.project import Command, ProjectState
from xstack.project.templates import new_state
from xstack.structure.coatings.graded import DepthGraded
from xstack.structure.coatings.periodic import Periodic
from xstack.structure.coatings.simple import SingleLayer
from xstack.targets import AngleScan, EnergyScan

pytestmark = pytest.mark.gui

DEG = math.pi / 180.0


class Recorder:
    """Collects the commands and errors emitted by a page."""

    def __init__(self, page: StepPage) -> None:
        self.commands: list[Command] = []
        self.errors: list[str] = []
        page.commandRequested.connect(self.commands.append)
        page.errorOccurred.connect(self.errors.append)

    def applied(self, state: ProjectState) -> ProjectState:
        assert len(self.commands) == 1
        return self.commands[-1].transform(state)


def make_page[P: StepPage](
    qtbot: QtBot, factory: Callable[[], P], state: ProjectState
) -> tuple[P, Recorder]:
    page = factory()
    qtbot.addWidget(page)
    page.set_state(state)
    return page, Recorder(page)


def combo_at(page: GoalPage | MaterialsPage, row: int, column: int) -> QComboBox:
    widget = page.table.cellWidget(row, column)
    assert isinstance(widget, QComboBox)
    return widget


def choose(combo: QComboBox, text: str) -> None:
    combo.setCurrentText(text)
    combo.activated.emit(combo.currentIndex())


# --- GoalPage ---------------------------------------------------------------------------


def energy_target() -> EnergyScan:
    return EnergyScan(
        angle=1.1 * DEG, energy_min=7900.0, energy_max=8200.0, resolution=2.5, name="mine"
    )


def test_goal_shows_units(qtbot: QtBot) -> None:
    page, _ = make_page(qtbot, GoalPage, new_state("periodic"))
    assert page.table.rowCount() == 1
    assert combo_at(page, 0, 0).currentText() == "Energy scan"
    texts = [page.table.item(0, c).text() for c in (1, 2, 3, 4)]
    assert texts == ["1.1", "7.9", "8.2", "deg | keV"]
    angle, _ = make_page(qtbot, GoalPage, new_state("single_layer"))
    assert combo_at(angle, 0, 0).currentText() == "Angle scan"
    assert angle.table.item(0, 1).text() == "8.048"
    assert angle.table.item(0, 4).text() == "keV | deg"


def test_goal_edit_to_converts_and_preserves(qtbot: QtBot) -> None:
    from xstack.targets import TargetSet

    state = new_state("periodic").evolve(targets=TargetSet(energy_target()))
    page, rec = make_page(qtbot, GoalPage, state)
    page.table.item(0, 3).setText("8.5")
    new = rec.applied(state).targets
    assert new is not None
    target = new.targets[0]
    assert isinstance(target, EnergyScan)
    assert target.energy_max == pytest.approx(8500.0)
    assert target.energy_min == pytest.approx(7900.0)
    assert target.angle == pytest.approx(1.1 * DEG)
    assert target.resolution == 2.5
    assert target.name == "mine"


def test_goal_scan_type_to_angle(qtbot: QtBot) -> None:
    state = new_state("periodic")
    page, rec = make_page(qtbot, GoalPage, state)
    choose(combo_at(page, 0, 0), "Angle scan")
    targets = rec.applied(state).targets
    assert targets is not None
    target = targets.targets[0]
    assert isinstance(target, AngleScan)
    assert target.energy == pytest.approx(1100.0)
    assert target.angle_max == pytest.approx(8.2 * DEG)


def test_goal_invalid_text(qtbot: QtBot) -> None:
    page, rec = make_page(qtbot, GoalPage, new_state("periodic"))
    page.table.item(0, 3).setText("abc")
    assert not rec.commands
    assert len(rec.errors) == 1
    assert page.table.item(0, 3).text() == "8.2"


def test_goal_add_scans_and_remove(qtbot: QtBot) -> None:
    state = new_state("periodic")
    page, rec = make_page(qtbot, GoalPage, state)
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
    page.table.setCurrentCell(1, 1)
    page.remove.click()
    removed = rec.applied(added).targets
    assert removed is not None
    assert state.targets is not None
    assert removed.targets == (state.targets.targets[0],)


def test_goal_remove_last_target_refused(qtbot: QtBot) -> None:
    page, rec = make_page(qtbot, GoalPage, new_state("periodic"))
    page.remove.click()
    assert not rec.commands
    assert len(rec.errors) == 1


def test_goal_combine_combo(qtbot: QtBot) -> None:
    state = new_state("periodic")
    page, rec = make_page(qtbot, GoalPage, state)
    assert page.combine.currentData() == "average_reflectivity"
    page.combine.setCurrentIndex(1)
    page.combine.activated.emit(1)
    targets = rec.applied(state).targets
    assert targets is not None
    assert targets.combine == "average_merit"


# --- MaterialsPage -----------------------------------------------------------------------


def test_materials_rows(qtbot: QtBot) -> None:
    page, _ = make_page(qtbot, MaterialsPage, new_state("periodic"))
    assert page.table.rowCount() == 3
    roles = [page.table.item(r, 0).text() for r in range(3)]
    assert roles == ["H", "L", "substrate"]
    assert combo_at(page, 0, 1).currentText() == "W"
    assert combo_at(page, 2, 1).currentText() == "Si"


def test_materials_choose_coating_material(qtbot: QtBot) -> None:
    state = new_state("periodic")
    page, rec = make_page(qtbot, MaterialsPage, state)
    choose(combo_at(page, 0, 1), "Pt")
    new = rec.applied(state)
    assert new.mirror is not None
    assert new.mirror.coating.materials["H"].name == "Pt"
    assert new.mirror.coating.materials["L"].name == "Si"


def test_materials_choose_substrate(qtbot: QtBot) -> None:
    state = new_state("periodic")
    page, rec = make_page(qtbot, MaterialsPage, state)
    choose(combo_at(page, 2, 1), "SiO2")
    new = rec.applied(state)
    assert new.mirror is not None
    assert new.mirror.substrate.name == "SiO2"
    assert state.mirror is not None
    assert new.mirror.coating == state.mirror.coating


def test_materials_edit_density(qtbot: QtBot) -> None:
    state = new_state("periodic")
    page, rec = make_page(qtbot, MaterialsPage, state)
    page.table.item(0, 2).setText("15.5")
    new = rec.applied(state)
    assert new.mirror is not None
    assert new.mirror.coating.materials["H"].density == pytest.approx(15.5)
    assert new.mirror.coating.materials["H"].name == "W"


@pytest.mark.parametrize("text", ["abc", "-3"])
def test_materials_invalid_density(qtbot: QtBot, text: str) -> None:
    page, rec = make_page(qtbot, MaterialsPage, new_state("periodic"))
    page.table.item(0, 2).setText(text)
    assert not rec.commands
    assert len(rec.errors) == 1


# --- CoatingPage and convert_coating -----------------------------------------------------


def test_convert_periodic_to_depth_graded_keeps_settings() -> None:
    w, si, c = get("W"), get("Si"), get("C")
    old = Periodic.create(w, si, N=30, sigma_H=0.5, sigma_L=0.3, interlayers={"HonL": c})
    new = convert_coating(old, "depth_graded")
    assert isinstance(new, DepthGraded)
    assert new.materials == old.materials
    assert new.parameters["sigma_H"].value == 0.5
    assert new.parameters["sigma_L"].value == 0.3


def test_convert_carries_vary_settings() -> None:
    state = new_state("periodic")
    assert state.mirror is not None
    old = state.mirror.coating
    new = convert_coating(old, "depth_graded")
    assert new.parameters["N"] == old.parameters["N"]


def test_convert_options() -> None:
    old = Periodic.create(get("W"), get("Si"))
    graded = convert_coating(old, "depth_graded", top="L", law="linear", cap=True)
    assert isinstance(graded, DepthGraded)
    assert (graded.top, graded.law, graded.cap) == ("L", "linear", True)
    periodic = convert_coating(graded, "periodic", top="H")
    assert isinstance(periodic, Periodic)
    assert periodic.top == "H"


def test_convert_single_layer() -> None:
    old = Periodic.create(get("W"), get("Si"))
    new = convert_coating(old, "single_layer")
    assert isinstance(new, SingleLayer)
    assert new.materials["M"].name == "W"


def test_convert_unknown_type() -> None:
    with pytest.raises(ValueError, match="cannot be created"):
        convert_coating(Periodic.create(get("W"), get("Si")), "nonsense")


def test_coating_page_shows_parameters(qtbot: QtBot) -> None:
    state = new_state("periodic")
    page, _ = make_page(qtbot, CoatingPage, state)
    assert state.mirror is not None
    assert page.parameter_table.table_model.rowCount() == len(state.mirror.coating.parameters)
    assert page.type_combo.currentData() == "periodic"
    assert not page.law_combo.isEnabled()
    assert page.top_combo.isEnabled()


def test_coating_page_switch_type(qtbot: QtBot) -> None:
    state = new_state("periodic")
    page, rec = make_page(qtbot, CoatingPage, state)
    index = page.type_combo.findData("depth_graded")
    page.type_combo.setCurrentIndex(index)
    page.type_combo.activated.emit(index)
    new = rec.applied(state)
    assert new.mirror is not None
    coating = new.mirror.coating
    assert isinstance(coating, DepthGraded)
    assert {r: m.name for r, m in coating.materials.items()} == {"H": "W", "L": "Si"}
    page.set_state(new)
    assert page.law_combo.isEnabled()
    assert page.type_combo.currentData() == "depth_graded"


def test_coating_page_options(qtbot: QtBot) -> None:
    state = new_state("depth_graded")
    page, rec = make_page(qtbot, CoatingPage, state)
    page.top_combo.setCurrentIndex(page.top_combo.findData("L"))
    page.top_combo.activated.emit(page.top_combo.currentIndex())
    new = rec.applied(state)
    assert new.mirror is not None
    assert isinstance(new.mirror.coating, DepthGraded)
    assert new.mirror.coating.top == "L"

    rec.commands.clear()
    page.law_combo.setCurrentText("linear")
    page.law_combo.activated.emit(page.law_combo.currentIndex())
    page.cap_check.setChecked(True)
    assert len(rec.commands) == 2
    final = rec.commands[-1].transform(state).mirror
    assert final is not None
    assert isinstance(final.coating, DepthGraded)
    assert final.coating.cap
    assert final.coating.law == "linear"


def test_coating_page_single_layer(qtbot: QtBot) -> None:
    state = new_state("periodic")
    page, rec = make_page(qtbot, CoatingPage, state)
    index = page.type_combo.findData("single_layer")
    page.type_combo.setCurrentIndex(index)
    page.type_combo.activated.emit(index)
    new = rec.applied(state)
    assert new.mirror is not None
    assert isinstance(new.mirror.coating, SingleLayer)


# --- ConstraintsPage ---------------------------------------------------------------------


def test_constraints_list_and_preview_marks(qtbot: QtBot) -> None:
    state = new_state("periodic")
    page, _ = make_page(qtbot, ConstraintsPage, state)
    assert page.list.count() == 2
    assert page.list.item(0).text().startswith("…")
    assert page.form.value() == MinLayerThickness(0.8)
    page.set_preview(compute_preview(state))
    marks = [page.list.item(i).text()[0] for i in range(2)]
    assert all(m in "✓✗" for m in marks)
    assert marks[0] == "✓"
    tight = state.evolve(constraints=(MinLayerThickness(0.8), MaxTotalThickness(10.0)))
    page.set_state(tight)
    page.set_preview(compute_preview(tight))
    assert page.list.item(1).text().startswith("✗")


def test_constraints_stale_preview_ignored(qtbot: QtBot) -> None:
    state = new_state("periodic")
    page, _ = make_page(qtbot, ConstraintsPage, state)
    page.set_preview(compute_preview(state.evolve(constraints=(MinLayerThickness(0.8),))))
    assert page.list.item(0).text().startswith("…")


@pytest.mark.parametrize("name", list(constraints))
def test_constraints_add_each_type(qtbot: QtBot, name: str) -> None:
    state = new_state("periodic")
    page, rec = make_page(qtbot, ConstraintsPage, state)
    page.type_combo.setCurrentIndex(page.type_combo.findData(name))
    page.add_button.click()
    new = rec.applied(state)
    assert len(new.constraints) == 3
    assert new.constraints[-1].type_name == name


def test_constraints_remove(qtbot: QtBot) -> None:
    state = new_state("periodic")
    page, rec = make_page(qtbot, ConstraintsPage, state)
    page.list.setCurrentRow(0)
    page.remove_button.click()
    assert rec.applied(state).constraints == (state.constraints[1],)


def test_constraints_edit_via_form(qtbot: QtBot) -> None:
    state = new_state("periodic")
    page, rec = make_page(qtbot, ConstraintsPage, state)
    page.list.setCurrentRow(1)
    editor = page.form.findChild(QLineEdit, "field_maximum")
    assert editor is not None
    editor.setText("500")
    editor.editingFinished.emit()
    new = rec.applied(state)
    assert new.constraints == (state.constraints[0], MaxTotalThickness(500.0))


def test_constraints_invalid_form_input_reports_error(qtbot: QtBot) -> None:
    state = new_state("periodic")
    page, rec = make_page(qtbot, ConstraintsPage, state)
    page.list.setCurrentRow(0)
    editor = page.form.findChild(QLineEdit, "field_minimum")
    assert editor is not None
    editor.setText("-1")
    editor.editingFinished.emit()
    assert not rec.commands
    assert len(rec.errors) == 1


# --- ProcessPage -------------------------------------------------------------------------


def test_process_enable_creates_default_process(qtbot: QtBot) -> None:
    state = new_state("periodic")
    page, rec = make_page(qtbot, ProcessPage, state)
    assert not page.enabled.isChecked()
    assert page.table.rowCount() == 2
    page.enabled.setChecked(True)
    process = rec.applied(state).process
    assert isinstance(process, DepositionProcess)
    assert dict(process.rates) == {"Si": 0.1, "W": 0.1}
    assert process.fixed_overhead == 0.0


def enabled_page(qtbot: QtBot) -> tuple[ProcessPage, Recorder, ProjectState]:
    state = new_state("periodic")
    process = DepositionProcess({"Si": 0.1, "W": 0.1})
    state = state.evolve(process=process)
    page, rec = make_page(qtbot, ProcessPage, state)
    return page, rec, state


def test_process_edit_rate(qtbot: QtBot) -> None:
    page, rec, state = enabled_page(qtbot)
    assert page.enabled.isChecked()
    assert page.table.item(1, 0).text() == "W"
    page.table.item(1, 1).setText("0.25")
    process = rec.applied(state).process
    assert process is not None
    assert dict(process.rates) == {"Si": 0.1, "W": 0.25}


def test_process_edit_overheads(qtbot: QtBot) -> None:
    page, rec, state = enabled_page(qtbot)
    page.fixed.setText("2")
    page.fixed.editingFinished.emit()
    process = rec.applied(state).process
    assert process is not None
    assert process.fixed_overhead == pytest.approx(7200.0)
    rec.commands.clear()
    page.per_layer.setText("1.5")
    page.per_layer.editingFinished.emit()
    process = rec.applied(state).process
    assert process is not None
    assert process.overhead_per_layer == pytest.approx(1.5)


def test_process_invalid_entry(qtbot: QtBot) -> None:
    page, rec, _ = enabled_page(qtbot)
    page.table.item(1, 1).setText("abc")
    assert not rec.commands
    assert len(rec.errors) == 1


def test_process_disable(qtbot: QtBot) -> None:
    page, rec, state = enabled_page(qtbot)
    page.enabled.setChecked(False)
    assert rec.applied(state).process is None


def test_process_summary(qtbot: QtBot) -> None:
    page, _, state = enabled_page(qtbot)
    assert page.summary.text() == ""
    page.set_preview(compute_preview(state))
    assert page.summary.text().startswith("Coating time:")
    page.set_preview(compute_preview(state.evolve(process=None)))
    assert page.summary.text() == ""
