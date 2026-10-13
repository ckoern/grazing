"""Tests for the workflow pages in `xstack.gui.steps`."""

import dataclasses
import math
from collections.abc import Callable

import pytest
from PySide6.QtWidgets import QComboBox, QLineEdit
from pytestqt.qtbot import QtBot

from xstack.gui.preview import compute_preview
from xstack.gui.steps import (
    CoatingPage,
    ConstraintsPage,
    MaterialsPage,
    ProcessPage,
    StepPage,
    convert_coating,
)
from xstack.gui.steps.coating import _restack
from xstack.manufacturing.constraints import MaxTotalThickness, MinLayerThickness, constraints
from xstack.manufacturing.process import DepositionProcess
from xstack.materials import get
from xstack.parameters import ParameterSet
from xstack.project import Command, ProjectState
from xstack.project.templates import new_state
from xstack.structure.coatings.graded import DepthGraded
from xstack.structure.coatings.periodic import Periodic
from xstack.structure.coatings.simple import NLayer, SingleLayer

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


def combo_at(page: MaterialsPage, row: int, column: int) -> QComboBox:
    widget = page.table.cellWidget(row, column)
    assert isinstance(widget, QComboBox)
    return widget


def choose(combo: QComboBox, text: str) -> None:
    combo.setCurrentText(text)
    combo.activated.emit(combo.currentIndex())


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
    old = Periodic.create(
        w, si, N=30, sigma_H=0.5, sigma_L=0.3, interlayers={"interlayer_H_on_L": c}
    )
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


def test_help_text() -> None:
    text = CoatingPage.help_text
    assert "input" in text
    assert "Vary" in text
    assert "expression" in text


def test_substrate_roughness_edit(qtbot: QtBot) -> None:
    state = new_state("periodic")
    page, rec = make_page(qtbot, CoatingPage, state)
    page.roughness_spin.setValue(0.7)
    page.roughness_spin.editingFinished.emit()
    new = rec.applied(state)
    assert new.mirror is not None
    assert state.mirror is not None
    assert new.mirror.substrate_roughness == pytest.approx(0.7)
    assert new.mirror.coating == state.mirror.coating
    assert new.mirror.substrate == state.mirror.substrate


def test_substrate_roughness_unchanged_emits_nothing(qtbot: QtBot) -> None:
    page, rec = make_page(qtbot, CoatingPage, new_state("periodic"))
    page.roughness_spin.editingFinished.emit()
    assert not rec.commands


def test_convert_to_n_layer_from_bilayer_and_single() -> None:
    old = Periodic.create(get("W"), get("Si"), d=5.0, gamma=0.4, sigma_H=0.5)
    new = convert_coating(old, "n_layer")
    assert isinstance(new, NLayer)
    assert [m.name for _, m in new.material_roles] == ["W", "Si"]
    assert new.parameters["t_W"].value == pytest.approx(2.0)
    assert new.parameters["t_Si"].value == pytest.approx(3.0)
    assert new.parameters["sigma_W"].value == pytest.approx(0.5)
    single = convert_coating(SingleLayer.create(get("W"), t=12.0), "n_layer")
    assert isinstance(single, NLayer)
    assert single.count == 1
    assert single.parameters["t_W"].value == pytest.approx(12.0)


def test_convert_from_n_layer() -> None:
    old = NLayer.create([get("Pt"), get("C"), get("Si")], [1.0, 2.0, 3.0])
    for name in ("periodic", "depth_graded"):
        new = convert_coating(old, name)
        assert new.materials["H"].name == "Pt"
        assert new.materials["L"].name == "C"
    single = convert_coating(old, "single_layer")
    assert single.materials["M"].name == "Pt"
    assert convert_coating(old, "n_layer") is old


def n_layer_state() -> ProjectState:
    state = new_state("periodic")
    assert state.mirror is not None
    coating = NLayer.create([get("Pt"), get("C")], [3.0, 4.0])
    return state.evolve(mirror=state.mirror.with_coating(coating))


def test_coating_page_select_n_layer(qtbot: QtBot) -> None:
    state = new_state("periodic")
    page, rec = make_page(qtbot, CoatingPage, state)
    assert page.layers_box.isHidden()
    index = page.type_combo.findData("n_layer")
    page.type_combo.setCurrentIndex(index)
    page.type_combo.activated.emit(index)
    new = rec.applied(state)
    assert new.mirror is not None
    assert isinstance(new.mirror.coating, NLayer)
    page.set_state(new)
    assert not page.layers_box.isHidden()
    assert page.layer_table.rowCount() == 2
    assert not page.top_combo.isEnabled()
    assert not page.law_combo.isEnabled()


def test_coating_page_add_remove_layer(qtbot: QtBot) -> None:
    state = n_layer_state()
    page, rec = make_page(qtbot, CoatingPage, state)
    page.add_layer_button.click()
    added = rec.commands[-1].transform(state).mirror
    assert added is not None
    assert isinstance(added.coating, NLayer)
    assert added.coating.count == 3
    assert added.coating.parameters["t_C_1"].value == pytest.approx(4.0)

    rec.commands.clear()
    page.layer_table.selectRow(0)
    page.remove_layer_button.click()
    removed = rec.commands[-1].transform(state).mirror
    assert removed is not None
    assert isinstance(removed.coating, NLayer)
    assert [m.name for _, m in removed.coating.material_roles] == ["C"]
    assert removed.coating.parameters["t_C"].value == pytest.approx(4.0)


def test_coating_page_cannot_remove_last_layer(qtbot: QtBot) -> None:
    state = new_state("periodic")
    assert state.mirror is not None
    single = state.evolve(mirror=state.mirror.with_coating(NLayer.create([get("W")], [3.0])))
    page, rec = make_page(qtbot, CoatingPage, single)
    page.remove_layer_button.click()
    assert not rec.commands
    assert rec.errors


def test_coating_page_layer_material_change(qtbot: QtBot) -> None:
    state = n_layer_state()
    page, rec = make_page(qtbot, CoatingPage, state)
    combo = page.layer_table.cellWidget(1, 0)
    assert isinstance(combo, QComboBox)
    combo.setCurrentText("Si")
    combo.activated.emit(combo.currentIndex())
    new = rec.applied(state)
    assert new.mirror is not None
    assert isinstance(new.mirror.coating, NLayer)
    assert [m.name for _, m in new.mirror.coating.material_roles] == ["Pt", "Si"]
    assert new.mirror.coating.parameters["t_Si"].value == pytest.approx(4.0)


def test_restack_renames_parameters_and_expressions() -> None:
    old = NLayer.create([get("Pt"), get("C")], [3.0, "t_Pt + 1"])
    new = _restack(old, [get("Si"), get("C")], [1, 2])
    assert list(new.parameters)[::2] == ["t_Si", "t_C"]
    assert new.parameters["t_C"].expr == "t_Si + 1"
    dropped = _restack(old, [get("C")], [2])
    assert dropped.parameters["t_C"].expr is None


def test_restack_of_legacy_names() -> None:
    new = NLayer.create([get("Pt"), get("C")], [3.0, 4.0])
    legacy = NLayer(
        new.name,
        new.material_roles,
        ParameterSet(
            dataclasses.replace(p, name=n)
            for n, p in zip(["t1", "sigma1", "t2", "sigma2"], new.parameters.values(), strict=True)
        ),
    )
    restacked = _restack(legacy, [get("Pt"), get("C"), get("Pt")], [1, 2, None])
    assert list(restacked.parameters)[::2] == ["t_Pt_1", "t_C", "t_Pt_2"]
    assert restacked.parameters["t_C"].value == pytest.approx(4.0)


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
