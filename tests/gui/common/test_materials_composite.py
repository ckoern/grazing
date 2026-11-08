"""Materials page with combined (composite) coatings, in both applications."""

import pytest
from PySide6.QtWidgets import QCheckBox, QComboBox
from pytestqt.qtbot import QtBot

from grazing.core.materials import get
from grazing.core.structure import DepthGraded, Mirror, NLayer, Periodic
from grazing.core.structure.coating import density_parameter_name
from grazing.core.structure.coatings.composite import Composite
from grazing.design.project import Project, ProjectState
from grazing.design.project.templates import new_state
from grazing.fit.project import FitProject, FitState
from grazing.fit.templates import new_state as new_fit_state
from grazing.gui.common.materials import INTERLAYER_START
from grazing.gui.design.steps import MaterialsPage
from grazing.gui.fit.steps.materials import MaterialsPage as FitMaterialsPage

pytestmark = pytest.mark.gui


def composite(interlayers: bool = False) -> Composite:
    wsi = Periodic.create(get("W"), get("Si"), N=20, d=4.0, name="wsi")
    if interlayers:
        wsi = wsi.with_interlayers(True)
    return Composite.create(
        [
            DepthGraded.create(get("Pt"), get("C"), N=10, d_min=3.0, d_max=6.0, name="ptc"),
            NLayer.create([get("B4C"), get("Mo")], [2.0, 3.0], name="mid"),
            wsi,
        ]
    )


def design_state(interlayers: bool = False) -> ProjectState:
    return new_state("periodic").evolve(mirror=Mirror(composite(interlayers), get("Si")))


def fit_state() -> FitState:
    return new_fit_state("single_layer").evolve(mirror=Mirror(composite(), get("Si")))


def page_with(qtbot: QtBot, state: ProjectState) -> tuple[MaterialsPage, list[object]]:
    page = MaterialsPage()
    qtbot.addWidget(page)
    commands: list[object] = []
    page.commandRequested.connect(commands.append)
    page.set_state(state)
    return page, commands


def role_rows(page: MaterialsPage) -> dict[str, int]:
    """Rows of the role names; heading rows are left out."""
    return {
        page.table.item(r, 0).text(): r
        for r in range(page.table.rowCount())
        if page.table.columnSpan(r, 0) == 1
    }


def choose(page: MaterialsPage, role: str, name: str) -> None:
    combo = page.table.cellWidget(role_rows(page)[role], 1)
    assert isinstance(combo, QComboBox)
    combo.setCurrentText(name)
    combo.activated.emit(combo.currentIndex())


def checkbox(page: MaterialsPage, role: str) -> QCheckBox:
    holder = page.table.cellWidget(role_rows(page)[role], 3)
    assert holder is not None
    box = holder.findChild(QCheckBox)
    assert box is not None
    return box


def apply(state: ProjectState, command: object) -> ProjectState:
    project = Project(state)
    project.execute(command)  # type: ignore[arg-type]
    return project.state


def coating_of(state: ProjectState | FitState) -> Composite:
    assert state.mirror is not None
    coating = state.mirror.coating
    assert isinstance(coating, Composite)
    return coating


def test_rows_have_headings_roles_and_substrate(qtbot: QtBot) -> None:
    page, _ = page_with(qtbot, design_state())
    texts = [page.table.item(r, 0).text() for r in range(page.table.rowCount())]
    assert texts[0].startswith("1  ptc")
    assert "Depth-graded multilayer" in texts[0]
    assert texts[3].startswith("2  mid")
    assert "wsi" in texts[-4]
    assert "Periodic multilayer" in texts[-4]
    assert [t for t in texts if "." in t and " " not in t] == [
        "ptc.H",
        "ptc.L",
        "mid.M1",
        "mid.M2",
        "wsi.H",
        "wsi.L",
    ]
    assert texts[-1] == "substrate"
    headings = [r for r in range(page.table.rowCount()) if page.table.columnSpan(r, 0) == 4]
    assert headings == [0, 3, 6]


def test_choosing_a_material_changes_only_that_block(qtbot: QtBot) -> None:
    state = design_state()
    page, commands = page_with(qtbot, state)
    choose(page, "wsi.H", "Mo")
    updated = coating_of(apply(state, commands[-1]))
    before = coating_of(state).material_map()
    after = updated.material_map()
    assert after["wsi.H"] == get("Mo")
    assert {r: m for r, m in after.items() if r != "wsi.H"} == {
        r: m for r, m in before.items() if r != "wsi.H"
    }


def test_choosing_substrate(qtbot: QtBot) -> None:
    state = design_state()
    page, commands = page_with(qtbot, state)
    choose(page, "substrate", "Mo")
    assert apply(state, commands[-1]).mirror is not None
    assert apply(state, commands[-1]).mirror.substrate == get("Mo")  # type: ignore[union-attr]


def test_density_edit_of_prefixed_role(qtbot: QtBot) -> None:
    state = design_state()
    page, commands = page_with(qtbot, state)
    page.table.item(role_rows(page)["wsi.L"], 2).setText("2.0")
    updated = coating_of(apply(state, commands[-1]))
    assert updated.material_map()["wsi.L"].density == 2.0
    assert updated.material_map()["wsi.H"] == get("W")


def test_density_parameter_toggle(qtbot: QtBot) -> None:
    state = design_state()
    page, commands = page_with(qtbot, state)
    name = density_parameter_name("wsi.H")
    checkbox(page, "wsi.H").setChecked(True)
    state = apply(state, commands[-1])
    assert state.mirror is not None
    assert list(state.mirror.densities) == [name]
    page.set_state(state)
    assert checkbox(page, "wsi.H").isChecked()
    assert not checkbox(page, "wsi.L").isChecked()
    page.table.item(role_rows(page)["wsi.H"], 2).setText("15.5")
    state = apply(state, commands[-1])
    assert state.mirror is not None
    assert state.mirror.densities[name].value == 15.5
    page.set_state(state)
    checkbox(page, "wsi.H").setChecked(False)
    state = apply(state, commands[-1])
    assert state.mirror is not None
    assert len(state.mirror.densities) == 0


def test_interlayer_rows_only_for_blocks_with_interlayers(qtbot: QtBot) -> None:
    page, _ = page_with(qtbot, design_state())
    assert not any("interlayer" in r for r in role_rows(page))
    page, _ = page_with(qtbot, design_state(interlayers=True))
    interlayers = [r for r in role_rows(page) if "interlayer" in r]
    assert interlayers == ["wsi.interlayer_H_on_L", "wsi.interlayer_L_on_H"]
    combo = page.table.cellWidget(role_rows(page)["wsi.interlayer_H_on_L"], 1)
    assert isinstance(combo, QComboBox)
    assert combo.currentText() == "(none)"


def test_interlayer_choose_and_remove(qtbot: QtBot) -> None:
    state = design_state(interlayers=True)
    thickness = "wsi.t_interlayer_H_on_L"
    state = state.evolve(
        mirror=Mirror(coating_of(state).with_parameter(thickness, value=0.0), get("Si"))
    )
    page, commands = page_with(qtbot, state)
    choose(page, "wsi.interlayer_H_on_L", "Mo")
    state = apply(state, commands[-1])
    coating = coating_of(state)
    assert coating.material_map()["wsi.interlayer_H_on_L"] == get("Mo")
    assert coating.parameters[thickness].value == INTERLAYER_START
    page.set_state(state)
    choose(page, "wsi.interlayer_H_on_L", "(none)")
    state = apply(state, commands[-1])
    coating = coating_of(state)
    assert "wsi.interlayer_H_on_L" not in coating.material_map()
    assert coating.parameters[thickness].value == 0.0


def test_plot_shows_each_material_once(qtbot: QtBot) -> None:
    page, _ = page_with(qtbot, design_state())
    names = {m.name for m in coating_of(design_state()).material_map().values()}
    names.add("Si")
    delta_curves = [i for i in page.plot.listDataItems() if i.name() is not None]
    assert len(delta_curves) == len(names)


def test_fit_application_materials_page(qtbot: QtBot) -> None:
    state = fit_state()
    page = FitMaterialsPage()
    qtbot.addWidget(page)
    commands: list[object] = []
    page.commandRequested.connect(commands.append)
    page.set_state(state)
    assert "ptc.H" in role_rows(page)
    choose(page, "wsi.L", "Mo")
    project = FitProject(state)
    project.execute(commands[-1])  # type: ignore[arg-type]
    assert coating_of(project.state).material_map()["wsi.L"] == get("Mo")
    page.set_state(project.state)
    checkbox(page, "mid.M2").setChecked(True)
    project.execute(commands[-1])  # type: ignore[arg-type]
    assert project.state.mirror is not None
    assert list(project.state.mirror.densities) == [density_parameter_name("mid.M2")]
