"""Combined coatings on the shared Coating page: blocks, block options and parameters."""

import pytest
from PySide6.QtCore import QPoint, Qt
from pytestqt.qtbot import QtBot

from grazing.core.materials import get
from grazing.core.structure import Composite, Mirror, Periodic
from grazing.core.structure.coatings import DepthGraded, NLayer
from grazing.design.project import Project, ProjectState
from grazing.design.project.templates import new_state
from grazing.fit.project import FitProject
from grazing.fit.templates import new_state as new_fit_state
from grazing.gui.design.steps import CoatingPage
from grazing.gui.fit.steps.structure import StructurePage

pytestmark = pytest.mark.gui


def example() -> Composite:
    return Composite.create(
        [
            DepthGraded.create(get("Pt"), get("C"), N=10, d_min=3.0, d_max=6.0, name="ptc"),
            NLayer.create([get("B4C"), get("Mo")], [2.0, 3.0], name="mid"),
            Periodic.create(get("W"), get("Si"), N=20, d=4.0, name="wsi"),
        ]
    ).with_parameter("wsi.d", vary=True, min=3.5, max=4.5)


def combined_state() -> ProjectState:
    return new_state("periodic").evolve(mirror=Mirror(example(), get("Si")))


class Editor:
    """A Coating page whose commands are applied to a project."""

    def __init__(self, qtbot: QtBot, state: ProjectState) -> None:
        self.page = CoatingPage()
        qtbot.addWidget(self.page)
        self.project = Project(state)
        self.errors: list[str] = []
        self.page.commandRequested.connect(self._apply)
        self.page.errorOccurred.connect(self.errors.append)
        self.page.set_state(state)

    def _apply(self, command: object) -> None:
        self.project.execute(command)  # type: ignore[arg-type]
        self.page.set_state(self.project.state)

    @property
    def coating(self) -> Composite:
        mirror = self.project.state.mirror
        assert mirror is not None
        assert isinstance(mirror.coating, Composite)
        return mirror.coating

    def rows(self) -> list[str]:
        return [self.page.block_list.item(r).text() for r in range(self.page.block_list.count())]

    def shown_parameters(self) -> list[str]:
        return list(self.page.parameter_table.table_model.parameters())


def choose(combo: object, data: str) -> None:
    index = combo.findData(data)  # type: ignore[attr-defined]
    combo.setCurrentIndex(index)  # type: ignore[attr-defined]
    combo.activated.emit(index)  # type: ignore[attr-defined]


def test_single_coating_hides_block_controls(qtbot: QtBot) -> None:
    editor = Editor(qtbot, new_state("periodic"))
    assert editor.page.blocks_box.isHidden()
    assert editor.page.scope_combo.isHidden()
    assert editor.page.selected_block is None


def test_choosing_combined_type(qtbot: QtBot) -> None:
    editor = Editor(qtbot, new_state("periodic"))
    choose(editor.page.type_combo, "composite")
    assert editor.coating.block_names() == ("periodic",)
    assert not editor.page.blocks_box.isHidden()
    assert editor.page.selected_block == "periodic"
    assert editor.rows()[0].startswith("1. periodic — Periodic multilayer")
    assert not editor.page.remove_block_button.isEnabled()
    choose(editor.page.type_combo, "depth_graded")
    mirror = editor.project.state.mirror
    assert mirror is not None
    assert isinstance(mirror.coating, DepthGraded)


def test_blocks_and_sketch(qtbot: QtBot) -> None:
    editor = Editor(qtbot, combined_state())
    assert [r.split(" —")[0] for r in editor.rows()] == ["1. ptc", "2. mid", "3. wsi"]
    sketch = editor.page.sketch.blocks
    assert [(n, count) for n, count, _ in sketch] == [("ptc", 20), ("mid", 2), ("wsi", 40)]
    assert sketch[2][2] == pytest.approx(80.0)
    assert editor.page.selected_block == "ptc"
    assert editor.page.block_type_combo.currentData() == "depth_graded"
    assert editor.page.options_box.title() == "Block “ptc”"


def test_select_block_and_parameter_scope(qtbot: QtBot) -> None:
    editor = Editor(qtbot, combined_state())
    editor.page.block_list.setCurrentRow(2)
    assert editor.page.selected_block == "wsi"
    assert editor.page.block_type_combo.currentData() == "periodic"
    assert editor.page.block_name_edit.text() == "wsi"
    names = editor.shown_parameters()
    assert "ptc.d_min" in names
    assert "wsi.d" in names
    choose(editor.page.scope_combo, "block")
    assert all(n.startswith("wsi.") for n in editor.shown_parameters())
    assert "wsi.t_interlayer_H_on_L" not in editor.shown_parameters()


def test_sketch_click_selects(qtbot: QtBot) -> None:
    editor = Editor(qtbot, combined_state())
    editor.page.show()
    qtbot.waitExposed(editor.page)
    sketch = editor.page.sketch
    sketch.repaint()
    bottom = QPoint(10, sketch.height() - 8)
    qtbot.mouseClick(sketch, Qt.MouseButton.LeftButton, pos=bottom)  # type: ignore[no-untyped-call]
    assert editor.page.selected_block == "wsi"
    qtbot.mouseClick(sketch, Qt.MouseButton.LeftButton, pos=QPoint(10, 8))  # type: ignore[no-untyped-call]
    assert editor.page.selected_block == "ptc"


def test_block_options_change_only_that_block(qtbot: QtBot) -> None:
    editor = Editor(qtbot, combined_state())
    editor.page.select_block("wsi")
    choose(editor.page.top_combo, "L")
    wsi = editor.coating.block("wsi")
    assert isinstance(wsi, Periodic)
    assert wsi.top == "L"
    assert editor.coating.parameters["wsi.d"].vary
    assert editor.page.selected_block == "wsi"
    editor.page.interlayer_check.setChecked(True)
    wsi = editor.coating.block("wsi")
    assert isinstance(wsi, Periodic)
    assert wsi.use_interlayers


def test_block_type_change_keeps_other_blocks(qtbot: QtBot) -> None:
    editor = Editor(qtbot, combined_state())
    editor.page.select_block("ptc")
    choose(editor.page.block_type_combo, "periodic")
    coating = editor.coating
    assert isinstance(coating.block("ptc"), Periodic)
    assert coating.block("ptc").material("H").name == "Pt"
    assert coating.parameters["wsi.d"].vary
    assert coating.block_names() == ("ptc", "mid", "wsi")


def test_constant_gamma_in_block(qtbot: QtBot) -> None:
    editor = Editor(qtbot, combined_state())
    editor.page.select_block("ptc")
    editor.page.constant_gamma_check.setChecked(True)
    assert editor.coating.parameters["ptc.gamma_bottom"].expr == "ptc.gamma_top"
    editor.page.constant_gamma_check.setChecked(False)
    assert editor.coating.parameters["ptc.gamma_bottom"].expr is None


def test_layer_stack_block(qtbot: QtBot) -> None:
    editor = Editor(qtbot, combined_state())
    editor.page.select_block("mid")
    assert not editor.page.layers_box.isHidden()
    assert editor.page.layer_table.rowCount() == 2
    editor.page.add_layer_button.click()
    mid = editor.coating.block("mid")
    assert isinstance(mid, NLayer)
    assert len(mid.material_roles) == 3
    assert editor.coating.parameters["wsi.d"].vary


@pytest.mark.parametrize(
    ("type_name", "name"),
    [
        ("periodic", "periodic"),
        ("depth_graded", "graded"),
        ("single_layer", "layer"),
        ("n_layer", "stack"),
    ],
)
def test_add_block(qtbot: QtBot, type_name: str, name: str) -> None:
    editor = Editor(qtbot, combined_state())
    editor.page.select_block("mid")
    editor.page.add_block(type_name)
    assert editor.coating.block_names() == ("ptc", "mid", name, "wsi")
    assert editor.coating.block(name).type_name == type_name
    assert editor.page.selected_block == name
    assert editor.coating.parameters["wsi.d"].vary


def test_duplicate_remove_and_move(qtbot: QtBot) -> None:
    editor = Editor(qtbot, combined_state())
    editor.page.select_block("wsi")
    editor.page.duplicate_block()
    assert editor.coating.block_names() == ("ptc", "mid", "wsi", "periodic")
    assert editor.coating.parameters["periodic.d"].vary
    editor.page.move_block(-1)
    assert editor.coating.block_names() == ("ptc", "mid", "periodic", "wsi")
    assert editor.page.selected_block == "periodic"
    editor.page.remove_block()
    assert editor.coating.block_names() == ("ptc", "mid", "wsi")
    assert editor.page.selected_block == "mid"
    editor.page.select_block("ptc")
    assert not editor.page.up_block_button.isEnabled()
    editor.project.undo()
    assert editor.coating.block_names() == ("ptc", "mid", "periodic", "wsi")


def test_rename_block(qtbot: QtBot) -> None:
    editor = Editor(qtbot, combined_state())
    editor.page.select_block("wsi")
    editor.page.block_name_edit.setText("ml")
    editor.page.block_name_edit.editingFinished.emit()
    assert editor.coating.block_names() == ("ptc", "mid", "ml")
    assert editor.coating.parameters["ml.d"].vary
    assert editor.page.selected_block == "ml"
    editor.page.block_name_edit.setText("ptc")
    editor.page.block_name_edit.editingFinished.emit()
    assert editor.errors
    assert editor.page.block_name_edit.text() == "ml"


def test_combined_to_single_converts_selected_block(qtbot: QtBot) -> None:
    editor = Editor(qtbot, combined_state())
    editor.page.select_block("wsi")
    choose(editor.page.type_combo, "periodic")
    mirror = editor.project.state.mirror
    assert mirror is not None
    assert isinstance(mirror.coating, Periodic)
    assert mirror.coating.material("H").name == "W"


def test_fit_application(qtbot: QtBot) -> None:
    state = new_fit_state("periodic").evolve(mirror=Mirror(example(), get("Si")))
    page = StructurePage()
    qtbot.addWidget(page)
    commands: list[object] = []
    page.commandRequested.connect(commands.append)
    page.set_state(state)
    assert not page.blocks_box.isHidden()
    assert page.scope_combo.isHidden()
    page.select_block("wsi")
    choose(page.top_combo, "L")
    project = FitProject(state)
    project.execute(commands[-1])  # type: ignore[arg-type]
    mirror = project.state.mirror
    assert mirror is not None
    assert isinstance(mirror.coating, Composite)
    assert mirror.coating.block("wsi").top == "L"  # type: ignore[attr-defined]


def test_options_of_other_types_are_unchecked(qtbot: QtBot) -> None:
    editor = Editor(qtbot, combined_state())
    editor.page.select_block("ptc")
    editor.page.constant_gamma_check.setChecked(True)
    editor.page.select_block("wsi")
    assert not editor.page.constant_gamma_check.isEnabled()
    assert not editor.page.constant_gamma_check.isChecked()
    assert not editor.page.cap_check.isChecked()
    editor.page.select_block("ptc")
    assert editor.page.constant_gamma_check.isChecked()
