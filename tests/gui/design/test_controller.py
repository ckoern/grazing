"""Tests for `grazing.gui.design.controller`."""

from pathlib import Path

import pytest
from pytestqt.qtbot import QtBot

from grazing.core.structure.coatings.periodic import Periodic
from grazing.design.project import Project
from grazing.design.project.commands import rename_project, set_parameter
from grazing.gui.design.controller import ProjectController

pytestmark = pytest.mark.gui


def value(c: ProjectController, name: str) -> float:
    assert c.state.mirror is not None
    return c.state.mirror.coating.parameters[name].value


def can_undo(c: ProjectController) -> bool:
    return c.project.history.can_undo


def dirty(c: ProjectController) -> bool:
    return c.is_dirty


@pytest.fixture
def controller(qtbot: QtBot) -> ProjectController:
    return ProjectController()


def test_default_project(controller: ProjectController) -> None:
    assert isinstance(controller.state.mirror.coating, Periodic)  # type: ignore[union-attr]
    assert controller.path is None
    assert not controller.is_dirty
    assert controller.undo_stack.count() == 0


def test_execute_changes_state_and_signals(controller: ProjectController, qtbot: QtBot) -> None:
    with (
        qtbot.waitSignal(controller.stateChanged),
        qtbot.waitSignal(controller.journalChanged),
    ):
        assert controller.execute(set_parameter("N", value=50))
    assert value(controller, "N") == 50
    assert controller.undo_stack.count() == 1


def test_noop_command(controller: ProjectController) -> None:
    before = controller.state
    assert not controller.execute(set_parameter("N", value=value(controller, "N")))
    assert controller.state is before
    assert controller.undo_stack.count() == 0
    assert not controller.is_dirty


def test_failing_command(controller: ProjectController, qtbot: QtBot) -> None:
    before = controller.state
    with qtbot.waitSignal(controller.errorOccurred) as blocker:
        assert not controller.execute(set_parameter("nonexistent", value=1))
    assert "nonexistent" in blocker.args[0]
    assert controller.state is before
    assert controller.undo_stack.count() == 0


def test_undo_redo_consistent_with_core(controller: ProjectController) -> None:
    original = controller.state
    controller.execute(set_parameter("N", value=50))
    controller.execute(rename_project("other"))
    edited = controller.state
    assert can_undo(controller)

    controller.undo()
    assert controller.state.name == original.name
    assert value(controller, "N") == 50
    controller.undo_stack.undo()
    assert controller.state == original
    assert not can_undo(controller)
    controller.redo()
    controller.undo_stack.redo()
    assert controller.state == edited
    assert not controller.project.history.can_redo


def test_undo_emits_signals(controller: ProjectController, qtbot: QtBot) -> None:
    controller.execute(set_parameter("N", value=50))
    with qtbot.waitSignal(controller.stateChanged), qtbot.waitSignal(controller.journalChanged):
        controller.undo()


def test_merging_same_parameter(controller: ProjectController) -> None:
    for v in (41, 42, 43):
        controller.execute(set_parameter("N", value=v))
    assert controller.undo_stack.count() == 1
    controller.undo()
    assert value(controller, "N") == 40
    assert not controller.project.history.can_undo


def test_different_parameters_do_not_merge(controller: ProjectController) -> None:
    controller.execute(set_parameter("N", value=41))
    controller.execute(set_parameter("d", value=4.5))
    assert controller.undo_stack.count() == 2
    controller.undo()
    assert value(controller, "N") == 41
    assert value(controller, "d") == 4.0
    controller.undo()
    assert value(controller, "N") == 40


def test_dirty_tracking(controller: ProjectController, tmp_path: Path) -> None:
    assert not dirty(controller)
    controller.execute(set_parameter("N", value=50))
    assert dirty(controller)
    controller.save_project(tmp_path / "p.grazing")
    assert not dirty(controller)
    controller.undo()
    assert dirty(controller)
    controller.redo()
    assert not dirty(controller)


def test_new_project(controller: ProjectController, qtbot: QtBot) -> None:
    controller.execute(set_parameter("N", value=50))
    old = controller.project
    with qtbot.waitSignal(controller.fileChanged) as blocker:
        controller.new_project("depth_graded")
    assert blocker.args == [None]
    assert controller.project is not old
    assert controller.undo_stack.count() == 0
    assert not controller.is_dirty
    assert "Depth" in controller.state.name


def test_save_project(controller: ProjectController, tmp_path: Path, qtbot: QtBot) -> None:
    path = tmp_path / "p.grazing"
    controller.execute(set_parameter("N", value=50))
    with qtbot.waitSignal(controller.fileChanged) as blocker:
        assert controller.save_project(path)
    assert blocker.args == [path]
    assert controller.path == path
    assert Project.load(path).state == controller.state
    assert controller.save_project()


def test_save_without_path(controller: ProjectController) -> None:
    with pytest.raises(ValueError):
        controller.save_project()


def test_save_failure(controller: ProjectController, tmp_path: Path, qtbot: QtBot) -> None:
    with qtbot.waitSignal(controller.errorOccurred):
        assert not controller.save_project(tmp_path / "missing_dir" / "p.grazing")


def test_open_project(controller: ProjectController, tmp_path: Path) -> None:
    path = tmp_path / "p.grazing"
    controller.execute(set_parameter("N", value=50))
    controller.save_project(path)
    other = ProjectController()
    assert other.open_project(path)
    assert other.state == controller.state
    assert other.path == path
    assert not other.is_dirty
    assert other.undo_stack.count() == 0


def test_open_invalid(controller: ProjectController, tmp_path: Path, qtbot: QtBot) -> None:
    bad = tmp_path / "bad.grazing"
    bad.write_text("not a project")
    before = controller.project
    with qtbot.waitSignal(controller.errorOccurred):
        assert not controller.open_project(bad)
    assert controller.project is before
    with qtbot.waitSignal(controller.errorOccurred):
        assert not controller.open_project(tmp_path / "absent.grazing")
