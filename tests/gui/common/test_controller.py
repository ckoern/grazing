"""Tests for the document controller and its Qt undo stack."""

import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from PySide6.QtWidgets import QApplication

from xstack.design.project import Project
from xstack.design.project.commands import rename_project, set_notes
from xstack.gui.common.controller import DocumentController


def make_controller() -> DocumentController:
    return DocumentController(Project, Project())


def assert_in_step(controller: DocumentController) -> None:
    history = controller.project.history
    undo_steps = len(history.undo_descriptions())
    redo_steps = len(history.redo_descriptions())
    assert controller.undo_stack.index() == undo_steps
    assert controller.undo_stack.count() == undo_steps + redo_steps
    assert controller.undo_stack.isClean() == history.is_at_save_point


class TestSavePoint:
    def test_edit_after_save_is_a_new_step(self, qapp: QApplication, tmp_path: Path) -> None:
        controller = make_controller()
        controller.execute(rename_project("renamed"))
        controller.execute(set_notes("one"))
        assert controller.save_project(tmp_path / "project.xstack")
        controller.execute(set_notes("two"))
        assert_in_step(controller)
        assert controller.undo_stack.count() == 3

        controller.undo()
        assert (controller.state.name, controller.state.notes) == ("renamed", "one")
        assert not controller.is_dirty
        assert_in_step(controller)
        controller.undo()
        assert (controller.state.name, controller.state.notes) == ("renamed", "")
        assert_in_step(controller)
        controller.undo()
        assert (controller.state.name, controller.state.notes) == ("untitled", "")
        assert_in_step(controller)
        controller.undo()
        assert (controller.state.name, controller.state.notes) == ("untitled", "")
        assert_in_step(controller)

        for _ in range(3):
            controller.redo()
        assert (controller.state.name, controller.state.notes) == ("renamed", "two")
        assert_in_step(controller)

    def test_edits_merge_again_after_leaving_save_point(
        self, qapp: QApplication, tmp_path: Path
    ) -> None:
        controller = make_controller()
        controller.execute(set_notes("one"))
        assert controller.save_project(tmp_path / "project.xstack")
        controller.execute(set_notes("two"))
        controller.execute(set_notes("three"))
        assert controller.undo_stack.count() == 2
        assert_in_step(controller)
        controller.undo()
        assert controller.state.notes == "one"
        assert not controller.is_dirty

    def test_save_point_lost_when_redo_steps_are_discarded(
        self, qapp: QApplication, tmp_path: Path
    ) -> None:
        controller = make_controller()
        controller.execute(set_notes("one"))
        assert controller.save_project(tmp_path / "project.xstack")
        controller.undo()
        controller.execute(set_notes("two"))
        controller.execute(set_notes("three"))
        assert controller.undo_stack.count() == 1
        assert controller.project.history.save_point is None
        assert controller.is_dirty
        assert_in_step(controller)
        controller.undo()
        assert controller.state.notes == ""
        assert controller.is_dirty

    def test_replace_project_resets_save_point(self, qapp: QApplication) -> None:
        controller = make_controller()
        controller.execute(set_notes("one"))
        controller.replace_project(Project())
        controller.execute(set_notes("two"))
        controller.execute(set_notes("three"))
        assert controller.undo_stack.count() == 1
        assert_in_step(controller)
        controller.undo()
        assert controller.state.notes == ""


@dataclass
class ReferenceStack:
    """Model of an undo stack of (name, notes) states with Qt's merge and clean rules."""

    states: list[tuple[str, str]] = field(default_factory=lambda: [("untitled", "")])
    keys: list[str] = field(default_factory=lambda: [""])
    index: int = 0
    clean: int | None = 0

    @property
    def state(self) -> tuple[str, str]:
        return self.states[self.index]

    def push(self, state: tuple[str, str], key: str) -> None:
        if state == self.state:
            return
        del self.states[self.index + 1 :]
        del self.keys[self.index + 1 :]
        if self.clean is not None and self.clean > self.index:
            self.clean = None
        if key and self.index > 0 and self.keys[self.index] == key and self.clean != self.index:
            self.states[self.index] = state
            return
        self.states.append(state)
        self.keys.append(key)
        self.index += 1

    def undo(self) -> None:
        self.index = max(self.index - 1, 0)

    def redo(self) -> None:
        self.index = min(self.index + 1, len(self.states) - 1)


operations = st.lists(
    st.one_of(
        st.tuples(st.just("notes"), st.sampled_from(["", "a", "b"])),
        st.tuples(st.just("rename"), st.sampled_from(["untitled", "x", "y"])),
        st.tuples(st.sampled_from(["undo", "redo", "save"]), st.just("")),
    ),
    max_size=25,
)


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(ops=operations)
def test_histories_stay_in_step(qapp: QApplication, ops: list[tuple[str, str]]) -> None:
    controller = make_controller()
    reference = ReferenceStack()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "project.xstack"
        for op, value in ops:
            match op:
                case "notes":
                    controller.execute(set_notes(value))
                    reference.push((reference.state[0], value), "notes")
                case "rename":
                    controller.execute(rename_project(value))
                    reference.push((value, reference.state[1]), "")
                case "undo":
                    controller.undo()
                    reference.undo()
                case "redo":
                    controller.redo()
                    reference.redo()
                case "save":
                    assert controller.save_project(path)
                    reference.clean = reference.index
            assert_in_step(controller)
            assert controller.undo_stack.index() == reference.index
            assert controller.undo_stack.count() == len(reference.states) - 1
            assert (controller.state.name, controller.state.notes) == reference.state
            assert controller.is_dirty == (reference.clean != reference.index)
