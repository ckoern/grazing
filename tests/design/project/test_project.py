"""Tests for the project state, undo history and persistence."""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from xstack.core import materials
from xstack.core.parameters import free
from xstack.core.structure import Mirror, Periodic
from xstack.core.targets import EnergyScan, TargetSet
from xstack.core.units import deg, kev
from xstack.design.manufacturing.process import DepositionProcess
from xstack.design.merit import MeanReflectivity
from xstack.design.optim import DesignProblem, Pipeline
from xstack.design.project import (
    SCHEMA_VERSION,
    Candidate,
    Command,
    History,
    Project,
    ProjectState,
)

W = materials.get("W")
SI = materials.get("Si")


def make_mirror() -> Mirror:
    return Mirror(Periodic.create(W, SI, N=20, d=free(4.0, 3.0, 5.0), gamma=0.4), SI)


def make_targets() -> TargetSet:
    return TargetSet(EnergyScan(angle=deg(0.5), energy_min=kev(10), energy_max=kev(12), points=5))


def rename(name: str) -> Command:
    return Command(f"rename {name}", lambda s: s.evolve(name=name))


def notes(text: str) -> Command:
    return Command("notes", lambda s: s.evolve(notes=text), merge_key="notes")


class TestProjectState:
    def test_defaults(self) -> None:
        state = ProjectState()
        assert state.name == "untitled"
        assert state.mirror is None
        assert state.candidates == ()

    def test_evolve(self) -> None:
        state = ProjectState().evolve(name="x")
        assert state.name == "x"
        assert ProjectState().name == "untitled"

    def test_problem_missing_parts(self) -> None:
        with pytest.raises(ValueError, match="mirror, targets, merit"):
            ProjectState().problem()
        state = ProjectState(mirror=make_mirror())
        with pytest.raises(ValueError) as info:
            state.problem()
        assert "targets, merit" in str(info.value)
        assert "mirror" not in str(info.value)
        with pytest.raises(ValueError, match="merit"):
            state.evolve(targets=make_targets()).problem()

    def test_problem(self) -> None:
        state = ProjectState(mirror=make_mirror(), targets=make_targets(), merit=MeanReflectivity())
        problem = state.problem()
        assert isinstance(problem, DesignProblem)
        assert problem.mirror == state.mirror
        assert problem.targets == state.targets
        assert problem.model == state.model

    def test_candidate_lookup(self) -> None:
        a = Candidate("a", make_mirror())
        b = Candidate("b", make_mirror())
        state = ProjectState(candidates=(a, b))
        assert state.candidate("b") is b
        with pytest.raises(KeyError):
            state.candidate("c")


class TestHistory:
    def states(self, n: int) -> list[ProjectState]:
        return [ProjectState(name=f"s{i}") for i in range(n)]

    def test_empty(self) -> None:
        history = History()
        assert not history.can_undo
        assert not history.can_redo
        with pytest.raises(IndexError):
            history.undo()
        with pytest.raises(IndexError):
            history.redo()

    def test_push_undo_redo(self) -> None:
        s = self.states(3)
        history = History()
        history.push("one", s[0], s[1])
        history.push("two", s[1], s[2])
        assert (history.can_undo, history.can_redo) == (True, False)
        assert history.undo() == ("two", s[1])
        assert (history.can_undo, history.can_redo) == (True, True)
        assert history.undo() == ("one", s[0])
        assert (history.can_undo, history.can_redo) == (False, True)
        assert history.redo() == ("one", s[1])
        assert history.redo() == ("two", s[2])
        assert (history.can_undo, history.can_redo) == (True, False)

    def test_push_clears_redo(self) -> None:
        s = self.states(4)
        history = History()
        history.push("one", s[0], s[1])
        history.undo()
        assert history.can_redo
        history.push("other", s[0], s[3])
        assert (history.can_undo, history.can_redo) == (True, False)
        assert history.undo_descriptions() == ("other",)

    def test_descriptions(self) -> None:
        s = self.states(4)
        history = History()
        for i, text in enumerate(("a", "b", "c")):
            history.push(text, s[i], s[i + 1])
        assert history.undo_descriptions() == ("a", "b", "c")
        assert history.redo_descriptions() == ()
        history.undo()
        history.undo()
        assert history.undo_descriptions() == ("a",)
        assert history.redo_descriptions() == ("b", "c")

    def test_merge_key(self) -> None:
        s = self.states(4)
        history = History()
        history.push("x1", s[0], s[1], "x")
        history.push("x2", s[1], s[2], "x")
        history.push("x3", s[2], s[3], "x")
        assert history.undo_descriptions() == ("x3",)
        assert history.undo() == ("x3", s[0])
        assert not history.can_undo
        assert history.redo() == ("x3", s[3])

    def test_merge_only_same_key(self) -> None:
        s = self.states(4)
        history = History()
        history.push("x", s[0], s[1], "x")
        history.push("y", s[1], s[2], "y")
        history.push("x again", s[2], s[3], "x")
        assert history.undo_descriptions() == ("x", "y", "x again")

    def test_empty_key_never_merges(self) -> None:
        s = self.states(3)
        history = History()
        history.push("a", s[0], s[1])
        history.push("b", s[1], s[2])
        assert len(history.undo_descriptions()) == 2

    def test_clear(self) -> None:
        s = self.states(3)
        history = History()
        history.push("a", s[0], s[1])
        history.push("b", s[1], s[2])
        history.undo()
        history.clear()
        assert not history.can_undo
        assert not history.can_redo
        assert history.undo_descriptions() == ()
        assert history.redo_descriptions() == ()


class TestProjectExecute:
    def test_execute(self) -> None:
        project = Project()
        project.execute(rename("A"))
        assert project.state.name == "A"
        assert project.history.undo_descriptions() == ("rename A",)
        entries = project.journal.entries("command")
        assert [e.message for e in entries] == ["rename A"]

    def test_initial_state_and_journal(self) -> None:
        state = ProjectState(name="start")
        project = Project(state)
        assert project.state is state
        assert len(project.journal) == 0

    def test_noop_not_recorded(self) -> None:
        project = Project(ProjectState(name="A"))
        project.execute(rename("A"))
        assert not project.history.can_undo
        assert len(project.journal) == 0

    def test_noop_does_not_notify(self) -> None:
        project = Project(ProjectState(name="A"))
        calls: list[ProjectState] = []
        project.subscribe(calls.append)
        project.execute(rename("A"))
        assert calls == []

    def test_failing_transform(self) -> None:
        project = Project()
        project.execute(rename("A"))
        calls: list[ProjectState] = []
        project.subscribe(calls.append)

        def fail(state: ProjectState) -> ProjectState:
            raise RuntimeError("boom")

        before = project.state
        with pytest.raises(RuntimeError, match="boom"):
            project.execute(Command("fail", fail))
        assert project.state is before
        assert project.history.undo_descriptions() == ("rename A",)
        assert len(project.journal) == 1
        assert calls == []

    def test_undo_redo(self) -> None:
        project = Project()
        project.execute(rename("A"))
        project.execute(rename("B"))
        project.undo()
        assert project.state.name == "A"
        project.undo()
        assert project.state.name == "untitled"
        project.redo()
        assert project.state.name == "A"
        project.redo()
        assert project.state.name == "B"
        kinds = [e.kind for e in project.journal]
        assert kinds == ["command", "command", "undo", "undo", "redo", "redo"]
        assert project.journal.entries("undo")[0].message == "Undo: rename B"
        assert project.journal.entries("redo")[0].message == "Redo: rename A"

    def test_undo_empty(self) -> None:
        project = Project()
        with pytest.raises(IndexError):
            project.undo()
        with pytest.raises(IndexError):
            project.redo()
        assert len(project.journal) == 0

    def test_new_command_clears_redo(self) -> None:
        project = Project()
        project.execute(rename("A"))
        project.undo()
        project.execute(rename("B"))
        assert not project.history.can_redo

    def test_merged_commands_single_undo(self) -> None:
        project = Project()
        project.execute(notes("a"))
        project.execute(notes("ab"))
        project.execute(notes("abc"))
        assert len(project.journal.entries("command")) == 3
        project.undo()
        assert project.state.notes == ""
        assert not project.history.can_undo

    def test_listeners(self) -> None:
        project = Project()
        calls: list[ProjectState] = []
        project.subscribe(calls.append)
        project.subscribe(lambda s: calls.append(s.evolve(notes="second")))
        project.execute(rename("A"))
        assert [c.name for c in calls] == ["A", "A"]
        assert calls[0] is project.state
        project.undo()
        assert len(calls) == 4
        assert calls[2].name == "untitled"
        project.redo()
        assert len(calls) == 6
        assert calls[4].name == "A"


names = st.text(alphabet="abc", min_size=0, max_size=3)
operations = st.lists(
    st.one_of(
        st.tuples(st.just("rename"), names),
        st.tuples(st.just("notes"), names),
    ),
    max_size=15,
)


@given(operations)
def test_undo_all_and_redo_all(ops: list[tuple[str, str]]) -> None:
    initial = ProjectState()
    project = Project(initial)
    for kind, text in ops:
        project.execute(rename(text) if kind == "rename" else notes(text))
    final = project.state
    while project.history.can_undo:
        project.undo()
    assert project.state == initial
    while project.history.can_redo:
        project.redo()
    assert project.state == final


class TestPersistence:
    def make_project(self) -> Project:
        project = Project(
            ProjectState(
                name="demo",
                mirror=make_mirror(),
                targets=make_targets(),
                merit=MeanReflectivity(),
                candidates=(Candidate("c", make_mirror(), "manual", "t", 0.1, 0.2, 3.0, "n"),),
            )
        )
        project.execute(rename("demo 2"))
        return project

    def test_to_data_is_strict_json(self) -> None:
        data = self.make_project().to_data()
        assert data["format"] == "xstack-project"
        assert data["schema_version"] == SCHEMA_VERSION
        json.dumps(data, allow_nan=False)

    def test_save_load(self, tmp_path: Path) -> None:
        project = self.make_project()
        path = tmp_path / "p.xstack"
        project.save(path)
        loaded = Project.load(path)
        assert loaded.state == project.state
        assert not loaded.history.can_undo
        assert not loaded.history.can_redo
        entries = loaded.journal.entries()
        assert [e.kind for e in entries[:2]] == ["command", "file"]
        assert entries[1].message == "Saved to p.xstack"
        assert entries[-1].kind == "file"
        assert entries[-1].message == "Loaded p.xstack"
        assert len(entries) == 3

    def test_load_in_fresh_interpreter(self, tmp_path: Path) -> None:
        project = self.make_project()
        state = project.state
        pipeline = Pipeline.default(state.problem(), seed=1)
        project = Project(
            state.evolve(pipeline=pipeline, process=DepositionProcess({"W": 0.1, "Si": 0.2}))
        )
        path = tmp_path / "fresh.xstack"
        project.save(path)
        script = (
            "import json, sys\n"
            "from xstack.design.project import Project\n"
            "print(json.dumps(Project.load(sys.argv[1]).to_data()['state']))\n"
        )
        out = subprocess.run(
            [sys.executable, "-c", script, str(path)], capture_output=True, text=True, check=True
        )
        assert json.loads(out.stdout) == json.loads(json.dumps(project.to_data()["state"]))

    def test_save_accepts_str(self, tmp_path: Path) -> None:
        project = self.make_project()
        project.save(str(tmp_path / "q.xstack"))
        assert Project.load(str(tmp_path / "q.xstack")).state == project.state

    def test_saved_file_is_json(self, tmp_path: Path) -> None:
        path = tmp_path / "p.xstack"
        self.make_project().save(path)
        data = json.loads(path.read_text())
        assert data["format"] == "xstack-project"

    def test_wrong_format(self) -> None:
        data = self.make_project().to_data()
        data["format"] = "something-else"
        with pytest.raises(ValueError, match="not an xstack project"):
            Project.from_data(data)

    def test_missing_format(self) -> None:
        with pytest.raises(ValueError):
            Project.from_data({})

    def test_wrong_schema_version(self) -> None:
        data = self.make_project().to_data()
        data["schema_version"] = SCHEMA_VERSION + 1
        with pytest.raises(ValueError, match="version"):
            Project.from_data(data)

    def test_state_of_wrong_type(self) -> None:
        data: dict[str, Any] = self.make_project().to_data()
        data["state"] = data["state"]["fields"]["mirror"]
        with pytest.raises(ValueError, match="project state"):
            Project.from_data(data)

    def test_missing_journal(self) -> None:
        data = self.make_project().to_data()
        del data["journal"]
        assert len(Project.from_data(data).journal) == 0
