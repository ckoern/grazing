"""Tests of the `remove_candidates` command."""

import pytest

from tests.design.project.test_commands import SI, W, make_project
from xstack.core.structure import Mirror, Periodic
from xstack.design.project import Candidate, Project, commands


def make_candidate_project(*names: str) -> Project:
    """Project with one candidate per name, nothing selected and an empty history."""
    mirror = Mirror(Periodic.create(W, SI, N=10, d=4.5, gamma=0.3), SI)
    candidates = tuple(Candidate(n, mirror, source="manual") for n in names)
    return Project(make_project().state.evolve(candidates=candidates))


def names(project: Project) -> list[str]:
    return [c.name for c in project.state.candidates]


def test_removes_several_candidates_in_one_step() -> None:
    project = make_candidate_project("a", "b", "c", "d")
    project.execute(commands.remove_candidates(["a", "c"]))
    assert names(project) == ["b", "d"]
    assert len(project.history.undo_descriptions()) == 1


def test_undo_restores_all_removed_candidates() -> None:
    project = make_candidate_project("a", "b", "c")
    before = project.state
    project.execute(commands.remove_candidates(["a", "b"]))
    project.undo()
    assert project.state == before
    assert names(project) == ["a", "b", "c"]


def test_clears_selection_when_selected_candidate_is_removed() -> None:
    project = make_candidate_project("a", "b")
    project.execute(commands.select_candidate("b"))
    project.execute(commands.remove_candidates(["a", "b"]))
    assert project.state.selected is None
    assert names(project) == []


def test_keeps_selection_when_selected_candidate_is_kept() -> None:
    project = make_candidate_project("a", "b", "c")
    project.execute(commands.select_candidate("b"))
    project.execute(commands.remove_candidates(["a", "c"]))
    assert project.state.selected == "b"
    assert names(project) == ["b"]


def test_unknown_name_raises_and_leaves_state_unchanged() -> None:
    project = make_candidate_project("a", "b")
    project.execute(commands.select_candidate("a"))
    before = project.state
    with pytest.raises(KeyError):
        project.execute(commands.remove_candidates(["b", "missing"]))
    assert project.state == before
    assert names(project) == ["a", "b"]


def test_description_for_one_candidate() -> None:
    assert commands.remove_candidates(["x"]).description == "Delete candidate 'x'"


def test_description_for_several_candidates() -> None:
    assert commands.remove_candidates(["a", "b", "c"]).description == "Delete 3 candidates"
    assert commands.remove_candidates(["a", "b"]).description == "Delete 2 candidates"
