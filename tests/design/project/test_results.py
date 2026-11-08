"""Tests for candidate selection and storing optimization results."""

import pytest

from grazing.core.optim import LmfitOptimizer
from grazing.core.structure import Mirror, Periodic
from grazing.design.optim import Pipeline, Stage
from grazing.design.project import Candidate, Project, ProjectState, commands
from tests.design.project.test_commands import SI, W, fast_pipeline, make_project


def make_other_mirror() -> Mirror:
    return Mirror(Periodic.create(W, SI, N=10, d=4.5, gamma=0.3), SI)


def make_candidate_project(*names: str) -> Project:
    """Project with one candidate per name, nothing selected and an empty history."""
    candidates = tuple(Candidate(n, make_other_mirror(), source="manual") for n in names)
    return Project(make_project().state.evolve(candidates=candidates))


class TestSelection:
    def test_select_and_clear(self) -> None:
        project = make_candidate_project("a")
        project.execute(commands.select_candidate("a"))
        assert project.state.selected == "a"
        project.execute(commands.select_candidate(None))
        assert project.state.selected is None

    def test_select_unknown_raises(self) -> None:
        project = make_candidate_project("a")
        with pytest.raises(KeyError):
            project.execute(commands.select_candidate("missing"))
        assert project.state.selected is None

    def test_consecutive_selections_merge(self) -> None:
        project = make_candidate_project("a", "b")
        project.execute(commands.select_candidate("a"))
        project.execute(commands.select_candidate("b"))
        assert project.state.selected == "b"
        assert len(project.history.undo_descriptions()) == 1
        project.undo()
        assert project.state.selected is None

    def test_selected_candidate(self) -> None:
        project = make_candidate_project("a")
        assert project.state.selected_candidate() is None
        project.execute(commands.select_candidate("a"))
        selected = project.state.selected_candidate()
        assert selected is not None
        assert selected.name == "a"
        assert selected == project.state.candidate("a")

    def test_selected_candidate_none_when_name_missing(self) -> None:
        state = ProjectState(selected="gone")
        assert state.selected_candidate() is None


class TestRenameAndRemove:
    def test_rename_keeps_selection(self) -> None:
        project = make_candidate_project("a", "b")
        project.execute(commands.select_candidate("a"))
        project.execute(commands.rename_candidate("a", "renamed"))
        names = [c.name for c in project.state.candidates]
        assert names == ["renamed", "b"]
        assert project.state.selected == "renamed"
        assert project.state.selected_candidate() is not None

    def test_rename_unselected_keeps_selection(self) -> None:
        project = make_candidate_project("a", "b")
        project.execute(commands.select_candidate("b"))
        project.execute(commands.rename_candidate("a", "renamed"))
        assert project.state.selected == "b"

    @pytest.mark.parametrize("new_name", ["", "   "])
    def test_rename_empty_raises(self, new_name: str) -> None:
        project = make_candidate_project("a")
        with pytest.raises(ValueError):
            project.execute(commands.rename_candidate("a", new_name))
        assert [c.name for c in project.state.candidates] == ["a"]

    def test_rename_duplicate_raises(self) -> None:
        project = make_candidate_project("a", "b")
        with pytest.raises(ValueError):
            project.execute(commands.rename_candidate("a", "b"))
        assert [c.name for c in project.state.candidates] == ["a", "b"]

    def test_rename_unknown_raises(self) -> None:
        project = make_candidate_project("a")
        with pytest.raises(KeyError):
            project.execute(commands.rename_candidate("missing", "new"))

    def test_remove_selected_clears_selection(self) -> None:
        project = make_candidate_project("a", "b")
        project.execute(commands.select_candidate("a"))
        project.execute(commands.remove_candidate("a"))
        assert project.state.selected is None
        assert [c.name for c in project.state.candidates] == ["b"]


class TestPromote:
    def test_promote_copies_mirror(self) -> None:
        project = make_candidate_project("a")
        candidate = project.state.candidate("a")
        assert project.state.mirror != candidate.mirror
        project.execute(commands.promote_candidate("a"))
        assert project.state.mirror == candidate.mirror


class TestRunPipelineResults:
    def test_run_pipeline_keeps_input_and_selects_candidate(self) -> None:
        project = make_project()
        initial_mirror = project.state.mirror
        commands.run_pipeline(project, fast_pipeline())
        assert project.state.mirror == initial_mirror
        assert [c.name for c in project.state.candidates] == ["optimized"]
        assert project.state.selected == "optimized"

    def test_run_pipeline_single_undo(self) -> None:
        project = make_project()
        initial = project.state
        commands.run_pipeline(project, fast_pipeline())
        project.undo()
        assert project.state == initial


class TestStorePipelineResult:
    def test_multi_stage_run_stores_only_final(self) -> None:
        project = make_project()
        problem = project.state.problem()
        pipeline = Pipeline(
            (
                Stage("coarse", LmfitOptimizer("nelder", max_evaluations=10)),
                Stage("fine", LmfitOptimizer("nelder", max_evaluations=10)),
            )
        )
        result = pipeline.run(problem)
        project.execute(commands.store_pipeline_result(problem, result, "run"))
        assert [c.name for c in project.state.candidates] == ["run"]
        assert project.state.candidate("run").source == "pipeline"
        assert project.state.selected == "run"

    def test_single_stage_run_stores_its_result(self) -> None:
        project = make_project()
        problem = project.state.problem()
        pipeline = Pipeline((Stage("only", LmfitOptimizer("nelder", max_evaluations=10)),))
        result = pipeline.run(problem)
        project.execute(commands.store_pipeline_result(problem, result, "run"))
        assert [c.name for c in project.state.candidates] == ["run"]
        assert project.state.candidate("run").source == "pipeline"

    def test_evaluate_stages_one_per_stage(self) -> None:
        problem = make_project().state.problem()
        result = fast_pipeline().run(problem)
        evaluations = commands.evaluate_stages(problem, result)
        assert len(evaluations) == 1


class TestStoreResults:
    def test_names_made_unique(self) -> None:
        project = make_project()
        problem = project.state.problem()
        params = problem.parameters
        evaluation = problem.evaluate()
        project.execute(
            commands.store_results(
                problem,
                [
                    ("x", params, evaluation, "manual"),
                    ("x", params, evaluation, "manual"),
                ],
            )
        )
        assert [c.name for c in project.state.candidates] == ["x", "x (2)"]
        assert project.state.selected == "x (2)"
