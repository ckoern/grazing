"""Tests for the project commands."""

from pathlib import Path

import pytest

from xstack import materials
from xstack.manufacturing import DepositionProcess, MaxLayers, MaxTotalThickness, MinLayerThickness
from xstack.merit import Flatness, MeanReflectivity
from xstack.optim import LmfitOptimizer, Pipeline, Stage
from xstack.parameters import free
from xstack.physics.models import KinematicModel
from xstack.project import Candidate, Command, Project, ProjectState, commands
from xstack.structure import Mirror, NLayer, Periodic, SingleLayer
from xstack.targets import EnergyScan, TargetSet
from xstack.units import deg, kev

W = materials.get("W")
SI = materials.get("Si")
B4C = materials.get("B4C")


def make_mirror() -> Mirror:
    return Mirror(
        Periodic.create(W, SI, N=20, d=free(4.0, 3.0, 5.0), gamma=free(0.4, 0.2, 0.8)), SI
    )


def make_targets() -> TargetSet:
    return TargetSet(EnergyScan(angle=deg(0.5), energy_min=kev(10), energy_max=kev(12), points=5))


def make_project() -> Project:
    return Project(
        ProjectState(
            name="demo",
            mirror=make_mirror(),
            targets=make_targets(),
            merit=MeanReflectivity(),
        )
    )


def check_undoable(project: Project, command: Command) -> ProjectState:
    """Execute a command, check it changes the state and that undo/redo restore states."""
    before = project.state
    project.execute(command)
    after = project.state
    assert after != before
    project.undo()
    assert project.state == before
    project.redo()
    assert project.state == after
    return after


class TestSimpleCommands:
    def test_rename(self) -> None:
        project = make_project()
        state = check_undoable(project, commands.rename_project("new"))
        assert state.name == "new"

    def test_set_notes_merges(self) -> None:
        project = make_project()
        project.execute(commands.set_notes("a"))
        project.execute(commands.set_notes("ab"))
        assert project.state.notes == "ab"
        project.undo()
        assert project.state.notes == ""
        assert not project.history.can_undo

    def test_set_mirror(self) -> None:
        project = Project()
        mirror = make_mirror()
        state = check_undoable(project, commands.set_mirror(mirror))
        assert state.mirror == mirror

    def test_set_coating(self) -> None:
        project = make_project()
        coating = SingleLayer.create(B4C, t=5.0)
        state = check_undoable(project, commands.set_coating(coating))
        assert state.mirror is not None
        assert state.mirror.coating == coating
        assert state.mirror.substrate == SI

    def test_set_coating_without_mirror(self) -> None:
        project = Project()
        with pytest.raises(ValueError):
            project.execute(commands.set_coating(SingleLayer.create(B4C)))
        assert project.state == ProjectState()

    def test_set_targets(self) -> None:
        project = Project()
        targets = make_targets()
        state = check_undoable(project, commands.set_targets(targets))
        assert state.targets == targets

    def test_set_merit(self) -> None:
        project = make_project()
        state = check_undoable(project, commands.set_merit(Flatness()))
        assert state.merit == Flatness()

    def test_set_process_and_remove(self) -> None:
        project = make_project()
        process = DepositionProcess({"W": 0.1, "Si": 0.2})
        state = check_undoable(project, commands.set_process(process))
        assert state.process == process
        state = check_undoable(project, commands.set_process(None))
        assert state.process is None

    def test_set_model(self) -> None:
        project = make_project()
        state = check_undoable(project, commands.set_model(KinematicModel()))
        assert state.model == KinematicModel()

    def test_set_pipeline(self) -> None:
        project = make_project()
        pipeline = Pipeline((Stage("refine", LmfitOptimizer("nelder", max_evaluations=10)),))
        state = check_undoable(project, commands.set_pipeline(pipeline))
        assert state.pipeline == pipeline
        state = check_undoable(project, commands.set_pipeline(None))
        assert state.pipeline is None


class TestParameterCommands:
    def test_set_parameter(self) -> None:
        project = make_project()
        state = check_undoable(project, commands.set_parameter("gamma", value=0.5, vary=False))
        assert state.mirror is not None
        assert state.mirror.parameters["gamma"].value == 0.5
        assert not state.mirror.parameters["gamma"].vary

    def test_set_parameter_leaves_others(self) -> None:
        project = make_project()
        assert project.state.mirror is not None
        before = project.state.mirror.parameters
        project.execute(commands.set_parameter("N", vary=True, min=10, max=40))
        assert project.state.mirror is not None
        after = project.state.mirror.parameters
        assert after["N"].vary
        assert (after["N"].min, after["N"].max) == (10, 40)
        assert after["d"] == before["d"]

    def test_set_parameter_merges_per_parameter(self) -> None:
        project = make_project()
        project.execute(commands.set_parameter("d", value=3.5))
        project.execute(commands.set_parameter("d", value=3.6))
        project.execute(commands.set_parameter("d", value=3.7))
        assert len(project.history.undo_descriptions()) == 1
        project.execute(commands.set_parameter("gamma", value=0.5))
        project.execute(commands.set_parameter("d", value=3.8))
        assert len(project.history.undo_descriptions()) == 3
        project.undo()
        project.undo()
        project.undo()
        assert project.state == make_project().state

    def test_set_parameter_without_mirror(self) -> None:
        project = Project()
        with pytest.raises(ValueError):
            project.execute(commands.set_parameter("d", value=3.0))
        assert not project.history.can_undo

    def test_set_parameter_unknown_name(self) -> None:
        project = make_project()
        with pytest.raises(KeyError):
            project.execute(commands.set_parameter("nope", value=3.0))
        assert project.state == make_project().state

    def test_set_parameters(self) -> None:
        project = make_project()
        assert project.state.mirror is not None
        params = project.state.mirror.parameters.replace("d", value=4.5)
        state = check_undoable(project, commands.set_parameters(params, "Edit d"))
        assert state.mirror is not None
        assert state.mirror.parameters["d"].value == 4.5
        assert project.history.undo_descriptions() == ("Edit d",)

    def test_set_parameters_without_mirror(self) -> None:
        project = Project()
        params = make_mirror().parameters
        with pytest.raises(ValueError):
            project.execute(commands.set_parameters(params))

    def test_n_layer_parameter(self) -> None:
        project = Project(
            ProjectState(mirror=Mirror(NLayer.create([W, SI], [free(1.0, 0.5, 2.0), 2.0]), SI))
        )
        project.execute(commands.set_parameter("t1", value=1.5))
        assert project.state.mirror is not None
        assert project.state.mirror.parameters["t1"].value == 1.5


class TestConstraintCommands:
    def test_add(self) -> None:
        project = make_project()
        state = check_undoable(project, commands.add_constraint(MaxLayers(50)))
        assert state.constraints == (MaxLayers(50),)
        state = check_undoable(project, commands.add_constraint(MaxTotalThickness(100.0)))
        assert state.constraints == (MaxLayers(50), MaxTotalThickness(100.0))

    def test_remove(self) -> None:
        project = make_project()
        project.execute(commands.add_constraint(MaxLayers(50)))
        project.execute(commands.add_constraint(MaxTotalThickness(100.0)))
        state = check_undoable(project, commands.remove_constraint(0))
        assert state.constraints == (MaxTotalThickness(100.0),)

    def test_remove_out_of_range(self) -> None:
        project = make_project()
        with pytest.raises(IndexError):
            project.execute(commands.remove_constraint(0))

    def test_replace(self) -> None:
        project = make_project()
        project.execute(commands.add_constraint(MinLayerThickness(1.0)))
        project.execute(commands.add_constraint(MaxLayers(50)))
        state = check_undoable(
            project, commands.replace_constraint(0, MinLayerThickness(2.0, severity="soft"))
        )
        assert state.constraints == (MinLayerThickness(2.0, severity="soft"), MaxLayers(50))

    def test_replace_merges_per_index(self) -> None:
        project = make_project()
        project.execute(commands.add_constraint(MinLayerThickness(1.0)))
        project.execute(commands.replace_constraint(0, MinLayerThickness(2.0)))
        project.execute(commands.replace_constraint(0, MinLayerThickness(3.0)))
        assert len(project.history.undo_descriptions()) == 2
        project.undo()
        assert project.state.constraints == (MinLayerThickness(1.0),)

    def test_replace_out_of_range(self) -> None:
        project = make_project()
        with pytest.raises(IndexError):
            project.execute(commands.replace_constraint(0, MaxLayers(5)))


class TestCandidateCommands:
    def candidate(self, name: str = "design") -> Candidate:
        return Candidate(name, make_mirror(), "manual", "now", 0.5, 0.6, notes="n")

    def test_add(self) -> None:
        project = make_project()
        candidate = self.candidate()
        state = check_undoable(project, commands.add_candidate(candidate))
        assert state.candidates == (candidate,)

    def test_add_unique_names(self) -> None:
        project = make_project()
        for _ in range(3):
            project.execute(commands.add_candidate(self.candidate()))
        names = [c.name for c in project.state.candidates]
        assert names == ["design", "design (2)", "design (3)"]
        assert project.state.candidate("design (2)").notes == "n"

    def test_add_unique_with_gap(self) -> None:
        project = make_project()
        project.execute(commands.add_candidate(self.candidate("a")))
        project.execute(commands.add_candidate(self.candidate("a (2)")))
        project.execute(commands.add_candidate(self.candidate("a")))
        assert [c.name for c in project.state.candidates] == ["a", "a (2)", "a (3)"]

    def test_remove(self) -> None:
        project = make_project()
        project.execute(commands.add_candidate(self.candidate("a")))
        project.execute(commands.add_candidate(self.candidate("b")))
        state = check_undoable(project, commands.remove_candidate("a"))
        assert [c.name for c in state.candidates] == ["b"]

    def test_remove_unknown(self) -> None:
        project = make_project()
        with pytest.raises(KeyError):
            project.execute(commands.remove_candidate("nope"))

    def test_promote(self) -> None:
        project = make_project()
        other = Mirror(SingleLayer.create(B4C, t=5.0), SI)
        project.execute(commands.add_candidate(Candidate("other", other)))
        state = check_undoable(project, commands.promote_candidate("other"))
        assert state.mirror == other

    def test_promote_unknown(self) -> None:
        project = make_project()
        with pytest.raises(KeyError):
            project.execute(commands.promote_candidate("nope"))
        assert project.state == make_project().state


def fast_pipeline() -> Pipeline:
    return Pipeline((Stage("refine", LmfitOptimizer("nelder", max_evaluations=30)),))


class TestRunPipeline:
    def test_run(self) -> None:
        project = make_project()
        initial = project.state
        result = commands.run_pipeline(project, fast_pipeline())

        assert len(result.stages) == 1
        assert result.final.objective <= project.state.problem().evaluate().objective
        # The input design is not changed; the result is stored as selected candidate.
        assert project.state.mirror == initial.mirror

        (candidate,) = project.state.candidates
        assert candidate.name == "optimized"
        assert candidate.source == "pipeline"
        assert candidate.mirror.parameters.value_dict() == result.parameters.value_dict()
        assert candidate.objective == result.final.objective
        assert candidate.merit == result.final.merit
        assert project.state.selected == "optimized"

    def test_journal(self) -> None:
        project = make_project()
        result = commands.run_pipeline(project, fast_pipeline())
        (entry,) = project.journal.entries("run")
        assert "1 stages" in entry.message
        assert entry.data["objective"] == result.final.objective
        assert entry.data["cancelled"] is False
        (stage,) = entry.data["stages"]
        assert stage["name"] == "refine"
        assert stage["evaluations"] == result.stages[0].result.evaluations
        assert len(project.journal.entries("command")) == 1

    def test_single_undo(self) -> None:
        project = make_project()
        initial = project.state
        commands.run_pipeline(project, fast_pipeline())
        assert len(project.history.undo_descriptions()) == 1
        project.undo()
        assert project.state == initial
        assert project.state.candidates == ()

    def test_uses_project_pipeline(self) -> None:
        project = make_project()
        project.execute(commands.set_pipeline(fast_pipeline()))
        result = commands.run_pipeline(project)
        assert [s.stage.name for s in result.stages] == ["refine"]

    def test_candidate_name_made_unique(self) -> None:
        project = make_project()
        commands.run_pipeline(project, fast_pipeline())
        commands.run_pipeline(project, fast_pipeline(), candidate_name="optimized")
        names = [c.name for c in project.state.candidates]
        assert names == ["optimized", "optimized (2)"]

    def test_incomplete_project(self) -> None:
        project = Project(ProjectState(mirror=make_mirror()))
        with pytest.raises(ValueError, match="targets"):
            commands.run_pipeline(project, fast_pipeline())
        assert len(project.journal) == 0


def test_rename_project_undo_and_round_trip(tmp_path: Path) -> None:
    project = Project()
    project.execute(commands.rename_project("mirror A"))
    assert project.state.name == "mirror A"
    path = tmp_path / "named.xstack"
    project.save(path)
    assert Project.load(path).state.name == "mirror A"
    project.undo()
    assert project.state.name == "untitled"
