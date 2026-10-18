"""Tests for `xstack.fit.project`."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from xstack.core import materials
from xstack.core.io.models import read_models
from xstack.core.measurement import MeasuredCurve
from xstack.core.optim.optimizers import LmfitOptimizer
from xstack.core.parameters import free
from xstack.core.physics.models import KinematicModel, ParrattModel
from xstack.core.structure import Mirror, SingleLayer
from xstack.design.plugin import INPUT_DESIGN
from xstack.design.project import Candidate, Project
from xstack.design.project.templates import new_state
from xstack.fit.dataset import Dataset
from xstack.fit.problem import FitProblem
from xstack.fit.project import (
    FILE_SUFFIX,
    SCHEMA_VERSION,
    FitProject,
    FitState,
    StoredFit,
    add_dataset,
    apply_fit,
    import_model,
    remove_dataset,
    remove_fit,
    rename_dataset,
    rename_fit,
    rename_project,
    replace_dataset,
    run_fit,
    set_coating,
    set_mirror,
    set_model,
    set_notes,
    set_optimizer,
    set_optimizers,
    set_parameter,
    set_parameters,
    set_ranges,
    set_weighting,
    store_result,
)
from xstack.fit.results import FitResult

PT, SI = materials.get("Pt"), materials.get("Si")
ANGLES = np.linspace(0.002, 0.012, 30)


def make_mirror(t: float = 30.0, start: float | None = None) -> Mirror:
    return Mirror(SingleLayer.create(PT, t=free(start if start is not None else t, 10.0, 60.0)), SI)


def make_curve() -> MeasuredCurve:
    blank = Dataset.create("x", MeasuredCurve("angle", ANGLES, np.ones(30), energy=8048.0))
    sim = FitProblem(make_mirror(), (blank,), weighting="linear").simulate(blank)
    return MeasuredCurve("angle", ANGLES, sim, 0.03 * sim + 1e-7, energy=8048.0)


def make_state(start: float = 30.0, scale: bool = False) -> FitState:
    dataset = Dataset.create("a", make_curve(), scale=free(1.0, 0.5, 2.0) if scale else 1.0)
    return FitState(name="demo", mirror=make_mirror(start=start), datasets=(dataset,))


def make_project(**kwargs: object) -> FitProject:
    return FitProject(make_state(**kwargs))  # type: ignore[arg-type]


def save_design(path: Path, with_candidate: bool = False, empty: bool = False) -> None:
    state = new_state("single_layer")
    if empty:
        state = state.evolve(mirror=None)
    mirror = state.mirror
    if with_candidate and mirror is not None:
        tuned = mirror.with_coating(mirror.coating.with_values(t=42.0))
        state = state.evolve(model=KinematicModel(), candidates=(Candidate("refined", tuned),))
    Project(state).save(path)


class TestState:
    def test_defaults(self) -> None:
        state = FitState()
        assert state.name == "untitled"
        assert state.mirror is None
        assert state.datasets == ()
        assert state.weighting == "log"
        assert state.result is None
        assert state.model == ParrattModel()
        assert state.optimizers == (LmfitOptimizer("least_squares"),)

    def test_evolve(self) -> None:
        state = FitState().evolve(name="x")
        assert state.name == "x"
        assert FitState().name == "untitled"

    def test_problem_requires_mirror(self) -> None:
        with pytest.raises(ValueError, match="no structure model"):
            FitState().problem()

    def test_problem_requires_data(self) -> None:
        with pytest.raises(ValueError, match="no measured data"):
            FitState(mirror=make_mirror()).problem()

    def test_problem(self) -> None:
        state = make_state().evolve(weighting="linear", model=KinematicModel())
        problem = state.problem()
        assert problem.weighting == "linear"
        assert problem.model == KinematicModel()
        assert problem.datasets == state.datasets

    def test_dataset_lookup(self) -> None:
        state = make_state()
        assert state.dataset("a") is state.datasets[0]
        with pytest.raises(KeyError, match="unknown dataset"):
            state.dataset("b")

    def test_file_constants(self) -> None:
        assert FILE_SUFFIX == ".xfit"
        assert FitProject.file_format == "xstack-fit"
        assert FitProject.schema_version == SCHEMA_VERSION == 1


class TestCommands:
    def test_rename(self) -> None:
        project = make_project()
        project.execute(rename_project("other"))
        assert project.state.name == "other"

    def test_notes_merge_into_one_undo_step(self) -> None:
        project = make_project()
        project.execute(set_notes("a"))
        project.execute(set_notes("ab"))
        assert project.state.notes == "ab"
        project.undo()
        assert project.state.notes == ""
        assert not project.history.can_undo

    def test_set_mirror_keeps_saved_fits_and_model(self) -> None:
        stored = StoredFit(
            "fit 1", FitResult(make_state().problem().parameters, None), make_mirror()
        )
        project = FitProject(make_state().evolve(model=KinematicModel(), results=(stored,)))
        project.execute(set_mirror(make_mirror(40.0), source="file.xstack: x"))
        assert project.state.results == (stored,)
        assert project.state.mirror == make_mirror(40.0)
        assert project.state.model == KinematicModel()
        assert project.state.model_source == "file.xstack: x"

    def test_set_mirror_without_source_keeps_source(self) -> None:
        project = FitProject(make_state().evolve(model_source="a.xstack: run 1"))
        project.execute(set_mirror(make_mirror(40.0)))
        assert project.state.model_source == "a.xstack: run 1"

    def test_set_coating(self) -> None:
        project = make_project()
        coating = SingleLayer.create(PT, t=25.0, name="thin")
        project.execute(set_coating(coating))
        assert project.state.mirror is not None
        assert project.state.mirror.coating == coating
        assert project.state.mirror.substrate == SI

    def test_set_coating_without_mirror(self) -> None:
        project = FitProject()
        with pytest.raises(ValueError, match="no structure model"):
            project.execute(set_coating(SingleLayer.create(PT)))

    def test_set_mirror_with_model(self) -> None:
        project = make_project()
        project.execute(set_mirror(make_mirror(), model=KinematicModel()))
        assert project.state.model == KinematicModel()

    def test_set_model(self) -> None:
        project = make_project()
        project.execute(set_model(KinematicModel()))
        assert project.state.model == KinematicModel()

    def test_set_weighting(self) -> None:
        project = make_project()
        project.execute(set_weighting("chi2"))
        assert project.state.weighting == "chi2"

    def test_set_optimizer(self) -> None:
        project = make_project()
        project.execute(set_optimizer(LmfitOptimizer("nelder", max_evaluations=50)))
        assert project.state.optimizers == (LmfitOptimizer("nelder", max_evaluations=50),)

    def test_set_optimizers(self) -> None:
        project = make_project()
        stages = (LmfitOptimizer("nelder"), LmfitOptimizer("least_squares"))
        project.execute(set_optimizers(stages))
        assert project.state.optimizers == stages
        with pytest.raises(ValueError, match="at least one"):
            project.execute(set_optimizers(()))

    def test_noop_not_recorded(self) -> None:
        project = make_project()
        project.execute(rename_project("demo"))
        assert not project.history.can_undo


class TestImportModel:
    def test_input_design(self, tmp_path: Path) -> None:
        path = tmp_path / "mirror.xstack"
        save_design(path)
        project = make_project()
        project.execute(import_model(path))
        assert project.state.mirror == read_models(path)[0].mirror
        assert project.state.model_source == f"mirror.xstack: {INPUT_DESIGN}"
        assert project.state.result is None

    def test_candidate_by_name(self, tmp_path: Path) -> None:
        path = tmp_path / "mirror.xstack"
        save_design(path, with_candidate=True)
        project = make_project()
        project.execute(import_model(path, "refined"))
        assert project.state.mirror is not None
        assert project.state.mirror.parameters["t"].value == 42.0
        assert project.state.model == KinematicModel()
        assert project.state.model_source == "mirror.xstack: refined"

    def test_default_is_first_model(self, tmp_path: Path) -> None:
        path = tmp_path / "mirror.xstack"
        save_design(path, with_candidate=True)
        project = make_project()
        project.execute(import_model(path))
        assert project.state.mirror is not None
        assert project.state.mirror.parameters["t"].value == 30.0

    def test_unknown_name(self, tmp_path: Path) -> None:
        path = tmp_path / "mirror.xstack"
        save_design(path, with_candidate=True)
        with pytest.raises(ValueError, match=r"no model 'nope'.*available.*'refined'"):
            import_model(path, "nope")

    def test_file_without_model(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.xstack"
        save_design(path, empty=True)
        with pytest.raises(ValueError, match="contains no structure model"):
            import_model(path)

    def test_unsupported_file(self, tmp_path: Path) -> None:
        path = tmp_path / "x.unknown"
        path.write_text("")
        with pytest.raises(ValueError):
            import_model(path)

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(OSError):
            import_model(tmp_path / "missing.xstack")


class TestDatasetCommands:
    def other(self, name: str = "b") -> Dataset:
        return Dataset.create(name, make_curve())

    def test_add(self) -> None:
        project = make_project()
        project.execute(add_dataset(self.other()))
        assert [d.name for d in project.state.datasets] == ["a", "b"]
        project.undo()
        assert [d.name for d in project.state.datasets] == ["a"]

    def test_add_duplicate(self) -> None:
        project = make_project()
        with pytest.raises(ValueError, match="already exists"):
            project.execute(add_dataset(self.other("a")))
        assert len(project.state.datasets) == 1

    def test_remove(self) -> None:
        project = make_project()
        project.execute(add_dataset(self.other()))
        project.execute(remove_dataset("a"))
        assert [d.name for d in project.state.datasets] == ["b"]

    def test_remove_unknown(self) -> None:
        with pytest.raises(KeyError, match="unknown dataset"):
            make_project().execute(remove_dataset("zzz"))

    def test_replace(self) -> None:
        project = make_project()
        project.execute(add_dataset(self.other()))
        new = replace(self.other("b"), weight=3.0)
        project.execute(replace_dataset(new, "Reweight"))
        assert project.state.datasets[1].weight == 3.0
        assert project.state.datasets[0].weight == 1.0
        assert project.history.undo_descriptions()[-1] == "Reweight"

    def test_replace_default_description_and_unknown(self) -> None:
        assert replace_dataset(self.other()).description == "Edit dataset 'b'"
        with pytest.raises(KeyError):
            make_project().execute(replace_dataset(self.other("b")))

    def test_set_ranges(self) -> None:
        project = make_project()
        project.execute(set_ranges("a", [(0.003, 0.006)], [(0.004, 0.0045)]))
        data = project.state.dataset("a")
        assert data.ranges == ((0.003, 0.006),)
        assert data.excluded == ((0.004, 0.0045),)

    def test_set_ranges_keeps_excluded_when_none(self) -> None:
        project = make_project()
        project.execute(set_ranges("a", [], [(0.004, 0.0045)]))
        project.execute(set_ranges("a", [(0.003, 0.006)]))
        assert project.state.dataset("a").excluded == ((0.004, 0.0045),)

    def test_set_ranges_merge(self) -> None:
        project = make_project()
        project.execute(set_ranges("a", [(0.003, 0.006)]))
        project.execute(set_ranges("a", [(0.003, 0.007)]))
        project.undo()
        assert project.state.dataset("a").ranges == ()

    def test_set_ranges_invalid_and_unknown(self) -> None:
        project = make_project()
        with pytest.raises(ValueError, match="invalid range"):
            project.execute(set_ranges("a", [(0.01, 0.0)]))
        with pytest.raises(KeyError):
            project.execute(set_ranges("b", []))


class TestParameterCommands:
    def test_coating_parameter(self) -> None:
        project = make_project()
        project.execute(set_parameter("sigma", vary=True, min=0.0, max=2.0))
        assert project.state.mirror is not None
        assert project.state.mirror.parameters["sigma"].is_free
        assert project.state.problem().parameters.free_names() == ("t", "sigma")

    def test_instrument_parameter(self) -> None:
        project = make_project()
        project.execute(set_parameter("a.scale", vary=True, min=0.5, max=2.0))
        assert project.state.dataset("a").instrument.parameters["scale"].is_free
        assert project.state.problem().parameters.free_names() == ("t", "a.scale")

    def test_expression_within_instrument(self) -> None:
        project = make_project()
        project.execute(set_parameter("a.background", expr="scale * 0.001"))
        problem = project.state.problem()
        assert problem.parameters["a.background"].expr == "a.scale * 0.001"
        assert problem.resolve()["a.background"] == pytest.approx(0.001)

    def test_consecutive_edits_merge(self) -> None:
        project = make_project()
        project.execute(set_parameter("t", value=31.0))
        project.execute(set_parameter("t", value=32.0))
        project.execute(set_parameter("a.offset", value=0.001, min=-1.0, max=1.0))
        project.undo()
        assert project.state.mirror is not None
        assert project.state.mirror.parameters["t"].value == 32.0
        project.undo()
        assert project.state.mirror.parameters["t"].value == 30.0
        project.redo()
        assert project.state.mirror.parameters["t"].value == 32.0

    def test_unknown_parameters(self) -> None:
        project = make_project()
        with pytest.raises(KeyError):
            project.execute(set_parameter("nothing", value=1.0))
        with pytest.raises(KeyError):
            project.execute(set_parameter("a.nothing", value=1.0))
        with pytest.raises(KeyError):
            project.execute(set_parameter("zzz.scale", value=1.0))

    def test_invalid_value(self) -> None:
        with pytest.raises(ValueError):
            make_project().execute(set_parameter("t", value=1000.0))

    def test_requires_mirror(self) -> None:
        with pytest.raises(ValueError, match="no structure model"):
            FitProject().execute(set_parameter("t", value=1.0))

    def test_set_parameters(self) -> None:
        project = make_project(scale=True)
        parameters = project.state.problem().parameters.with_values({"t": 35.0, "a.scale": 1.2})
        project.execute(set_parameters(parameters, "Restore"))
        assert project.state.problem().parameters == parameters
        assert project.state.dataset("a").instrument.parameters["scale"].value == 1.2
        assert project.history.undo_descriptions()[-1] == "Restore"
        project.undo()
        assert project.state.dataset("a").instrument.parameters["scale"].value == 1.0


class TestStoreResult:
    def make_result(self, project: FitProject) -> FitResult:
        parameters = project.state.problem().parameters.with_values({"t": 35.0})
        return FitResult(parameters, None, method="manual")

    def test_apply(self) -> None:
        project = make_project()
        result = self.make_result(project)
        project.execute(store_result(result))
        assert project.state.result == result
        assert project.state.mirror is not None
        assert project.state.mirror.parameters["t"].value == 35.0

    def test_without_apply(self) -> None:
        project = make_project()
        result = self.make_result(project)
        project.execute(store_result(result, apply=False))
        assert project.state.result == result
        assert project.state.mirror is not None
        assert project.state.mirror.parameters["t"].value == 30.0

    def test_undo(self) -> None:
        project = make_project()
        project.execute(store_result(self.make_result(project)))
        project.undo()
        assert project.state.result is None
        assert project.state.mirror is not None
        assert project.state.mirror.parameters["t"].value == 30.0

    def test_description_names_method(self) -> None:
        assert (
            "manual"
            in store_result(
                FitResult(make_state().problem().parameters, None, "manual")
            ).description
        )


class TestSavedFits:
    def store(self, project: FitProject, t: float, name: str | None = None) -> FitResult:
        result = FitResult(
            project.state.problem().parameters.with_values({"t": t}), None, method="manual"
        )
        project.execute(store_result(result, name=name))
        return result

    def test_names_count_up(self) -> None:
        project = make_project()
        self.store(project, 35.0)
        self.store(project, 36.0)
        assert [f.name for f in project.state.results] == ["fit 1", "fit 2"]
        assert project.state.result is project.state.results[-1].result

    def test_name_skips_used_names(self) -> None:
        project = make_project()
        self.store(project, 35.0, name="fit 2")
        self.store(project, 36.0)
        assert [f.name for f in project.state.results] == ["fit 2", "fit 3"]

    def test_duplicate_name_rejected(self) -> None:
        project = make_project()
        self.store(project, 35.0, name="best")
        with pytest.raises(ValueError, match="already exists"):
            self.store(project, 36.0, name="best")

    def test_stored_fit_keeps_fitted_structure_and_model(self) -> None:
        project = FitProject(make_state().evolve(model=KinematicModel()))
        self.store(project, 35.0)
        stored = project.state.stored("fit 1")
        assert stored.mirror.parameters["t"].value == 35.0
        assert stored.model == KinematicModel()

    def test_apply_restores_structure_model_and_instrument(self) -> None:
        project = make_project(scale=True)
        parameters = project.state.problem().parameters.with_values({"t": 35.0, "a.scale": 1.5})
        project.execute(store_result(FitResult(parameters, None)))
        project.execute(set_parameter("t", value=20.0))
        project.execute(set_parameter("a.scale", value=1.0))
        project.execute(set_model(KinematicModel()))
        project.execute(apply_fit("fit 1"))
        state = project.state
        assert state.mirror is not None
        assert state.mirror.parameters["t"].value == 35.0
        assert state.dataset("a").instrument.parameters["scale"].value == 1.5
        assert state.model == ParrattModel()

    def test_apply_restores_changed_coating(self) -> None:
        project = make_project()
        self.store(project, 35.0)
        project.execute(set_coating(SingleLayer.create(PT, t=20.0, name="other")))
        project.execute(apply_fit("fit 1"))
        assert project.state.mirror is not None
        assert project.state.mirror.coating.name == "layer"

    def test_rename_and_remove(self) -> None:
        project = make_project()
        self.store(project, 35.0)
        self.store(project, 36.0)
        project.execute(rename_fit("fit 1", "first"))
        assert [f.name for f in project.state.results] == ["first", "fit 2"]
        with pytest.raises(ValueError, match="already exists"):
            project.execute(rename_fit("first", "fit 2"))
        with pytest.raises(ValueError, match="empty"):
            project.execute(rename_fit("first", " "))
        project.execute(remove_fit("first"))
        assert [f.name for f in project.state.results] == ["fit 2"]
        project.undo()
        assert [f.name for f in project.state.results] == ["first", "fit 2"]

    def test_unknown_saved_fit(self) -> None:
        project = make_project()
        for command in (apply_fit("x"), rename_fit("x", "y"), remove_fit("x")):
            with pytest.raises(KeyError, match="unknown saved fit"):
                project.execute(command)

    def test_saved_fits_survive_save_and_load(self, tmp_path: Path) -> None:
        project = make_project()
        self.store(project, 35.0)
        project.save(tmp_path / "p.xfit")
        assert FitProject.load(tmp_path / "p.xfit").state.results == project.state.results


class TestRunStages:
    def test_two_stages_start_from_previous_best(self) -> None:
        project = make_project(start=25.0)
        project.execute(
            set_optimizers(
                (
                    LmfitOptimizer("nelder", max_evaluations=10),
                    LmfitOptimizer("least_squares"),
                )
            )
        )
        result = run_fit(project.state)
        assert result.method == "nelder + least_squares"
        assert result.parameters["t"].value == pytest.approx(30.0, abs=1e-3)
        assert result.evaluations > 10
        assert result.statistics is not None


class TestRunFit:
    def test_fits_thickness(self) -> None:
        project = make_project(start=34.0)
        state = project.state.evolve(weighting="chi2")
        result = run_fit(state)
        assert result.method == "least_squares"
        assert result.parameters["t"].value == pytest.approx(30.0, abs=0.05)
        assert result.statistics is not None
        assert result.statistics.names == ("t",)
        assert result.statistics.determined
        # The state is unchanged until the result is stored.
        assert state.mirror is not None
        assert state.mirror.parameters["t"].value == 34.0

    @pytest.mark.parametrize("weighting", ["log", "linear"])
    def test_other_weightings(self, weighting: str) -> None:
        state = make_state(start=33.0).evolve(weighting=weighting)
        result = run_fit(state)
        assert result.parameters["t"].value == pytest.approx(30.0, abs=0.05)

    def test_store_after_run(self) -> None:
        project = make_project(start=34.0)
        result = run_fit(project.state)
        project.execute(store_result(result))
        assert project.state.mirror is not None
        assert project.state.mirror.parameters["t"].value == pytest.approx(30.0, abs=0.05)
        assert project.state.result is result

    def test_invalid_state(self) -> None:
        with pytest.raises(ValueError):
            run_fit(FitState())

    def test_nothing_free(self) -> None:
        state = make_state()
        mirror = state.mirror
        assert mirror is not None
        fixed = mirror.with_coating(mirror.coating.with_parameter("t", vary=False))
        with pytest.raises(ValueError):
            run_fit(state.evolve(mirror=fixed))


class TestFiles:
    def test_round_trip_with_result(self, tmp_path: Path) -> None:
        project = make_project(start=33.0, scale=True)
        project.execute(set_ranges("a", [(0.003, 0.011)], [(0.006, 0.0065)]))
        project.execute(set_notes("hello"))
        project.execute(store_result(run_fit(project.state)))
        assert project.state.result is not None
        assert project.state.result.statistics is not None
        path = tmp_path / f"demo{FILE_SUFFIX}"
        project.save(path)
        loaded = FitProject.load(path)
        assert loaded.state == project.state
        assert loaded.state.result == project.state.result
        assert not loaded.history.can_undo

    def test_round_trip_energy_scan(self, tmp_path: Path) -> None:
        curve = MeasuredCurve("energy", [7000.0, 8000.0], [0.9, 0.8], angle=0.006, name="e")
        state = make_state().evolve(
            datasets=(Dataset.create("e", curve, offset=free(0.0, -5.0, 5.0)),),
            optimizers=(LmfitOptimizer("nelder", max_evaluations=10),),
            model=KinematicModel(),
            weighting="linear",
        )
        path = tmp_path / "e.xfit"
        FitProject(state).save(path)
        assert FitProject.load(path).state == state

    def test_rejects_design_file(self, tmp_path: Path) -> None:
        path = tmp_path / "design.xstack"
        save_design(path)
        with pytest.raises(ValueError, match="not an xstack fit project file"):
            FitProject.load(path)

    def test_rejects_other_version(self, tmp_path: Path) -> None:
        data = make_project().to_data()
        data["schema_version"] = SCHEMA_VERSION + 1
        with pytest.raises(ValueError, match="version"):
            FitProject.from_data(data)


class TestRenameDataset:
    def test_rename_updates_expressions(self) -> None:
        state = make_state()
        b = Dataset.create("b", make_curve())
        b = b.with_parameters(b.parameters.replace("b.offset", expr="a.offset"))
        project = FitProject(state.evolve(datasets=(*state.datasets, b)))
        project.execute(rename_dataset("a", "cu"))
        names = [d.name for d in project.state.datasets]
        assert names == ["cu", "b"]
        assert project.state.dataset("b").parameters["b.offset"].expr == "cu.offset"
        assert "cu.scale" in project.state.problem().parameters

    def test_rename_keeps_unrelated_names(self) -> None:
        state = make_state()
        b = Dataset.create("b", make_curve())
        b = b.with_parameters(b.parameters.replace("b.offset", expr="aa.offset + a.offset"))
        aa = Dataset.create("aa", make_curve())
        project = FitProject(state.evolve(datasets=(*state.datasets, aa, b)))
        project.execute(rename_dataset("a", "x"))
        assert project.state.dataset("b").parameters["b.offset"].expr == "aa.offset + x.offset"

    def test_rename_errors(self) -> None:
        state = make_state()
        project = FitProject(
            state.evolve(datasets=(*state.datasets, Dataset.create("b", make_curve())))
        )
        with pytest.raises(ValueError, match="already exists"):
            project.execute(rename_dataset("a", "b"))
        with pytest.raises(KeyError):
            project.execute(rename_dataset("zz", "c"))
        with pytest.raises(ValueError):
            project.execute(rename_dataset("a", "1x"))
