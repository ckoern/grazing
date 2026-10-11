"""Tests for the command-line interface."""

from pathlib import Path

import pytest

from xstack import materials
from xstack.__main__ import main
from xstack.merit import MeanReflectivity
from xstack.optim import LmfitOptimizer, Pipeline, Stage
from xstack.parameters import free
from xstack.project import Project, ProjectState, commands
from xstack.structure import Mirror, Periodic
from xstack.targets import EnergyScan, TargetSet
from xstack.units import deg, kev

W = materials.get("W")
SI = materials.get("Si")


def make_project(complete: bool = True) -> Project:
    mirror = Mirror(
        Periodic.create(W, SI, N=20, d=free(4.0, 3.0, 5.0), gamma=free(0.4, 0.2, 0.8)), SI
    )
    targets = TargetSet(
        EnergyScan(angle=deg(0.5), energy_min=kev(10), energy_max=kev(12), points=5)
    )
    project = Project(ProjectState(name="cli demo", mirror=mirror, merit=MeanReflectivity()))
    if complete:
        project.execute(commands.set_targets(targets))
    project.execute(
        commands.set_pipeline(
            Pipeline((Stage("refine", LmfitOptimizer("nelder", max_evaluations=30)),))
        )
    )
    return project


@pytest.fixture
def project_file(tmp_path: Path) -> Path:
    path = tmp_path / "demo.xstack"
    make_project().save(path)
    return path


def test_run_writes_output(
    project_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "out.xstack"
    assert main(["run", str(project_file), "--output", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "refine" in printed
    assert "final objective" in printed

    result = Project.load(out)
    (candidate,) = result.state.candidates
    assert candidate.source == "pipeline"
    assert len(result.journal.entries("run")) == 1
    assert Project.load(project_file).state.candidates == ()


def test_run_overwrites_input_by_default(project_file: Path) -> None:
    assert main(["run", str(project_file)]) == 0
    assert len(Project.load(project_file).state.candidates) == 1


def test_run_verbose(
    project_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "out.xstack"
    assert main(["run", str(project_file), "--output", str(out), "--verbose"]) == 0
    assert "evaluations" in capsys.readouterr().out


def test_info(project_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["info", str(project_file)]) == 0
    printed = capsys.readouterr().out
    assert "cli demo" in printed
    assert "periodic" in printed
    assert "targets: 1" in printed
    assert "candidates: none" in printed


def test_missing_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["run", str(tmp_path / "missing.xstack")]) == 1
    captured = capsys.readouterr()
    assert captured.err.startswith("error:")
    assert "missing.xstack" in captured.err
    assert main(["info", str(tmp_path / "missing.xstack")]) == 1


def test_incomplete_project(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "incomplete.xstack"
    make_project(complete=False).save(path)
    assert main(["run", str(path)]) == 1
    assert "targets" in capsys.readouterr().err


def test_invalid_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "bad.xstack"
    path.write_text('{"format": "other"}')
    assert main(["info", str(path)]) == 1
    assert "error:" in capsys.readouterr().err


def test_no_command() -> None:
    with pytest.raises(SystemExit) as info:
        main([])
    assert info.value.code == 2
