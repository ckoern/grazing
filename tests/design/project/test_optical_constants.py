"""The optical-constants provider as an option of the design project."""

from pathlib import Path

import pytest

from grazing.core.materials import DEFAULT_PROVIDER, default_library, provider_library
from grazing.design.project import Project
from grazing.design.project.commands import set_optical_constants
from grazing.design.project.templates import new_state


def test_design_default_and_command() -> None:
    project = Project(new_state("periodic"))
    assert project.state.optical_constants == DEFAULT_PROVIDER
    assert project.state.problem().library is default_library()
    reference = project.state.problem().evaluate().objective
    project.execute(set_optical_constants("chantler"))
    state = project.state
    assert state.optical_constants == "chantler"
    assert state.problem().library is provider_library("chantler")
    assert state.problem().evaluate().objective != reference
    project.undo()
    assert project.state.optical_constants == DEFAULT_PROVIDER


def test_design_save_and_load(tmp_path: Path) -> None:
    project = Project(new_state("periodic"))
    project.execute(set_optical_constants("henke_photo"))
    path = tmp_path / "p.grazing"
    project.save(path)
    assert Project.load(path).state.optical_constants == "henke_photo"


def test_design_old_data_uses_default() -> None:
    data = Project(new_state("periodic")).to_data()
    fields = data["state"]["fields"]
    assert "optical_constants" in fields
    del fields["optical_constants"]
    assert Project.from_data(data).state.optical_constants == DEFAULT_PROVIDER


def test_unknown_provider_rejected() -> None:
    with pytest.raises(KeyError):
        set_optical_constants("nonexistent")
