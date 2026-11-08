"""The optical-constants provider as an option of the fit project."""

from pathlib import Path

import numpy as np
import pytest

from grazing.core.materials import provider_library
from grazing.fit import project as fit_project
from grazing.fit.project import FitProject
from tests.gui.fit.helpers import make_state


def test_fit_problem_uses_provider(tmp_path: Path) -> None:
    project = FitProject(make_state())
    dataset = project.state.datasets[0]
    default = project.state.problem().simulate(dataset)
    project.execute(fit_project.set_optical_constants("chantler"))
    chantler = project.state.problem().simulate(dataset)
    assert project.state.problem().library is provider_library("chantler")
    assert not np.allclose(default, chantler, rtol=1e-6)
    path = tmp_path / "f.gfit"
    project.save(path)
    loaded = FitProject.load(path)
    assert loaded.state.optical_constants == "chantler"
    np.testing.assert_allclose(loaded.state.problem().simulate(dataset), chantler)


def test_unknown_provider_rejected() -> None:
    with pytest.raises(KeyError):
        fit_project.set_optical_constants("nonexistent")
