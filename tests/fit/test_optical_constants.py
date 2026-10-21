"""The optical-constants provider as an option of the fit project."""

from pathlib import Path

import numpy as np
import pytest

from tests.gui.fit.helpers import make_state
from xstack.core.materials import provider_library
from xstack.fit import project as fit_project
from xstack.fit.project import FitProject


def test_fit_problem_uses_provider(tmp_path: Path) -> None:
    project = FitProject(make_state())
    dataset = project.state.datasets[0]
    default = project.state.problem().simulate(dataset)
    project.execute(fit_project.set_optical_constants("chantler"))
    chantler = project.state.problem().simulate(dataset)
    assert project.state.problem().library is provider_library("chantler")
    assert not np.allclose(default, chantler, rtol=1e-6)
    path = tmp_path / "f.xfit"
    project.save(path)
    loaded = FitProject.load(path)
    assert loaded.state.optical_constants == "chantler"
    np.testing.assert_allclose(loaded.state.problem().simulate(dataset), chantler)


def test_unknown_provider_rejected() -> None:
    with pytest.raises(KeyError):
        fit_project.set_optical_constants("nonexistent")
