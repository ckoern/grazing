"""Tests for `xstack.design.plugin`."""

import subprocess
import sys
from pathlib import Path

import pytest

from xstack.core.io.models import read_models, reader_for
from xstack.core.physics.models import KinematicModel
from xstack.design.plugin import INPUT_DESIGN, XstackModelReader
from xstack.design.project import Candidate, Project
from xstack.design.project.templates import new_state


def save_project(path: Path, with_mirror: bool = True) -> Project:
    state = new_state("single_layer")
    mirror = state.mirror
    assert mirror is not None
    tuned = mirror.with_coating(mirror.coating.with_values(t=42.0))
    state = state.evolve(
        mirror=mirror if with_mirror else None,
        model=KinematicModel(),
        candidates=(Candidate("refined", tuned, source="refine"),),
    )
    project = Project(state)
    project.save(path)
    return project


def test_reads_input_design_and_candidates(tmp_path: Path) -> None:
    path = tmp_path / "mirror.xstack"
    state = save_project(path).state
    models = read_models(path)
    assert [m.name for m in models] == [INPUT_DESIGN, "refined"]
    assert models[0].mirror == state.mirror
    assert models[1].mirror == state.candidates[0].mirror
    assert all(m.model == KinematicModel() for m in models)


def test_project_without_input_design(tmp_path: Path) -> None:
    path = tmp_path / "mirror.xstack"
    save_project(path, with_mirror=False)
    assert [m.name for m in read_models(path)] == ["refined"]


def test_registered_for_xstack_suffix() -> None:
    assert isinstance(reader_for("a/b.XSTACK"), XstackModelReader)


def test_invalid_file(tmp_path: Path) -> None:
    path = tmp_path / "other.xstack"
    path.write_text('{"format": "something-else"}')
    with pytest.raises(ValueError, match="not an xstack project file"):
        read_models(path)


def test_found_through_entry_point_without_importing_design(tmp_path: Path) -> None:
    path = tmp_path / "mirror.xstack"
    save_project(path)
    script = (
        "import sys\n"
        "from xstack.core.io import read_models\n"
        "assert 'xstack.design' not in sys.modules\n"
        "print([m.name for m in read_models(sys.argv[1])])\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", script, str(path)], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == str([INPUT_DESIGN, "refined"])
