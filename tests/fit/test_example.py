"""Smoke test of the fitting example script."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

EXAMPLES = Path(__file__).parents[2] / "examples"


def load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.slow
def test_fit_reflectivity(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    example = load("fit_reflectivity")
    project = tmp_path / "fit.xfit"
    example.main(["--quick", "--project", str(project)])
    out = capsys.readouterr().out
    assert "χ²_red" in out
    assert project.exists()
