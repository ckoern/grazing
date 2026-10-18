"""Smoke tests that run the example scripts with small budgets."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import matplotlib
import pytest

matplotlib.use("Agg")

EXAMPLES = Path(__file__).parents[1] / "examples"


def load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive")
def test_reflectivity_demo() -> None:
    load("reflectivity_demo").main()


def test_graded_time_penalty(tmp_path: Path) -> None:
    recipe = tmp_path / "recipe.csv"
    load("graded_time_penalty").main(["--quick", "--recipe", str(recipe)])
    assert recipe.exists()


@pytest.mark.slow
def test_coupled_angles() -> None:
    load("coupled_angles").main(["--quick"])


@pytest.mark.slow
def test_periodic_fixed_angle() -> None:
    load("periodic_fixed_angle").main(["--quick"])
