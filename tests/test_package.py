import importlib

import pytest

import xstack

SUBPACKAGES = [
    "materials",
    "physics",
    "structure",
    "structure.coatings",
    "merit",
    "manufacturing",
    "optim",
    "project",
    "io",
]


def test_version_is_str() -> None:
    assert isinstance(xstack.__version__, str)


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_imports(name: str) -> None:
    assert importlib.import_module(f"xstack.{name}") is not None
