import ast
import importlib
from pathlib import Path

import pytest

import xstack

SUBPACKAGES = [
    "core",
    "core.materials",
    "core.physics",
    "core.structure",
    "core.structure.coatings",
    "core.io",
    "design",
    "design.merit",
    "design.manufacturing",
    "design.optim",
    "design.project",
]

SRC = Path(xstack.__file__).parent

FORBIDDEN = {
    "core": ("xstack.design", "xstack.fit", "xstack.gui", "PySide6", "pyqtgraph"),
    "design": ("xstack.fit", "xstack.gui", "PySide6", "pyqtgraph"),
}
"""Imports that a subpackage must not use: core is shared, the GUI sits on top."""


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_version_is_str() -> None:
    assert isinstance(xstack.__version__, str)


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_imports(name: str) -> None:
    assert importlib.import_module(f"xstack.{name}") is not None


@pytest.mark.parametrize("package", sorted(FORBIDDEN))
def test_layering(package: str) -> None:
    violations = [
        f"{path.relative_to(SRC)}: {module}"
        for path in sorted((SRC / package).rglob("*.py"))
        for module in imported_modules(path)
        if any(module == f or module.startswith(f + ".") for f in FORBIDDEN[package])
    ]
    assert not violations
