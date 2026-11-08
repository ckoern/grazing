"""Tests of file dialogs that complete file names."""

import csv
import json
from pathlib import Path

import pytest
from PySide6.QtWidgets import QFileDialog

from grazing.gui.common import files
from grazing.gui.common.files import ask_save_path, with_suffix
from grazing.gui.design.main_window import MainWindow


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("curves", "curves.csv"),
        ("curves.csv", "curves.csv"),
        ("curves.CSV", "curves.CSV"),
        ("run.1", "run.1.csv"),
        ("/tmp/dir.v2/out", "/tmp/dir.v2/out.csv"),
    ],
)
def test_with_suffix(name: str, expected: str) -> None:
    assert with_suffix(name, ".csv") == expected


def fake_dialog(monkeypatch: pytest.MonkeyPatch, chosen: str) -> None:
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", staticmethod(lambda *args, **kwargs: (chosen, ""))
    )


def test_ask_save_path_appends_suffix(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_dialog(monkeypatch, "/data/out")
    assert ask_save_path(None, "Save", "CSV files (*.csv)", ".csv") == "/data/out.csv"


def test_ask_save_path_cancelled(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_dialog(monkeypatch, "")
    assert files.ask_save_path(None, "Save", "CSV files (*.csv)", ".csv") is None


def test_reflectivity_export_appends_csv(
    monkeypatch: pytest.MonkeyPatch, window: MainWindow, tmp_path: Path
) -> None:
    fake_dialog(monkeypatch, str(tmp_path / "curves"))
    assert window.export_reflectivity()
    with (tmp_path / "curves.csv").open() as f:
        assert list(csv.DictReader(f))


@pytest.mark.parametrize("kind", ["csv", "json"])
def test_recipe_export_appends_suffix(
    monkeypatch: pytest.MonkeyPatch, window: MainWindow, tmp_path: Path, kind: str
) -> None:
    fake_dialog(monkeypatch, str(tmp_path / "recipe"))
    assert window.analyze_page.export(None, kind)
    path = tmp_path / f"recipe.{kind}"
    assert path.exists()
    if kind == "json":
        assert json.loads(path.read_text())["layers"]


def test_save_project_appends_suffix(
    monkeypatch: pytest.MonkeyPatch, window: MainWindow, tmp_path: Path
) -> None:
    fake_dialog(monkeypatch, str(tmp_path / "project"))
    assert window.save_project_as()
    assert (tmp_path / "project.grazing").exists()
