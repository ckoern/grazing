"""Tests of editing the project name in the main window."""

import pytest
from PySide6.QtWidgets import QInputDialog

from grazing.gui.design.main_window import MainWindow

pytestmark = pytest.mark.gui


def test_edit_name_field(window: MainWindow) -> None:
    window.name_edit.setText("  My mirror ")
    window.name_edit.editingFinished.emit()
    assert window.controller.state.name == "My mirror"
    assert window.name_edit.text() == "My mirror"
    assert "My mirror" in window.windowTitle()
    assert window.isWindowModified()
    window.controller.undo()
    assert window.name_edit.text() != "My mirror"


def test_empty_name_is_rejected(window: MainWindow) -> None:
    old = window.controller.state.name
    window.name_edit.setText("   ")
    window.name_edit.editingFinished.emit()
    assert window.controller.state.name == old
    assert window.name_edit.text() == old
    assert not window.isWindowModified()


def test_rename_action(window: MainWindow, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("Renamed", True))
    window.rename_action.trigger()
    assert window.controller.state.name == "Renamed"
    assert window.name_edit.text() == "Renamed"


def test_rename_action_cancelled(window: MainWindow, monkeypatch: pytest.MonkeyPatch) -> None:
    old = window.controller.state.name
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("Other", False))
    window.rename_action.trigger()
    assert window.controller.state.name == old
