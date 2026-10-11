"""Fixtures for the GUI tests."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from collections.abc import Callable
from pathlib import Path

import pytest
from PySide6.QtCore import QSettings
from pytestqt.qtbot import QtBot

from xstack.gui.main_window import MainWindow


@pytest.fixture
def settings(tmp_path: Path) -> QSettings:
    """Settings backed by an ini file in a temporary directory."""
    return QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)


@pytest.fixture
def make_window(qtbot: QtBot, settings: QSettings) -> Callable[[], MainWindow]:
    """Factory of main windows sharing the test settings."""

    def make() -> MainWindow:
        w = MainWindow(settings=settings, confirm_close=False)
        qtbot.addWidget(w)
        return w

    return make


@pytest.fixture
def window(make_window: Callable[[], MainWindow]) -> MainWindow:
    """A main window with a periodic default project."""
    return make_window()
