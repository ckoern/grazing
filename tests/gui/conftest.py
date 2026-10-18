"""Fixtures for the GUI tests."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication
from pytestqt.qtbot import QtBot

from xstack.gui.common.garbage import GarbageCollector
from xstack.gui.design.main_window import MainWindow


@pytest.fixture(scope="session", autouse=True)
def gui_thread_garbage_collection(qapp: QApplication) -> Iterator[None]:
    """Collect garbage in the GUI thread only, as the applications do."""
    collector = GarbageCollector(parent=qapp)
    yield
    collector.stop()


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
