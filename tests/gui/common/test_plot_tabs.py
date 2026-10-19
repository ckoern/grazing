"""Tests for `xstack.gui.common.plot_tabs`."""

import pytest
from PySide6.QtWidgets import QLabel
from pytestqt.qtbot import QtBot

from xstack.gui.common.plot_tabs import PlotTabs

pytestmark = pytest.mark.gui


def make(qtbot: QtBot) -> tuple[PlotTabs, list[QLabel]]:
    tabs = PlotTabs("test")
    qtbot.addWidget(tabs)
    views = [QLabel(name) for name in ("a", "b", "c")]
    for view, name in zip(views, ("A", "B", "C"), strict=True):
        tabs.addTab(view, name)
    tabs.show()
    return tabs, views


def titles(tabs: PlotTabs) -> list[str]:
    return [tabs.tabText(i) for i in range(tabs.count())]


def test_detach_moves_tab_into_window(qtbot: QtBot) -> None:
    tabs, (a, b, _c) = make(qtbot)
    window = tabs.detach(b)
    assert titles(tabs) == ["A", "C"]
    assert tabs.is_detached(b)
    assert tabs.window_of(b) is window
    assert window.windowTitle() == "B — test"
    assert window.isVisible()
    assert b.parent() is window
    assert not tabs.is_detached(a)
    window.close()


def test_closing_window_reattaches_at_original_position(qtbot: QtBot) -> None:
    tabs, (a, b, c) = make(qtbot)
    window_b = tabs.detach(b)
    window_a = tabs.detach(a)
    assert titles(tabs) == ["C"]
    window_b.close()
    assert titles(tabs) == ["B", "C"]
    window_a.close()
    assert titles(tabs) == ["A", "B", "C"]
    assert not tabs.is_detached(a)
    assert [tabs.widget(i) for i in range(3)] == [a, b, c]


def test_detach_current_and_attach_all(qtbot: QtBot) -> None:
    tabs, (_a, _b, c) = make(qtbot)
    tabs.setCurrentWidget(c)
    window = tabs.detach_current()
    assert window is not None
    assert tabs.is_detached(c)
    tabs.detach(tabs.widget(0))
    tabs.attach_all()
    assert titles(tabs) == ["A", "B", "C"]


def test_detach_twice_returns_same_window(qtbot: QtBot) -> None:
    tabs, (a, _b, _c) = make(qtbot)
    assert tabs.detach(a) is tabs.detach(a)
    tabs.attach_all()


def test_show_view_selects_tab_or_keeps_detached(qtbot: QtBot) -> None:
    tabs, (_a, b, c) = make(qtbot)
    tabs.show_view(c)
    assert tabs.currentWidget() is c
    tabs.detach(b)
    tabs.show_view(b)
    assert tabs.currentWidget() is not b
    assert tabs.window_of(b) is not None
    tabs.attach_all()


def test_detached_changed_signal(qtbot: QtBot) -> None:
    tabs, (a, _b, _c) = make(qtbot)
    with qtbot.waitSignal(tabs.detachedChanged):
        tabs.detach(a)
    with qtbot.waitSignal(tabs.detachedChanged):
        tabs.attach(a)


def test_detach_button(qtbot: QtBot) -> None:
    tabs, (a, _b, _c) = make(qtbot)
    tabs.setCurrentWidget(a)
    tabs.detach_button.click()
    assert tabs.is_detached(a)
    tabs.attach_all()
