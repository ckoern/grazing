"""Tests of the live behaviour of `xstack.gui.main_window`: pages and previews."""

import pytest
from PySide6.QtWidgets import QLineEdit
from pytestqt.qtbot import QtBot

from xstack.gui.main_window import WORKFLOW_STEPS, MainWindow

pytestmark = pytest.mark.gui


def wait_for_preview(qtbot: QtBot, window: MainWindow, previous: object = None) -> None:
    qtbot.waitUntil(
        lambda: (
            window.preview is not None
            and window.preview is not previous
            and window.preview.state is window.controller.state
        ),
        timeout=10000,
    )


def test_pages_in_workflow_order(window: MainWindow) -> None:
    assert window.step_tabs.count() == len(WORKFLOW_STEPS)
    for index, step in enumerate(WORKFLOW_STEPS):
        assert window.step_tabs.widget(index).isAncestorOf(window.pages[step])


def test_selecting_workflow_row_switches_page(window: MainWindow) -> None:
    for index, step in enumerate(WORKFLOW_STEPS):
        window.step_tabs.setCurrentIndex(index)
        assert window.step_tabs.currentIndex() == index
        assert window.step_tabs.currentWidget().isAncestorOf(window.pages[step])


def test_initial_preview(qtbot: QtBot, window: MainWindow) -> None:
    qtbot.waitUntil(
        lambda: window.preview is not None and "layers" in window.summary_label.text(),
        timeout=10000,
    )
    assert window.preview is not None
    assert window.preview.error is None
    assert "80 layers" in window.summary_label.text()
    assert window.reflectivity_plot.plot_items()


def test_parameter_edit_triggers_new_preview(qtbot: QtBot, window: MainWindow) -> None:
    wait_for_preview(qtbot, window)
    first = window.preview
    model = window.parameter_table.table_model
    names = list(model.parameters())
    assert model.setData(model.index(names.index("N"), 1), "55")
    wait_for_preview(qtbot, window, first)
    assert window.preview is not first
    assert "110 layers" in window.summary_label.text()
    items = window.reflectivity_plot.plot_items()
    assert items
    assert all(p.listDataItems() for p in items.values())


def test_page_error_shown_in_status_bar(window: MainWindow) -> None:
    window.goal_page.errorOccurred.emit("Invalid target: nonsense")
    assert window.statusBar().currentMessage() == "Invalid target: nonsense"


def test_page_command_updates_project(qtbot: QtBot, window: MainWindow) -> None:
    wait_for_preview(qtbot, window)
    page = window.goal_page
    edit = page.table.cellWidget(0, 4)
    assert isinstance(edit, QLineEdit)
    edit.setText("8.4")
    edit.editingFinished.emit()
    targets = window.controller.state.targets
    assert targets is not None
    assert targets.targets[0].scan_range[1] == pytest.approx(8400.0)
    assert window.controller.undo_stack.count() == 1
