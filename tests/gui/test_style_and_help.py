"""Tests for the explanatory texts, tooltips and `xstack.gui.style`."""

import pytest
from PySide6.QtWidgets import QApplication, QPushButton
from pytestqt.qtbot import QtBot

from xstack.gui.forms import DataclassForm
from xstack.gui.main_window import MainWindow
from xstack.gui.steps import (
    AnalyzePage,
    ConstraintsPage,
    MaterialsPage,
    ProcessPage,
    ScanPage,
    StepPage,
)
from xstack.gui.style import HELP_STYLE, apply_style, set_variant
from xstack.manufacturing.constraints import MinLayerThickness

pytestmark = pytest.mark.gui


@pytest.mark.parametrize(
    "factory", [MaterialsPage, ConstraintsPage, ProcessPage, ScanPage, AnalyzePage]
)
def test_pages_have_help_and_tooltips(qtbot: QtBot, factory: type[StepPage]) -> None:
    page = factory()
    qtbot.addWidget(page)
    assert len(page.help_text) > 100
    for button in page.findChildren(QPushButton):
        assert button.toolTip()


def test_scan_help_mentions_start_values() -> None:
    assert "start values" in ScanPage.help_text


def test_apply_style_sets_stylesheet_once() -> None:
    app = QApplication.instance()
    assert isinstance(app, QApplication)
    try:
        apply_style(app)
        assert "gridline-color" in app.styleSheet()
        assert 'QPushButton[variant="primary"]' in app.styleSheet()
        app.setStyleSheet("marker")
        apply_style(app)
        assert app.styleSheet() == "marker"
    finally:
        app.setStyleSheet("")
        app.setProperty("xstack_styled", False)
    assert "border" in HELP_STYLE


def test_set_variant_sets_property(qtbot: QtBot) -> None:
    button = QPushButton("Run")
    qtbot.addWidget(button)
    set_variant(button, "primary")
    assert button.property("variant") == "primary"


def test_primary_actions_are_marked(qtbot: QtBot, window: MainWindow) -> None:
    for button in (
        window.estimate_page.estimate_button,
        window.scan_page.run_button,
        window.optimize_page.run_button,
        window.analyze_page.run_button,
    ):
        assert button.property("variant") == "primary"


def test_form_editors_have_tooltips(qtbot: QtBot) -> None:
    form = DataclassForm()
    qtbot.addWidget(form)
    form.set_value(MinLayerThickness())
    assert form.findChild(object, "field_minimum").toolTip()
