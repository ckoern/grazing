"""Tests for the explanatory texts, tooltips and `xstack.gui.style`."""

import pytest
from PySide6.QtWidgets import QApplication, QPushButton
from pytestqt.qtbot import QtBot

from xstack.gui.forms import DataclassForm
from xstack.gui.steps import (
    AnalyzePage,
    ConstraintsPage,
    MaterialsPage,
    ProcessPage,
    ScanPage,
    StepPage,
)
from xstack.gui.style import HELP_STYLE, apply_style
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


def test_apply_style_sets_stylesheet() -> None:
    app = QApplication.instance()
    assert isinstance(app, QApplication)
    apply_style(app)
    assert "gridline-color" in app.styleSheet()
    assert "border" in HELP_STYLE


def test_form_editors_have_tooltips(qtbot: QtBot) -> None:
    form = DataclassForm()
    qtbot.addWidget(form)
    form.set_value(MinLayerThickness())
    assert form.findChild(object, "field_minimum").toolTip()
