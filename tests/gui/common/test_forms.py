"""Tests for `grazing.gui.common.forms`."""

import pytest
from PySide6.QtWidgets import QCheckBox, QComboBox, QLineEdit
from pytestqt.qtbot import QtBot

from grazing.core.optim.optimizers import LmfitOptimizer
from grazing.design.manufacturing.constraints import MaxCoatingTime, MinLayerThickness
from grazing.design.merit.composite import TimePenalized
from grazing.design.merit.optical import ProfileDeviation, SoftMinReflectivity
from grazing.gui.common.forms import DataclassForm, editable_fields

pytestmark = pytest.mark.gui


def kinds(cls: type) -> dict[str, str]:
    return {f.name: kind for f, kind, _ in editable_fields(cls)}


def test_editable_fields_kinds() -> None:
    assert kinds(MinLayerThickness) == {
        "severity": "literal",
        "minimum": "float",
        "materials": "str_tuple",
    }
    assert kinds(MaxCoatingTime) == {"severity": "literal", "maximum": "float"}
    assert kinds(TimePenalized) == {"weight": "float", "reference_time": "float"}
    assert kinds(SoftMinReflectivity) == {"softness": "float"}
    assert kinds(ProfileDeviation) == {"mode": "literal"}
    assert kinds(LmfitOptimizer) == {"method": "str", "max_evaluations": "int"}


def line(form: DataclassForm, name: str) -> QLineEdit:
    editor = form.findChild(QLineEdit, f"field_{name}")
    assert editor is not None
    return editor


def enter(form: DataclassForm, name: str, text: str) -> None:
    editor = line(form, name)
    editor.setText(text)
    editor.editingFinished.emit()


@pytest.fixture
def form(qtbot: QtBot) -> DataclassForm:
    f = DataclassForm()
    qtbot.addWidget(f)
    return f


def test_builds_editor_per_field(form: DataclassForm) -> None:
    form.set_value(MinLayerThickness(2.0))
    assert isinstance(form.findChild(QComboBox, "field_severity"), QComboBox)
    assert line(form, "minimum").text() == "2"
    assert line(form, "materials").text() == ""


def test_scaled_display(form: DataclassForm) -> None:
    form.set_value(MaxCoatingTime())
    assert line(form, "maximum").text() == "24"


def test_scaled_edit_unscales(form: DataclassForm, qtbot: QtBot) -> None:
    form.set_value(MaxCoatingTime())
    with qtbot.waitSignal(form.valueChanged, timeout=1000) as blocker:
        enter(form, "maximum", "48")
    new = blocker.args[0]
    assert isinstance(new, MaxCoatingTime)
    assert new.maximum == pytest.approx(48 * 3600)
    assert form.value() is new


def test_literal_combo_edit(form: DataclassForm, qtbot: QtBot) -> None:
    form.set_value(MinLayerThickness(2.0))
    combo = form.findChild(QComboBox, "field_severity")
    assert isinstance(combo, QComboBox)
    other = 1 - combo.currentIndex()
    expected = combo.itemData(other)
    combo.setCurrentIndex(other)
    with qtbot.waitSignal(form.valueChanged, timeout=1000) as blocker:
        combo.activated.emit(other)
    assert blocker.args[0] == MinLayerThickness(2.0, severity=expected)


def test_str_tuple_parsing(form: DataclassForm, qtbot: QtBot) -> None:
    form.set_value(MinLayerThickness(2.0))
    with qtbot.waitSignal(form.valueChanged, timeout=1000) as blocker:
        enter(form, "materials", "W, Si")
    assert blocker.args[0].materials == ("W", "Si")
    form.set_value(MinLayerThickness(2.0, materials=("C", "B4C")))
    assert line(form, "materials").text() == "C, B4C"


def test_optional_int_and_str(form: DataclassForm, qtbot: QtBot) -> None:
    form.set_value(LmfitOptimizer())
    assert line(form, "max_evaluations").text() == ""
    with qtbot.waitSignal(form.valueChanged, timeout=1000) as blocker:
        enter(form, "max_evaluations", "300")
    assert blocker.args[0].max_evaluations == 300
    with qtbot.waitSignal(form.valueChanged, timeout=1000) as blocker:
        enter(form, "method", "nelder")
    assert blocker.args[0] == LmfitOptimizer(method="nelder", max_evaluations=300)
    with qtbot.waitSignal(form.valueChanged, timeout=1000) as blocker:
        enter(form, "max_evaluations", "")
    assert blocker.args[0].max_evaluations is None


def test_bool_checkbox(form: DataclassForm, qtbot: QtBot) -> None:
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Flag:
        on: bool = False

    form.set_value(Flag())
    box = form.findChild(QCheckBox, "field_on")
    assert isinstance(box, QCheckBox)
    assert not box.isChecked()
    with qtbot.waitSignal(form.valueChanged, timeout=1000) as blocker:
        box.setChecked(True)
    assert blocker.args[0] == Flag(True)


@pytest.mark.parametrize("text", ["-1", "abc"])
def test_invalid_input_restores(form: DataclassForm, qtbot: QtBot, text: str) -> None:
    form.set_value(MinLayerThickness(2.0))
    changed: list[object] = []
    form.valueChanged.connect(changed.append)
    with qtbot.waitSignal(form.invalidInput, timeout=1000) as blocker:
        enter(form, "minimum", text)
    assert "minimum" in blocker.args[0]
    assert line(form, "minimum").text() == "2"
    assert form.value() == MinLayerThickness(2.0)
    assert not changed


def test_unchanged_value_does_not_emit(form: DataclassForm) -> None:
    form.set_value(MinLayerThickness(2.0))
    changed: list[object] = []
    form.valueChanged.connect(changed.append)
    enter(form, "minimum", "2")
    assert not changed


def test_set_value_none_clears(form: DataclassForm) -> None:
    form.set_value(MinLayerThickness(2.0))
    form.set_value(None)
    assert form.value() is None
    assert form.findChild(QLineEdit, "field_minimum") is None


def test_type_change_rebuilds(form: DataclassForm) -> None:
    form.set_value(MinLayerThickness(2.0))
    form.set_value(SoftMinReflectivity(0.05))
    assert form.findChild(QLineEdit, "field_minimum") is None
    assert line(form, "softness").text() == "0.05"


def test_same_type_reloads_values(form: DataclassForm) -> None:
    form.set_value(MinLayerThickness(2.0))
    editor = line(form, "minimum")
    form.set_value(MinLayerThickness(3.0))
    assert line(form, "minimum") is editor
    assert editor.text() == "3"
