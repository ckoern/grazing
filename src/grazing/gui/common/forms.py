"""Editor forms generated from dataclass fields.

Settings objects of grazing (constraints, figures of merit, processes, optimizers, ...) are
immutable dataclasses. `DataclassForm` creates an editor for their simple fields and emits a
modified copy on every edit, so that new plugin classes are editable without GUI code.

Supported field types: ``float``, ``int``, ``bool``, ``str``, ``Literal[...]``, ``tuple[str,
...]`` and optional variants of these. Other fields are shown read-only. Field metadata may
provide ``"label"``, ``"unit"``, ``"scale"`` (displayed value = stored value · scale) and
``"help"`` (tooltip text of the editor).
"""

import dataclasses
import types
import typing
from typing import Any, Literal, Union, get_args, get_origin

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QWidget,
)

__all__ = ["DataclassForm", "editable_fields"]


def _unwrap_optional(tp: Any) -> tuple[Any, bool]:
    origin = get_origin(tp)
    if origin in (Union, types.UnionType):
        args = [a for a in get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return args[0], True
    return tp, False


def _kind(tp: Any) -> str | None:
    tp, _ = _unwrap_optional(tp)
    if tp is bool:
        return "bool"
    if tp is int:
        return "int"
    if tp is float:
        return "float"
    if tp is str:
        return "str"
    if get_origin(tp) is Literal:
        return "literal"
    if get_origin(tp) is tuple and get_args(tp)[:1] == (str,):
        return "str_tuple"
    if isinstance(tp, typing.TypeAliasType):
        return _kind(tp.__value__)
    return None


def editable_fields(cls: type) -> list[tuple[dataclasses.Field[Any], str, Any]]:
    """Return the init fields of a dataclass that `DataclassForm` can edit.

    Args:
        cls: A dataclass type.

    Returns:
        Tuples of field, kind (``"float"``, ``"int"``, ``"bool"``, ``"str"``, ``"literal"``,
        ``"str_tuple"``) and the resolved type annotation.
    """
    hints = typing.get_type_hints(cls)
    result = []
    for f in dataclasses.fields(cls):
        if not f.init:
            continue
        kind = _kind(hints.get(f.name))
        if kind is not None:
            result.append((f, kind, hints[f.name]))
    return result


def _literal_choices(tp: Any) -> tuple[Any, ...]:
    tp, _ = _unwrap_optional(tp)
    if isinstance(tp, typing.TypeAliasType):
        tp = tp.__value__
    return get_args(tp)


class DataclassForm(QWidget):
    """Form editing the simple fields of an immutable dataclass instance.

    Signals:
        valueChanged(object): A modified copy of the edited object.
        invalidInput(str): An entry could not be applied; the message describes why.

    Args:
        parent: Parent widget.
    """

    valueChanged = Signal(object)
    invalidInput = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._layout = QFormLayout(self)
        self._value: Any = None
        self._editors: dict[str, tuple[QWidget, str, Any, float]] = {}

    def value(self) -> Any:
        """Return the edited object."""
        return self._value

    def set_value(self, value: Any) -> None:
        """Show an object; rebuilds the form if its type changed.

        Args:
            value: A dataclass instance, or ``None`` to clear the form.
        """
        rebuild = value is None or self._value is None or type(value) is not type(self._value)
        self._value = value
        if rebuild:
            self._build()
        self._load()

    def _build(self) -> None:
        while self._layout.rowCount():
            self._layout.removeRow(0)
        self._editors.clear()
        if self._value is None:
            return
        for f, kind, tp in editable_fields(type(self._value)):
            label = f.metadata.get("label", f.name.replace("_", " "))
            unit = f.metadata.get("unit", "")
            scale = float(f.metadata.get("scale", 1.0))
            editor: QWidget
            if kind == "bool":
                box = QCheckBox()
                box.toggled.connect(lambda _=False, n=f.name: self._commit(n))
                editor = box
            elif kind == "literal":
                combo = QComboBox()
                for choice in _literal_choices(tp):
                    combo.addItem(str(choice), choice)
                combo.activated.connect(lambda _=0, n=f.name: self._commit(n))
                editor = combo
            else:
                line = QLineEdit()
                line.editingFinished.connect(lambda n=f.name: self._commit(n))
                editor = line
            editor.setObjectName(f"field_{f.name}")
            editor.setToolTip(str(f.metadata.get("help", f"{label}{f' ({unit})' if unit else ''}")))
            self._editors[f.name] = (editor, kind, tp, scale)
            self._layout.addRow(f"{label}{f' ({unit})' if unit else ''}", editor)
        if not self._editors:
            self._layout.addRow(QLabel("No editable settings."))

    def _load(self) -> None:
        if self._value is None:
            return
        for name, (editor, kind, _, scale) in self._editors.items():
            current = getattr(self._value, name)
            editor.blockSignals(True)
            if isinstance(editor, QCheckBox):
                editor.setChecked(bool(current))
            elif isinstance(editor, QComboBox):
                editor.setCurrentIndex(max(0, editor.findData(current)))
            elif isinstance(editor, QLineEdit):
                if current is None:
                    editor.setText("")
                elif kind == "float":
                    editor.setText(f"{current * scale:.6g}")
                elif kind == "str_tuple":
                    editor.setText(", ".join(current))
                else:
                    editor.setText(str(current))
            editor.blockSignals(False)

    def _parse(self, name: str) -> Any:
        editor, kind, tp, scale = self._editors[name]
        _, optional = _unwrap_optional(tp)
        if isinstance(editor, QCheckBox):
            return editor.isChecked()
        if isinstance(editor, QComboBox):
            return editor.currentData()
        assert isinstance(editor, QLineEdit)
        text = editor.text().strip()
        if not text and optional:
            return None
        if kind == "float":
            return float(text) / scale
        if kind == "int":
            return int(text)
        if kind == "str_tuple":
            return tuple(part.strip() for part in text.split(",") if part.strip())
        return text

    def _commit(self, name: str) -> None:
        if self._value is None:
            return
        try:
            new = dataclasses.replace(self._value, **{name: self._parse(name)})
        except (ValueError, TypeError) as exc:
            self.invalidInput.emit(f"{name}: {exc}")
            self._load()
            return
        if new != self._value:
            self._value = new
            self.valueChanged.emit(new)
