"""Conversion of model objects to and from JSON-compatible data.

Frozen dataclasses are written as ``{"type": <tag>, "fields": {...}}`` with one entry per
constructor field. Classes registered in the component registries (coatings, figures of merit,
constraints, optimizers) are known automatically under tags such as ``"coating.periodic"``;
other classes are added with `register_serializable` and registries with
`register_component_registry`, both by the modules that define them. Tuples are written as lists
and read back as tuples, non-finite floats are written as ``{"float": "inf"}``.

Callables (e.g. in `grazing.design.merit.CallableMerit`) are written as a reference
``"module:qualified_name"`` and must be importable when reading.

Examples:
    >>> from grazing.core.materials import get
    >>> data = to_data(get("B4C"))
    >>> data["type"], data["fields"]["density"]
    ('Material', 2.52)
    >>> from_data(data) == get("B4C")
    True
"""

import dataclasses
import importlib
import math
from collections.abc import Callable
from typing import Any

import numpy as np

from grazing.core.parameters import Parameter, ParameterSet
from grazing.core.plugins import Registry

__all__ = [
    "SerializationError",
    "from_data",
    "register_component_registry",
    "register_serializable",
    "to_data",
]


class SerializationError(ValueError):
    """Raised when an object cannot be converted to or from data."""


_TAG_TO_CLASS: dict[str, type] = {}
_CLASS_TO_TAG: dict[type, str] = {}
_REGISTRIES: dict[str, Registry[Any]] = {}


def register_serializable(tag: str, cls: type) -> None:
    """Make a dataclass serializable under a tag.

    Args:
        tag: Unique tag written to the data.
        cls: A dataclass whose constructor accepts all its init fields as keyword arguments.

    Raises:
        ValueError: If the tag or class is already registered differently, or `cls` is not a
            dataclass.
    """
    if not dataclasses.is_dataclass(cls):
        raise ValueError(f"{cls.__name__} is not a dataclass")
    if _TAG_TO_CLASS.get(tag, cls) is not cls or _CLASS_TO_TAG.get(cls, tag) != tag:
        raise ValueError(f"conflicting registration for tag {tag!r} / class {cls.__name__}")
    _TAG_TO_CLASS[tag] = cls
    _CLASS_TO_TAG[cls] = tag


register_serializable("Parameter", Parameter)


def register_component_registry(prefix: str, registry: Registry[Any]) -> None:
    """Make all classes of a component registry serializable under ``"<prefix>.<name>"``.

    The registry is consulted on every lookup, so classes registered later are included.

    Args:
        prefix: Unique tag prefix, e.g. ``"coating"``.
        registry: Registry mapping type names to classes with a ``type_name`` attribute.

    Raises:
        ValueError: If the prefix is already used by another registry.
    """
    if _REGISTRIES.get(prefix, registry) is not registry:
        raise ValueError(f"conflicting registration for prefix {prefix!r}")
    _REGISTRIES[prefix] = registry


def _tag(cls: type) -> str:
    if cls in _CLASS_TO_TAG:
        return _CLASS_TO_TAG[cls]
    type_name = getattr(cls, "type_name", None)
    if isinstance(type_name, str):
        for prefix, registry in _REGISTRIES.items():
            if type_name in registry and registry.get(type_name) is cls:
                return f"{prefix}.{type_name}"
    raise SerializationError(f"{cls.__module__}.{cls.__qualname__} is not serializable")


def _class_for(tag: str) -> type:
    if tag in _TAG_TO_CLASS:
        return _TAG_TO_CLASS[tag]
    prefix, _, name = tag.partition(".")
    registry = _REGISTRIES.get(prefix)
    if registry is not None and name in registry:
        cls: type = registry.get(name)
        return cls
    raise SerializationError(f"unknown type tag {tag!r}")


def to_data(obj: Any) -> Any:
    """Convert an object to JSON-compatible data.

    Args:
        obj: The object: a registered dataclass, a `ParameterSet`, a primitive, a tuple, list or
            dict of these, a numpy array, or an importable function.

    Returns:
        Nested dicts, lists, strings, numbers, booleans and ``None``.

    Raises:
        SerializationError: If the object (or a part of it) is not serializable.
    """
    if obj is None or isinstance(obj, bool | str | int):
        return obj
    if isinstance(obj, float | np.floating):
        value = float(obj)
        return value if math.isfinite(value) else {"float": repr(value)}
    if isinstance(obj, np.integer | np.bool_):
        return obj.item()
    if isinstance(obj, tuple | list):
        return [to_data(v) for v in obj]
    if isinstance(obj, dict):
        return {"type": "dict", "items": [[to_data(k), to_data(v)] for k, v in obj.items()]}
    if isinstance(obj, np.ndarray):
        return {"type": "ndarray", "dtype": str(obj.dtype), "data": to_data(obj.tolist())}
    if isinstance(obj, ParameterSet):
        return {"type": "ParameterSet", "parameters": [to_data(p) for p in obj.values()]}
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        tag = _tag(type(obj))
        fields = {
            f.name: to_data(getattr(obj, f.name))
            for f in dataclasses.fields(obj)
            if f.init and f.metadata.get("serialize", True)
        }
        return {"type": tag, "fields": fields}
    if callable(obj):
        return {"type": "callable", "ref": _callable_ref(obj)}
    raise SerializationError(f"cannot serialize object of type {type(obj).__name__}")


def from_data(data: Any) -> Any:
    """Convert data produced by `to_data` back to objects.

    Args:
        data: The data.

    Returns:
        The reconstructed object.

    Raises:
        SerializationError: If the data refers to unknown types or is malformed.
    """
    if data is None or isinstance(data, bool | str | int | float):
        return data
    if isinstance(data, list):
        return tuple(from_data(v) for v in data)
    if not isinstance(data, dict):
        raise SerializationError(f"unexpected data of type {type(data).__name__}")
    if "float" in data and len(data) == 1:
        return float(data["float"])
    kind = data.get("type")
    if kind == "dict":
        return {from_data(k): from_data(v) for k, v in data["items"]}
    if kind == "ndarray":
        return np.array(_lists(data["data"]), dtype=data["dtype"])
    if kind == "ParameterSet":
        return ParameterSet(from_data(p) for p in data["parameters"])
    if kind == "callable":
        return _resolve_callable(data["ref"])
    if not isinstance(kind, str):
        raise SerializationError("missing type tag")
    cls = _class_for(kind)
    kwargs = {name: from_data(value) for name, value in data.get("fields", {}).items()}
    try:
        return cls(**kwargs)
    except (TypeError, ValueError) as exc:
        raise SerializationError(f"cannot construct {kind}: {exc}") from exc


def _lists(data: Any) -> Any:
    if isinstance(data, list):
        return [_lists(v) for v in data]
    if isinstance(data, dict) and "float" in data:
        return float(data["float"])
    return data


def _callable_ref(func: Callable[..., Any]) -> str:
    module = getattr(func, "__module__", None)
    name = getattr(func, "__qualname__", None)
    if not module or not name or "<" in name:
        raise SerializationError(f"callable {func!r} is not importable (lambda or local function)")
    ref = f"{module}:{name}"
    if _resolve_callable(ref) is not func:
        raise SerializationError(f"callable {ref} does not resolve to the same object")
    return ref


def _resolve_callable(ref: str) -> Callable[..., Any]:
    module_name, _, name = ref.partition(":")
    try:
        obj: Any = importlib.import_module(module_name)
        for part in name.split("."):
            obj = getattr(obj, part)
    except (ImportError, AttributeError) as exc:
        raise SerializationError(f"cannot import callable {ref!r}: {exc}") from exc
    if not callable(obj):
        raise SerializationError(f"{ref!r} is not callable")
    return obj  # type: ignore[no-any-return]
