"""Tests for `xstack.core.plugins`."""

from importlib.metadata import entry_points

import pytest

from xstack.core import plugins
from xstack.core.plugins import Registry


def make_registry() -> Registry[int]:
    registry: Registry[int] = Registry("number")
    registry.register("one", 1)
    registry.register("two", 2)
    registry.register("three", 3)
    return registry


def test_kind() -> None:
    assert Registry[int]("number").kind == "number"


def test_register_and_get() -> None:
    registry = make_registry()
    assert registry.get("one") == 1
    assert registry.get("three") == 3


def test_names_in_registration_order() -> None:
    registry: Registry[int] = Registry("number")
    for name in ("zeta", "alpha", "mid"):
        registry.register(name, 0)
    assert registry.names() == ("zeta", "alpha", "mid")


def test_contains_len_iter() -> None:
    registry = make_registry()
    assert "one" in registry
    assert "four" not in registry
    assert 5 not in registry
    assert len(registry) == 3
    assert list(registry) == ["one", "two", "three"]


def test_empty_registry() -> None:
    registry: Registry[int] = Registry("number")
    assert len(registry) == 0
    assert registry.names() == ()
    assert list(registry) == []


def test_duplicate_raises() -> None:
    registry = make_registry()
    with pytest.raises(ValueError, match="already registered"):
        registry.register("one", 10)
    assert registry.get("one") == 1


def test_replace() -> None:
    registry = make_registry()
    registry.register("one", 10, replace=True)
    assert registry.get("one") == 10
    assert len(registry) == 3
    assert registry.names() == ("one", "two", "three")


def test_replace_new_name() -> None:
    registry = make_registry()
    registry.register("four", 4, replace=True)
    assert registry.get("four") == 4


def test_empty_name_raises() -> None:
    registry: Registry[int] = Registry("number")
    with pytest.raises(ValueError, match="empty"):
        registry.register("", 1)
    assert len(registry) == 0


def test_unknown_name_lists_available() -> None:
    registry = make_registry()
    with pytest.raises(KeyError) as excinfo:
        registry.get("nope")
    message = str(excinfo.value)
    assert "nope" in message
    assert "number" in message
    for name in ("one", "two", "three"):
        assert name in message


def test_unknown_name_in_empty_registry() -> None:
    registry: Registry[int] = Registry("number")
    with pytest.raises(KeyError, match="none"):
        registry.get("x")


def test_unregister() -> None:
    registry = make_registry()
    registry.unregister("two")
    assert "two" not in registry
    assert registry.names() == ("one", "three")
    with pytest.raises(KeyError):
        registry.get("two")


def test_unregister_unknown_raises() -> None:
    registry = make_registry()
    with pytest.raises(KeyError):
        registry.unregister("nope")
    assert len(registry) == 3


def test_reregister_after_unregister() -> None:
    registry = make_registry()
    registry.unregister("one")
    registry.register("one", 11)
    assert registry.get("one") == 11
    assert registry.names() == ("two", "three", "one")


class FakeEntryPoint:
    def __init__(self, name: str, error: Exception | None = None) -> None:
        self.name = name
        self.value = f"fake.{name}"
        self.error = error
        self.calls = 0

    def load(self) -> None:
        self.calls += 1
        if self.error is not None:
            raise self.error


@pytest.fixture
def fake_plugins(monkeypatch: pytest.MonkeyPatch) -> list[FakeEntryPoint]:
    entry_points: list[FakeEntryPoint] = []

    def select(*, group: str) -> list[FakeEntryPoint]:
        return entry_points if group == plugins.PLUGIN_GROUP else []

    monkeypatch.setattr(plugins, "entry_points", select)
    monkeypatch.setattr(plugins, "_loaded", set())
    return entry_points


def test_load_plugins_imports_each_plugin_once(fake_plugins: list[FakeEntryPoint]) -> None:
    good = FakeEntryPoint("good")
    fake_plugins.append(good)
    assert plugins.load_plugins() == ("good",)
    assert plugins.load_plugins() == ()
    assert good.calls == 1


def test_broken_plugin_warns_and_is_retried(fake_plugins: list[FakeEntryPoint]) -> None:
    broken = FakeEntryPoint("broken", ImportError("missing dependency"))
    good = FakeEntryPoint("good")
    fake_plugins.extend([broken, good])
    with pytest.warns(plugins.PluginWarning, match="broken.*missing dependency"):
        assert plugins.load_plugins() == ("good",)
    with pytest.warns(plugins.PluginWarning):
        assert plugins.load_plugins() == ()
    assert (broken.calls, good.calls) == (2, 1)


def test_design_plugin_is_installed() -> None:
    names = {ep.name: ep.value for ep in entry_points(group=plugins.PLUGIN_GROUP)}
    assert names.get("design") == "xstack.design.plugin"
