"""Named registries for extensible components and loading of plugins.

Every extension axis of xstack (optical-constant providers, coatings, constraints, figures of
merit, optimizers, model readers, ...) has a `Registry` that maps a unique name to an
implementation. New implementations become available everywhere, including serialization and the
GUI, by registering them.

Packages contribute to registries owned by other packages through entry points in the group
`PLUGIN_GROUP`. Each entry point names a module that registers its components when imported;
`load_plugins` imports all of them.

Examples:
    >>> registry: Registry[int] = Registry("number")
    >>> registry.register("answer", 42)
    >>> registry.get("answer")
    42
"""

import warnings
from collections.abc import Iterator
from importlib.metadata import EntryPoint, entry_points

__all__ = ["PLUGIN_GROUP", "PluginWarning", "Registry", "load_plugins"]

PLUGIN_GROUP = "xstack.plugins"
"""Entry point group of xstack plugins."""


class Registry[T]:
    """A mapping from unique names to registered items.

    Args:
        kind: Human-readable name of the kind of item stored, used in error messages.
    """

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._items: dict[str, T] = {}

    @property
    def kind(self) -> str:
        """Human-readable name of the kind of item stored."""
        return self._kind

    def register(self, name: str, item: T, *, replace: bool = False) -> None:
        """Register an item under a name.

        Args:
            name: Unique name of the item.
            item: The item to register.
            replace: Allow replacing an item that is already registered under `name`.

        Raises:
            ValueError: If `name` is empty, or already registered and `replace` is False.
        """
        if not name:
            raise ValueError(f"{self._kind} name must not be empty")
        if name in self._items and not replace:
            raise ValueError(f"{self._kind} {name!r} is already registered")
        self._items[name] = item

    def unregister(self, name: str) -> None:
        """Remove a registered item.

        Args:
            name: Name of the item to remove.

        Raises:
            KeyError: If no item is registered under `name`.
        """
        self.get(name)
        del self._items[name]

    def get(self, name: str) -> T:
        """Return the item registered under a name.

        Args:
            name: Name of the item.

        Returns:
            The registered item.

        Raises:
            KeyError: If no item is registered under `name`.
        """
        try:
            return self._items[name]
        except KeyError:
            available = ", ".join(sorted(self._items)) or "none"
            raise KeyError(f"unknown {self._kind} {name!r} (available: {available})") from None

    def names(self) -> tuple[str, ...]:
        """Return the names of all registered items in registration order."""
        return tuple(self._items)

    def __contains__(self, name: object) -> bool:
        return name in self._items

    def __iter__(self) -> Iterator[str]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)


class PluginWarning(UserWarning):
    """Issued when a plugin cannot be loaded."""


_loaded: set[str] = set()


def load_plugins() -> tuple[str, ...]:
    """Import all plugin modules registered in the entry point group `PLUGIN_GROUP`.

    Plugins already loaded are skipped, so the function can be called before every lookup in a
    registry that plugins contribute to. A plugin that fails to load is skipped with a warning
    and tried again on the next call.

    Returns:
        Names of the entry points loaded by this call.

    Warns:
        PluginWarning: For every plugin that cannot be loaded.
    """
    loaded: list[str] = []
    for entry_point in entry_points(group=PLUGIN_GROUP):
        if entry_point.name in _loaded:
            continue
        if _load(entry_point):
            _loaded.add(entry_point.name)
            loaded.append(entry_point.name)
    return tuple(loaded)


def _load(entry_point: EntryPoint) -> bool:
    try:
        entry_point.load()
    except Exception as exc:
        warnings.warn(
            f"cannot load xstack plugin {entry_point.name!r} ({entry_point.value}): {exc}",
            PluginWarning,
            stacklevel=3,
        )
        return False
    return True
