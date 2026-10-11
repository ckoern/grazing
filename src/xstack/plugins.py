"""Named registries for extensible components.

Every extension axis of xstack (optical-constant providers, coatings, constraints, figures of
merit, optimizers, ...) has a `Registry` that maps a unique name to an implementation. New
implementations become available everywhere, including serialization and the GUI, by
registering them.

Examples:
    >>> registry: Registry[int] = Registry("number")
    >>> registry.register("answer", 42)
    >>> registry.get("answer")
    42
"""

from collections.abc import Iterator

__all__ = ["Registry"]


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
