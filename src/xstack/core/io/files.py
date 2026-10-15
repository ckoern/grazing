"""Choice of file handlers by file suffix."""

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from xstack.core.plugins import Registry, load_plugins

__all__ = ["FileHandler", "handler_for"]


class FileHandler(Protocol):
    """Common interface of file readers chosen by suffix."""

    @property
    def description(self) -> str:
        """Human-readable name of the file type."""
        ...

    @property
    def suffixes(self) -> Sequence[str]:
        """File suffixes the handler accepts, lower case with leading dot, e.g. ``".csv"``."""
        ...


def handler_for[H: FileHandler](registry: Registry[H], path: Path | str) -> H:
    """Return the handler for a file, chosen by its suffix.

    Plugins are loaded first, so that handlers contributed by other packages are available.

    Args:
        registry: Registry of handlers to choose from.
        path: The file.

    Returns:
        The first registered handler that accepts the suffix of `path`.

    Raises:
        ValueError: If no registered handler accepts the suffix.
    """
    load_plugins()
    suffix = Path(path).suffix.lower()
    handlers = [registry.get(name) for name in registry]
    for handler in handlers:
        if suffix in handler.suffixes:
            return handler
    known = sorted({s for handler in handlers for s in handler.suffixes})
    raise ValueError(
        f"no {registry.kind} for {suffix or 'files without suffix'!r} "
        f"(supported: {', '.join(known) or 'none'})"
    )
