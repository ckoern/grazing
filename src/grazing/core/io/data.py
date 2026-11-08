"""Reading measured reflectivity curves from data files.

Parsers implement `DataParser` and register themselves in `data_parsers`, either in the module
that defines them or from a plugin module (see `grazing.core.plugins.load_plugins`).
`read_curves` picks the parser by the file suffix.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from grazing.core.io.files import handler_for
from grazing.core.measurement import MeasuredCurve
from grazing.core.plugins import Registry

__all__ = ["DataParser", "data_parsers", "parser_for", "read_curves"]


class DataParser(Protocol):
    """Interface of parsers of measured data files.

    A parser may take options that describe the file layout (columns, units, ...). Their type is
    `options_type`; ``None`` asks the parser to detect the layout or use its defaults.
    """

    @property
    def description(self) -> str:
        """Human-readable name of the file type, e.g. ``"CSV table"``."""
        ...

    @property
    def suffixes(self) -> Sequence[str]:
        """File suffixes the parser handles, lower case with leading dot, e.g. ``".csv"``."""
        ...

    @property
    def options_type(self) -> type[Any] | None:
        """Type of the options accepted by `read`, or ``None`` if the parser takes none."""
        ...

    def read(self, path: Path, options: Any = None) -> tuple[MeasuredCurve, ...]:
        """Read all curves contained in a file.

        Args:
            path: The file.
            options: Layout of the file as an instance of `options_type`, or ``None`` to
                detect it.

        Returns:
            The curves in the order the file lists them.

        Raises:
            ValueError: If the file cannot be parsed or `options` has the wrong type.
            OSError: If the file cannot be read.
        """
        ...


data_parsers: Registry[DataParser] = Registry("data parser")
"""Registered data parsers by name."""


def parser_for(path: Path | str) -> DataParser:
    """Return the parser for a file, chosen by its suffix.

    Plugins are loaded first, so that parsers contributed by other packages are available.

    Args:
        path: The file.

    Returns:
        The first registered parser that handles the suffix of `path`.

    Raises:
        ValueError: If no registered parser handles the suffix.
    """
    return handler_for(data_parsers, path)


def read_curves(path: Path | str, options: Any = None) -> tuple[MeasuredCurve, ...]:
    """Read all measured curves from a data file.

    Args:
        path: The file.
        options: File layout for the parser chosen by the suffix (see `DataParser`), or
            ``None`` to detect it.

    Returns:
        The curves contained in the file.

    Raises:
        ValueError: If no parser handles the file type or the file cannot be parsed.
        OSError: If the file cannot be read.
    """
    return parser_for(path).read(Path(path), options)
