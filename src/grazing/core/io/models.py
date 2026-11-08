"""Reading structure models from files of other applications.

A fit starts from a model of the measured mirror, e.g. a design stored in a ``.grazing`` file.
Readers for such files implement `ModelReader` and register themselves in `model_readers`,
usually from a plugin module (see `grazing.core.plugins.load_plugins`). `read_models` picks the
reader by the file suffix.

Examples:
    >>> from grazing.core.materials import get
    >>> from grazing.core.structure.coatings import SingleLayer
    >>> mirror = Mirror(SingleLayer.create(get("Pt"), t=30.0), get("Si"))
    >>> ImportedModel("single", mirror).mirror.substrate.name
    'Si'
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from grazing.core.io.files import handler_for
from grazing.core.physics.models import ParrattModel, ReflectivityModel
from grazing.core.plugins import Registry
from grazing.core.structure.coating import Mirror

__all__ = ["ImportedModel", "ModelReader", "model_readers", "read_models", "reader_for"]


@dataclass(frozen=True)
class ImportedModel:
    """A structure model read from a file.

    Attributes:
        name: Name of the model within its file, e.g. ``"input design"`` or a candidate name.
        mirror: The mirror with its parameters and their values.
        model: Reflectivity model the file uses for the mirror.
    """

    name: str
    mirror: Mirror
    model: ReflectivityModel = field(default_factory=ParrattModel)


class ModelReader(Protocol):
    """Interface of readers of structure models."""

    @property
    def description(self) -> str:
        """Human-readable name of the file type, e.g. ``"grazing design project"``."""
        ...

    @property
    def suffixes(self) -> Sequence[str]:
        """File suffixes the reader handles, lower case with leading dot, e.g. ``".grazing"``."""
        ...

    def read(self, path: Path) -> tuple[ImportedModel, ...]:
        """Read all models contained in a file.

        Args:
            path: The file.

        Returns:
            The models in the order the file lists them; empty if the file contains none.

        Raises:
            ValueError: If the file is not a valid file of this type.
        """
        ...


model_readers: Registry[ModelReader] = Registry("model reader")
"""Registered model readers by name."""


def reader_for(path: Path | str) -> ModelReader:
    """Return the reader for a file, chosen by its suffix.

    Plugins are loaded first, so that readers contributed by other packages are available.

    Args:
        path: The file.

    Returns:
        The first registered reader that handles the suffix of `path`.

    Raises:
        ValueError: If no registered reader handles the suffix.
    """
    return handler_for(model_readers, path)


def read_models(path: Path | str) -> tuple[ImportedModel, ...]:
    """Read all structure models from a file.

    Args:
        path: The file.

    Returns:
        The models contained in the file.

    Raises:
        ValueError: If no reader handles the file type or the file is invalid.
        OSError: If the file cannot be read.
    """
    return reader_for(path).read(Path(path))
