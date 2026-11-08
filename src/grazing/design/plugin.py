"""Plugin of the design tool for other grazing applications.

Registers `GrazingModelReader`, so that the fitting tool can start from a design stored in a
``.grazing`` project file. The module is loaded through the ``grazing.plugins`` entry point.
"""

from collections.abc import Sequence
from pathlib import Path

from grazing.core.io.models import ImportedModel, model_readers
from grazing.design.project import Project

__all__ = ["INPUT_DESIGN", "GrazingModelReader"]

INPUT_DESIGN = "input design"
"""Name of the imported model holding the input design of a project."""


class GrazingModelReader:
    """Reader of the designs in a ``.grazing`` project file.

    The input design comes first under the name `INPUT_DESIGN`, followed by the stored
    candidates under their names. All use the reflectivity model of the project.
    """

    description = "grazing design project"
    suffixes: Sequence[str] = (".grazing",)

    def read(self, path: Path) -> tuple[ImportedModel, ...]:
        """Read the designs of a project file.

        Args:
            path: The project file.

        Returns:
            The input design, if defined, followed by the candidates.

        Raises:
            ValueError: If the file is not a valid grazing project.
        """
        state = Project.load(path).state
        models = [ImportedModel(INPUT_DESIGN, state.mirror, state.model)] if state.mirror else []
        models += [ImportedModel(c.name, c.mirror, state.model) for c in state.candidates]
        return tuple(models)


model_readers.register("grazing", GrazingModelReader())
