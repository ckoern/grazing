"""Plugin of the design tool for other xstack applications.

Registers `XstackModelReader`, so that the fitting tool can start from a design stored in an
``.xstack`` project file. The module is loaded through the ``xstack.plugins`` entry point.
"""

from collections.abc import Sequence
from pathlib import Path

from xstack.core.io.models import ImportedModel, model_readers
from xstack.design.project import Project

__all__ = ["INPUT_DESIGN", "XstackModelReader"]

INPUT_DESIGN = "input design"
"""Name of the imported model holding the input design of a project."""


class XstackModelReader:
    """Reader of the designs in an ``.xstack`` project file.

    The input design comes first under the name `INPUT_DESIGN`, followed by the stored
    candidates under their names. All use the reflectivity model of the project.
    """

    description = "xstack design project"
    suffixes: Sequence[str] = (".xstack",)

    def read(self, path: Path) -> tuple[ImportedModel, ...]:
        """Read the designs of a project file.

        Args:
            path: The project file.

        Returns:
            The input design, if defined, followed by the candidates.

        Raises:
            ValueError: If the file is not a valid xstack project.
        """
        state = Project.load(path).state
        models = [ImportedModel(INPUT_DESIGN, state.mirror, state.model)] if state.mirror else []
        models += [ImportedModel(c.name, c.mirror, state.model) for c in state.candidates]
        return tuple(models)


model_readers.register("xstack", XstackModelReader())
