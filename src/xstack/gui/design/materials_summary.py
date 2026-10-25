"""Short text descriptions of the coating and its materials."""

from xstack.core.structure.coating import Coating
from xstack.core.structure.coatings.composite import Composite

__all__ = ["coating_title", "materials_text"]


def coating_title(coating: Coating) -> str:
    """Return the name of the coating kind.

    Args:
        coating: The coating.

    Returns:
        ``"Combined coating"`` for a composite, else the class name of the coating.
    """
    return "Combined coating" if isinstance(coating, Composite) else type(coating).__name__


def materials_text(coating: Coating) -> str:
    """Describe the materials of a coating's roles.

    Args:
        coating: The coating.

    Returns:
        ``"role: material"`` entries joined by " · ". For a composite, one entry per block with
        the block name and the materials of its roles, separated by "/" for a bilayer block
        (roles H and L) and by ", " otherwise.
    """
    if not isinstance(coating, Composite):
        return " · ".join(f"{role}: {m.name}" for role, m in coating.materials.items())
    entries = []
    for block in coating.blocks:
        names = [m.name for m in block.materials.values()]
        separator = "/" if set(block.roles) == {"H", "L"} else ", "
        entries.append(f"{block.name}: {separator.join(names)}")
    return " · ".join(entries)
