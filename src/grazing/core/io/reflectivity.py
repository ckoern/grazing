"""Export of reflectivity curves."""

import csv
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from grazing.core.materials import MaterialLibrary
from grazing.core.physics.models import ReflectivityModel
from grazing.core.structure.stack import Stack
from grazing.core.targets import Target
from grazing.core.units import to_deg, to_kev

__all__ = ["REFLECTIVITY_COLUMNS", "reflectivity_table", "write_reflectivity_csv"]

REFLECTIVITY_COLUMNS = (
    "target",
    "axis",
    "reflections",
    "energy_keV",
    "angle_deg",
    "reflectivity",
    "reflectivity^N",
)
"""Columns of the table returned by `reflectivity_table`."""


def reflectivity_table(
    stack: Stack,
    targets: Sequence[Target],
    model: ReflectivityModel,
    library: MaterialLibrary | None = None,
) -> list[dict[str, Any]]:
    """Return the reflectivity of a stack on the grids of targets, one row per grid point.

    The columns are `REFLECTIVITY_COLUMNS`: the target label (see `Target.label`), the scanned
    axis (``"energy"`` or ``"angle"``), the target's number of reflections N, the photon energy
    in keV, the grazing angle in degrees, the reflectivity R of one reflection and the
    reflectivity Rᴺ after N reflections. Both include the target's resolution.

    Args:
        stack: The stack.
        targets: Targets defining grids, polarization, resolution and number of reflections.
        model: The reflectivity model.
        library: Material library for optical constants.

    Returns:
        The rows, target by target.
    """
    rows = []
    for index, target in enumerate(targets):
        single = model.evaluate(stack, replace(target, reflections=1), library=library)
        multiple = (
            single if target.reflections == 1 else model.evaluate(stack, target, library=library)
        )
        energies = to_kev(single.energy)
        angles = to_deg(single.theta)
        for energy, angle, r, r_n in zip(
            energies, angles, single.reflectivity, multiple.reflectivity, strict=True
        ):
            rows.append(
                {
                    "target": target.label(index),
                    "axis": target.axis,
                    "reflections": target.reflections,
                    "energy_keV": float(energy),
                    "angle_deg": float(angle),
                    "reflectivity": float(r),
                    "reflectivity^N": float(r_n),
                }
            )
    return rows


def write_reflectivity_csv(
    path: Path | str,
    stack: Stack,
    targets: Sequence[Target],
    model: ReflectivityModel,
    library: MaterialLibrary | None = None,
) -> None:
    """Write the reflectivity of a stack on the grids of targets as a CSV file.

    The file has a header row with `REFLECTIVITY_COLUMNS` and the rows of
    `reflectivity_table`.

    Args:
        path: Output file.
        stack: The stack.
        targets: Targets defining grids, polarization, resolution and number of reflections.
        model: The reflectivity model.
        library: Material library for optical constants.
    """
    rows = reflectivity_table(stack, targets, model, library)
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REFLECTIVITY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
