"""Export of layer tables and deposition recipes."""

import csv
import json
from pathlib import Path
from typing import Any, Literal

from grazing.core.structure.stack import Stack
from grazing.design.manufacturing.process import DepositionProcess

__all__ = ["layer_table", "write_recipe_csv", "write_recipe_json"]

type Order = Literal["deposition", "top_down"]
"""Row order: from the substrate up (``"deposition"``) or from the surface down."""


def layer_table(
    stack: Stack, process: DepositionProcess | None = None, order: Order = "deposition"
) -> list[dict[str, Any]]:
    """Return one row per layer of non-zero thickness.

    Each row contains ``step`` (deposition order, 1 = on the substrate), ``layer_index`` (index
    in `Stack.layers`, 0 = top), ``tag``, ``material``, ``formula``, ``thickness_nm``,
    ``roughness_nm`` and, if a process is given, ``duration_s`` and ``start_s``.

    Args:
        stack: The stack.
        process: Deposition process for timing information.
        order: Row order.

    Returns:
        The rows.

    Raises:
        KeyError: If the process has no rate for a material of the stack.
    """
    indices = [i for i in range(stack.num_layers - 1, -1, -1) if stack.layers[i].thickness > 0]
    timing = {s.layer_index: s for s in process.schedule(stack)} if process else {}
    rows = []
    for step, index in enumerate(indices, start=1):
        layer = stack.layers[index]
        row: dict[str, Any] = {
            "step": step,
            "layer_index": index,
            "tag": layer.tag,
            "material": layer.material.name,
            "formula": layer.material.formula,
            "thickness_nm": layer.thickness,
            "roughness_nm": layer.roughness,
        }
        if process is not None:
            row["duration_s"] = timing[index].duration
            row["start_s"] = timing[index].start_time
        rows.append(row)
    return rows if order == "deposition" else rows[::-1]


def write_recipe_csv(
    path: Path | str,
    stack: Stack,
    process: DepositionProcess | None = None,
    order: Order = "deposition",
) -> None:
    """Write the layer table as a CSV file.

    Args:
        path: Output file.
        stack: The stack.
        process: Deposition process for timing information.
        order: Row order.
    """
    rows = layer_table(stack, process, order)
    with Path(path).open("w", newline="") as f:
        if not rows:
            return
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_recipe_json(
    path: Path | str,
    stack: Stack,
    process: DepositionProcess | None = None,
    order: Order = "deposition",
) -> None:
    """Write the layer table together with substrate and summary information as JSON.

    Args:
        path: Output file.
        stack: The stack.
        process: Deposition process for timing information.
        order: Row order.
    """
    data: dict[str, Any] = {
        "substrate": {"material": stack.substrate.name, "roughness_nm": stack.substrate_roughness},
        "ambient": stack.ambient.name,
        "total_thickness_nm": stack.total_thickness,
        "order": order,
        "layers": layer_table(stack, process, order),
    }
    if process is not None:
        data["total_time_s"] = process.time(stack)
    Path(path).write_text(json.dumps(data, indent=2))
