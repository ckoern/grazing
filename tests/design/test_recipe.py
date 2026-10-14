"""Tests for recipe export."""

import csv
import json
from pathlib import Path

import pytest

from xstack.core import materials
from xstack.core.structure import Layer, Stack
from xstack.design.manufacturing.process import DepositionProcess
from xstack.design.recipe import layer_table, write_recipe_csv, write_recipe_json

W = materials.get("W")
SI = materials.get("Si")
B4C = materials.get("B4C")

BASE_KEYS = [
    "step",
    "layer_index",
    "tag",
    "material",
    "formula",
    "thickness_nm",
    "roughness_nm",
]


def make_stack() -> Stack:
    layers = [
        Layer(W, 2.0, 0.3, "top"),
        Layer(SI, 0.0, 0.1, "empty"),
        Layer(W, 1.0, 0.2, "mid"),
        Layer(SI, 4.0, 0.0, "bottom"),
    ]
    return Stack(layers, B4C, 0.5, SI)


def make_process() -> DepositionProcess:
    return DepositionProcess({"W": 0.5, "Si": 2.0}, overhead_per_layer=1.0, fixed_overhead=10.0)


class TestLayerTable:
    def test_rows_deposition_order(self) -> None:
        rows = layer_table(make_stack())
        assert [r["tag"] for r in rows] == ["bottom", "mid", "top"]
        assert [r["step"] for r in rows] == [1, 2, 3]
        assert [r["layer_index"] for r in rows] == [3, 2, 0]
        assert list(rows[0]) == BASE_KEYS
        assert rows[0]["material"] == "Si"
        assert rows[0]["formula"] == SI.formula
        assert rows[0]["thickness_nm"] == 4.0
        assert rows[2]["roughness_nm"] == 0.3

    def test_top_down(self) -> None:
        rows = layer_table(make_stack(), order="top_down")
        assert [r["tag"] for r in rows] == ["top", "mid", "bottom"]
        assert [r["step"] for r in rows] == [3, 2, 1]

    def test_zero_thickness_omitted(self) -> None:
        assert all(r["thickness_nm"] > 0 for r in layer_table(make_stack()))

    def test_no_timing_without_process(self) -> None:
        for row in layer_table(make_stack()):
            assert "duration_s" not in row
            assert "start_s" not in row

    def test_timing_matches_schedule(self) -> None:
        stack, process = make_stack(), make_process()
        rows = layer_table(stack, process)
        steps = process.schedule(stack)
        assert list(rows[0]) == [*BASE_KEYS, "duration_s", "start_s"]
        assert len(rows) == len(steps)
        for row, step in zip(rows, steps, strict=True):
            assert row["step"] == step.step
            assert row["layer_index"] == step.layer_index
            assert row["duration_s"] == pytest.approx(step.duration)
            assert row["start_s"] == pytest.approx(step.start_time)

    def test_timing_top_down(self) -> None:
        stack, process = make_stack(), make_process()
        rows = layer_table(stack, process, order="top_down")
        assert rows[-1]["start_s"] == 10.0
        assert rows[0]["layer_index"] == 0

    def test_missing_rate(self) -> None:
        with pytest.raises(KeyError, match="W"):
            layer_table(make_stack(), DepositionProcess({"Si": 1.0}))

    def test_empty_stack(self) -> None:
        assert layer_table(Stack([], SI)) == []


class TestCsv:
    def read(self, path: Path) -> list[dict[str, str]]:
        with path.open(newline="") as f:
            return list(csv.DictReader(f))

    def test_roundtrip(self, tmp_path: Path) -> None:
        path = tmp_path / "recipe.csv"
        write_recipe_csv(path, make_stack())
        rows = self.read(path)
        assert len(rows) == 3
        assert list(rows[0]) == BASE_KEYS
        assert [r["tag"] for r in rows] == ["bottom", "mid", "top"]
        assert float(rows[0]["thickness_nm"]) == 4.0

    def test_str_path_and_timing(self, tmp_path: Path) -> None:
        path = tmp_path / "recipe.csv"
        write_recipe_csv(str(path), make_stack(), make_process(), order="top_down")
        rows = self.read(path)
        assert list(rows[0]) == [*BASE_KEYS, "duration_s", "start_s"]
        assert [r["tag"] for r in rows] == ["top", "mid", "bottom"]
        assert float(rows[-1]["start_s"]) == pytest.approx(10.0)

    def test_empty_stack(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.csv"
        write_recipe_csv(path, Stack([], SI))
        assert path.read_text() == ""


class TestJson:
    def test_without_process(self, tmp_path: Path) -> None:
        path = tmp_path / "recipe.json"
        stack = make_stack()
        write_recipe_json(path, stack)
        data = json.loads(path.read_text())
        assert data["substrate"] == {"material": "B4C", "roughness_nm": 0.5}
        assert data["ambient"] == "Si"
        assert data["total_thickness_nm"] == pytest.approx(7.0)
        assert data["order"] == "deposition"
        assert data["layers"] == layer_table(stack)
        assert "total_time_s" not in data

    def test_with_process(self, tmp_path: Path) -> None:
        path = tmp_path / "recipe.json"
        stack, process = make_stack(), make_process()
        write_recipe_json(path, stack, process, order="top_down")
        data = json.loads(path.read_text())
        assert data["order"] == "top_down"
        # fixed 10 + W 2/0.5+1 + W 1/0.5+1 + Si 4/2+1 = 10 + 5 + 3 + 3
        assert data["total_time_s"] == pytest.approx(21.0)
        assert data["total_time_s"] == pytest.approx(process.time(stack))
        assert [layer["tag"] for layer in data["layers"]] == ["top", "mid", "bottom"]
        assert "duration_s" in data["layers"][0]

    def test_empty_stack(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.json"
        write_recipe_json(path, Stack([], SI))
        data = json.loads(path.read_text())
        assert data["layers"] == []
        assert data["total_thickness_nm"] == 0.0
