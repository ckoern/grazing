"""Tests for reflectivity CSV export."""

import csv
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from grazing.core import materials
from grazing.core.io.reflectivity import (
    REFLECTIVITY_COLUMNS,
    reflectivity_table,
    write_reflectivity_csv,
)
from grazing.core.physics.models import ParrattModel
from grazing.core.structure import Layer, Stack
from grazing.core.targets import AngleScan, EnergyScan, Target

W = materials.get("W")
SI = materials.get("Si")
MODEL = ParrattModel()


def make_stack() -> Stack:
    layers = [Layer(W, 2.0, 0.3, "top"), Layer(SI, 3.0, 0.2, "bottom")]
    return Stack(layers, SI, 0.5, SI)


def energy_target(name: str = "scan", reflections: int = 2, points: int = 7) -> Target:
    return EnergyScan(
        angle=math.radians(0.5),
        energy_min=5000.0,
        energy_max=20000.0,
        name=name,
        reflections=reflections,
        points=points,
    )


def angle_target(name: str = "angles", points: int = 5) -> Target:
    return AngleScan(
        energy=8000.0,
        angle_min=math.radians(0.2),
        angle_max=math.radians(1.0),
        name=name,
        points=points,
    )


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def test_header(tmp_path: Path) -> None:
    path = tmp_path / "r.csv"
    write_reflectivity_csv(path, make_stack(), [energy_target()], MODEL)
    with path.open(newline="") as f:
        assert next(csv.reader(f)) == list(REFLECTIVITY_COLUMNS)


def test_no_targets_writes_header_only(tmp_path: Path) -> None:
    path = tmp_path / "r.csv"
    write_reflectivity_csv(str(path), make_stack(), [], MODEL)
    assert read(path) == []
    assert path.read_text().splitlines() == [",".join(REFLECTIVITY_COLUMNS)]


def test_row_count_is_sum_of_target_points(tmp_path: Path) -> None:
    targets = [energy_target("first", points=7), angle_target("second", points=5)]
    path = tmp_path / "r.csv"
    write_reflectivity_csv(path, make_stack(), targets, MODEL)
    rows = read(path)
    assert len(rows) == 12
    assert [r["target"] for r in rows] == ["first"] * 7 + ["second"] * 5
    assert len(reflectivity_table(make_stack(), targets, MODEL)) == 12


def test_single_reflection_columns_are_equal() -> None:
    rows = reflectivity_table(make_stack(), [angle_target()], MODEL)
    assert {r["reflections"] for r in rows} == {1}
    for r in rows:
        assert r["reflectivity"] == r["reflectivity^N"]


def test_two_reflections_columns() -> None:
    stack = make_stack()
    target = energy_target(reflections=2)
    rows = reflectivity_table(stack, [target], MODEL)
    assert {r["reflections"] for r in rows} == {2}
    r_n = np.array([r["reflectivity^N"] for r in rows])
    r_1 = np.array([r["reflectivity"] for r in rows])
    full = MODEL.evaluate(stack, target)
    single = MODEL.evaluate(stack, replace(target, reflections=1))
    np.testing.assert_allclose(r_n, full.reflectivity, rtol=1e-9)
    np.testing.assert_allclose(r_1, single.reflectivity, rtol=1e-9)
    assert np.all(r_n <= r_1 + 1e-15)


def test_energy_scan_units(tmp_path: Path) -> None:
    path = tmp_path / "r.csv"
    write_reflectivity_csv(path, make_stack(), [energy_target()], MODEL)
    rows = read(path)
    assert {r["axis"] for r in rows} == {"energy"}
    energy = np.array([float(r["energy_keV"]) for r in rows])
    angle = np.array([float(r["angle_deg"]) for r in rows])
    assert energy[0] == pytest.approx(5.0)
    assert energy[-1] == pytest.approx(20.0)
    np.testing.assert_allclose(angle, 0.5, rtol=1e-9)


def test_angle_scan_units(tmp_path: Path) -> None:
    path = tmp_path / "r.csv"
    write_reflectivity_csv(path, make_stack(), [angle_target()], MODEL)
    rows = read(path)
    assert {r["axis"] for r in rows} == {"angle"}
    energy = np.array([float(r["energy_keV"]) for r in rows])
    angle = np.array([float(r["angle_deg"]) for r in rows])
    np.testing.assert_allclose(energy, 8.0, rtol=1e-9)
    assert angle[0] == pytest.approx(0.2)
    assert angle[-1] == pytest.approx(1.0)


def test_unnamed_target_gets_numbered_label(tmp_path: Path) -> None:
    path = tmp_path / "r.csv"
    write_reflectivity_csv(path, make_stack(), [energy_target(name="")], MODEL)
    assert {r["target"] for r in read(path)} == {"target 1"}


def test_second_unnamed_target_numbered_by_position(tmp_path: Path) -> None:
    path = tmp_path / "r.csv"
    targets = [energy_target(name="named"), angle_target(name="")]
    write_reflectivity_csv(path, make_stack(), targets, MODEL)
    assert {r["target"] for r in read(path)} == {"named", "target 2"}
