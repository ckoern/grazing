"""Tests of the display ranges and candidate overlays in `grazing.gui.design.preview`."""

import numpy as np
import pytest

import grazing.core.materials
from grazing.core.parameters import ParameterSet
from grazing.core.structure.coating import Mirror
from grazing.core.targets import AngleScan, EnergyScan, TargetSet
from grazing.design.manufacturing import ErrorModel, ToleranceResult, tolerance_analysis
from grazing.design.project import Candidate, ProjectState
from grazing.design.project.templates import new_state
from grazing.gui.design.preview import (
    compute_curves,
    compute_preview,
    display_targets,
    lowest_energy,
    tolerance_curves,
)

pytestmark = pytest.mark.gui


def mixed_targets() -> tuple[EnergyScan, AngleScan]:
    energy = EnergyScan(angle=0.02, energy_min=7000.0, energy_max=8000.0, points=200)
    angle = AngleScan(energy=8000.0, angle_min=0.01, angle_max=0.03, points=100)
    return energy, angle


def test_display_targets_without_display_keeps_targets() -> None:
    targets = mixed_targets()
    assert display_targets(targets, None) == targets
    assert display_targets(targets, {}) == targets


def test_display_targets_replaces_range_and_scales_points() -> None:
    energy, angle = mixed_targets()
    shown = display_targets([energy, angle], {"energy": (6000.0, 8000.0)})
    assert isinstance(shown[0], EnergyScan)
    assert shown[0].scan_range == (6000.0, 8000.0)
    assert shown[0].points == 400
    assert shown[1] is angle


def test_display_targets_replaces_angle_range() -> None:
    _, angle = mixed_targets()
    shown = display_targets([angle], {"angle": (0.01, 0.02)})
    assert isinstance(shown[0], AngleScan)
    assert shown[0].scan_range == (0.01, 0.02)
    assert shown[0].points == 50


def test_display_targets_clips_points() -> None:
    energy, _ = mixed_targets()
    narrow = display_targets([energy], {"energy": (7000.0, 7010.0)})[0]
    wide = display_targets([energy], {"energy": (1000.0, 100000.0)})[0]
    assert narrow.points == 50
    assert wide.points == 1000


def test_compute_curves_counts_and_empty_cases() -> None:
    state = new_state("periodic")
    energy, angle = mixed_targets()
    state = state.evolve(targets=TargetSet((energy, angle), "average_merit"))
    assert state.mirror is not None
    curves = compute_curves(state, state.mirror)
    assert len(curves) == 2
    shown = compute_curves(state, state.mirror, display={"energy": (7500.0, 8500.0)})
    assert shown[0].x[0] == pytest.approx(7500.0)
    assert shown[0].x[-1] == pytest.approx(8500.0)
    assert shown[1].x[0] == pytest.approx(0.01)
    assert compute_curves(state.evolve(targets=None), state.mirror) == ()


def test_compute_curves_uses_values_and_rejects_unbuildable() -> None:
    state = new_state("periodic")
    assert state.mirror is not None
    base = compute_curves(state, state.mirror)
    changed = compute_curves(state, state.mirror, {"d": 4.5})
    assert not np.allclose(base[0].reflectivity, changed[0].reflectivity)
    assert compute_curves(state, state.mirror, {"gamma": 1.5}) == ()
    assert compute_curves(state, state.mirror, {"unknown": 1.0}) == ()


def test_compute_curves_unbuildable_coating() -> None:
    state = new_state("periodic")
    assert state.mirror is not None
    params = state.mirror.coating.parameters
    broken = state.mirror.coating.with_parameters(
        ParameterSet(
            p.evolve(expr="1.5", vary=False) if name == "gamma" else p for name, p in params.items()
        )
    )
    mirror = Mirror(broken, state.mirror.substrate)
    assert compute_curves(state, mirror) == ()


def test_compute_preview_display_grid_and_target_grid() -> None:
    state = new_state("periodic")
    target = state.targets.targets[0]  # type: ignore[union-attr]
    lo, hi = target.scan_range
    preview = compute_preview(state, display={"energy": (lo - 500.0, hi + 500.0)})
    assert preview.error is None
    assert preview.curves[0].x[0] == pytest.approx(lo - 500.0)
    assert preview.curves[0].x[-1] == pytest.approx(hi + 500.0)
    assert preview.results[0].x[0] == pytest.approx(lo)
    assert preview.results[0].x[-1] == pytest.approx(hi)
    assert preview.evaluation is not None
    plain = compute_preview(state)
    assert plain.curves == plain.results


def test_compute_preview_overlays() -> None:
    state = new_state("periodic")
    assert state.mirror is not None
    state = state.evolve(
        candidates=(Candidate("a", state.mirror, "manual"), Candidate("b", state.mirror, "manual"))
    )
    preview = compute_preview(state, ("b", "missing", "a"), {"energy": (7800.0, 8300.0)})
    assert [name for name, _ in preview.overlays] == ["b", "a"]
    for _, results in preview.overlays:
        assert len(results) == 1
        assert results[0].x[0] == pytest.approx(7800.0)
        assert results[0].x[-1] == pytest.approx(8300.0)
    assert compute_preview(state).overlays == ()


def periodic_media() -> frozenset[grazing.core.materials.Material]:
    state = new_state("periodic")
    assert state.mirror is not None
    return frozenset(state.mirror.stack().media)


def test_lowest_energy_of_periodic_stack() -> None:
    value = lowest_energy(periodic_media())
    assert 10.0 < value < 100.0
    assert lowest_energy(periodic_media()) == value


def test_lowest_energy_of_vacuum_is_zero() -> None:
    assert lowest_energy(frozenset([grazing.core.materials.VACUUM])) == 0.0


def test_display_targets_energy_from_zero_starts_at_first_grid_point() -> None:
    energy, _ = mixed_targets()
    shown = display_targets([energy], {"energy": (0.0, 10e3)}, energy_floor=0.0)[0]
    assert isinstance(shown, EnergyScan)
    assert shown.energy_min == pytest.approx(10e3 / (shown.points - 1))
    assert shown.energy_max == pytest.approx(10e3)


def test_display_targets_energy_from_zero_is_raised_to_floor() -> None:
    energy, _ = mixed_targets()
    shown = display_targets([energy], {"energy": (0.0, 10e3)}, energy_floor=50.0)[0]
    assert isinstance(shown, EnergyScan)
    assert shown.energy_min == pytest.approx(50.0)
    assert shown.energy_max == pytest.approx(10e3)


def test_display_targets_angle_from_zero_starts_above_zero() -> None:
    _, angle = mixed_targets()
    shown = display_targets([angle], {"angle": (0.0, 0.02)})[0]
    assert isinstance(shown, AngleScan)
    assert shown.angle_min == pytest.approx(0.02 / (shown.points - 1))
    assert shown.angle_min > 0.0
    assert shown.angle_max == pytest.approx(0.02)


def tolerance_setup() -> tuple[ProjectState, ToleranceResult]:
    state = new_state("periodic")
    result = tolerance_analysis(
        state.problem().evolve(points=20),
        errors=ErrorModel(thickness_random=0.02),
        samples=3,
        seed=0,
    )
    return state, result


def test_tolerance_curves_without_display_returns_analysis_curves() -> None:
    state, result = tolerance_setup()
    curves = tolerance_curves(state, result)
    assert curves.nominal is result.nominal.results
    assert len(curves.samples) == len(result.samples)
    for drawn, sample in zip(curves.samples, result.samples, strict=True):
        assert drawn is sample.results


def test_tolerance_curves_with_display_cover_display_range() -> None:
    state, result = tolerance_setup()
    curves = tolerance_curves(state, result, {"energy": (0.0, 12e3)})
    stacks = [result.nominal.stack, *(sample.stack for sample in result.samples)]
    floor = lowest_energy(frozenset(m for stack in stacks if stack for m in stack.media))
    assert len(curves.samples) == 3
    drawn = [*curves.nominal, *(r for sample in curves.samples for r in sample)]
    assert drawn
    for curve in drawn:
        assert curve.x[0] >= floor
        assert curve.x[-1] == pytest.approx(12e3)
