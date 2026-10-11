"""Tests of the display ranges and candidate overlays in `xstack.gui.preview`."""

import numpy as np
import pytest

from xstack.gui.preview import compute_curves, compute_preview, display_targets
from xstack.parameters import ParameterSet
from xstack.project import Candidate
from xstack.project.templates import new_state
from xstack.structure.coating import Mirror
from xstack.targets import AngleScan, EnergyScan, TargetSet

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
