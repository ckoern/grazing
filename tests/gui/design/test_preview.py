"""Tests for `xstack.gui.design.preview`."""

import pytest

from xstack.core.parameters import ParameterSet
from xstack.core.structure.coating import Mirror
from xstack.design.project.templates import new_state
from xstack.gui.design.preview import compute_preview

pytestmark = pytest.mark.gui


def test_complete_state() -> None:
    state = new_state("periodic")
    preview = compute_preview(state)
    assert preview.error is None
    assert preview.stack is not None
    assert preview.stack.num_layers == 80
    assert len(preview.results) == 1
    assert preview.evaluation is not None
    assert 0.0 < preview.evaluation.merit <= 1.0
    assert preview.state is state


def test_results_per_target() -> None:
    from xstack.core.targets import AngleScan, EnergyScan, TargetSet

    state = new_state("periodic")
    first = state.targets.targets[0]  # type: ignore[union-attr]
    second = EnergyScan(angle=0.03, energy_min=7000.0, energy_max=9000.0, points=50)
    third = AngleScan(energy=8048.0, angle_min=0.01, angle_max=0.03)
    state = state.evolve(targets=TargetSet((first, second, third), "average_merit"))
    preview = compute_preview(state)
    assert preview.error is None
    assert len(preview.results) == 3


def test_no_targets() -> None:
    preview = compute_preview(new_state("periodic").evolve(targets=None))
    assert preview.stack is not None
    assert preview.results == ()
    assert preview.evaluation is None
    assert preview.error is None


def test_no_merit() -> None:
    preview = compute_preview(new_state("periodic").evolve(merit=None))
    assert preview.stack is not None
    assert len(preview.results) == 1
    assert preview.evaluation is None
    assert preview.error is None


def test_unbuildable_design_sets_error() -> None:
    state = new_state("periodic")
    assert state.mirror is not None
    params = state.mirror.coating.parameters
    broken = state.mirror.coating.with_parameters(
        ParameterSet(
            p.evolve(expr="1.5", vary=False) if name == "gamma" else p for name, p in params.items()
        )
    )
    preview = compute_preview(state.evolve(mirror=Mirror(broken, state.mirror.substrate)))
    assert preview.stack is None
    assert preview.error is not None
    assert "gamma" in preview.error
    assert preview.results == ()


def test_no_mirror() -> None:
    preview = compute_preview(new_state("periodic").evolve(mirror=None))
    assert preview.stack is None
    assert preview.error is not None
