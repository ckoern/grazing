"""Tests for `xstack.gui.plots`."""

import numpy as np
import pytest
from pytestqt.qtbot import QtBot

from xstack.gui.plots import Curve, LayerProfilePlot, ReflectivityPlot, bilayer_profile, scan_axis
from xstack.materials import get
from xstack.project.templates import new_state
from xstack.structure.coating import Mirror
from xstack.structure.coatings.periodic import Periodic
from xstack.structure.stack import Stack
from xstack.targets import AngleScan, EnergyScan

pytestmark = pytest.mark.gui

ENERGY = EnergyScan(angle=0.02, energy_min=7000.0, energy_max=9000.0, points=20)
ANGLE = AngleScan(energy=8048.0, angle_min=0.005, angle_max=0.03, points=20)


def test_scan_axis_energy() -> None:
    label, factor = scan_axis(ENERGY)
    assert "keV" in label
    assert factor == pytest.approx(1e-3)


def test_scan_axis_angle() -> None:
    label, factor = scan_axis(ANGLE)
    assert "deg" in label
    assert factor == pytest.approx(180 / np.pi)


def curves_for(*targets: EnergyScan | AngleScan) -> list[Curve]:
    state = new_state("periodic")
    assert state.mirror is not None
    stack = state.mirror.stack()
    return [Curve(f"t{i}", state.model.evaluate(stack, t, 20)) for i, t in enumerate(targets)]


def test_reflectivity_one_panel_per_axis(qtbot: QtBot) -> None:
    plot = ReflectivityPlot()
    qtbot.addWidget(plot)
    plot.set_curves(curves_for(ENERGY, ANGLE), [ENERGY, ANGLE])
    assert set(plot.plot_items()) == {"energy", "angle"}
    plot.set_curves(curves_for(ENERGY), [ENERGY])
    assert set(plot.plot_items()) == {"energy"}


def test_reflectivity_curve_data(qtbot: QtBot) -> None:
    plot = ReflectivityPlot()
    qtbot.addWidget(plot)
    curves = curves_for(ENERGY)
    plot.set_curves(curves, [ENERGY])
    item = plot.plot_items()["energy"].listDataItems()[0]
    x, y = item.getData()
    assert x is not None
    assert x[0] == pytest.approx(7.0)
    assert np.allclose(y, curves[0].result.reflectivity)


def test_reflectivity_empty_curves(qtbot: QtBot) -> None:
    plot = ReflectivityPlot()
    qtbot.addWidget(plot)
    plot.set_curves([])
    assert plot.plot_items() == {}
    plot.set_curves([], [ENERGY])
    assert set(plot.plot_items()) == {"energy"}


def test_reflectivity_log_scale(qtbot: QtBot) -> None:
    plot = ReflectivityPlot()
    qtbot.addWidget(plot)
    plot.log_scale = True
    plot.set_curves(curves_for(ENERGY), [ENERGY])
    assert set(plot.plot_items()) == {"energy"}


def _stack(coating: Periodic) -> Stack:
    return Mirror(coating, get("Si")).stack()


def test_bilayer_profile_periodic() -> None:
    coating = Periodic.create(get("W"), get("Si"), N=5, d=4.0, gamma=0.3)
    profile = bilayer_profile(_stack(coating))
    assert profile is not None
    periods, gammas = profile
    assert len(periods) == 5
    assert np.allclose(periods, 4.0)
    assert np.allclose(gammas, 0.3)


def test_bilayer_profile_counts_interlayers_in_period() -> None:
    coating = Periodic.create(
        get("W"),
        get("Si"),
        N=4,
        d=4.0,
        gamma=0.4,
        interlayers={"HonL": get("C")},
        t_HonL=0.5,
    )
    profile = bilayer_profile(_stack(coating))
    assert profile is not None
    periods, gammas = profile
    assert len(periods) == 4
    assert np.allclose(periods, 4.0)
    assert np.allclose(gammas, 0.4)


def test_bilayer_profile_single_layer_none() -> None:
    state = new_state("single_layer")
    assert state.mirror is not None
    assert bilayer_profile(state.mirror.stack()) is None


def test_layer_profile_plot(qtbot: QtBot) -> None:
    plot = LayerProfilePlot()
    qtbot.addWidget(plot)
    plot.set_stack(None)
    for name in ("periodic", "depth_graded", "single_layer"):
        state = new_state(name)
        assert state.mirror is not None
        plot.set_stack(state.mirror.stack())
    plot.set_stack(None)
