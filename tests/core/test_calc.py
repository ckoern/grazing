"""Tests of the direct reflectivity calculations."""

import numpy as np
import pytest

from xstack.core.calc import reflectivity_vs_angle, reflectivity_vs_energy
from xstack.core.materials import get
from xstack.core.physics.models import KinematicModel, ParrattModel
from xstack.core.structure import Composite, Mirror, Periodic, SingleLayer
from xstack.core.structure.coating import CoatingBuildError
from xstack.core.targets import AngleScan, EnergyScan
from xstack.core.units import deg, kev

MIRROR = Mirror(
    Periodic.create(get("W"), get("Si"), N=30, d=4.0, gamma=0.4, sigma_H=0.3, sigma_L=0.3),
    get("Si"),
)
ANGLES = np.linspace(deg(0.2), deg(3.0), 200)
ENERGIES = np.linspace(kev(15.0), kev(70.0), 150)


def test_angle_scan_matches_model_on_target() -> None:
    target = AngleScan(energy=8048.0, angle_min=ANGLES[0], angle_max=ANGLES[-1], points=200)
    expected = ParrattModel().evaluate(MIRROR.stack(), target).reflectivity
    np.testing.assert_allclose(
        reflectivity_vs_angle(MIRROR, ANGLES, energy=8048.0), expected, rtol=1e-10
    )


def test_energy_scan_matches_model_on_target() -> None:
    target = EnergyScan(angle=deg(0.2), energy_min=ENERGIES[0], energy_max=ENERGIES[-1], points=150)
    expected = ParrattModel().evaluate(MIRROR.stack(), target).reflectivity
    np.testing.assert_allclose(
        reflectivity_vs_energy(MIRROR, ENERGIES, angle=deg(0.2)), expected, rtol=1e-10
    )


def test_fixed_value_must_be_keyword() -> None:
    with pytest.raises(TypeError):
        reflectivity_vs_angle(MIRROR, ANGLES, 8048.0)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        reflectivity_vs_energy(MIRROR, ENERGIES, deg(0.2))  # type: ignore[call-arg]


def test_stack_input() -> None:
    np.testing.assert_array_equal(
        reflectivity_vs_angle(MIRROR.stack(), ANGLES, energy=8048.0),
        reflectivity_vs_angle(MIRROR, ANGLES, energy=8048.0),
    )


def test_combined_coating() -> None:
    coating = Composite.create(
        [
            SingleLayer.create(get("B4C"), t=2.0, name="cap"),
            Periodic.create(get("W"), get("Si"), N=10, d=4.0, name="ml"),
        ]
    )
    mirror = Mirror(coating, get("Si"))
    expected = reflectivity_vs_angle(mirror.stack(), ANGLES, energy=8048.0)
    np.testing.assert_array_equal(reflectivity_vs_angle(mirror, ANGLES, energy=8048.0), expected)


def test_shapes() -> None:
    assert reflectivity_vs_angle(MIRROR, deg(1.0), energy=8048.0).shape == ()
    assert reflectivity_vs_angle(MIRROR, [deg(1.0), deg(2.0)], energy=8048.0).shape == (2,)
    grid = reflectivity_vs_angle(MIRROR, ANGLES[None, :5], energy=ENERGIES[:3, None])
    assert grid.shape == (3, 5)
    single = reflectivity_vs_angle(MIRROR, ANGLES[1], energy=ENERGIES[2])
    assert grid[2, 1] == pytest.approx(float(single))


@pytest.mark.parametrize("name", ["kinematic", "kinematic_no_refraction", "parratt"])
def test_model_by_name(name: str) -> None:
    assert np.all(reflectivity_vs_angle(MIRROR, ANGLES, energy=8048.0, model=name) >= 0.0)


def test_model_choice_is_used() -> None:
    by_name = reflectivity_vs_angle(MIRROR, ANGLES, energy=8048.0, model="kinematic")
    by_object = reflectivity_vs_angle(MIRROR, ANGLES, energy=8048.0, model=KinematicModel())
    np.testing.assert_array_equal(by_name, by_object)
    assert not np.allclose(by_name, reflectivity_vs_angle(MIRROR, ANGLES, energy=8048.0))


def test_polarization() -> None:
    s = reflectivity_vs_angle(MIRROR, ANGLES, energy=8048.0, polarization="s")
    p = reflectivity_vs_angle(MIRROR, ANGLES, energy=8048.0, polarization="p")
    mean = reflectivity_vs_angle(MIRROR, ANGLES, energy=8048.0)
    np.testing.assert_allclose(mean, 0.5 * (s + p))
    assert not np.array_equal(s, p)


def test_provider() -> None:
    default = reflectivity_vs_angle(MIRROR, ANGLES, energy=8048.0)
    chantler = reflectivity_vs_angle(MIRROR, ANGLES, energy=8048.0, provider="chantler")
    assert not np.allclose(default, chantler, rtol=1e-6)


@pytest.mark.parametrize("angle", [0.0, -0.01, 2.0])
def test_invalid_angle(angle: float) -> None:
    with pytest.raises(ValueError, match="grazing angles"):
        reflectivity_vs_angle(MIRROR, [0.01, angle], energy=8048.0)


def test_invalid_energy_and_polarization() -> None:
    with pytest.raises(ValueError, match="energies"):
        reflectivity_vs_energy(MIRROR, [8048.0, 0.0], angle=0.01)
    with pytest.raises(ValueError, match="polarization"):
        reflectivity_vs_angle(MIRROR, ANGLES, energy=8048.0, polarization="x")  # type: ignore[arg-type]


def test_unknown_names() -> None:
    with pytest.raises(KeyError):
        reflectivity_vs_angle(MIRROR, ANGLES, energy=8048.0, model="nope")
    with pytest.raises(KeyError):
        reflectivity_vs_angle(MIRROR, ANGLES, energy=8048.0, provider="nope")


def test_invalid_mirror() -> None:
    mirror = MIRROR.with_parameter("gamma", expr="d")
    with pytest.raises(CoatingBuildError):
        reflectivity_vs_angle(mirror, ANGLES, energy=8048.0)
