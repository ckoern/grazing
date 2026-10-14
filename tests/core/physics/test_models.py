"""Tests for the reflectivity models `ParrattModel` and `KinematicModel`."""

import dataclasses

import numpy as np
import pytest

from xstack.core.materials import delta_beta, get
from xstack.core.physics import reflectivity
from xstack.core.physics.bragg import bragg_angle, mean_delta
from xstack.core.physics.models import (
    KinematicModel,
    ParrattModel,
    ReflectivityResult,
    models,
)
from xstack.core.structure import Mirror, Periodic, Stack
from xstack.core.targets import AngleScan, EnergyScan
from xstack.core.units import deg, energy_to_wavelength, kev, mrad

W = get("W")
SI = get("Si")
C = get("C")
ENERGY = 8048.0


def periodic_stack(h: object, low: object, n: int, d: float, gamma: float = 0.4) -> Stack:
    coating = Periodic.create(h, low, N=n, d=d, gamma=gamma)  # type: ignore[arg-type]
    return Mirror(coating, SI).stack()


def wsi_stack() -> Stack:
    return periodic_stack(W, SI, 30, 4.0)


class TestRegistry:
    @pytest.mark.parametrize("name", ["parratt", "kinematic", "kinematic_no_refraction"])
    def test_registered(self, name: str) -> None:
        assert name in models
        assert models.get(name).name == name

    def test_registered_types(self) -> None:
        assert isinstance(models.get("parratt"), ParrattModel)
        assert models.get("kinematic") == KinematicModel()
        assert models.get("kinematic_no_refraction") == KinematicModel(refraction=False)

    def test_names(self) -> None:
        assert ParrattModel().name == "parratt"
        assert KinematicModel().name == "kinematic"
        assert KinematicModel(refraction=False).name == "kinematic_no_refraction"


class TestParrattModel:
    @pytest.mark.parametrize("polarization", ["s", "p", "unpolarized"])
    def test_angle_scan_matches_reflectivity(self, polarization: str) -> None:
        stack = wsi_stack()
        target = AngleScan(
            energy=ENERGY,
            angle_min=mrad(5),
            angle_max=mrad(40),
            points=60,
            polarization=polarization,  # type: ignore[arg-type]
        )
        result = ParrattModel().evaluate(stack, target)
        expected = reflectivity(stack, result.energy, result.theta, polarization)  # type: ignore[arg-type]
        np.testing.assert_allclose(result.reflectivity, expected, rtol=1e-12, atol=1e-15)

    @pytest.mark.parametrize("polarization", ["s", "p", "unpolarized"])
    def test_energy_scan_matches_reflectivity(self, polarization: str) -> None:
        stack = wsi_stack()
        target = EnergyScan(
            angle=deg(0.2),
            energy_min=kev(15),
            energy_max=kev(70),
            points=40,
            polarization=polarization,  # type: ignore[arg-type]
        )
        result = ParrattModel().evaluate(stack, target)
        expected = reflectivity(stack, result.energy, result.theta, polarization)  # type: ignore[arg-type]
        np.testing.assert_allclose(result.reflectivity, expected, rtol=1e-12, atol=1e-15)

    def test_interface_model_passed_on(self) -> None:
        coating = Periodic.create(W, SI, N=10, d=4.0, sigma_H=0.5, sigma_L=0.5)
        stack = Mirror(coating, SI).stack()
        target = AngleScan(energy=ENERGY, angle_min=mrad(10), angle_max=mrad(30), points=20)
        result = ParrattModel(interface_model="debye_waller").evaluate(stack, target)
        expected = reflectivity(
            stack, result.energy, result.theta, "unpolarized", interface_model="debye_waller"
        )
        np.testing.assert_allclose(result.reflectivity, expected, rtol=1e-12, atol=1e-15)


class TestResult:
    def test_fields_and_shapes(self) -> None:
        target = EnergyScan(angle=deg(0.2), energy_min=kev(15), energy_max=kev(70), points=25)
        result = ParrattModel().evaluate(wsi_stack(), target)
        assert isinstance(result, ReflectivityResult)
        assert result.target is target
        for array in (result.x, result.energy, result.theta, result.reflectivity):
            assert array.shape == (25,)
        np.testing.assert_allclose(result.x, target.scan_values())
        np.testing.assert_allclose(result.energy, result.x)
        np.testing.assert_allclose(result.theta, deg(0.2))
        assert np.all((result.reflectivity >= 0.0) & (result.reflectivity <= 1.0 + 1e-12))

    def test_angle_scan_fields(self) -> None:
        target = AngleScan(energy=ENERGY, angle_min=mrad(5), angle_max=mrad(40), points=15)
        result = ParrattModel().evaluate(wsi_stack(), target)
        np.testing.assert_allclose(result.x, result.theta)
        np.testing.assert_allclose(result.energy, ENERGY)

    @pytest.mark.parametrize("model", [ParrattModel(), KinematicModel()])
    def test_points_override(self, model: ParrattModel | KinematicModel) -> None:
        target = AngleScan(energy=ENERGY, angle_min=mrad(5), angle_max=mrad(40), points=15)
        result = model.evaluate(wsi_stack(), target, points=33)
        for array in (result.x, result.energy, result.theta, result.reflectivity):
            assert array.shape == (33,)
        np.testing.assert_allclose(result.x, target.scan_values(33))

    def test_points_override_with_resolution(self) -> None:
        target = AngleScan(
            energy=ENERGY, angle_min=mrad(5), angle_max=mrad(40), points=15, resolution=mrad(0.5)
        )
        result = ParrattModel().evaluate(wsi_stack(), target, points=21)
        assert result.reflectivity.shape == (21,)
        np.testing.assert_allclose(result.x, target.scan_values(21))


class TestResolution:
    def bragg_target(self, resolution: float) -> AngleScan:
        return AngleScan(
            energy=ENERGY,
            angle_min=mrad(12),
            angle_max=mrad(30),
            points=400,
            polarization="s",
            resolution=resolution,
        )

    def test_peak_is_smoothed(self) -> None:
        stack = wsi_stack()
        sharp = ParrattModel().evaluate(stack, self.bragg_target(0.0))
        smooth = ParrattModel().evaluate(stack, self.bragg_target(mrad(1.0)))
        assert sharp.reflectivity.max() > 0.3
        assert smooth.reflectivity.max() < 0.9 * sharp.reflectivity.max()
        # the integrated reflectivity is preserved by the convolution
        assert smooth.reflectivity.sum() == pytest.approx(sharp.reflectivity.sum(), rel=0.05)

    def test_close_away_from_features(self) -> None:
        stack = Stack([], SI)
        kwargs = dict(energy=ENERGY, angle_min=mrad(1), angle_max=mrad(3), points=100)
        sharp = ParrattModel().evaluate(stack, AngleScan(**kwargs))  # type: ignore[arg-type]
        smooth = ParrattModel().evaluate(
            stack,
            AngleScan(resolution=mrad(0.1), **kwargs),  # type: ignore[arg-type]
        )
        interior = slice(20, 80)
        np.testing.assert_allclose(
            smooth.reflectivity[interior], sharp.reflectivity[interior], atol=2e-3
        )

    def test_zero_resolution_is_unsmeared(self) -> None:
        stack = wsi_stack()
        target = self.bragg_target(0.0)
        result = ParrattModel().evaluate(stack, target)
        expected = reflectivity(stack, result.energy, result.theta, "s")
        np.testing.assert_allclose(result.reflectivity, expected, rtol=1e-12, atol=1e-15)

    def test_energy_resolution(self) -> None:
        stack = Stack([], SI)
        base = EnergyScan(angle=deg(0.2), energy_min=kev(15), energy_max=kev(20), points=30)
        smooth = dataclasses.replace(base, resolution=50.0)
        a = ParrattModel().evaluate(stack, base)
        b = ParrattModel().evaluate(stack, smooth)
        assert b.reflectivity.shape == (30,)
        np.testing.assert_allclose(b.reflectivity, a.reflectivity, atol=5e-3)


class TestKinematicModel:
    def test_agrees_with_parratt_for_weak_reflection(self) -> None:
        stack = periodic_stack(SI, C, 5, 4.0)
        target = AngleScan(
            energy=ENERGY, angle_min=mrad(15), angle_max=mrad(40), points=60, polarization="s"
        )
        parratt = ParrattModel().evaluate(stack, target).reflectivity
        kinematic = KinematicModel().evaluate(stack, target).reflectivity
        assert parratt.max() > 1e-8
        np.testing.assert_allclose(kinematic, parratt, rtol=0.05)

    def peak_angle(self, model: KinematicModel) -> float:
        target = AngleScan(
            energy=ENERGY,
            angle_min=mrad(12),
            angle_max=mrad(30),
            points=6000,
            polarization="s",
        )
        result = model.evaluate(wsi_stack(), target)
        return float(result.x[np.argmax(result.reflectivity)])

    def test_no_refraction_peak_at_plain_bragg_angle(self) -> None:
        wavelength = float(energy_to_wavelength(ENERGY))
        plain = float(np.arcsin(wavelength / (2 * 4.0)))
        found = self.peak_angle(KinematicModel(refraction=False))
        assert found == pytest.approx(plain, rel=5e-3)

    def test_refraction_peak_at_corrected_angle(self) -> None:
        wavelength = float(energy_to_wavelength(ENERGY))
        delta = mean_delta(delta_beta(W, ENERGY).real, delta_beta(SI, ENERGY).real, 0.4)
        corrected = float(bragg_angle(4.0, wavelength, 1, delta))
        plain = float(np.arcsin(wavelength / (2 * 4.0)))
        found = self.peak_angle(KinematicModel(refraction=True))
        assert found == pytest.approx(corrected, rel=1e-2)
        assert abs(found - corrected) < abs(found - plain)

    @pytest.mark.parametrize("model", [KinematicModel(), KinematicModel(refraction=False)])
    @pytest.mark.parametrize("polarization", ["s", "p", "unpolarized"])
    def test_reflectivity_in_unit_interval(self, model: KinematicModel, polarization: str) -> None:
        target = AngleScan(
            energy=ENERGY,
            angle_min=mrad(2),
            angle_max=mrad(60),
            points=300,
            polarization=polarization,  # type: ignore[arg-type]
        )
        r = model.evaluate(wsi_stack(), target).reflectivity
        assert np.all(np.isfinite(r))
        assert np.all((r >= 0.0) & (r <= 1.0))

    @pytest.mark.parametrize("refraction", [True, False])
    def test_unpolarized_is_mean_of_s_and_p(self, refraction: bool) -> None:
        model = KinematicModel(refraction=refraction)
        stack = wsi_stack()

        def run(polarization: str) -> np.ndarray:
            target = AngleScan(
                energy=ENERGY,
                angle_min=mrad(5),
                angle_max=mrad(60),
                points=80,
                polarization=polarization,  # type: ignore[arg-type]
            )
            return model.evaluate(stack, target).reflectivity

        np.testing.assert_allclose(run("unpolarized"), 0.5 * (run("s") + run("p")), rtol=1e-12)

    def test_energy_scan(self) -> None:
        target = EnergyScan(angle=deg(0.2), energy_min=kev(15), energy_max=kev(70), points=30)
        r = KinematicModel().evaluate(wsi_stack(), target).reflectivity
        assert r.shape == (30,)
        assert np.all((r >= 0.0) & (r <= 1.0))

    def test_name_property(self) -> None:
        assert KinematicModel().name == "kinematic"
        assert KinematicModel(refraction=False).name == "kinematic_no_refraction"
