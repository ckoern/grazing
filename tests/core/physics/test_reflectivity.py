"""Tests for `reflectivity` and `stack_response` with real materials."""

import numpy as np
import pytest
from numpy.typing import ArrayLike, NDArray

from grazing.core import materials
from grazing.core.materials import MaterialLibrary
from grazing.core.physics.reflectivity import reflectivity, stack_response
from grazing.core.structure.stack import Layer, Stack
from grazing.core.units import energy_to_wavelength

ENERGY = 8048.0
SI = materials.get("Si")
W = materials.get("W")


def multilayer(period: float = 4.0, gamma: float = 0.4, n_periods: int = 30) -> Stack:
    layers = [Layer(W, gamma * period, 0.2), Layer(SI, (1 - gamma) * period, 0.2)] * n_periods
    return Stack(layers, SI, substrate_roughness=0.2)


def small_stack() -> Stack:
    return Stack([Layer(W, 2.0, 0.3), Layer(SI, 3.0, 0.3)] * 3, SI, 0.3)


def test_energy_array_with_scalar_theta() -> None:
    stack = small_stack()
    energy = np.linspace(7000.0, 9000.0, 7)
    result = reflectivity(stack, energy, 0.015)
    assert result.shape == (7,)
    expected = [float(reflectivity(stack, e, 0.015)) for e in energy]
    np.testing.assert_allclose(result, expected, rtol=1e-12)


def test_scalar_energy_with_theta_array() -> None:
    stack = small_stack()
    theta = np.linspace(0.005, 0.04, 9)
    result = reflectivity(stack, ENERGY, theta)
    assert result.shape == (9,)
    expected = [float(reflectivity(stack, ENERGY, t)) for t in theta]
    np.testing.assert_allclose(result, expected, rtol=1e-12)


def test_scalar_inputs_give_zero_dimensional_result() -> None:
    result = reflectivity(small_stack(), ENERGY, 0.015)
    assert np.ndim(result) == 0


@pytest.mark.parametrize("polarization", ["s", "p", "unpolarized"])
def test_two_dimensional_map(polarization: str) -> None:
    stack = small_stack()
    energy = np.linspace(7000.0, 9000.0, 4)
    theta = np.linspace(0.005, 0.04, 6)
    result = reflectivity(stack, energy[:, None], theta[None, :], polarization)  # type: ignore[arg-type]
    assert result.shape == (4, 6)
    for i, e in enumerate(energy):
        for j, t in enumerate(theta):
            single = reflectivity(stack, e, t, polarization)  # type: ignore[arg-type]
            np.testing.assert_allclose(result[i, j], single, rtol=1e-12)


def test_bare_silicon_reflectivity() -> None:
    stack = Stack([], SI)
    r = reflectivity(stack, ENERGY, np.array([1e-3, 20e-3]))
    assert r[0] > 0.97
    assert r[1] < 1e-3


def test_default_polarization_is_unpolarized() -> None:
    stack = small_stack()
    theta = np.linspace(0.005, 0.04, 6)
    r_s = reflectivity(stack, ENERGY, theta, "s")
    r_p = reflectivity(stack, ENERGY, theta, "p")
    np.testing.assert_allclose(reflectivity(stack, ENERGY, theta), 0.5 * (r_s + r_p), rtol=1e-12)


def test_interface_model_is_forwarded() -> None:
    stack = small_stack()
    theta = np.linspace(0.005, 0.04, 6)
    sharp = reflectivity(stack, ENERGY, theta, interface_model="sharp")
    rough = reflectivity(stack, ENERGY, theta, interface_model="nevot_croce")
    assert not np.allclose(sharp, rough, rtol=1e-3)
    with pytest.raises(KeyError):
        reflectivity(stack, ENERGY, theta, interface_model="bogus")


def test_bragg_peak_of_w_si_multilayer() -> None:
    period, gamma = 4.0, 0.4
    stack = multilayer(period, gamma)
    wavelength = energy_to_wavelength(ENERGY)
    theta_bragg = float(np.arcsin(wavelength / (2 * period)))
    theta = np.linspace(0.8 * theta_bragg, 1.3 * theta_bragg, 2001)
    r = reflectivity(stack, ENERGY, theta)
    peak = theta[int(np.argmax(r))]
    assert r.max() > 0.3
    assert peak > theta_bragg
    assert peak == pytest.approx(theta_bragg, rel=0.08)

    # Refraction-corrected first-order Bragg condition with the mean index of the period.
    n_w = materials.refractive_index(W, ENERGY)
    n_si = materials.refractive_index(SI, ENERGY)
    delta_mean = 1.0 - (gamma * n_w + (1 - gamma) * n_si).real
    sin_corrected = np.sqrt((wavelength / (2 * period)) ** 2 + 2 * delta_mean)
    assert peak == pytest.approx(float(np.arcsin(sin_corrected)), rel=0.03)


def test_stack_response_returns_flattened_arrays() -> None:
    stack = small_stack()
    energy = np.linspace(7000.0, 9000.0, 4)
    theta = np.linspace(0.005, 0.04, 6)
    res = stack_response(stack, energy[:, None], theta[None, :], "s")
    for array in (res.r, res.t, res.reflectance, res.transmittance):
        assert array.shape == (24,)
    maps = reflectivity(stack, energy[:, None], theta[None, :], "s")
    np.testing.assert_allclose(res.reflectance.reshape(4, 6), maps, rtol=1e-12)
    np.testing.assert_allclose(np.abs(res.r) ** 2, res.reflectance, rtol=1e-12)


def test_stack_response_scalar_input_has_one_point() -> None:
    res = stack_response(small_stack(), ENERGY, 0.015, "p")
    assert res.r.shape == (1,)


class ScaledLibrary(MaterialLibrary):
    """Library whose optical constants are scaled, to check that it is used."""

    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = scale
        self.calls = 0

    def delta_beta(
        self, material: materials.Material | str, energy: ArrayLike
    ) -> NDArray[np.complex128]:
        self.calls += 1
        return self.scale * super().delta_beta(material, energy)


def test_custom_library_is_used() -> None:
    stack = Stack([], SI)
    theta = np.array([2e-3, 3e-3, 4e-3, 5e-3])
    default = reflectivity(stack, ENERGY, theta)
    library = ScaledLibrary(0.5)
    custom = reflectivity(stack, ENERGY, theta, library=library)
    assert library.calls > 0
    assert not np.allclose(custom, default, rtol=1e-3)
    response = stack_response(stack, ENERGY, theta, library=library)
    assert not np.allclose(response.reflectance, stack_response(stack, ENERGY, theta).reflectance)


def test_library_with_modified_density_changes_reflectivity() -> None:
    lighter = SI.with_density(1.0)
    library = MaterialLibrary([lighter])
    theta = np.array([2e-3, 3e-3, 4e-3, 5e-3])
    default = reflectivity(Stack([], SI), ENERGY, theta)
    custom = reflectivity(Stack([], library.get("Si")), ENERGY, theta, library=library)
    assert not np.allclose(custom, default, rtol=1e-3)
