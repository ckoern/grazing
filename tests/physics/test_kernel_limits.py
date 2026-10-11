"""Tier 1 kernel tests: analytic limits with synthetic refractive indices."""

import numpy as np
import pytest
from numpy.typing import NDArray

from xstack.physics.fresnel import Polarization
from xstack.physics.kernel import parratt, reflectance

TOL = 1e-12
LAM = 0.1  # nm
K = 2.0 * np.pi / LAM
POLS: tuple[Polarization, ...] = ("s", "p")
MODELS = ("sharp", "nevot_croce", "debye_waller")

ComplexArr = NDArray[np.complex128]
FloatArr = NDArray[np.float64]


def make_n(delta: object, beta: object = 0.0, points: int | None = None) -> ComplexArr:
    """Build n = 1 - delta + i beta; shape (M,) per point, tiled to (P, M) if requested."""
    d = np.atleast_1d(np.asarray(delta, dtype=float))
    b = np.broadcast_to(np.asarray(beta, dtype=float), d.shape)
    n = 1.0 - d + 1j * b
    if points is not None:
        n = np.tile(n, (points, 1))
    return np.asarray(n, dtype=np.complex128)


def kz_over_k(n: complex, theta: FloatArr) -> ComplexArr:
    """k_z/k in a medium for vacuum ambient, without cancellation."""
    kz = np.sqrt(np.sin(theta) ** 2 + (n - 1.0) * (n + 1.0) + 0j)
    return np.asarray(np.where(kz.imag < 0, -kz, kz), dtype=np.complex128)


def fresnel(n0: complex, n1: complex, theta: FloatArr, pol: str) -> ComplexArr:
    k0, k1 = kz_over_k(n0, theta), kz_over_k(n1, theta)
    if pol == "s":
        return np.asarray((k0 - k1) / (k0 + k1), dtype=np.complex128)
    return np.asarray((n1**2 * k0 - n0**2 * k1) / (n1**2 * k0 + n0**2 * k1), dtype=np.complex128)


def random_stack(
    rng: np.random.Generator, layers: int, points: int, absorbing: bool
) -> tuple[ComplexArr, FloatArr, FloatArr]:
    delta = rng.uniform(1e-6, 1e-4, size=layers + 1)
    beta = rng.uniform(1e-8, 1e-5, size=layers + 1) if absorbing else np.zeros(layers + 1)
    n = np.ones((points, layers + 2), dtype=np.complex128)
    n[:, 1:] = (1.0 - delta + 1j * beta)[None, :]
    thickness = rng.uniform(0.5, 30.0, size=layers)
    return n, thickness, np.zeros(layers + 1)


@pytest.mark.parametrize("pol", POLS)
def test_bare_substrate_matches_fresnel(pol: Polarization) -> None:
    theta = np.linspace(0.001, 1.5, 60)
    ns = 1.0 - 2e-5 + 3e-6j
    n = np.tile([1.0, ns], (theta.size, 1))
    res = parratt(n, [], [0.0], theta, LAM, pol)
    expected = fresnel(1.0, ns, theta, pol)
    np.testing.assert_allclose(res.r, expected, rtol=TOL, atol=TOL)
    np.testing.assert_allclose(res.reflectance, np.abs(expected) ** 2, rtol=TOL, atol=TOL)


@pytest.mark.parametrize("pol", POLS)
def test_total_external_reflection(pol: Polarization) -> None:
    delta = 1e-5
    theta_c = np.sqrt(2 * delta)
    theta = theta_c * np.linspace(0.05, 0.9, 30)
    n = np.tile(make_n([0.0, delta]), (theta.size, 1))
    res = parratt(n, [], [0.0], theta, LAM, pol)
    np.testing.assert_allclose(res.reflectance, 1.0, atol=TOL)
    np.testing.assert_allclose(res.transmittance, 0.0, atol=TOL)


@pytest.mark.parametrize("pol", POLS)
@pytest.mark.parametrize("position", [1, 2, 3, 4])
def test_zero_thickness_layer_leaves_r_unchanged(pol: Polarization, position: int) -> None:
    theta = np.linspace(0.002, 0.2, 25)
    delta = [0.0, 3e-5, 8e-6, 2e-5, 5e-6]
    beta = [0.0, 3e-6, 1e-7, 2e-6, 1e-8]
    n = np.tile(make_n(delta, beta), (theta.size, 1))
    thickness = [2.0, 3.0, 4.0]
    base = parratt(n, thickness, np.zeros(4), theta, LAM, pol)
    inserted_n = make_n(1.5e-5, 4e-6)[0]
    n_new = np.insert(n, position, inserted_n, axis=1)
    t_new = np.insert(thickness, position - 1, 0.0)
    res = parratt(n_new, t_new, np.zeros(5), theta, LAM, pol)
    np.testing.assert_allclose(res.r, base.r, rtol=TOL, atol=TOL)


@pytest.mark.parametrize("pol", POLS)
def test_ambient_material_top_layer_shifts_phase_only(pol: Polarization) -> None:
    theta = np.linspace(0.002, 0.2, 25)
    n = np.tile(make_n([0.0, 3e-5, 8e-6], [0.0, 3e-6, 1e-7]), (theta.size, 1))
    base = parratt(n, [2.5], [0.3, 0.2], theta, LAM, pol)
    d = 7.3
    n_new = np.insert(n, 1, 1.0 + 0j, axis=1)
    res = parratt(n_new, [d, 2.5], [0.0, 0.3, 0.2], theta, LAM, pol)
    np.testing.assert_allclose(res.reflectance, base.reflectance, rtol=TOL, atol=TOL)
    expected = base.r * np.exp(2j * K * np.sin(theta) * d)
    np.testing.assert_allclose(res.r, expected, rtol=1e-10, atol=TOL)


@pytest.mark.parametrize("pol", POLS)
def test_substrate_material_bottom_layer_leaves_r_unchanged(pol: Polarization) -> None:
    theta = np.linspace(0.002, 0.2, 25)
    sub = 1.0 - 8e-6 + 1e-7j
    n = np.tile([1.0, 1.0 - 3e-5 + 3e-6j, sub], (theta.size, 1))
    base = parratt(n, [2.5], [0.3, 0.2], theta, LAM, pol)
    n_new = np.insert(n, 2, sub, axis=1)
    res = parratt(n_new, [2.5, 11.0], [0.3, 0.2, 0.0], theta, LAM, pol)
    np.testing.assert_allclose(res.r, base.r, rtol=TOL, atol=TOL)


def test_normal_incidence_s_equals_p() -> None:
    n = np.tile(make_n([0.0, 3e-5, 8e-6, 2e-5], [0.0, 3e-6, 1e-7, 2e-6]), (3, 1))
    args = (n, [3.0, 4.0], [0.2, 0.2, 0.2], np.pi / 2, LAM)
    r_s = parratt(*args, "s")
    r_p = parratt(*args, "p")
    np.testing.assert_allclose(r_s.reflectance, r_p.reflectance, rtol=TOL, atol=TOL)


@pytest.mark.parametrize("pol", POLS)
def test_energy_conservation_lossless(pol: Polarization) -> None:
    rng = np.random.default_rng(1234)
    theta = np.concatenate([np.linspace(1e-3, 0.05, 40), np.linspace(0.05, 1.5, 20)])
    for layers in (1, 2, 5, 10):
        n, thickness, sigma = random_stack(rng, layers, theta.size, absorbing=False)
        res = parratt(n, thickness, sigma, theta, LAM, pol)
        np.testing.assert_allclose(res.reflectance + res.transmittance, 1.0, atol=TOL)


@pytest.mark.parametrize("pol", POLS)
def test_energy_conservation_covers_both_sides_of_critical_angle(pol: Polarization) -> None:
    delta = 2e-5
    theta_c = np.sqrt(2 * delta)
    theta = theta_c * np.array([0.3, 0.9, 1.1, 2.0, 5.0])
    n = np.tile(make_n([0.0, delta, 5e-6]), (theta.size, 1))
    res = parratt(n, [10.0], [0.0, 0.0], theta, LAM, pol)
    np.testing.assert_allclose(res.reflectance + res.transmittance, 1.0, atol=TOL)


@pytest.mark.parametrize("pol", POLS)
def test_absorption_reduces_r_plus_t(pol: Polarization) -> None:
    rng = np.random.default_rng(99)
    theta = np.linspace(1e-3, 0.3, 50)
    n, thickness, sigma = random_stack(rng, 4, theta.size, absorbing=True)
    res = parratt(n, thickness, sigma, theta, LAM, pol)
    assert np.all(res.reflectance + res.transmittance < 1.0)


@pytest.mark.parametrize("pol", POLS)
def test_single_film_airy_formula(pol: Polarization) -> None:
    theta = np.linspace(0.002, 0.3, 40)
    n1, n2 = 1.0 - 4e-5 + 2e-6j, 1.0 - 1e-5 + 1e-7j
    d = 6.5
    n = np.tile([1.0, n1, n2], (theta.size, 1))
    r01 = fresnel(1.0, n1, theta, pol)
    r12_k0 = kz_over_k(n1, theta)
    r12_k1 = kz_over_k(n2, theta)
    if pol == "s":
        r12 = (r12_k0 - r12_k1) / (r12_k0 + r12_k1)
    else:
        r12 = (n2**2 * r12_k0 - n1**2 * r12_k1) / (n2**2 * r12_k0 + n1**2 * r12_k1)
    phase = np.exp(2j * K * r12_k0 * d)
    expected = (r01 + r12 * phase) / (1 + r01 * r12 * phase)
    res = parratt(n, [d], [0.0, 0.0], theta, LAM, pol)
    np.testing.assert_allclose(res.r, expected, rtol=TOL, atol=TOL)


@pytest.mark.parametrize("pol", POLS)
def test_interface_models_identical_without_roughness(pol: Polarization) -> None:
    rng = np.random.default_rng(5)
    theta = np.linspace(1e-3, 0.1, 30)
    n, thickness, sigma = random_stack(rng, 4, theta.size, absorbing=True)
    results = [parratt(n, thickness, sigma, theta, LAM, pol, m) for m in MODELS]
    for other in results[1:]:
        np.testing.assert_allclose(other.r, results[0].r, rtol=TOL, atol=TOL)
        np.testing.assert_allclose(other.t, results[0].t, rtol=TOL, atol=TOL)


@pytest.mark.parametrize("pol", POLS)
def test_sharp_model_ignores_roughness(pol: Polarization) -> None:
    rng = np.random.default_rng(6)
    theta = np.linspace(1e-3, 0.1, 30)
    n, thickness, _ = random_stack(rng, 4, theta.size, absorbing=True)
    rough = parratt(n, thickness, np.full(5, 0.5), theta, LAM, pol, "sharp")
    smooth = parratt(n, thickness, np.zeros(5), theta, LAM, pol, "sharp")
    np.testing.assert_array_equal(rough.r, smooth.r)
    nc = parratt(n, thickness, np.full(5, 0.5), theta, LAM, pol, "nevot_croce")
    assert not np.allclose(nc.reflectance, smooth.reflectance, rtol=1e-3)


@pytest.mark.parametrize("pol", POLS)
def test_nevot_croce_single_interface(pol: Polarization) -> None:
    theta = np.linspace(0.002, 0.2, 40)
    ns = 1.0 - 2e-5 + 3e-6j
    sigma = 0.6
    n = np.tile([1.0, ns], (theta.size, 1))
    res = parratt(n, [], [sigma], theta, LAM, pol, "nevot_croce")
    kz0 = K * kz_over_k(1.0, theta)
    kz1 = K * kz_over_k(ns, theta)
    expected = fresnel(1.0, ns, theta, pol) * np.exp(-2 * kz0 * kz1 * sigma**2)
    np.testing.assert_allclose(res.r, expected, rtol=TOL, atol=TOL)


class TestValidation:
    n = np.tile([1.0, 1.0 - 1e-5, 1.0 - 2e-5], (3, 1))
    theta = np.array([0.01, 0.02, 0.03])

    def call(self, **overrides: object) -> None:
        args: dict[str, object] = {
            "n": self.n,
            "thickness": [1.0],
            "roughness": [0.0, 0.0],
            "theta": self.theta,
            "wavelength": LAM,
        }
        args.update(overrides)
        parratt(**args)  # type: ignore[arg-type]

    def test_valid_call(self) -> None:
        self.call()

    @pytest.mark.parametrize(
        "overrides",
        [
            {"n": np.array([1.0, 0.99])},
            {"n": np.ones((3, 1))},
            {"thickness": [1.0, 2.0]},
            {"thickness": []},
            {"roughness": [0.0]},
            {"roughness": [0.0, 0.0, 0.0]},
            {"thickness": [-1.0]},
            {"roughness": [0.0, -0.1]},
            {"roughness": [-0.1, 0.0]},
            {"theta": 0.0},
            {"theta": -0.1},
            {"theta": np.pi / 2 + 0.01},
            {"theta": np.array([0.01, 0.0, 0.03])},
            {"wavelength": 0.0},
            {"wavelength": -0.1},
        ],
    )
    def test_value_errors(self, overrides: dict[str, object]) -> None:
        with pytest.raises(ValueError):
            self.call(**overrides)

    def test_theta_upper_bound_inclusive(self) -> None:
        self.call(theta=np.pi / 2)

    @pytest.mark.parametrize("function", [parratt, reflectance])
    def test_unknown_polarization(self, function: object) -> None:
        with pytest.raises(ValueError, match="polarization"):
            function(self.n, [1.0], [0.0, 0.0], self.theta, LAM, "circular")  # type: ignore[operator]

    @pytest.mark.parametrize("function", [parratt, reflectance])
    def test_unknown_interface_model(self, function: object) -> None:
        with pytest.raises(KeyError):
            function(self.n, [1.0], [0.0, 0.0], self.theta, LAM, "s", "bogus")  # type: ignore[operator]


def test_unpolarized_is_mean_of_s_and_p() -> None:
    rng = np.random.default_rng(7)
    theta = np.linspace(1e-3, 0.5, 40)
    n, thickness, _ = random_stack(rng, 3, theta.size, absorbing=True)
    sigma = np.full(4, 0.3)
    r_s = reflectance(n, thickness, sigma, theta, LAM, "s")
    r_p = reflectance(n, thickness, sigma, theta, LAM, "p")
    r_u = reflectance(n, thickness, sigma, theta, LAM, "unpolarized")
    np.testing.assert_allclose(r_u, 0.5 * (r_s + r_p), rtol=TOL, atol=TOL)
    np.testing.assert_allclose(reflectance(n, thickness, sigma, theta, LAM), r_u)
