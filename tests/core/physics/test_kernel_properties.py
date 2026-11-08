"""Tier 2 kernel tests: invariances, property-based checks and numerical stress cases."""

import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.typing import ArrayLike, NDArray

from grazing.core.physics.fresnel import Polarization
from grazing.core.physics.kernel import KernelResult, parratt

FloatArr = NDArray[np.float64]
ComplexArr = NDArray[np.complex128]
POLS: tuple[Polarization, ...] = ("s", "p")
SLACK = 1e-12

PROPERTY_SETTINGS = settings(max_examples=50, deadline=None)


@dataclass(frozen=True)
class Case:
    n: ComplexArr
    thickness: FloatArr
    roughness: FloatArr
    theta: FloatArr
    wavelength: FloatArr


@st.composite
def cases(draw: st.DrawFn, with_roughness: bool = False) -> Case:
    layers = draw(st.integers(1, 6))
    points = draw(st.integers(1, 6))
    delta = draw(
        st.lists(
            st.lists(st.floats(1e-7, 1e-4), min_size=layers + 1, max_size=layers + 1),
            min_size=points,
            max_size=points,
        )
    )
    beta = draw(
        st.lists(
            st.lists(
                st.one_of(st.just(0.0), st.floats(1e-9, 1e-4)),
                min_size=layers + 1,
                max_size=layers + 1,
            ),
            min_size=points,
            max_size=points,
        )
    )
    n = np.ones((points, layers + 2), dtype=np.complex128)
    n[:, 1:] = 1.0 - np.array(delta) + 1j * np.array(beta)
    thickness = np.array(draw(st.lists(st.floats(0.1, 200.0), min_size=layers, max_size=layers)))
    if with_roughness:
        sigma = np.array(
            draw(st.lists(st.floats(0.0, 1.0), min_size=layers + 1, max_size=layers + 1))
        )
    else:
        sigma = np.zeros(layers + 1)
    theta = np.array(draw(st.lists(st.floats(1e-4, np.pi / 2), min_size=points, max_size=points)))
    wavelength = np.array(draw(st.lists(st.floats(0.01, 5.0), min_size=points, max_size=points)))
    return Case(n, thickness, sigma, theta, wavelength)


def run(case: Case, pol: Polarization = "s", model: str = "nevot_croce") -> KernelResult:
    return parratt(case.n, case.thickness, case.roughness, case.theta, case.wavelength, pol, model)


@PROPERTY_SETTINGS
@given(case=cases(), data=st.data(), pol=st.sampled_from(POLS))
def test_splitting_a_layer_preserves_r(case: Case, data: st.DataObject, pol: Polarization) -> None:
    layers = case.thickness.size
    i = data.draw(st.integers(0, layers - 1))
    frac = data.draw(st.floats(0.05, 0.95))
    n_split = np.insert(case.n, i + 1, case.n[:, i + 1], axis=1)
    t_split = np.concatenate(
        [
            case.thickness[:i],
            [frac * case.thickness[i], (1.0 - frac) * case.thickness[i]],
            case.thickness[i + 1 :],
        ]
    )
    sigma_split = np.insert(case.roughness, i + 1, 0.0)
    split = Case(n_split, t_split, sigma_split, case.theta, case.wavelength)
    np.testing.assert_allclose(run(split, pol).r, run(case, pol).r, rtol=1e-10, atol=1e-12)


@PROPERTY_SETTINGS
@given(
    case=cases(with_roughness=True),
    data=st.data(),
    pol=st.sampled_from(POLS),
    model=st.sampled_from(["sharp", "nevot_croce", "debye_waller"]),
)
def test_point_order_and_subsets_are_independent(
    case: Case, data: st.DataObject, pol: Polarization, model: str
) -> None:
    points = case.theta.size
    full = run(case, pol, model)
    perm = np.array(data.draw(st.permutations(range(points))))
    subset = np.array(
        sorted(data.draw(st.sets(st.integers(0, points - 1), min_size=1, max_size=points)))
    )
    for idx, res in (
        (perm, run(sub(case, perm), pol, model)),
        (subset, run(sub(case, subset), pol, model)),
    ):
        np.testing.assert_allclose(res.r, full.r[idx], rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(res.t, full.t[idx], rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(
            res.transmittance, full.transmittance[idx], rtol=1e-12, atol=1e-15
        )


def sub(case: Case, idx: NDArray[np.intp]) -> Case:
    return Case(case.n[idx], case.thickness, case.roughness, case.theta[idx], case.wavelength[idx])


@PROPERTY_SETTINGS
@given(case=cases(), pol=st.sampled_from(POLS))
def test_output_is_finite_bounded_and_shaped(case: Case, pol: Polarization) -> None:
    res = run(case, pol)
    points = case.theta.size
    for array in (res.r, res.t, res.reflectance, res.transmittance):
        assert array.shape == (points,)
        assert np.all(np.isfinite(array))
    assert res.r.dtype == np.complex128
    assert res.reflectance.dtype == np.float64
    assert res.transmittance.dtype == np.float64
    assert np.all(res.reflectance >= -SLACK)
    assert np.all(res.reflectance <= 1.0 + SLACK)


# Stress cases


@contextmanager
def strict_floating_point() -> Iterator[None]:
    """Turn floating-point errors and warnings into exceptions.

    Underflow of exponentials of strongly damped waves to zero is exact enough and allowed.
    """
    with warnings.catch_warnings(), np.errstate(all="raise", under="ignore"):
        warnings.simplefilter("error")
        yield


def check(res: KernelResult) -> None:
    for array in (res.r, res.t, res.reflectance, res.transmittance):
        assert np.all(np.isfinite(array))
    assert np.all(res.reflectance >= 0.0)
    assert np.all(res.reflectance <= 1.0 + SLACK)


def tile(n_media: ArrayLike, points: int) -> ComplexArr:
    return np.tile(np.asarray(n_media, dtype=np.complex128), (points, 1))


@pytest.mark.parametrize("pol", POLS)
def test_stress_5000_bilayers(pol: Polarization) -> None:
    bilayers = 5000
    n_h = 1.0 - 3e-5 + 3e-6j
    n_l = 1.0 - 5e-6 + 1e-8j
    theta = np.concatenate([np.linspace(1e-5, 0.02, 120), np.linspace(0.02, 0.1, 60)])
    media = np.concatenate([[1.0], np.tile([n_h, n_l], bilayers), [n_l]])
    thickness = np.tile([1.6, 2.4], bilayers)
    sigma = np.full(2 * bilayers + 1, 0.3)
    with strict_floating_point():
        res = parratt(tile(media, theta.size), thickness, sigma, theta, 0.1, pol)
    check(res)


@pytest.mark.parametrize("pol", POLS)
@pytest.mark.parametrize("model", ["sharp", "nevot_croce", "debye_waller"])
def test_stress_grazing_angle_down_to_1e_minus_5(pol: Polarization, model: str) -> None:
    theta = np.geomspace(1e-5, 1e-2, 50)
    n = tile([1.0, 1.0 - 3e-5 + 3e-6j, 1.0 - 5e-6 + 1e-8j], theta.size)
    with strict_floating_point():
        res = parratt(n, [3.0], [0.3, 0.3], theta, 0.1, pol, model)
    check(res)
    assert res.reflectance[0] > 0.99 or model != "sharp"


@pytest.mark.parametrize("pol", POLS)
@pytest.mark.parametrize("beta", [0.0, 1e-8, 1e-5])
def test_stress_near_critical_angle(pol: Polarization, beta: float) -> None:
    delta = 5e-6
    theta_c = np.sqrt(2 * delta)
    theta = theta_c * np.concatenate(
        [1.0 + np.linspace(-1e-3, 1e-3, 41), [1 - 1e-9, 1.0, 1 + 1e-9, 0.5, 2.0]]
    )
    n = tile([1.0, 1.0 - delta + 1j * beta, 1.0 - 2 * delta + 1j * beta], theta.size)
    with strict_floating_point():
        res = parratt(n, [20.0], [0.0, 0.0], theta, 0.1, pol)
    check(res)


@pytest.mark.parametrize("pol", POLS)
def test_stress_100_kev(pol: Polarization) -> None:
    wavelength = 0.0124
    theta = np.geomspace(1e-5, 0.01, 60)
    media = np.concatenate([[1.0], np.tile([1.0 - 2e-7 + 1e-10j, 1.0 - 4e-8 + 1e-11j], 200)])
    n = tile(np.append(media, 1.0 - 4e-8 + 1e-11j), theta.size)
    thickness = np.tile([2.0, 3.0], 200)
    sigma = np.full(401, 0.2)
    with strict_floating_point():
        res = parratt(n, thickness, sigma, theta, wavelength, pol)
    check(res)


@pytest.mark.parametrize("pol", POLS)
def test_stress_strongly_absorbing_thick_layers(pol: Polarization) -> None:
    theta = np.geomspace(1e-4, np.pi / 2, 60)
    media = [1.0, 0.95 + 0.05j, 0.95 + 0.05j, 0.98 + 0.02j, 0.95 + 0.05j]
    n = tile(media, theta.size)
    with strict_floating_point():
        res = parratt(n, [1000.0, 1000.0, 1000.0], [0.5, 0.5, 0.5, 0.5], theta, 5.0, pol)
    check(res)


@pytest.mark.parametrize("pol", POLS)
@pytest.mark.parametrize("beta", [0.0, 1e-7])
def test_stress_very_thick_layers(pol: Polarization, beta: float) -> None:
    theta = np.geomspace(1e-4, 0.5, 80)
    media = [1.0, 1.0 - 3e-5 + 1j * beta, 1.0 - 5e-6 + 1j * beta, 1.0 - 2e-5 + 1j * beta]
    n = tile(media, theta.size)
    with strict_floating_point():
        res = parratt(n, [1e4, 1e4], [0.0, 0.0, 0.0], theta, 0.1, pol)
    check(res)
    if beta == 0.0:
        np.testing.assert_allclose(res.reflectance + res.transmittance, 1.0, atol=1e-9)
