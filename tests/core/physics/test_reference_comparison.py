"""Tier 3: comparison of the Parratt kernel with independent implementations."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from grazing.core.physics.fresnel import Polarization
from grazing.core.physics.kernel import parratt
from tests.reference.matrix_methods import abeles, transfer_matrix, transfer_matrix_mp


@st.composite
def stacks(draw: st.DrawFn, rough: bool) -> dict[str, object]:
    layers = draw(st.integers(0, 20))
    media = layers + 2
    wavelength = draw(st.floats(0.01, 15.0))
    # Physical scaling of the optical constants: δ ≈ c·λ² with c up to ~3e-3 nm⁻² (dense
    # metals), β/δ between 0 and ~2.
    c = np.array(draw(st.lists(st.floats(1e-5, 3e-3), min_size=media, max_size=media)))
    ratio = np.array(draw(st.lists(st.floats(0.0, 2.0), min_size=media, max_size=media)))
    delta = np.minimum(c * wavelength**2, 0.2)
    n = 1 - delta + 1j * delta * ratio
    n[0] = 1.0
    thickness = np.array(draw(st.lists(st.floats(0.5, 50.0), min_size=layers, max_size=layers)))
    roughness = (
        np.array(draw(st.lists(st.floats(0.0, 1.0), min_size=media - 1, max_size=media - 1)))
        if rough
        else np.zeros(media - 1)
    )
    return {
        "n": n,
        "thickness": thickness,
        "roughness": roughness,
        "theta": draw(st.floats(1e-4, np.pi / 2)),
        "wavelength": wavelength,
        "polarization": draw(st.sampled_from(["s", "p"])),
    }


def kernel(case: dict[str, object]) -> tuple[complex, complex]:
    res = parratt(
        np.asarray(case["n"])[None, :],
        case["thickness"],  # type: ignore[arg-type]
        case["roughness"],  # type: ignore[arg-type]
        case["theta"],  # type: ignore[arg-type]
        case["wavelength"],  # type: ignore[arg-type]
        case["polarization"],  # type: ignore[arg-type]
    )
    return complex(res.r[0]), complex(res.t[0])


def assert_close(actual: complex, expected: complex, rtol: float, atol: float) -> None:
    assert abs(actual - expected) <= atol + rtol * abs(expected), (actual, expected)


@settings(max_examples=300, deadline=None)
@given(case=stacks(rough=False))
def test_parratt_matches_abeles(case: dict[str, object]) -> None:
    r, _ = kernel(case)
    expected = abeles(
        case["n"],  # type: ignore[arg-type]
        case["thickness"],  # type: ignore[arg-type]
        case["theta"],  # type: ignore[arg-type]
        case["wavelength"],  # type: ignore[arg-type]
        case["polarization"],  # type: ignore[arg-type]
    )
    assert_close(r, expected, rtol=1e-8, atol=1e-10)


@settings(max_examples=300, deadline=None)
@given(case=stacks(rough=True))
def test_parratt_matches_transfer_matrix_with_roughness(case: dict[str, object]) -> None:
    r, t = kernel(case)
    r_ref, t_ref = transfer_matrix(**case)  # type: ignore[arg-type]
    assert_close(r, r_ref, rtol=1e-8, atol=1e-10)
    assert_close(t, t_ref, rtol=1e-8, atol=1e-10)


W_LIKE = 1 - 3.0e-5 + 2.5e-6j
SI_LIKE = 1 - 7.6e-6 + 1.8e-7j
C_LIKE = 1 - 6.5e-6 + 9e-9j

HARD_CASES = {
    "grazing_1e-5": dict(n=[1, W_LIKE, SI_LIKE], thickness=[3.0], theta=1e-5, wavelength=0.154),
    "near_critical": dict(
        n=[1, SI_LIKE], thickness=[], theta=np.sqrt(2 * 7.6e-6), wavelength=0.154
    ),
    "evanescent_layers": dict(
        n=[1, W_LIKE, C_LIKE, W_LIKE, SI_LIKE],
        thickness=[2.0, 3.0, 2.0],
        theta=2e-3,
        wavelength=0.154,
    ),
    "periodic_bragg": dict(
        n=[1, *([W_LIKE, C_LIKE] * 10), SI_LIKE],
        thickness=[1.6, 2.4] * 10,
        theta=float(np.arcsin(0.154 / 8.0)),
        wavelength=0.154,
    ),
    "soft_xray_absorbing": dict(
        n=[1, 0.93 + 0.006j, 1.0 - 0.001 + 0.0018j, 0.99 + 0.002j],
        thickness=[2.8, 4.1],
        theta=1.2,
        wavelength=13.5,
    ),
    "hard_xray_100kev": dict(
        n=[1, 1 - 1.2e-7 + 1e-10j, 1 - 2.5e-8 + 1e-13j, 1 - 3e-8 + 1e-12j],
        thickness=[1.0, 2.0],
        theta=1e-3,
        wavelength=0.0124,
    ),
}


@pytest.mark.parametrize("name", HARD_CASES)
@pytest.mark.parametrize("polarization", ["s", "p"])
@pytest.mark.parametrize("sigma", [0.0, 0.4])
def test_parratt_matches_arbitrary_precision(
    name: str, polarization: Polarization, sigma: float
) -> None:
    case = HARD_CASES[name]
    n = np.asarray(case["n"], dtype=complex)
    thickness = np.asarray(case["thickness"], dtype=float)
    roughness = np.full(len(n) - 1, sigma)
    res = parratt(n[None, :], thickness, roughness, case["theta"], case["wavelength"], polarization)
    r_ref, t_ref = transfer_matrix_mp(
        n, thickness, roughness, case["theta"], case["wavelength"], polarization
    )
    assert_close(complex(res.r[0]), r_ref, rtol=1e-11, atol=1e-14)
    assert_close(complex(res.t[0]), t_ref, rtol=1e-11, atol=1e-14)
