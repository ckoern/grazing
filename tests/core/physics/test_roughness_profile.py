"""Roughness factors against an explicitly sliced error-function interface profile.

A Gaussian interface roughness of width σ corresponds to a refractive-index profile
n(z) = n₁ + (n₂ − n₁)·[1 + erf(z/(√2·σ))]/2. Slicing that profile into thin layers with sharp
interfaces gives a reference reflectivity without any roughness approximation. The Névot–Croce
factor reproduces it closely; its error grows with the index contrast and with σ. The Debye–Waller
factor ignores refraction at the interface and is less accurate near the critical angle.
"""

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.special import erf

from xstack.core.materials import default_library, get
from xstack.core.physics.kernel import reflectance
from xstack.core.units import energy_to_wavelength

ENERGY = 8048.0
WAVELENGTH = float(energy_to_wavelength(ENERGY))
THETA = np.radians(np.linspace(0.1, 4.0, 400))


def index(name: str) -> complex:
    db = default_library().delta_beta(get(name), np.array([ENERGY]))[0]
    return complex(1.0 - db.real + 1j * db.imag)


def smooth(n_bottom: complex, sigma: float, model: str) -> NDArray[np.float64]:
    """Reflectivity of vacuum on a medium with an interface roughness model."""
    points = THETA.size
    n = np.tile([1.0 + 0j, n_bottom], (points, 1))
    return reflectance(n, [], [sigma], THETA, np.full(points, WAVELENGTH), "s", model)


def sliced(n_bottom: complex, sigma: float, dz: float = 0.005) -> NDArray[np.float64]:
    """Reflectivity of the error-function profile sliced into sharp layers of thickness dz."""
    z = np.arange(-6.0 * sigma, 6.0 * sigma + dz, dz)
    centre = z[:-1] + dz / 2
    fraction = 0.5 * (1.0 + erf(centre / (np.sqrt(2.0) * sigma)))
    media = np.concatenate([[1.0 + 0j], 1.0 + (n_bottom - 1.0) * fraction, [n_bottom]])
    points = THETA.size
    return reflectance(
        np.tile(media, (points, 1)),
        np.full(media.size - 2, dz),
        np.zeros(media.size - 1),
        THETA,
        np.full(points, WAVELENGTH),
        "s",
        "sharp",
    )


def max_relative(result: NDArray[np.float64], reference: NDArray[np.float64]) -> float:
    shown = reference > 1e-9
    return float(np.max(np.abs(result[shown] / reference[shown] - 1.0)))


def test_sliced_reference_has_converged() -> None:
    """The slicing error is O(dz²): the 0.005 nm reference is about 4× better than 0.01 nm.

    A change below 3·10⁻⁴ between 0.01 and 0.005 nm bounds the reference error to ~10⁻⁴, well
    below the tolerances of the comparisons.
    """
    n_si = index("Si")
    assert max_relative(sliced(n_si, 0.5, dz=0.01), sliced(n_si, 0.5, dz=0.005)) < 3e-4


@pytest.mark.parametrize(
    ("material", "sigma", "tolerance"),
    [
        ("Si", 0.5, 1e-3),
        ("Si", 1.0, 6e-3),
        ("Pt", 0.5, 3e-2),
    ],
)
def test_nevot_croce_matches_erf_profile(material: str, sigma: float, tolerance: float) -> None:
    n = index(material)
    assert max_relative(smooth(n, sigma, "nevot_croce"), sliced(n, sigma)) < tolerance


@pytest.mark.parametrize("material", ["Si", "Pt"])
def test_nevot_croce_closer_to_profile_than_debye_waller(material: str) -> None:
    n = index(material)
    reference = sliced(n, 0.5)
    nc = max_relative(smooth(n, 0.5, "nevot_croce"), reference)
    dw = max_relative(smooth(n, 0.5, "debye_waller"), reference)
    assert nc < dw / 5


def test_debye_waller_and_nevot_croce_differ_by_their_exponents() -> None:
    """For one interface, R_DW / R_NC = exp(−4σ²·Re(k_z0² − k_z0·k_z1))."""
    n, sigma = index("Si"), 0.7
    k = 2.0 * np.pi / WAVELENGTH
    kz0 = k * np.sin(THETA)
    kz1 = k * np.sqrt(n**2 - np.cos(THETA) ** 2 + 0j)
    expected = np.exp(-4.0 * sigma**2 * np.real(kz0**2 - kz0 * kz1))
    ratio = smooth(n, sigma, "debye_waller") / smooth(n, sigma, "nevot_croce")
    np.testing.assert_allclose(ratio, expected, rtol=1e-10)


def test_roughness_models_reduce_to_sharp_interface() -> None:
    n = index("Si")
    sharp = smooth(n, 0.0, "sharp")
    for model in ("nevot_croce", "debye_waller"):
        np.testing.assert_allclose(smooth(n, 0.0, model), sharp, rtol=1e-14)
        assert max_relative(smooth(n, 1e-4, model), sharp) < 1e-6
