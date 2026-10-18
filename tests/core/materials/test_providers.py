from importlib import resources

import numpy as np
import pytest
import xraylib

from xstack.core.materials import (
    HenkeProvider,
    Material,
    XraylibProvider,
    delta_beta,
    providers,
    refractive_index,
)
from xstack.core.materials.providers import compton_f2

SI = Material("Si", "Si", 2.33)
W = Material("W", "W", 19.25)
C = Material("C", "C", 2.2)
AU = Material("Au", "Au", 19.32)


def raw_henke_table(symbol: str) -> np.ndarray:
    path = resources.files("xstack.core.materials").joinpath("data", "henke", f"{symbol}.nff")
    with resources.as_file(path) as p:
        return np.loadtxt(p, skiprows=1)


@pytest.mark.parametrize("z", range(1, 93))
def test_all_henke_elements_load(z: int) -> None:
    f = HenkeProvider().scattering_factors(z, np.array([1000.0, 10000.0]))
    assert np.all(np.isfinite(f))
    assert np.all(f.imag > 0)


def test_henke_reproduces_table_nodes() -> None:
    table = raw_henke_table("w")
    nodes = table[(table[:, 1] > -9000) & (table[:, 0] > 50)][::25]
    f = HenkeProvider().scattering_factors(74, nodes[:, 0])
    np.testing.assert_allclose(f.real, nodes[:, 1], rtol=1e-12)
    np.testing.assert_allclose(f.imag, nodes[:, 2], rtol=1e-12)


def test_henke_duplicate_energies_are_averaged() -> None:
    table = raw_henke_table("pt")
    rows = table[np.isclose(table[:, 0], 6.199)]
    assert len(rows) == 2
    f = HenkeProvider().scattering_factors(78, np.array([6.199]))
    assert f.real[0] == pytest.approx(rows[:, 1].mean())
    assert f.imag[0] == pytest.approx(rows[:, 2].mean())


@pytest.mark.parametrize("energy", [5.0, 30001.0])
def test_henke_out_of_range(energy: float) -> None:
    with pytest.raises(ValueError, match="Henke data for W"):
        delta_beta(W, energy, provider="henke")


def test_henke_extended_range_elements() -> None:
    low, high = HenkeProvider().element_range(24)
    assert low < 10.0
    assert high == pytest.approx(30000.0)


def test_henke_rejects_unknown_element() -> None:
    with pytest.raises(ValueError, match="Z = 1-92"):
        HenkeProvider().scattering_factors(93, np.array([1000.0]))


def test_xraylib_high_energy() -> None:
    db = delta_beta(W, np.array([50e3, 100e3, 200e3]), provider="xraylib")
    assert np.all(db.real > 0)
    assert np.all(db.imag > 0)
    assert np.all(np.diff(db.real) < 0)


@pytest.mark.parametrize(
    "material",
    [
        W,
        SI,
        C,
        Material("B4C", "B4C", 2.52),
        Material("Pt", "Pt", 21.45),
        Material("Ni", "Ni", 8.9),
    ],
    ids=lambda m: m.name,
)
def test_henke_xraylib_continuity_at_30kev(material: Material) -> None:
    henke = delta_beta(material, 30000.0, provider="henke")
    xrl = delta_beta(material, 30000.0, provider="xraylib")
    assert xrl.real == pytest.approx(henke.real, rel=3e-3)
    assert xrl.imag == pytest.approx(henke.imag, rel=5e-2)


def test_combined_provider_switches_at_30kev() -> None:
    energy = np.array([[40e3, 20e3], [30e3, 31e3]])
    combined = delta_beta(W, energy, provider="henke_xraylib")
    low = energy <= 30e3
    np.testing.assert_array_equal(combined[low], delta_beta(W, energy[low], provider="henke"))
    np.testing.assert_array_equal(combined[~low], delta_beta(W, energy[~low], provider="xraylib"))


def test_default_provider_is_combined_with_compton() -> None:
    np.testing.assert_array_equal(
        delta_beta(SI, np.array([1e3, 50e3])),
        delta_beta(SI, np.array([1e3, 50e3]), provider="henke_xraylib"),
    )
    assert providers.get("henke_xraylib").name == "henke_xraylib"


@pytest.mark.parametrize("base", ["henke", "xraylib", "henke_xraylib"])
def test_compton_contribution(base: str) -> None:
    energy = np.array([500.0, 8000.0, 25000.0])
    with_compton = delta_beta(C, energy, provider=base)
    photo = delta_beta(C, energy, provider=f"{base}_photo")
    np.testing.assert_array_equal(with_compton.real, photo.real)
    lam = 1239.8419843320026 / energy
    expected = (
        2.8179403262e-6 * lam**2 / (2 * np.pi) * C.atom_densities()[6] * compton_f2(6, energy)
    )
    np.testing.assert_allclose(with_compton.imag - photo.imag, expected, rtol=1e-8)
    # Negligible in the soft X-ray range, dominant for carbon at high energies.
    assert expected[0] / photo.imag[0] < 1e-3
    assert expected[2] / photo.imag[2] > 1.0


def test_compton_f2_matches_xraylib_cross_section() -> None:
    energy = np.array([20e3])
    sigma_cm2 = xraylib.CS_Compt(14, 20.0) * xraylib.AtomicWeight(14) / 6.02214076e23
    r_e_cm, lam_cm = 2.8179403262e-13, 1239.8419843320026 / 20e3 * 1e-7
    assert compton_f2(14, energy)[0] == pytest.approx(sigma_cm2 / (2 * r_e_cm * lam_cm))


def test_compton_zero_below_xraylib_range() -> None:
    np.testing.assert_array_equal(compton_f2(6, np.array([30.0, 99.0])), 0.0)


def test_provider_names_reflect_compton_setting() -> None:
    assert HenkeProvider(include_compton=False).name == "henke_photo"
    assert XraylibProvider().name == "xraylib"
    assert not providers.get("xraylib_photo").include_compton  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("material", "theta_c_mrad"),
    [(SI, 3.84), (AU, 9.6)],
    ids=["Si", "Au"],
)
def test_critical_angle_cu_kalpha(material: Material, theta_c_mrad: float) -> None:
    n = refractive_index(material, 8047.8)
    theta_c = np.sqrt(2 * (1 - n.real)) * 1e3
    assert theta_c == pytest.approx(theta_c_mrad, rel=3e-2)


def test_unknown_provider() -> None:
    with pytest.raises(KeyError, match="available"):
        delta_beta(SI, 1000.0, provider="nonexistent")
