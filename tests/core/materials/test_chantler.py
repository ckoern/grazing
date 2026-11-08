"""Tests of the Chantler (NIST FFAST) provider."""

import numpy as np
import pytest

from grazing.core.materials import ChantlerProvider, Material, delta_beta, providers

SI = Material("Si", "Si", 2.33)
W = Material("W", "W", 19.25)
B4C = Material("B4C", "B4C", 2.52)


@pytest.mark.parametrize("z", range(1, 93))
def test_all_elements_load(z: int) -> None:
    f = ChantlerProvider().scattering_factors(z, np.array([10.69, 1000.0, 1e5, 432945.0]))
    assert np.all(np.isfinite(f))
    assert np.all(f.imag >= 0)
    assert np.all(f.imag[1:] > 0)


def test_reproduces_table_values() -> None:
    """FFAST f1 of Si at 7.902609 keV (14.2709) plus f_rel (3/5 CL) and f_NT (−0.01943), and f2."""
    f = ChantlerProvider().scattering_factors(14, np.array([7902.609]))
    assert f.real[0] == pytest.approx(14.2514716, rel=1e-9)
    assert f.imag[0] == pytest.approx(0.34129, rel=1e-9)


def test_high_energy_limit_of_f1() -> None:
    """Far above the edges, f1 approaches Z plus the small relativistic corrections."""
    f = ChantlerProvider().scattering_factors(14, np.array([4e5]))
    assert f.real[0] == pytest.approx(14.0, abs=0.03)


@pytest.mark.parametrize("energy", [5.0, 440000.0])
def test_out_of_range(energy: float) -> None:
    with pytest.raises(ValueError, match="Chantler data for W"):
        delta_beta(W, energy, provider="chantler")


def test_hydrogen_below_ionization_has_no_absorption() -> None:
    f = ChantlerProvider().scattering_factors(1, np.array([11.0]))
    assert 0.0 <= f.imag[0] < 1e-20


def test_rejects_unknown_element() -> None:
    with pytest.raises(ValueError, match="Z = 1-92"):
        ChantlerProvider().scattering_factors(93, np.array([1000.0]))


@pytest.mark.parametrize("material", [SI, W, B4C], ids=lambda m: m.name)
@pytest.mark.parametrize(("energy", "other"), [(8048.0, "henke"), (17479.0, "henke")])
def test_close_to_henke_in_hard_xray_range(material: Material, energy: float, other: str) -> None:
    """δ agrees with the Henke tables within 1 % and β within 6 % at Cu and Mo Kα."""
    chantler = delta_beta(material, energy, provider="chantler")
    reference = delta_beta(material, energy, provider=other)
    assert chantler.real == pytest.approx(reference.real, rel=1e-2)
    assert chantler.imag == pytest.approx(reference.imag, rel=6e-2)


def test_close_to_xraylib_at_high_energy() -> None:
    chantler = delta_beta(W, 60000.0, provider="chantler")
    reference = delta_beta(W, 60000.0, provider="xraylib")
    assert chantler.real == pytest.approx(reference.real, rel=5e-3)
    assert chantler.imag == pytest.approx(reference.imag, rel=5e-2)


def test_registered_with_and_without_compton() -> None:
    assert providers.get("chantler").name == "chantler"
    assert not providers.get("chantler_photo").include_compton  # type: ignore[attr-defined]
    energy = np.array([30000.0])
    assert (
        delta_beta(B4C, energy, provider="chantler").imag
        > delta_beta(B4C, energy, provider="chantler_photo").imag
    )
