"""Tests for `grazing.core.materials.material`."""

import dataclasses

import pytest
import xraylib

from grazing.core.materials import VACUUM, Material


def weight(z: int) -> float:
    return float(xraylib.AtomicWeight(z))


def test_b4c_composition_and_molar_mass() -> None:
    m = Material("B4C", "B4C", 2.52)
    assert m.composition == ((5, 4.0), (6, 1.0))
    assert m.molar_mass == pytest.approx(4 * weight(5) + weight(6), rel=1e-12)


def test_sio2_composition_and_molar_mass() -> None:
    m = Material("SiO2", "SiO2", 2.2)
    assert m.composition == ((8, 2.0), (14, 1.0))
    assert m.molar_mass == pytest.approx(weight(14) + 2 * weight(8), rel=1e-12)


def test_parentheses_formula() -> None:
    m = Material("lime", "Ca(OH)2", 2.2)
    assert dict(m.composition) == {1: 2.0, 8: 2.0, 20: 1.0}
    assert m.molar_mass == pytest.approx(weight(20) + 2 * weight(8) + 2 * weight(1), rel=1e-12)


def test_composition_sorted_by_atomic_number() -> None:
    m = Material("x", "SiO2", 2.2)
    zs = [z for z, _ in m.composition]
    assert zs == sorted(zs)


def test_vacuum() -> None:
    assert VACUUM.is_vacuum
    assert VACUUM.composition == ()
    assert VACUUM.molar_mass == 0.0
    assert VACUUM.density == 0.0
    assert VACUUM.atom_densities() == {}


def test_non_vacuum_is_not_vacuum() -> None:
    assert not Material("Si", "Si", 2.33).is_vacuum


def test_vacuum_with_density_raises() -> None:
    with pytest.raises(ValueError, match="density 0"):
        Material("vac", "", 1.0)


def test_invalid_formula_raises() -> None:
    with pytest.raises(ValueError, match="bad"):
        Material("bad", "Xx9Qq", 1.0)


@pytest.mark.parametrize("density", [0.0, -1.0, float("nan")])
def test_non_positive_density_raises(density: float) -> None:
    with pytest.raises(ValueError, match="positive density"):
        Material("Si", "Si", density)


def test_empty_name_raises() -> None:
    with pytest.raises(ValueError, match="name"):
        Material("", "Si", 2.33)


def test_silicon_atom_density() -> None:
    densities = Material("Si", "Si", 2.33).atom_densities()
    assert set(densities) == {14}
    assert densities[14] == pytest.approx(49.95, rel=1e-3)


def test_b4c_atom_density_ratio() -> None:
    densities = Material("B4C", "B4C", 2.52).atom_densities()
    assert densities[5] / densities[6] == pytest.approx(4.0, rel=1e-12)


def test_atom_densities_reproduce_mass_density() -> None:
    m = Material("SiO2", "SiO2", 2.2)
    mass_per_nm3 = sum(n * weight(z) for z, n in m.atom_densities().items()) / 6.02214076e23
    assert mass_per_nm3 / 1e-21 == pytest.approx(2.2, rel=1e-6)


def test_with_density_keeps_other_fields() -> None:
    m = Material("B4C", "B4C", 2.52, provider="xraylib")
    n = m.with_density(2.0)
    assert n.density == 2.0
    assert (n.name, n.formula, n.provider) == ("B4C", "B4C", "xraylib")
    assert n.composition == m.composition
    assert m.density == 2.52


def test_with_density_validates() -> None:
    with pytest.raises(ValueError, match="positive"):
        Material("Si", "Si", 2.33).with_density(-1.0)


def test_hashable_and_equality() -> None:
    a = Material("Si", "Si", 2.33)
    b = Material("Si", "Si", 2.33)
    assert a == b
    assert hash(a) == hash(b)
    assert len({a, b}) == 1
    assert a != Material("Si", "Si", 2.34)
    assert a != Material("Si2", "Si", 2.33)
    assert a != Material("Si", "Si", 2.33, provider="henke")


def test_equality_uses_only_init_fields() -> None:
    fields = [f for f in dataclasses.fields(Material) if f.compare]
    assert [f.name for f in fields] == ["name", "formula", "density", "provider"]


def test_frozen() -> None:
    m = Material("Si", "Si", 2.33)
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.density = 3.0  # type: ignore[misc]
