"""Tests for `TabulatedProvider`, `CombinedProvider` and the module-level functions."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from grazing.core import materials
from grazing.core.materials import (
    CombinedProvider,
    Material,
    TabulatedProvider,
    delta_beta,
    providers,
    refractive_index,
)

SI = Material("Si", "Si", 2.33)
ENERGY = np.array([100.0, 200.0, 400.0, 800.0, 1600.0])
DELTA = np.array([1e-3, 8e-4, 5e-4, 2e-4, 1e-4])
BETA = 1e8 * ENERGY**-3.0


def make() -> TabulatedProvider:
    return TabulatedProvider("tab", ENERGY, DELTA, BETA)


@pytest.fixture
def registered() -> Iterator[str]:
    name = "test_tabulated_unique"
    providers.register(name, TabulatedProvider(name, ENERGY, DELTA, BETA))
    try:
        yield name
    finally:
        providers.unregister(name)


def test_name_and_range() -> None:
    p = make()
    assert p.name == "tab"
    assert p.energy_range == pytest.approx((100.0, 1600.0), rel=1e-12)


@pytest.mark.parametrize(
    ("energy", "delta", "beta"),
    [
        ([1.0, 2.0, 3.0], [1.0, 1.0], [1.0, 1.0, 1.0]),
        ([1.0, 2.0], [1.0, 1.0], [1.0, 1.0, 1.0]),
        ([1.0, 1.0, 2.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]),
        ([2.0, 1.0], [1.0, 1.0], [1.0, 1.0]),
        ([0.0, 1.0], [1.0, 1.0], [1.0, 1.0]),
        ([1.0, 2.0], [1.0, 1.0], [1.0, 0.0]),
        ([1.0, 2.0], [1.0, 1.0], [1.0, -1.0]),
        ([1.0], [1.0], [1.0]),
        ([], [], []),
    ],
)
def test_validation_errors(energy: list[float], delta: list[float], beta: list[float]) -> None:
    with pytest.raises(ValueError, match="must"):
        TabulatedProvider("bad", energy, delta, beta)


def test_reproduces_nodes() -> None:
    db = make().delta_beta(SI, ENERGY)
    np.testing.assert_allclose(db.real, DELTA, rtol=1e-12)
    np.testing.assert_allclose(db.imag, BETA, rtol=1e-12)


def test_power_law_beta_between_nodes() -> None:
    e = np.array([123.0, 250.0, 555.5, 1234.5])
    db = make().delta_beta(SI, e)
    np.testing.assert_allclose(db.imag, 1e8 * e**-3.0, rtol=1e-12)


def test_delta_log_linear() -> None:
    e = np.sqrt(ENERGY[0] * ENERGY[1])
    db = make().delta_beta(SI, np.array([e]))
    assert db.real[0] == pytest.approx(0.5 * (DELTA[0] + DELTA[1]), rel=1e-12)


@pytest.mark.parametrize("energy", [50.0, 1700.0])
def test_out_of_range(energy: float) -> None:
    with pytest.raises(ValueError, match="covers"):
        make().delta_beta(SI, np.array([energy]))


def test_shape_preserved() -> None:
    e = np.array([[100.0, 300.0], [700.0, 1600.0]])
    assert make().delta_beta(SI, e).shape == (2, 2)


def test_from_file(tmp_path: Path) -> None:
    path = tmp_path / "table.csv"
    rows = zip(ENERGY.tolist(), DELTA.tolist(), BETA.tolist(), strict=True)
    lines = ["energy,delta,beta"] + [f"{e!r},{d!r},{b!r}" for e, d, b in rows]
    path.write_text("\n".join(lines) + "\n")
    p = TabulatedProvider.from_file("file_tab", path, skiprows=1)
    assert p.name == "file_tab"
    db = p.delta_beta(SI, ENERGY)
    np.testing.assert_allclose(db.real, DELTA, rtol=1e-12)
    np.testing.assert_allclose(db.imag, BETA, rtol=1e-12)


def test_from_file_whitespace_separated(tmp_path: Path) -> None:
    path = tmp_path / "table.txt"
    path.write_text("100 1e-3 1e-6\n200 5e-4 1e-7\n")
    p = TabulatedProvider.from_file("ws", path)
    assert p.energy_range == pytest.approx((100.0, 200.0), rel=1e-12)


def test_registered_provider_via_material(registered: str) -> None:
    material = Material("Si-tab", "Si", 2.33, provider=registered)
    n = refractive_index(material, ENERGY)
    np.testing.assert_allclose(n.real, 1.0 - DELTA, rtol=1e-12)
    np.testing.assert_allclose(n.imag, BETA, rtol=1e-12)
    with pytest.raises(ValueError, match="covers"):
        refractive_index(material, 10000.0)


def test_registered_provider_via_library(registered: str) -> None:
    library = materials.MaterialLibrary([Material("Si-tab", "Si", 2.33, provider=registered)])
    n = library.refractive_index("Si-tab", ENERGY)
    np.testing.assert_allclose(n.real, 1.0 - DELTA, rtol=1e-12)


def test_explicit_provider_overrides_material(registered: str) -> None:
    material = Material("Si", "Si", 2.33, provider="xraylib")
    n = refractive_index(material, ENERGY, registered)
    np.testing.assert_allclose(n.imag, BETA, rtol=1e-12)


def constant_provider(
    name: str, delta: float, beta: float, lo: float, hi: float
) -> TabulatedProvider:
    return TabulatedProvider(name, [lo, hi], [delta, delta], [beta, beta])


def combined() -> CombinedProvider:
    low = constant_provider("low", 1e-3, 1e-5, 10.0, 1000.0)
    high = constant_provider("high", 2e-6, 3e-8, 100.0, 1e5)
    return CombinedProvider(low, high, 1000.0, "combo")


def test_combined_properties() -> None:
    c = combined()
    assert c.name == "combo"
    assert c.switch_energy == 1000.0
    assert c.energy_range == pytest.approx((10.0, 1e5), rel=1e-12)


def test_combined_switching() -> None:
    e = np.array([10.0, 500.0, 1000.0, 1000.5, 5000.0, 1e5])
    db = combined().delta_beta(SI, e)
    np.testing.assert_allclose(db[:3], 1e-3 + 1e-5j, rtol=1e-12)
    np.testing.assert_allclose(db[3:], 2e-6 + 3e-8j, rtol=1e-12)


def test_combined_preserves_2d_shape() -> None:
    e = np.array([[500.0, 2000.0, 1000.0], [3000.0, 20.0, 1e4]])
    db = combined().delta_beta(SI, e)
    assert db.shape == (2, 3)
    expected_low = e <= 1000.0
    assert np.all(db.real[expected_low] == pytest.approx(1e-3))
    assert np.all(db.real[~expected_low] == pytest.approx(2e-6))


def test_combined_all_one_side() -> None:
    assert combined().delta_beta(SI, np.array([20.0, 30.0])).shape == (2,)
    assert combined().delta_beta(SI, np.array([2000.0, 3000.0])).shape == (2,)


def test_combined_registered_name_in_defaults() -> None:
    assert materials.DEFAULT_PROVIDER in providers


@pytest.mark.parametrize("bad", [0.0, -1.0, [100.0, -5.0]])
def test_delta_beta_rejects_non_positive_energy(bad: float | list[float]) -> None:
    with pytest.raises(ValueError, match="positive"):
        delta_beta(SI, bad)
    with pytest.raises(ValueError, match="positive"):
        refractive_index(SI, bad)


def test_delta_beta_non_positive_energy_rejected_for_vacuum() -> None:
    with pytest.raises(ValueError, match="positive"):
        delta_beta(materials.VACUUM, 0.0)


def test_delta_beta_preserves_2d_shape() -> None:
    e: NDArray[np.float64] = np.array([[1000.0, 2000.0, 3000.0], [4000.0, 5000.0, 6000.0]])
    assert delta_beta(SI, e).shape == (2, 3)
    assert refractive_index(SI, e).shape == (2, 3)
    assert delta_beta(materials.VACUUM, e).shape == (2, 3)


def test_delta_beta_scalar_is_0d() -> None:
    db = delta_beta(SI, 5000.0)
    assert isinstance(db, np.ndarray)
    assert db.shape == ()
    assert db.dtype == np.complex128
    assert refractive_index(SI, 5000.0).shape == ()
    assert delta_beta(materials.VACUUM, 5000.0).shape == ()


def test_unknown_provider_raises() -> None:
    with pytest.raises(KeyError):
        delta_beta(SI, 5000.0, "nope")
