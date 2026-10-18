"""Tests for `xstack.core.materials.library`."""

import numpy as np
import pytest

from xstack.core import materials
from xstack.core.materials import (
    DEFAULT_PROVIDER,
    PRESETS,
    VACUUM,
    Material,
    MaterialLibrary,
    default_library,
)

ENERGIES = np.array([100.0, 500.0, 5000.0, 20000.0, 50000.0])


def test_defaults_to_presets() -> None:
    library = MaterialLibrary()
    assert len(library) == len(PRESETS)
    for name in ("W", "Si", "B4C", "vacuum"):
        assert name in library
    assert library.default_provider == DEFAULT_PROVIDER
    assert library.names() == tuple(m.name for m in PRESETS)
    assert [m.name for m in library] == list(library.names())


def test_custom_materials_only() -> None:
    library = MaterialLibrary([Material("A", "Si", 2.0)])
    assert library.names() == ("A",)
    assert "W" not in library


def test_add_get_remove() -> None:
    library = MaterialLibrary([])
    m = Material("Foo", "SiO2", 2.2)
    library.add(m)
    assert library.get("Foo") is m
    library.remove("Foo")
    assert "Foo" not in library
    with pytest.raises(KeyError):
        library.get("Foo")


def test_add_duplicate_raises_and_replace() -> None:
    library = MaterialLibrary()
    with pytest.raises(ValueError, match="already exists"):
        library.add(Material("W", "W", 15.0))
    assert library.get("W").density == 19.25
    library.add(Material("W", "W", 15.0), replace=True)
    assert library.get("W").density == 15.0


def test_remove_unknown_raises() -> None:
    with pytest.raises(KeyError):
        MaterialLibrary().remove("nope")


def test_get_unknown_raises() -> None:
    with pytest.raises(KeyError, match="nope"):
        MaterialLibrary().get("nope")


def test_unknown_default_provider_raises() -> None:
    with pytest.raises(KeyError):
        MaterialLibrary(default_provider="nope")
    library = MaterialLibrary()
    with pytest.raises(KeyError):
        library.default_provider = "nope"
    assert library.default_provider == DEFAULT_PROVIDER


def test_default_provider_setter() -> None:
    library = MaterialLibrary()
    library.default_provider = "xraylib"
    assert library.default_provider == "xraylib"


def test_provider_for() -> None:
    library = MaterialLibrary(default_provider="xraylib")
    assert library.provider_for(Material("Si", "Si", 2.33)) == "xraylib"
    assert library.provider_for(Material("Si", "Si", 2.33, provider="henke")) == "henke"


def test_vacuum_refractive_index_is_one() -> None:
    n = MaterialLibrary().refractive_index(VACUUM, ENERGIES)
    assert n.shape == ENERGIES.shape
    assert np.all(n == 1.0)
    assert np.all(MaterialLibrary().refractive_index("vacuum", 1000.0) == 1.0)


def test_by_name_equals_by_object() -> None:
    library = MaterialLibrary()
    by_name = library.refractive_index("W", ENERGIES)
    by_object = library.refractive_index(library.get("W"), ENERGIES)
    assert np.array_equal(by_name, by_object)


def test_refractive_index_physical() -> None:
    n = MaterialLibrary().refractive_index("Si", 8048.0)
    assert n.shape == (1,) or n.shape == ()
    n = n.reshape(())
    assert 0.0 < 1.0 - n.real < 1e-4
    assert n.imag > 0.0


def test_scalar_energy_gives_0d_result() -> None:
    assert MaterialLibrary().refractive_index("Si", 8048.0).shape == ()


def test_cache_returns_identical_readonly_object() -> None:
    library = MaterialLibrary()
    first = library.delta_beta("Si", ENERGIES)
    second = library.delta_beta("Si", ENERGIES.copy())
    assert first is second
    assert not first.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        first[0] = 0.0


def test_cache_distinguishes_shape_and_provider() -> None:
    library = MaterialLibrary()
    flat = library.delta_beta("Si", ENERGIES)
    shaped = library.delta_beta("Si", ENERGIES.reshape(5, 1))
    assert flat is not shaped
    assert shaped.shape == (5, 1)
    other = library.delta_beta(Material("Si", "Si", 2.33, provider="xraylib"), ENERGIES)
    assert other is not flat


def test_clear_cache() -> None:
    library = MaterialLibrary()
    first = library.delta_beta("Si", ENERGIES)
    library.clear_cache()
    second = library.delta_beta("Si", ENERGIES)
    assert first is not second
    assert np.array_equal(first, second)


def test_cache_matches_uncached() -> None:
    library = MaterialLibrary()
    for name in ("Si", "W", "B4C"):
        material = library.get(name)
        expected = materials.refractive_index(material, ENERGIES)
        assert np.array_equal(library.refractive_index(name, ENERGIES), expected)
        assert np.array_equal(library.refractive_index(name, ENERGIES), expected)


def test_library_default_provider_is_used() -> None:
    library = MaterialLibrary(default_provider="xraylib")
    expected = materials.refractive_index(library.get("Si"), ENERGIES, "xraylib")
    assert np.array_equal(library.refractive_index("Si", ENERGIES), expected)


def test_dict_round_trip() -> None:
    library = MaterialLibrary(default_provider="xraylib")
    library.add(Material("Custom", "SiO2", 2.0, provider="henke"))
    restored = MaterialLibrary.from_dict(library.to_dict())
    assert restored.default_provider == "xraylib"
    assert restored.names() == library.names()
    assert list(restored) == list(library)
    assert restored.get("Custom").provider == "henke"


def test_to_dict_is_json_serializable() -> None:
    import json

    data = MaterialLibrary().to_dict()
    assert json.loads(json.dumps(data)) == data


def test_from_dict_default_provider_optional() -> None:
    data = {"materials": [{"name": "Si", "formula": "Si", "density": 2.33}]}
    library = MaterialLibrary.from_dict(data)
    assert library.default_provider == DEFAULT_PROVIDER
    assert library.get("Si").provider is None


def test_default_library_is_shared() -> None:
    assert default_library() is default_library()
    assert "W" in default_library()


def test_module_get() -> None:
    w = materials.get("W")
    assert w.formula == "W"
    assert w.density == 19.25
    with pytest.raises(KeyError):
        materials.get("nope")
