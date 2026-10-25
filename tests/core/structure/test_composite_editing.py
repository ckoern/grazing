"""Editing combined coatings: materials, blocks and the parameter settings they keep."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from xstack.core.materials import get
from xstack.core.serialization import from_data, to_data
from xstack.core.structure import Composite, Mirror, Periodic, SingleLayer
from xstack.core.structure.coatings import DepthGraded, NLayer

PT, C, W, SI, B4C, MO, RU = (get(n) for n in ("Pt", "C", "W", "Si", "B4C", "Mo", "Ru"))


def example() -> Composite:
    """The example of issue #40: graded Pt/C, five single layers, periodic W/Si."""
    return Composite.create(
        [
            DepthGraded.create(PT, C, N=10, d_min=3.0, d_max=6.0, name="ptc"),
            NLayer.create([B4C, MO, SI, RU, C], [2.0, 3.0, 4.0, 2.0, 3.0], name="mid"),
            Periodic.create(W, SI, N=20, d=4.0, name="wsi"),
        ]
    )


def tuned() -> Composite:
    """The example with settings made on the composite, including a cross-block expression."""
    c = example()
    c = c.with_parameter("wsi.d", vary=True, min=3.5, max=4.5)
    return c.with_parameter("ptc.d_max", expr="1.5 * wsi.d")


def test_stack_of_the_example() -> None:
    layers = Mirror(example(), SI).stack().layers
    assert len(layers) == 2 * 10 + 5 + 2 * 20
    assert layers[0].tag.startswith("ptc")
    assert layers[-1].tag.startswith("wsi")


def test_materials_are_prefixed_roles() -> None:
    c = example()
    assert {r: m.name for r, m in c.materials.items()} == {
        "ptc.H": "Pt",
        "ptc.L": "C",
        "mid.M1": "B4C",
        "mid.M2": "Mo",
        "mid.M3": "Si",
        "mid.M4": "Ru",
        "mid.M5": "C",
        "wsi.H": "W",
        "wsi.L": "Si",
    }
    assert c.material("wsi.H") == W


def test_with_and_without_material() -> None:
    c = tuned().with_material("wsi.H", MO)
    assert c.block("wsi").material("H") == MO
    assert c.parameters["wsi.d"].vary
    periodic = c.block("wsi")
    assert isinstance(periodic, Periodic)
    c = c.with_block("wsi", periodic.with_interlayers(True))
    c = c.with_material("wsi.interlayer_H_on_L", get("MoSi2"))
    assert c.materials["wsi.interlayer_H_on_L"].name == "MoSi2"
    c = c.without_material("wsi.interlayer_H_on_L")
    assert "wsi.interlayer_H_on_L" not in c.materials
    with pytest.raises(KeyError):
        c.with_material("nope.H", W)
    with pytest.raises(ValueError):
        c.without_material("wsi.H")


def test_block_strips_only_its_own_prefix() -> None:
    c = Composite.create(
        [SingleLayer.create(SI, t=2.0, name="si"), Periodic.create(W, SI, name="wsi")]
    )
    c = c.with_parameter("si.t", expr="wsi.d / 2")
    c = c.with_parameter("si.sigma", expr="si.t / 10")
    block = c.block("si")
    assert block.parameters["t"].expr == "wsi.d / 2"
    assert block.parameters["sigma"].expr == "t / 10"


def test_block_round_trip_is_lossless() -> None:
    c = tuned()
    edited = c.move_block("ptc", 0)
    for name in c.block_names():
        assert c.with_block(name, c.block(name)).parameters == c.parameters
        assert edited.with_block(name, edited.block(name)) == edited


def test_with_block_keeps_other_blocks() -> None:
    c = tuned()
    graded = c.block("ptc")
    periodic = Periodic.create(PT, C, N=12, d=5.0)
    new = c.with_block("ptc", periodic)
    assert new.block_names() == ("ptc", "mid", "wsi")
    assert isinstance(new.blocks[0], Periodic)
    assert new.parameters["ptc.N"].value == 12
    assert new.parameters["wsi.d"] == c.parameters["wsi.d"]
    assert "ptc.d_max" not in new.parameters
    assert new.with_block("ptc", graded).parameters["ptc.d_max"].expr == "1.5 * wsi.d"


@pytest.mark.parametrize("index", [0, 1, 3])
def test_insert_block(index: int) -> None:
    c = tuned()
    new = c.insert_block(index, SingleLayer.create(B4C, t=2.5, name="cap"))
    assert new.block_names()[index] == "cap"
    assert new.parameters["cap.t"].value == 2.5
    for name, p in c.parameters.items():
        assert new.parameters[name] == p
    assert list(new.parameters)[: len(new.blocks[0].parameters)] == [
        f"{new.blocks[0].name}.{n}" for n in new.blocks[0].parameters
    ]
    with pytest.raises(ValueError, match="already exists"):
        new.insert_block(0, SingleLayer.create(B4C, name="cap"))


def test_remove_block_clears_references() -> None:
    c = tuned()
    d_max = c.parameters.evaluate()["ptc.d_max"]
    new = c.remove_block("wsi")
    assert new.block_names() == ("ptc", "mid")
    assert not any(n.startswith("wsi.") for n in new.parameters)
    assert new.parameters["ptc.d_max"].expr is None
    assert new.parameters["ptc.d_max"].value == pytest.approx(d_max)
    kept = c.remove_block("mid")
    assert kept.parameters["ptc.d_max"].expr == "1.5 * wsi.d"
    with pytest.raises(ValueError, match="at least one block"):
        Composite.create([SingleLayer.create(SI, name="only")]).remove_block("only")


def test_move_block_keeps_settings() -> None:
    c = tuned()
    new = c.move_block("wsi", 0)
    assert new.block_names() == ("wsi", "ptc", "mid")
    assert dict(new.parameters.items()) == dict(c.parameters.items())
    assert next(iter(new.parameters)).startswith("wsi.")
    layers = Mirror(new, SI).stack().layers
    assert layers[0].tag.startswith("wsi")


def test_rename_block_renames_names_and_expressions() -> None:
    c = tuned()
    new = c.rename_block("wsi", "ml")
    assert new.block_names() == ("ptc", "mid", "ml")
    assert new.parameters["ml.d"].vary
    assert new.parameters["ptc.d_max"].expr == "1.5 * ml.d"
    assert "ml.H" in new.materials
    assert new.rename_block("ml", "ml") == new
    with pytest.raises(ValueError, match="already exists"):
        new.rename_block("ml", "ptc")
    with pytest.raises(ValueError, match="must be an identifier"):
        new.rename_block("ml", "not valid")


def test_edited_composite_round_trips() -> None:
    c = tuned().insert_block(0, SingleLayer.create(B4C, name="cap")).rename_block("mid", "glue")
    mirror = Mirror(c, SI)
    assert from_data(to_data(mirror)) == mirror


@given(st.lists(st.tuples(st.sampled_from(["ptc", "mid", "wsi"]), st.integers(0, 2)), max_size=6))
def test_moves_never_change_settings(moves: list[tuple[str, int]]) -> None:
    c = tuned()
    new = c
    for name, index in moves:
        new = new.move_block(name, index)
    assert dict(new.parameters.items()) == dict(c.parameters.items())
    assert sorted(new.block_names()) == sorted(c.block_names())
