"""Tests for `grazing.core.structure.presets`."""

import pytest

from grazing.core.structure.coatings.graded import DepthGraded
from grazing.core.structure.coatings.periodic import Periodic
from grazing.core.structure.coatings.simple import NLayer, SingleLayer
from grazing.core.structure.presets import (
    COATING_KINDS,
    MATERIAL_PAIRS,
    StructureChoice,
    build_coating,
    build_mirror,
)


def test_default_is_periodic_w_si() -> None:
    mirror = build_mirror(StructureChoice())
    assert isinstance(mirror.coating, Periodic)
    assert mirror.coating.materials["H"].name == "W"
    assert mirror.coating.materials["L"].name == "Si"
    assert mirror.substrate.name == "Si"


@pytest.mark.parametrize(
    ("kind", "materials", "cls"),
    [
        ("single_layer", ("Ir",), SingleLayer),
        ("layers", ("Ru", "B4C"), NLayer),
        ("periodic", ("Mo", "Si"), Periodic),
        ("graded_linear", ("Pt", "C"), DepthGraded),
        ("graded_power_law", ("W", "B4C"), DepthGraded),
    ],
)
def test_every_kind_builds_a_stack_with_free_parameters(
    kind: str, materials: tuple[str, ...], cls: type
) -> None:
    mirror = build_mirror(StructureChoice(kind, materials))  # type: ignore[arg-type]
    assert isinstance(mirror.coating, cls)
    assert mirror.stack().layers
    assert mirror.parameters.free_names()


@pytest.mark.parametrize(
    ("kind", "law"), [("graded_linear", "linear"), ("graded_power_law", "power_law")]
)
def test_grading_law(kind: str, law: str) -> None:
    coating = build_coating(StructureChoice(kind, ("W", "Si")))  # type: ignore[arg-type]
    assert isinstance(coating, DepthGraded)
    assert coating.law == law


@pytest.mark.parametrize("kind", ["periodic", "graded_linear", "graded_power_law"])
@pytest.mark.parametrize("top", ["H", "L"])
def test_top_layer(kind: str, top: str) -> None:
    coating = build_coating(StructureChoice(kind, ("W", "Si"), top=top))  # type: ignore[arg-type]
    first = coating.layers()[0]
    assert coating.top == top  # type: ignore[attr-defined]
    assert first.material.name == ("W" if top == "H" else "Si")


def test_layers_keep_order_from_the_top() -> None:
    coating = build_coating(StructureChoice("layers", ("Cr", "Pt", "C")))
    assert [layer.material.name for layer in coating.layers()] == ["Cr", "Pt", "C"]


def test_all_pairs_use_library_materials() -> None:
    for label, pair in MATERIAL_PAIRS.items():
        assert label == "/".join(pair)
        build_mirror(StructureChoice("periodic", pair))


def test_labels_cover_all_kinds() -> None:
    assert list(COATING_KINDS) == [
        "single_layer",
        "layers",
        "periodic",
        "graded_linear",
        "graded_power_law",
    ]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"kind": "bad"}, "unknown coating kind"),
        ({"kind": "single_layer", "materials": ("W", "Si")}, "exactly one"),
        ({"kind": "layers", "materials": ()}, "at least one"),
        ({"kind": "periodic", "materials": ("W",)}, "two materials"),
        ({"top": "X"}, "top layer"),
    ],
)
def test_validation(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        StructureChoice(**kwargs)  # type: ignore[arg-type]


def test_unknown_material() -> None:
    with pytest.raises(KeyError):
        build_mirror(StructureChoice("single_layer", ("Unobtainium",)))
