"""Tests of the conversion between coating types and of layer stack edits."""

import dataclasses

import pytest

from xstack.core.materials import get
from xstack.core.parameters import ParameterSet
from xstack.core.structure import Composite, DepthGraded, NLayer, Periodic, SingleLayer
from xstack.core.structure.conversion import (
    constant_gamma_changes,
    convert_coating,
    has_constant_gamma,
    restack_layers,
    unique_block_name,
)


def test_convert_periodic_to_depth_graded_keeps_settings() -> None:
    w, si, c = get("W"), get("Si"), get("C")
    old = Periodic.create(
        w, si, N=30, sigma_H=0.5, sigma_L=0.3, interlayers={"interlayer_H_on_L": c}
    )
    new = convert_coating(old, "depth_graded")
    assert isinstance(new, DepthGraded)
    assert new.materials == old.materials
    assert new.parameters["sigma_H"].value == 0.5
    assert new.parameters["sigma_L"].value == 0.3


def test_convert_carries_vary_settings() -> None:
    old = Periodic.create(get("W"), get("Si")).with_parameter("N", vary=True, min=5, max=50)
    new = convert_coating(old, "depth_graded")
    assert new.parameters["N"] == old.parameters["N"]


def test_convert_options() -> None:
    old = Periodic.create(get("W"), get("Si"))
    graded = convert_coating(old, "depth_graded", top="L", law="linear", cap=True)
    assert isinstance(graded, DepthGraded)
    assert (graded.top, graded.law, graded.cap) == ("L", "linear", True)
    periodic = convert_coating(graded, "periodic", top="H")
    assert isinstance(periodic, Periodic)
    assert periodic.top == "H"


def test_convert_single_layer() -> None:
    old = Periodic.create(get("W"), get("Si"))
    new = convert_coating(old, "single_layer")
    assert isinstance(new, SingleLayer)
    assert new.materials["M"].name == "W"


def test_convert_unknown_type() -> None:
    with pytest.raises(ValueError, match="cannot be created"):
        convert_coating(Periodic.create(get("W"), get("Si")), "nonsense")


def test_convert_to_n_layer_from_bilayer_and_single() -> None:
    old = Periodic.create(get("W"), get("Si"), d=5.0, gamma=0.4, sigma_H=0.5)
    new = convert_coating(old, "n_layer")
    assert isinstance(new, NLayer)
    assert [m.name for _, m in new.material_roles] == ["W", "Si"]
    assert new.parameters["t_W"].value == pytest.approx(2.0)
    assert new.parameters["t_Si"].value == pytest.approx(3.0)
    assert new.parameters["sigma_W"].value == pytest.approx(0.5)
    single = convert_coating(SingleLayer.create(get("W"), t=12.0), "n_layer")
    assert isinstance(single, NLayer)
    assert single.count == 1
    assert single.parameters["t_W"].value == pytest.approx(12.0)


def test_convert_from_n_layer() -> None:
    old = NLayer.create([get("Pt"), get("C"), get("Si")], [1.0, 2.0, 3.0])
    for name in ("periodic", "depth_graded"):
        new = convert_coating(old, name)
        assert new.materials["H"].name == "Pt"
        assert new.materials["L"].name == "C"
    single = convert_coating(old, "single_layer")
    assert single.materials["M"].name == "Pt"
    assert convert_coating(old, "n_layer") is old


def test_restack_renames_parameters_and_expressions() -> None:
    old = NLayer.create([get("Pt"), get("C")], [3.0, "t_Pt + 1"])
    new = restack_layers(old, [get("Si"), get("C")], [1, 2])
    assert list(new.parameters)[::2] == ["t_Si", "t_C"]
    assert new.parameters["t_C"].expr == "t_Si + 1"
    dropped = restack_layers(old, [get("C")], [2])
    assert dropped.parameters["t_C"].expr is None


def test_restack_of_legacy_names() -> None:
    new = NLayer.create([get("Pt"), get("C")], [3.0, 4.0])
    legacy = NLayer(
        new.name,
        new.material_roles,
        ParameterSet(
            dataclasses.replace(p, name=n)
            for n, p in zip(["t1", "sigma1", "t2", "sigma2"], new.parameters.values(), strict=True)
        ),
    )
    restacked = restack_layers(legacy, [get("Pt"), get("C"), get("Pt")], [1, 2, None])
    assert list(restacked.parameters)[::2] == ["t_Pt_1", "t_C", "t_Pt_2"]
    assert restacked.parameters["t_C"].value == pytest.approx(4.0)


def test_unique_block_name() -> None:
    assert unique_block_name("periodic", ()) == "periodic"
    assert unique_block_name("depth_graded", ["graded", "graded2"]) == "graded3"


def test_convert_to_and_from_combined() -> None:
    periodic = Periodic.create(get("W"), get("Si"), d=3.5).with_parameter("d", vary=True)
    combined = convert_coating(periodic, "composite")
    assert isinstance(combined, Composite)
    assert combined.block_names() == ("periodic",)
    assert combined.parameters["periodic.d"].vary
    assert convert_coating(combined, "composite") is combined
    back = convert_coating(combined, "periodic")
    assert isinstance(back, Periodic)
    assert back.parameters["d"].value == 3.5


def test_convert_keeps_option() -> None:
    on = Periodic.create(get("W"), get("Si")).with_interlayers(True)
    converted = convert_coating(on, "depth_graded")
    assert isinstance(converted, DepthGraded)
    assert converted.use_interlayers
    assert not convert_coating(on.with_interlayers(False), "depth_graded").use_interlayers  # type: ignore[attr-defined]


def tied() -> DepthGraded:
    return DepthGraded.create(get("W"), get("Si")).with_parameter(
        "gamma_bottom", expr="gamma_top", vary=False
    )


def test_has_constant_gamma() -> None:
    assert not has_constant_gamma(DepthGraded.create(get("W"), get("Si"), gamma_bottom=0.3))
    assert has_constant_gamma(tied())


def test_constant_gamma_changes_tie() -> None:
    coating = DepthGraded.create(get("W"), get("Si"), gamma_bottom=0.3)
    new = coating.with_parameter("gamma_bottom", **constant_gamma_changes(coating, True))
    assert new.parameters["gamma_bottom"].expr == "gamma_top"
    assert has_constant_gamma(new)


def test_constant_gamma_changes_untie_copies_top_settings() -> None:
    coating = tied()
    top = coating.parameters["gamma_top"]
    new = coating.with_parameter("gamma_bottom", **constant_gamma_changes(coating, False))
    bottom = new.parameters["gamma_bottom"]
    assert bottom.expr is None
    assert bottom.value == pytest.approx(top.value)
    assert (bottom.vary, bottom.min, bottom.max) == (top.vary, top.min, top.max)
    assert not has_constant_gamma(new)
