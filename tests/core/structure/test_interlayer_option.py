"""The interlayer option of bilayer coatings."""

import pytest

from xstack.core.materials import get
from xstack.core.serialization import from_data, to_data
from xstack.core.structure import Mirror, Periodic, SingleLayer
from xstack.core.structure.coatings import DepthGraded
from xstack.core.structure.coatings.bilayer import INTERLAYER_PARAMETERS
from xstack.core.structure.coatings.composite import Composite

W, SI, WSI2 = get("W"), get("Si"), get("WSi2")


@pytest.mark.parametrize(
    "make",
    [
        lambda **kw: Periodic.create(W, SI, N=3, d=4.0, gamma=0.4, **kw),
        lambda **kw: DepthGraded.create(W, SI, N=5, d_min=3.0, d_max=6.0, c=0.25, **kw),
    ],
    ids=["periodic", "graded"],
)
class TestOption:
    def test_off_by_default(self, make) -> None:
        coating = make()
        assert not coating.use_interlayers
        assert coating.inactive_parameters() == frozenset(INTERLAYER_PARAMETERS)

    def test_on_with_materials(self, make) -> None:
        coating = make(interlayers={"interlayer_H_on_L": WSI2}, t_interlayer_H_on_L=0.5)
        assert coating.use_interlayers
        assert coating.inactive_parameters() == frozenset()
        names = [layer.material.name for layer in coating.layers()]
        assert "WSi2" in names

    def test_switching_off_removes_materials_and_thickness(self, make) -> None:
        coating = make(interlayers={"interlayer_H_on_L": WSI2}, t_interlayer_H_on_L=0.5)
        off = coating.with_interlayers(False)
        assert not off.use_interlayers
        assert "interlayer_H_on_L" not in off.materials
        for name in INTERLAYER_PARAMETERS:
            assert off.parameters[name].value == 0.0
            assert not off.parameters[name].vary
        assert "WSi2" not in [layer.material.name for layer in off.layers()]

    def test_switching_on_without_materials_builds(self, make) -> None:
        on = make().with_interlayers(True)
        assert on.use_interlayers
        assert len(on.layers()) == len(make().layers())

    def test_thickness_inactive_when_off(self, make) -> None:
        coating = make().with_parameter("t_interlayer_H_on_L", value=0.7)
        assert not coating.use_interlayers
        assert coating.layers() == make().layers()


def test_composite_prefixes_inactive_parameters() -> None:
    cap = SingleLayer.create(get("Pt"), t=2.0, name="cap")
    ml = Periodic.create(W, SI, N=3, name="ml")
    composite = Composite.create([cap, ml])
    assert composite.inactive_parameters() == frozenset(
        {"ml.t_interlayer_H_on_L", "ml.t_interlayer_L_on_H"}
    )
    assert SingleLayer.create(get("Pt")).inactive_parameters() == frozenset()


def test_serialization_keeps_option() -> None:
    on = Periodic.create(W, SI, N=3).with_interlayers(True)
    assert from_data(to_data(on)) == on
    mirror = Mirror(Periodic.create(W, SI, N=3), SI)
    assert from_data(to_data(mirror)) == mirror


def test_old_data_with_interlayer_materials_enables_option() -> None:
    coating = Periodic.create(
        W, SI, N=3, interlayers={"interlayer_L_on_H": WSI2}, t_interlayer_L_on_H=0.3
    )
    data = to_data(coating)
    del data["fields"]["use_interlayers"]
    loaded = from_data(data)
    assert loaded.use_interlayers
    assert loaded == coating


def test_old_data_without_interlayers_has_option_off() -> None:
    data = to_data(Periodic.create(W, SI, N=3))
    del data["fields"]["use_interlayers"]
    assert not from_data(data).use_interlayers
