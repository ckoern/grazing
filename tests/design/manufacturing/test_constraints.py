"""Tests for the manufacturing constraints."""

import pytest

from xstack.core import materials
from xstack.core.parameters import free
from xstack.core.structure import DepthGraded, Layer, Mirror, Periodic, SingleLayer, Stack
from xstack.design.manufacturing.constraints import (
    MaxCoatingTime,
    MaxLayers,
    MaxTotalThickness,
    MinLayerThickness,
    constraints,
)
from xstack.design.manufacturing.process import DepositionProcess

W = materials.get("W")
SI = materials.get("Si")
B4C = materials.get("B4C")


def stack_of(*thicknesses: float) -> Stack:
    """Alternating W/Si stack, top first."""
    return Stack([Layer(W if i % 2 == 0 else SI, t) for i, t in enumerate(thicknesses)], SI)


class TestRegistry:
    @pytest.mark.parametrize(
        ("name", "cls"),
        [
            ("min_layer_thickness", MinLayerThickness),
            ("max_total_thickness", MaxTotalThickness),
            ("max_layers", MaxLayers),
            ("max_coating_time", MaxCoatingTime),
        ],
    )
    def test_registered(self, name: str, cls: type) -> None:
        assert name in constraints
        assert constraints.get(name) is cls
        assert cls.type_name == name


class TestSeverityAndSettings:
    def test_default_severity(self) -> None:
        assert MinLayerThickness().severity == "hard"
        assert MaxLayers().severity == "hard"

    def test_soft_by_keyword(self) -> None:
        assert MinLayerThickness(2.0, severity="soft").severity == "soft"
        assert MaxTotalThickness(severity="soft").severity == "soft"
        assert MaxCoatingTime(severity="soft").severity == "soft"

    @pytest.mark.parametrize("value", [0.0, -1.0])
    def test_invalid_minimum(self, value: float) -> None:
        with pytest.raises(ValueError):
            MinLayerThickness(value)

    @pytest.mark.parametrize("value", [0.0, -1.0])
    def test_invalid_max_total(self, value: float) -> None:
        with pytest.raises(ValueError):
            MaxTotalThickness(value)

    @pytest.mark.parametrize("value", [0, -3])
    def test_invalid_max_layers(self, value: int) -> None:
        with pytest.raises(ValueError):
            MaxLayers(value)

    @pytest.mark.parametrize("value", [0.0, -1.0])
    def test_invalid_max_time(self, value: float) -> None:
        with pytest.raises(ValueError):
            MaxCoatingTime(value)


class TestMinLayerThickness:
    def test_satisfied(self) -> None:
        result = MinLayerThickness(1.0).evaluate(stack_of(1.0, 2.0, 3.0))
        assert result.satisfied
        assert result.violation == 0.0
        assert result.offending_layers == ()

    def test_violated(self) -> None:
        result = MinLayerThickness(2.0).evaluate(stack_of(3.0, 1.5, 0.5, 4.0))
        assert not result.satisfied
        assert result.violation == pytest.approx(0.75)
        assert result.offending_layers == (1, 2)
        assert "2" in result.message
        assert "0.5" in result.message

    def test_zero_thickness_ignored(self) -> None:
        result = MinLayerThickness(2.0).evaluate(stack_of(3.0, 0.0, 4.0))
        assert result.satisfied
        assert result.offending_layers == ()

    def test_materials_filter(self) -> None:
        stack = stack_of(0.5, 0.5, 0.5, 0.5)  # W, Si, W, Si
        result = MinLayerThickness(1.0, materials=("W",)).evaluate(stack)
        assert result.offending_layers == (0, 2)
        result = MinLayerThickness(1.0, materials=("B4C",)).evaluate(stack)
        assert result.satisfied

    def test_exactly_at_minimum(self) -> None:
        assert MinLayerThickness(2.0).evaluate(stack_of(2.0, 2.0)).satisfied

    def test_empty_stack(self) -> None:
        assert MinLayerThickness().evaluate(Stack([], SI)).satisfied


class TestMinLayerThicknessBoundsHint:
    def test_periodic_fixed_d(self) -> None:
        coating = Periodic.create(W, SI, N=5, d=4.0, gamma=free(0.4, 0.1, 0.9))
        hint = MinLayerThickness(1.0).bounds_hint(coating)
        assert list(hint) == ["gamma"]
        assert hint["gamma"] == pytest.approx((0.25, 0.75))

    def test_varying_d(self) -> None:
        coating = Periodic.create(W, SI, N=5, d=free(4.0, 2.0, 6.0))
        assert MinLayerThickness(1.0).bounds_hint(coating) == {}

    def test_too_thick_minimum(self) -> None:
        coating = Periodic.create(W, SI, N=5, d=4.0)
        assert MinLayerThickness(2.5).bounds_hint(coating) == {}

    def test_half_period_minimum_gives_point(self) -> None:
        coating = Periodic.create(W, SI, N=5, d=4.0)
        assert MinLayerThickness(2.0).bounds_hint(coating) == {"gamma": (0.5, 0.5)}

    @pytest.mark.parametrize("names", [("B4C",), ("W",), ("Si",)])
    def test_materials_filter_excludes_one_role(self, names: tuple[str, ...]) -> None:
        coating = Periodic.create(W, SI, N=5, d=4.0)
        hint = MinLayerThickness(1.0, materials=names).bounds_hint(coating)
        assert hint == {}

    def test_materials_filter_includes_both(self) -> None:
        coating = Periodic.create(W, SI, N=5, d=4.0)
        hint = MinLayerThickness(1.0, materials=("W", "Si")).bounds_hint(coating)
        assert set(hint) == {"gamma"}

    def test_single_layer(self) -> None:
        assert MinLayerThickness(1.0).bounds_hint(SingleLayer.create(W, t=5.0)) == {}

    def test_depth_graded_fixed_d_min(self) -> None:
        coating = DepthGraded.create(
            W, SI, N=10, gamma_top=0.4, gamma_bottom=0.5, d_min=3.0, d_max=6.0
        )
        hint = MinLayerThickness(0.6).bounds_hint(coating)
        assert set(hint) == {"gamma_top", "gamma_bottom"}
        for bounds in hint.values():
            assert bounds == pytest.approx((0.2, 0.8))

    def test_depth_graded_varying_d_min(self) -> None:
        coating = DepthGraded.create(W, SI, N=10, d_min=free(3.0, 2.0, 4.0), d_max=6.0)
        assert MinLayerThickness(0.6).bounds_hint(coating) == {}

    @pytest.mark.parametrize("which", ["lower", "upper"])
    def test_hinted_bounds_just_satisfy_constraint(self, which: str) -> None:
        d, minimum = 4.0, 1.0
        constraint = MinLayerThickness(minimum)
        coating = Periodic.create(W, SI, N=5, d=d, gamma=free(0.4, 0.0, 1.0))
        low, high = constraint.bounds_hint(coating)["gamma"]
        mirror = Mirror(coating, SI)
        inside = mirror.stack({"gamma": low if which == "lower" else high})
        assert constraint.evaluate(inside).satisfied
        eps = 1e-3
        outside = mirror.stack({"gamma": low - eps if which == "lower" else high + eps})
        assert not constraint.evaluate(outside).satisfied


class TestMaxTotalThickness:
    def test_satisfied(self) -> None:
        result = MaxTotalThickness(10.0).evaluate(stack_of(4.0, 6.0))
        assert result.satisfied
        assert result.violation == 0.0

    def test_violated(self) -> None:
        result = MaxTotalThickness(10.0).evaluate(stack_of(6.0, 6.0))
        assert not result.satisfied
        assert result.violation == pytest.approx(0.2)
        assert "12" in result.message
        assert "10" in result.message


class TestMaxLayers:
    def test_satisfied(self) -> None:
        assert MaxLayers(3).evaluate(stack_of(1.0, 1.0, 1.0)).satisfied

    def test_violated(self) -> None:
        result = MaxLayers(4).evaluate(stack_of(*[1.0] * 5))
        assert not result.satisfied
        assert result.violation == pytest.approx(0.25)
        assert "5" in result.message
        assert "4" in result.message

    def test_counts_only_non_zero(self) -> None:
        result = MaxLayers(2).evaluate(stack_of(1.0, 0.0, 1.0, 0.0))
        assert result.satisfied
        assert "2 layers" in result.message


class TestMaxCoatingTime:
    process = DepositionProcess({"W": 1.0, "Si": 1.0})

    def test_requires_process(self) -> None:
        with pytest.raises(ValueError, match="process"):
            MaxCoatingTime(10.0).evaluate(stack_of(1.0))

    def test_satisfied(self) -> None:
        result = MaxCoatingTime(3600.0).evaluate(stack_of(1000.0, 2000.0), self.process)
        assert result.satisfied
        assert result.violation == 0.0

    def test_violated(self) -> None:
        result = MaxCoatingTime(3600.0).evaluate(stack_of(3600.0, 3600.0), self.process)
        assert not result.satisfied
        assert result.violation == pytest.approx(1.0)
        assert "2 h" in result.message
        assert "1 h" in result.message

    def test_includes_overheads(self) -> None:
        process = DepositionProcess({"W": 1.0}, fixed_overhead=100.0)
        result = MaxCoatingTime(100.0).evaluate(Stack([Layer(W, 50.0)], SI), process)
        assert result.violation == pytest.approx(0.5)
