"""Tests for coatings, the coating registry, composites and `Mirror`."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from xstack import materials
from xstack.materials import VACUUM
from xstack.parameters import Parameter, ParameterSet, free
from xstack.structure.coating import Coating, CoatingBuildError, Mirror, coatings
from xstack.structure.coatings import (
    Composite,
    DepthGraded,
    NLayer,
    Periodic,
    SingleLayer,
    grading_laws,
)
from xstack.structure.coatings.simple import layer_labels
from xstack.structure.stack import Layer

W = materials.get("W")
SI = materials.get("Si")
B4C = materials.get("B4C")
WSI2 = materials.get("WSi2")
MOSI2 = materials.get("MoSi2")


def suffixes(layers: tuple[Layer, ...]) -> list[str]:
    return [layer.tag.rsplit(".", 1)[-1] for layer in layers]


def thicknesses(layers: tuple[Layer, ...]) -> list[float]:
    return [layer.thickness for layer in layers]


class TestRegistry:
    def test_all_types_registered(self) -> None:
        expected = {
            "single_layer": SingleLayer,
            "n_layer": NLayer,
            "periodic": Periodic,
            "depth_graded": DepthGraded,
            "composite": Composite,
        }
        for name, cls in expected.items():
            assert name in coatings
            assert coatings.get(name) is cls
            assert cls.type_name == name


class TestSingleLayer:
    def test_create(self) -> None:
        c = SingleLayer.create(W, t=5.0, sigma=0.3, name="top")
        assert c.name == "top"
        assert c.material("M") is W
        assert list(c.parameters) == ["t", "sigma"]
        (layer,) = c.layers()
        assert layer.material is W
        assert layer.thickness == 5.0
        assert layer.roughness == 0.3
        assert layer.tag == "top.M"

    def test_defaults(self) -> None:
        c = SingleLayer.create(W)
        assert c.name == "layer"
        assert c.parameters["t"].value == 30.0
        assert c.parameters["sigma"].value == 0.0

    def test_free_thickness(self) -> None:
        c = SingleLayer.create(W, t=free(10.0, 5.0, 20.0))
        assert c.parameters.free_names() == ("t",)

    def test_values_override(self) -> None:
        c = SingleLayer.create(W, t=5.0)
        assert c.layers({"t": 7.0})[0].thickness == 7.0
        assert c.layers()[0].thickness == 5.0

    def test_unknown_value_name(self) -> None:
        with pytest.raises(KeyError):
            SingleLayer.create(W).layers({"x": 1.0})

    def test_negative_thickness_rejected(self) -> None:
        with pytest.raises(ValueError):
            SingleLayer.create(W).layers({"t": -1.0})


class TestNLayer:
    def test_create(self) -> None:
        c = NLayer.create([W, SI, B4C], [1.0, 2.0, 3.0], [0.1, 0.2, 0.3])
        assert [role for role, _ in c.material_roles] == ["M1", "M2", "M3"]
        assert c.count == 3
        assert list(c.parameters) == ["t_W", "sigma_W", "t_Si", "sigma_Si", "t_B4C", "sigma_B4C"]
        layers = c.layers()
        assert [layer.material for layer in layers] == [W, SI, B4C]
        assert thicknesses(layers) == [1.0, 2.0, 3.0]
        assert [layer.roughness for layer in layers] == [0.1, 0.2, 0.3]
        assert [layer.tag for layer in layers] == ["layers.M1", "layers.M2", "layers.M3"]

    def test_default_roughness(self) -> None:
        layers = NLayer.create([W, SI], [1.0, 2.0], name="x").layers()
        assert [layer.roughness for layer in layers] == [0.0, 0.0]
        assert layers[0].tag == "x.M1"

    def test_specs(self) -> None:
        c = NLayer.create([W, SI], [free(2.0, 1.0, 3.0), "t_W + 1"])
        assert c.parameters.free_names() == ("t_W",)
        assert thicknesses(c.layers()) == [2.0, 3.0]
        assert thicknesses(c.layers({"t_W": 1.5})) == [1.5, 2.5]

    def test_parameters_named_after_materials(self) -> None:
        c = NLayer.create([W, SI, B4C, SI], [1.0] * 4)
        assert list(c.parameters)[::2] == ["t_W", "t_Si_1", "t_B4C", "t_Si_2"]
        assert list(c.parameters)[1::2] == ["sigma_W", "sigma_Si_1", "sigma_B4C", "sigma_Si_2"]

    def test_legacy_positional_names_still_build(self) -> None:
        new = NLayer.create([W, SI], [1.5, 2.5], [0.1, 0.2])
        legacy = ParameterSet(
            Parameter(old, p.value, min=p.min, max=p.max, unit=p.unit)
            for old, p in zip(
                ["t1", "sigma1", "t2", "sigma2"], new.parameters.values(), strict=True
            )
        )
        old = NLayer("layers", new.material_roles, legacy)
        assert thicknesses(old.layers()) == [1.5, 2.5]
        assert [layer.roughness for layer in old.layers()] == [0.1, 0.2]
        assert old.layers({"t2": 4.0})[1].thickness == 4.0

    @pytest.mark.parametrize(
        ("n_mat", "n_t", "n_s"),
        [(2, 3, 2), (3, 2, 3), (2, 2, 3), (2, 2, 1)],
    )
    def test_length_mismatch(self, n_mat: int, n_t: int, n_s: int) -> None:
        with pytest.raises(ValueError):
            NLayer.create([W] * n_mat, [1.0] * n_t, [0.0] * n_s)

    def test_empty(self) -> None:
        with pytest.raises(ValueError):
            NLayer.create([], [])

    def test_roles_out_of_order(self) -> None:
        with pytest.raises(ValueError):
            NLayer("x", (("M2", W), ("M1", SI)), ParameterSet())

    def test_with_material(self) -> None:
        c = NLayer.create([W, SI], [1.0, 2.0]).with_material("M2", B4C)
        assert c.layers()[1].material is B4C


class TestPeriodicBasics:
    def test_layer_count_and_thicknesses(self) -> None:
        c = Periodic.create(W, SI, N=7, d=5.0, gamma=0.3)
        layers = c.layers()
        assert len(layers) == 14
        for i in range(7):
            assert layers[2 * i].material is W
            assert layers[2 * i + 1].material is SI
            assert layers[2 * i].thickness == pytest.approx(1.5)
            assert layers[2 * i + 1].thickness == pytest.approx(3.5)

    def test_tags(self) -> None:
        layers = Periodic.create(W, SI, N=2, name="ml").layers()
        assert [layer.tag for layer in layers] == ["ml[1].H", "ml[1].L", "ml[2].H", "ml[2].L"]

    def test_top_l(self) -> None:
        layers = Periodic.create(W, SI, N=3, d=5.0, gamma=0.3, top="L").layers()
        assert len(layers) == 6
        assert [layer.material for layer in layers] == [SI, W] * 3
        assert thicknesses(layers[:2]) == pytest.approx([3.5, 1.5])

    def test_total_thickness(self) -> None:
        layers = Periodic.create(W, SI, N=12, d=4.5, gamma=0.45).layers()
        assert sum(thicknesses(layers)) == pytest.approx(12 * 4.5)

    @pytest.mark.parametrize("top", ["H", "L"])
    def test_roughness_assignment(self, top: str) -> None:
        c = Periodic.create(W, SI, N=3, sigma_H=0.3, sigma_L=0.5, top=top)  # type: ignore[arg-type]
        for layer in c.layers():
            if layer.material is W:
                assert layer.roughness == 0.3
            else:
                assert layer.roughness == 0.5

    def test_n_rounding(self) -> None:
        c = Periodic.create(W, SI, N=free(10, 5, 20))
        assert len(c.layers({"N": 10.4})) == 20
        assert len(c.layers({"N": 10.6})) == 22

    def test_n_with_values_rounded(self) -> None:
        assert (
            Periodic.create(W, SI, N=free(10, 5, 20)).with_values(N=10.4).parameters["N"].value
            == 10
        )

    def test_values_override_does_not_modify(self) -> None:
        c = Periodic.create(W, SI, N=4, d=5.0, gamma=0.4)
        layers = c.layers({"d": 6.0, "gamma": 0.5, "N": 2})
        assert thicknesses(layers) == pytest.approx([3.0, 3.0, 3.0, 3.0])
        assert len(c.layers()) == 8

    def test_override_out_of_bounds(self) -> None:
        with pytest.raises(ValueError):
            Periodic.create(W, SI).layers({"gamma": 1.5})

    def test_expression_parameter(self) -> None:
        c = Periodic.create(W, SI, N=1, d=4.0, gamma="d / 10")
        assert thicknesses(c.layers()) == pytest.approx([1.6, 2.4])
        assert thicknesses(c.layers({"d": 5.0})) == pytest.approx([2.5, 2.5])

    def test_expression_parameter_forward_reference(self) -> None:
        c = Periodic.create(W, SI, N=1, gamma=0.5, d="2*gamma + 3")
        assert thicknesses(c.layers()) == pytest.approx([2.0, 2.0])
        assert thicknesses(c.layers({"gamma": 0.25})) == pytest.approx([0.875, 2.625])

    def test_free_names(self) -> None:
        c = Periodic.create(W, SI, N=free(10, 5, 20), d=free(4.0, 3.0, 5.0))
        assert c.parameters.free_names() == ("N", "d")

    def test_interlayer_thickness_ignored_without_material(self) -> None:
        c = Periodic.create(
            W, SI, N=2, d=5.0, gamma=0.4, t_interlayer_H_on_L=1.0, t_interlayer_L_on_H=1.0
        )
        assert len(c.layers()) == 4
        assert thicknesses(c.layers()[:2]) == pytest.approx([2.0, 3.0])


class TestPeriodicInterlayers:
    def make(self, top: str = "H", n: int = 3, **overrides: float) -> Periodic:
        kwargs: dict[str, object] = {
            "t_interlayer_H_on_L": 0.3,
            "t_interlayer_L_on_H": 0.2,
            **overrides,
        }
        return Periodic.create(
            W,
            SI,
            N=n,
            d=5.0,
            gamma=0.4,
            interlayers={"interlayer_H_on_L": B4C, "interlayer_L_on_H": MOSI2},
            top=top,  # type: ignore[arg-type]
            **kwargs,  # type: ignore[arg-type]
        )

    def test_top_h_sequence(self) -> None:
        layers = self.make("H").layers()
        assert suffixes(layers) == ["H", "interlayer_H_on_L", "L", "interlayer_L_on_H"] * 2 + [
            "H",
            "interlayer_H_on_L",
            "L",
        ]
        assert thicknesses(layers[:4]) == pytest.approx([2.0, 0.3, 2.5, 0.2])
        assert [layer.material for layer in layers[:4]] == [W, B4C, SI, MOSI2]

    def test_top_l_sequence(self) -> None:
        layers = self.make("L").layers()
        assert suffixes(layers) == ["L", "interlayer_L_on_H", "H", "interlayer_H_on_L"] * 2 + [
            "L",
            "interlayer_L_on_H",
            "H",
        ]
        assert thicknesses(layers[:4]) == pytest.approx([2.5, 0.2, 2.0, 0.3])
        assert [layer.material for layer in layers[:4]] == [SI, MOSI2, W, B4C]

    @pytest.mark.parametrize("top", ["H", "L"])
    def test_period_includes_interlayers(self, top: str) -> None:
        layers = self.make(top, n=4).layers()
        total = sum(thicknesses(layers))
        trailing = layers[-1]
        assert trailing.tag.endswith(("L", "H"))
        assert total == pytest.approx(4 * 5.0 - (0.2 if top == "H" else 0.3))
        first_bilayer = [layer for layer in layers if layer.tag.startswith("periodic[1]")]
        assert sum(thicknesses(tuple(first_bilayer))) == pytest.approx(5.0)

    def test_bottom_trailing_interlayer_omitted(self) -> None:
        h = self.make("H", n=1).layers()
        assert suffixes(h) == ["H", "interlayer_H_on_L", "L"]
        ll = self.make("L", n=1).layers()
        assert suffixes(ll) == ["L", "interlayer_L_on_H", "H"]

    def test_interlayer_roughness(self) -> None:
        layers = self.make("H", sigma_H=0.7, sigma_L=0.9).layers()
        by_suffix = {layer.tag.rsplit(".", 1)[-1]: layer.roughness for layer in layers}
        assert by_suffix == {"H": 0.7, "interlayer_H_on_L": 0.9, "L": 0.9, "interlayer_L_on_H": 0.7}

    def test_only_honl_top_h(self) -> None:
        c = Periodic.create(
            W, SI, N=3, d=5.0, interlayers={"interlayer_H_on_L": B4C}, t_interlayer_H_on_L=0.5
        )
        layers = c.layers()
        assert suffixes(layers) == ["H", "interlayer_H_on_L", "L"] * 3
        assert sum(thicknesses(layers)) == pytest.approx(15.0)

    def test_only_honl_top_l(self) -> None:
        c = Periodic.create(
            W,
            SI,
            N=3,
            d=5.0,
            interlayers={"interlayer_H_on_L": B4C},
            t_interlayer_H_on_L=0.5,
            top="L",
        )
        assert suffixes(c.layers()) == ["L", "H", "interlayer_H_on_L"] * 2 + ["L", "H"]

    def test_only_lonh_top_h(self) -> None:
        c = Periodic.create(
            W, SI, N=3, d=5.0, interlayers={"interlayer_L_on_H": MOSI2}, t_interlayer_L_on_H=0.5
        )
        assert suffixes(c.layers()) == ["H", "L", "interlayer_L_on_H"] * 2 + ["H", "L"]

    def test_only_lonh_top_l(self) -> None:
        c = Periodic.create(
            W,
            SI,
            N=3,
            d=5.0,
            interlayers={"interlayer_L_on_H": MOSI2},
            t_interlayer_L_on_H=0.5,
            top="L",
        )
        assert suffixes(c.layers()) == ["L", "interlayer_L_on_H", "H"] * 3

    def test_zero_thickness_omitted(self) -> None:
        c = Periodic.create(
            W,
            SI,
            N=3,
            d=5.0,
            gamma=0.4,
            interlayers={"interlayer_H_on_L": B4C, "interlayer_L_on_H": WSI2},
        )
        layers = c.layers()
        assert suffixes(layers) == ["H", "L"] * 3
        assert thicknesses(layers[:2]) == pytest.approx([2.0, 3.0])

    def test_zero_thickness_omitted_for_one_role(self) -> None:
        c = self.make("H", t_interlayer_H_on_L=0.0)
        assert suffixes(c.layers()[:3]) == ["H", "L", "interlayer_L_on_H"]

    def test_thickness_via_values(self) -> None:
        c = Periodic.create(
            W,
            SI,
            N=2,
            d=5.0,
            gamma=0.4,
            interlayers={"interlayer_H_on_L": B4C},
            t_interlayer_H_on_L=free(0.0, 0.0, 1.0),
        )
        assert suffixes(c.layers()) == ["H", "L", "H", "L"]
        layers = c.layers({"t_interlayer_H_on_L": 0.5})
        assert suffixes(layers) == ["H", "interlayer_H_on_L", "L", "H", "interlayer_H_on_L", "L"]
        assert thicknesses(layers[:3]) == pytest.approx([2.0, 0.5, 2.5])


class TestPeriodicBuildErrors:
    def test_negative_spacer_thickness(self) -> None:
        c = Periodic.create(
            W,
            SI,
            N=3,
            d=5.0,
            gamma=0.5,
            interlayers={"interlayer_H_on_L": B4C, "interlayer_L_on_H": MOSI2},
            t_interlayer_H_on_L=free(0.5, 0.0, 5.0),
            t_interlayer_L_on_H=0.5,
        )
        assert len(c.layers()) > 0
        with pytest.raises(CoatingBuildError):
            c.with_values(t_interlayer_H_on_L=2.5).layers()
        with pytest.raises(CoatingBuildError):
            c.layers({"t_interlayer_H_on_L": 2.5})

    def test_negative_period(self) -> None:
        c = Periodic.create(W, SI, N=2, d=Parameter("d", -1.0, min=-5.0), gamma=0.5)
        with pytest.raises(CoatingBuildError):
            c.layers()

    def test_gamma_outside_unit_interval(self) -> None:
        for gamma in (1.5, -0.1):
            c = Periodic.create(W, SI, N=2, d=5.0, gamma=Parameter("gamma", gamma, min=-1, max=2))
            with pytest.raises(CoatingBuildError):
                c.layers()

    def test_gamma_bounds_enforced_by_with_values(self) -> None:
        c = Periodic.create(W, SI)
        with pytest.raises(ValueError):
            c.with_values(gamma=1.5)
        with pytest.raises(ValueError):
            c.with_values(gamma=-0.1)

    def test_gamma_limits_allowed(self) -> None:
        layers = Periodic.create(W, SI, N=1, d=5.0, gamma=1.0).layers()
        assert thicknesses(layers) == pytest.approx([5.0, 0.0])
        layers = Periodic.create(W, SI, N=1, d=5.0, gamma=0.0).layers()
        assert thicknesses(layers) == pytest.approx([0.0, 5.0])

    def test_error_is_value_error(self) -> None:
        assert issubclass(CoatingBuildError, ValueError)


def graded(**kwargs: object) -> DepthGraded:
    return DepthGraded.create(W, SI, **kwargs)  # type: ignore[arg-type]


def bilayer_periods(layers: tuple[Layer, ...], n: int) -> np.ndarray:
    t = np.array(thicknesses(layers))
    return t.reshape(n, 2).sum(axis=1)


class TestDepthGradedPowerLaw:
    @pytest.mark.parametrize("c", [0.1, 0.25, 0.5, 1.0, 2.0, 5.0])
    def test_end_points_and_monotonic(self, c: float) -> None:
        d = graded(N=50, d_min=2.0, d_max=9.0, c=c).periods()
        assert d.shape == (50,)
        assert d[0] == pytest.approx(9.0)
        assert d[-1] == pytest.approx(2.0)
        assert np.all(np.diff(d) < 0)

    @pytest.mark.parametrize("c", [0.1, 0.25, 1.0, 5.0])
    def test_formula(self, c: float) -> None:
        n, d_min, d_max = 40, 2.5, 8.0
        d = graded(N=n, d_min=d_min, d_max=d_max, c=c).periods()
        # Solve d_1 = a/(b+1)^c and d_N = a/(b+N)^c for a and b independently.
        ratio = (d_max / d_min) ** (1.0 / c)
        b = (n - ratio) / (ratio - 1.0)
        a = d_max * (b + 1.0) ** c
        i = np.arange(1, n + 1)
        np.testing.assert_allclose(d, a / (b + i) ** c, rtol=1e-12)

    def test_larger_c_changes_profile(self) -> None:
        small = graded(N=30, d_min=2.0, d_max=9.0, c=0.2).periods()
        large = graded(N=30, d_min=2.0, d_max=9.0, c=3.0).periods()
        assert small[0] == pytest.approx(large[0])
        assert small[-1] == pytest.approx(large[-1])
        assert not np.allclose(small, large)
        assert abs(small[15] - large[15]) > 0.1

    def test_single_bilayer_constant(self) -> None:
        d = graded(N=1, d_min=2.0, d_max=9.0).periods()
        np.testing.assert_allclose(d, [9.0])

    def test_equal_limits_constant(self) -> None:
        d = graded(N=10, d_min=4.0, d_max=4.0).periods()
        np.testing.assert_allclose(d, 4.0)

    def test_d_min_greater_than_d_max(self) -> None:
        c = graded(N=10, d_min=12.0, d_max=10.0)
        with pytest.raises(CoatingBuildError):
            c.layers()
        with pytest.raises(CoatingBuildError):
            c.periods()

    def test_zero_d_min(self) -> None:
        with pytest.raises(CoatingBuildError):
            graded(N=10, d_min=0.0, d_max=10.0).layers()

    def test_layers_follow_periods(self) -> None:
        c = graded(N=12, d_min=2.0, d_max=9.0, c=0.3)
        layers = c.layers()
        assert len(layers) == 24
        np.testing.assert_allclose(bilayer_periods(layers, 12), c.periods())

    def test_layer_tags_and_order(self) -> None:
        layers = graded(N=2, name="g").layers()
        assert [layer.tag for layer in layers] == ["g[1].H", "g[1].L", "g[2].H", "g[2].L"]
        assert [layer.material for layer in layers] == [W, SI, W, SI]

    def test_top_l(self) -> None:
        layers = graded(N=2, top="L").layers()
        assert [layer.material for layer in layers] == [SI, W, SI, W]

    def test_n_via_values(self) -> None:
        c = graded(N=free(10, 5, 20), d_min=2.0, d_max=9.0)
        assert len(c.layers({"N": 7.4})) == 14
        assert c.periods({"N": 7.4}).shape == (7,)

    def test_periods_values_override(self) -> None:
        c = graded(N=10, d_min=2.0, d_max=9.0)
        assert c.periods({"d_max": 8.0})[0] == pytest.approx(8.0)
        assert c.periods()[0] == pytest.approx(9.0)

    def test_params_present(self) -> None:
        names = list(graded().parameters)
        for name in ["N", "d_min", "d_max", "c", "gamma_top", "gamma_bottom", "sigma_H"]:
            assert name in names


class TestDepthGradedLinearLaw:
    def test_periods_linspace(self) -> None:
        d = graded(N=9, law="linear", d_min=3.0, d_max=10.0).periods()
        np.testing.assert_allclose(d, np.linspace(10.0, 3.0, 9))

    def test_layers_follow_periods(self) -> None:
        c = graded(N=9, law="linear", d_min=3.0, d_max=10.0)
        np.testing.assert_allclose(bilayer_periods(c.layers(), 9), np.linspace(10.0, 3.0, 9))

    def test_no_exponent_parameter(self) -> None:
        assert "c" not in graded(law="linear").parameters

    def test_unknown_law(self) -> None:
        with pytest.raises(KeyError):
            graded(law="nonexistent")

    def test_unknown_law_parameter(self) -> None:
        with pytest.raises(ValueError):
            graded(law="linear", c=0.3)
        with pytest.raises(ValueError):
            graded(bogus=1.0)

    def test_laws_registered(self) -> None:
        assert "power_law" in grading_laws
        assert "linear" in grading_laws


class TestDepthGradedGamma:
    def test_default_bottom_tied_to_top(self) -> None:
        c = graded(N=6, gamma_top=0.3)
        assert c.parameters["gamma_bottom"].expr == "gamma_top"
        assert not c.parameters["gamma_bottom"].is_free
        np.testing.assert_allclose(c.gammas(), 0.3)

    def test_changing_top_changes_all(self) -> None:
        c = graded(N=6, gamma_top=0.3).with_values(gamma_top=0.6)
        np.testing.assert_allclose(c.gammas(), 0.6)
        layers = c.layers()
        periods = c.periods()
        for i in range(6):
            assert layers[2 * i].thickness == pytest.approx(0.6 * periods[i])

    def test_linear_interpolation(self) -> None:
        c = graded(N=5, gamma_top=0.2, gamma_bottom=0.6)
        np.testing.assert_allclose(c.gammas(), [0.2, 0.3, 0.4, 0.5, 0.6])

    def test_layers_use_gammas(self) -> None:
        c = graded(N=5, gamma_top=0.2, gamma_bottom=0.6)
        layers = c.layers()
        periods = c.periods()
        gammas = c.gammas()
        for i in range(5):
            assert layers[2 * i].thickness == pytest.approx(gammas[i] * periods[i])
            assert layers[2 * i + 1].thickness == pytest.approx((1 - gammas[i]) * periods[i])

    def test_gammas_values_override(self) -> None:
        c = graded(N=3, gamma_top=0.2, gamma_bottom=0.6)
        np.testing.assert_allclose(c.gammas({"gamma_bottom": 0.4}), [0.2, 0.3, 0.4])

    def test_single_bilayer_gamma(self) -> None:
        np.testing.assert_allclose(graded(N=1, gamma_top=0.3).gammas(), [0.3])

    def test_free_gamma_bottom(self) -> None:
        c = graded(gamma_bottom=free(0.4, 0.2, 0.5))
        assert "gamma_bottom" in c.parameters.free_names()


class TestDepthGradedCap:
    def test_cap_parameters(self) -> None:
        c = graded(N=4, cap=True, cap_d=12.0, cap_gamma=0.35)
        assert c.parameters["cap_d"].value == 12.0
        assert c.parameters["cap_gamma"].value == 0.35

    def test_no_cap_parameters(self) -> None:
        c = graded(N=4, cap=False)
        assert "cap_d" not in c.parameters
        assert "cap_gamma" not in c.parameters

    def test_cap_layers(self) -> None:
        c = graded(N=4, cap=True, cap_d=12.0, cap_gamma=0.35, name="graded")
        layers = c.layers()
        assert len(layers) == 2 * 4 + 2
        assert [layer.tag for layer in layers[:2]] == ["graded.cap.H", "graded.cap.L"]
        assert layers[0].thickness == pytest.approx(0.35 * 12.0)
        assert layers[1].thickness == pytest.approx(0.65 * 12.0)
        assert layers[2].tag == "graded[1].H"
        np.testing.assert_allclose(bilayer_periods(layers[2:], 4), c.periods())

    def test_cap_top_l(self) -> None:
        layers = graded(N=2, cap=True, cap_d=12.0, cap_gamma=0.25, top="L").layers()
        assert [layer.tag for layer in layers[:2]] == ["graded.cap.L", "graded.cap.H"]
        assert thicknesses(layers[:2]) == pytest.approx([9.0, 3.0])

    def test_cap_values_override(self) -> None:
        c = graded(N=2, cap=True, cap_d=free(12.0, 5.0, 20.0), cap_gamma=0.5)
        layers = c.layers({"cap_d": 8.0})
        assert thicknesses(layers[:2]) == pytest.approx([4.0, 4.0])

    def test_cap_with_interlayers_not_bottom(self) -> None:
        c = DepthGraded.create(
            W,
            SI,
            N=1,
            cap=True,
            cap_d=10.0,
            cap_gamma=0.4,
            interlayers={"interlayer_H_on_L": B4C},
            t_interlayer_H_on_L=0.5,
            d_min=5.0,
            d_max=5.0,
        )
        assert suffixes(c.layers()) == [
            "H",
            "interlayer_H_on_L",
            "L",
            "H",
            "interlayer_H_on_L",
            "L",
        ]

    def test_layer_count_without_cap(self) -> None:
        assert len(graded(N=9).layers()) == 18


class TestComposite:
    def make(self) -> Composite:
        cap = SingleLayer.create(B4C, t=2.0, sigma=0.2, name="cap")
        ml = Periodic.create(W, SI, N=free(3, 1, 10), d=4.0, gamma=0.5, name="ml")
        return Composite.create([cap, ml], name="comp")

    def test_parameters_prefixed(self) -> None:
        c = self.make()
        assert list(c.parameters) == [
            "cap.t",
            "cap.sigma",
            "ml.N",
            "ml.d",
            "ml.gamma",
            "ml.sigma_H",
            "ml.sigma_L",
            "ml.t_interlayer_H_on_L",
            "ml.t_interlayer_L_on_H",
        ]
        assert c.parameters["ml.N"].integer
        assert c.parameters.free_names() == ("ml.N",)

    def test_layers_in_block_order(self) -> None:
        layers = self.make().layers()
        assert len(layers) == 1 + 6
        assert layers[0].tag == "cap.M"
        assert layers[0].material is B4C
        assert layers[0].thickness == 2.0
        assert layers[0].roughness == 0.2
        assert [layer.tag for layer in layers[1:3]] == ["ml[1].H", "ml[1].L"]

    def test_block_order_reversed(self) -> None:
        c = self.make()
        rev = Composite.create(list(reversed(c.blocks)))
        assert rev.layers()[-1].tag == "cap.M"
        assert len(rev.layers()) == 7

    def test_duplicate_names(self) -> None:
        a = SingleLayer.create(W, name="x")
        b = SingleLayer.create(SI, name="x")
        with pytest.raises(ValueError):
            Composite.create([a, b])

    def test_empty(self) -> None:
        with pytest.raises(ValueError):
            Composite.create([])

    def test_with_values_changes_right_block(self) -> None:
        c = self.make()
        new = c.with_values(**{"ml.N": 5, "cap.t": 3.0})
        layers = new.layers()
        assert len(layers) == 1 + 10
        assert layers[0].thickness == 3.0
        assert len(c.layers()) == 7

    def test_with_values_period(self) -> None:
        layers = self.make().with_values(**{"ml.d": 6.0}).layers()
        assert thicknesses(layers[1:3]) == pytest.approx([3.0, 3.0])
        assert layers[0].thickness == 2.0

    def test_layers_values_override(self) -> None:
        c = self.make()
        assert len(c.layers({"ml.N": 4.4})) == 1 + 8
        assert c.layers({"cap.t": 9.0})[0].thickness == 9.0

    def test_block_returns_current_values(self) -> None:
        c = self.make().with_values(**{"ml.d": 5.0, "ml.N": 6})
        block = c.block("ml")
        assert isinstance(block, Periodic)
        assert block.name == "ml"
        assert block.parameters["d"].value == 5.0
        assert block.parameters["N"].value == 6.0
        assert block.parameters["N"].vary
        assert block.material("H") is W
        assert c.blocks[1].parameters["d"].value == 4.0

    def test_block_layers_match(self) -> None:
        c = self.make().with_values(**{"ml.d": 5.0})
        assert c.block("ml").layers() == c.layers()[1:]
        assert c.block("cap").layers() == c.layers()[:1]

    def test_block_with_expression(self) -> None:
        g = DepthGraded.create(W, SI, N=4, gamma_top=0.3, name="graded")
        c = Composite.create([SingleLayer.create(B4C, name="cap"), g])
        assert c.parameters["graded.gamma_bottom"].expr == "graded.gamma_top"
        block = c.block("graded")
        assert block.parameters["gamma_bottom"].expr == "gamma_top"
        assert block.parameters == g.parameters

    def test_expression_in_composite_follows(self) -> None:
        g = DepthGraded.create(W, SI, N=4, gamma_top=0.3, name="graded")
        c = Composite.create([g]).with_values(**{"graded.gamma_top": 0.5})
        block = c.block("graded")
        assert isinstance(block, DepthGraded)
        np.testing.assert_allclose(block.gammas(), 0.5)

    def test_cross_block_expression(self) -> None:
        ml = Periodic.create(W, SI, N=2, d=4.0, name="ml")
        cap = SingleLayer.create(B4C, t="ml.d / 2", name="cap")
        c = Composite.create([ml, cap])
        assert c.layers()[-1].thickness == pytest.approx(2.0)
        assert c.with_values(**{"ml.d": 6.0}).layers()[-1].thickness == pytest.approx(3.0)

    def test_cross_block_expression_forward_reference(self) -> None:
        c = self.make().with_parameter("cap.t", expr="ml.d / 2")
        assert c.layers()[0].thickness == pytest.approx(2.0)

    def test_unknown_block(self) -> None:
        with pytest.raises(KeyError):
            self.make().block("nope")

    def test_unknown_parameter(self) -> None:
        with pytest.raises(KeyError):
            self.make().with_values(d=1.0)

    def test_nested_composite(self) -> None:
        inner = Composite.create([SingleLayer.create(B4C, name="a")], name="inner")
        outer = Composite.create([inner, SingleLayer.create(W, name="b")])
        assert "inner.a.t" in outer.parameters
        assert len(outer.layers()) == 2

    def test_hashable_and_equal(self) -> None:
        assert self.make() == self.make()
        assert hash(self.make()) == hash(self.make())


class TestCoatingCommon:
    def make(self) -> Periodic:
        return Periodic.create(W, SI, N=free(10, 5, 20), d=4.0)

    def test_with_parameter_returns_new(self) -> None:
        c = self.make()
        new = c.with_parameter("d", vary=True, min=3.0, max=5.0)
        assert new is not c
        assert new.parameters["d"].is_free
        assert not c.parameters["d"].is_free
        assert isinstance(new, Periodic)

    def test_with_values_returns_new(self) -> None:
        c = self.make()
        new = c.with_values(d=3.5, gamma=0.5)
        assert new.parameters["d"].value == 3.5
        assert c.parameters["d"].value == 4.0
        assert new.top == c.top

    def test_with_values_unknown(self) -> None:
        with pytest.raises(KeyError):
            self.make().with_values(x=1.0)

    def test_with_material_returns_new(self) -> None:
        c = self.make()
        new = c.with_material("L", B4C)
        assert new.material("L") is B4C
        assert c.material("L") is SI
        assert new.layers()[1].material is B4C

    def test_with_material_adds_optional_role(self) -> None:
        c = (
            self.make()
            .with_material("interlayer_H_on_L", B4C)
            .with_parameter("t_interlayer_H_on_L", value=0.5)
        )
        assert "interlayer_H_on_L" in c.materials
        assert "interlayer_H_on_L" in suffixes(c.layers())

    def test_with_material_replacing_keeps_roles(self) -> None:
        new = self.make().with_material("H", WSI2)
        assert [r for r, _ in new.material_roles] == ["H", "L"]

    def test_with_material_unknown_role(self) -> None:
        with pytest.raises(ValueError):
            self.make().with_material("X", B4C)

    def test_renamed_returns_new(self) -> None:
        c = self.make()
        new = c.renamed("other")
        assert new.name == "other"
        assert c.name == "periodic"
        assert new.layers()[0].tag == "other[1].H"

    def test_renamed_invalid(self) -> None:
        with pytest.raises(ValueError):
            self.make().renamed("not valid")

    def test_with_parameters_same_names(self) -> None:
        c = self.make()
        new = c.with_parameters(c.parameters.replace("d", value=3.0))
        assert new.parameters["d"].value == 3.0
        assert c.parameters["d"].value == 4.0

    def test_with_parameters_different_names(self) -> None:
        c = self.make()
        with pytest.raises(ValueError):
            c.with_parameters(ParameterSet([Parameter("d", 1.0)]))
        with pytest.raises(ValueError):
            c.with_parameters(ParameterSet(list(c.parameters.values())[:-1]))

    def test_missing_required_role(self) -> None:
        with pytest.raises(ValueError):
            Periodic("p", (("H", W),), ParameterSet())
        with pytest.raises(ValueError):
            SingleLayer("s", (), ParameterSet())

    def test_unknown_role(self) -> None:
        with pytest.raises(ValueError):
            Periodic("p", (("H", W), ("L", SI), ("X", B4C)), ParameterSet())
        with pytest.raises(ValueError):
            Periodic.create(W, SI, interlayers={"foo": B4C})

    @pytest.mark.parametrize("name", ["", "1a", "a.b", "a b", "a-b"])
    def test_invalid_name(self, name: str) -> None:
        with pytest.raises(ValueError):
            Periodic.create(W, SI, name=name)
        with pytest.raises(ValueError):
            SingleLayer.create(W, name=name)

    def test_material_lookup(self) -> None:
        c = self.make()
        assert c.materials == {"H": W, "L": SI}
        assert c.material("H") is W
        with pytest.raises(KeyError):
            c.material("interlayer_H_on_L")

    def test_hashable_and_equal(self) -> None:
        a, b = self.make(), self.make()
        assert a == b
        assert hash(a) == hash(b)
        assert len({a, b}) == 1

    def test_not_equal_when_different(self) -> None:
        a = self.make()
        assert a != a.with_values(d=3.0)
        assert a != a.with_material("L", B4C)
        assert a != a.renamed("x")
        assert a != Periodic.create(W, SI, N=free(10, 5, 20), d=4.0, top="L")

    def test_frozen(self) -> None:
        with pytest.raises(AttributeError):
            self.make().name = "x"  # type: ignore[misc]

    def test_is_coating(self) -> None:
        assert isinstance(self.make(), Coating)


class TestMirror:
    def make(self) -> Mirror:
        return Mirror(Periodic.create(W, SI, N=free(5, 1, 10), d=4.0), B4C, 0.4)

    def test_stack(self) -> None:
        stack = self.make().stack()
        assert stack.num_layers == 10
        assert stack.substrate is B4C
        assert stack.substrate_roughness == 0.4
        assert stack.ambient == VACUUM
        assert stack.layers == self.make().coating.layers()

    def test_ambient(self) -> None:
        m = Mirror(Periodic.create(W, SI, N=2), B4C, ambient=SI)
        assert m.stack().ambient is SI
        assert m.stack().substrate_roughness == 0.0

    def test_stack_values(self) -> None:
        m = self.make()
        stack = m.stack({"N": 3, "d": 6.0})
        assert stack.num_layers == 6
        assert stack.total_thickness == pytest.approx(18.0)
        assert m.stack().num_layers == 10

    def test_stack_invalid_values(self) -> None:
        m = Mirror(Periodic.create(W, SI, d=Parameter("d", -1.0, min=-5.0)), B4C)
        with pytest.raises(CoatingBuildError):
            m.stack()

    def test_parameters(self) -> None:
        m = self.make()
        assert m.parameters is m.coating.parameters
        assert m.parameters.free_names() == ("N",)

    def test_with_coating(self) -> None:
        m = self.make()
        new = m.with_coating(SingleLayer.create(W))
        assert new.stack().num_layers == 1
        assert new.substrate is B4C
        assert m.stack().num_layers == 10

    def test_hashable_and_equal(self) -> None:
        assert self.make() == self.make()
        assert hash(self.make()) == hash(self.make())


def test_expression_outside_bounds_raises_build_error() -> None:
    from xstack.materials import get
    from xstack.structure import CoatingBuildError, Periodic

    coating = Periodic.create(get("W"), get("Si"), gamma="1.5")
    with pytest.raises(CoatingBuildError, match="outside"):
        coating.layers()
    coating = Periodic.create(get("W"), get("Si"), d=4.0, gamma="d / 10")
    assert coating.layers()[0].thickness == pytest.approx(1.6)


class TestLayerLabels:
    def test_unique_material_keeps_name(self) -> None:
        assert layer_labels([W, SI, B4C]) == ["W", "Si", "B4C"]

    def test_repeated_material_is_indexed(self) -> None:
        assert layer_labels([SI, W, SI, SI]) == ["Si_1", "W", "Si_2", "Si_3"]

    def test_sanitizing(self) -> None:
        odd = [materials.get("Si"), materials.get("B4C")]
        assert layer_labels(odd) == ["Si", "B4C"]
        assert layer_labels([_named("2-x"), _named("a b"), _named("-")]) == [
            "m_2_x",
            "a_b",
            "layer",
        ]

    def test_collision_with_indexed_label(self) -> None:
        labels = layer_labels([_named("Si"), _named("Si"), _named("Si_1")])
        assert len(set(labels)) == 3
        assert "Si_1" in labels

    @given(st.lists(st.text(min_size=1, max_size=6), min_size=1, max_size=12))
    def test_labels_are_unique_identifiers(self, names: list[str]) -> None:
        labels = layer_labels([_named(n) for n in names])
        assert len(labels) == len(names)
        assert len(set(labels)) == len(labels)
        for label in labels:
            assert f"t_{label}".isidentifier()
            assert f"sigma_{label}".isidentifier()
            assert "__" not in label

    @given(st.lists(st.sampled_from(["Si", "W", "B4C", "Si_1", "x"]), min_size=1, max_size=10))
    def test_nlayer_parameters_unique(self, names: list[str]) -> None:
        mats = [_named(n) for n in names]
        c = NLayer.create(mats, [1.0] * len(mats))
        assert len(c.parameters) == 2 * len(mats)


def _named(name: str) -> materials.Material:
    return materials.Material(name, "Si", 2.33)
