"""Tests for `Parameter`, `ParameterSet` and the specification helpers."""

import math

import lmfit
import pytest

from xstack.core.parameters import Parameter, ParameterSet, Range, free, resolve


class TestParameterValidation:
    @pytest.mark.parametrize("name", ["", "a..b", "1x", "a b", ".a", "a.", "a__b", "x.y__z"])
    def test_invalid_names(self, name: str) -> None:
        with pytest.raises(ValueError):
            Parameter(name, 1.0)

    @pytest.mark.parametrize("name", ["d", "cap.d", "_x", "a.b.c", "N1"])
    def test_valid_names(self, name: str) -> None:
        assert Parameter(name, 1.0).name == name

    def test_min_greater_than_max(self) -> None:
        with pytest.raises(ValueError):
            Parameter("d", 1.0, min=2.0, max=1.0)

    @pytest.mark.parametrize("value", [-0.1, 10.1])
    def test_value_out_of_bounds(self, value: float) -> None:
        with pytest.raises(ValueError):
            Parameter("d", value, min=0.0, max=10.0)

    def test_nan_value(self) -> None:
        with pytest.raises(ValueError):
            Parameter("d", math.nan)

    def test_non_integral_integer_value(self) -> None:
        with pytest.raises(ValueError):
            Parameter("N", 5.5, integer=True)

    def test_integral_float_integer_value(self) -> None:
        assert Parameter("N", 5.0, integer=True).value == 5.0

    def test_values_converted_to_float(self) -> None:
        p = Parameter("N", 3, min=1, max=5)
        assert isinstance(p.value, float)
        assert isinstance(p.min, float)
        assert isinstance(p.max, float)

    def test_defaults(self) -> None:
        p = Parameter("d", 1.0)
        assert not p.vary
        assert p.min == -math.inf
        assert p.max == math.inf
        assert p.expr is None


class TestIsFree:
    def test_vary_without_expr(self) -> None:
        assert Parameter("d", 1.0, vary=True).is_free

    def test_fixed(self) -> None:
        assert not Parameter("d", 1.0).is_free

    def test_vary_with_expr(self) -> None:
        assert not Parameter("d", 1.0, vary=True, expr="2*x").is_free

    def test_expr_only(self) -> None:
        assert not Parameter("d", 1.0, expr="2*x").is_free


class TestEvolve:
    def test_explicit_value_out_of_range(self) -> None:
        p = Parameter("d", 1.0, min=0.0, max=2.0)
        with pytest.raises(ValueError):
            p.evolve(value=3.0)

    def test_explicit_value_in_range(self) -> None:
        p = Parameter("d", 1.0, min=0.0, max=2.0)
        assert p.evolve(value=1.5).value == 1.5

    def test_original_unchanged(self) -> None:
        p = Parameter("d", 1.0, min=0.0, max=2.0)
        p.evolve(value=1.5)
        assert p.value == 1.0

    def test_bounds_clip_value_up(self) -> None:
        p = Parameter("d", 1.0, min=0.0, max=2.0).evolve(min=1.5)
        assert (p.value, p.min, p.max) == (1.5, 1.5, 2.0)

    def test_bounds_clip_value_down(self) -> None:
        p = Parameter("d", 1.0, min=0.0, max=2.0).evolve(max=0.5)
        assert p.value == 0.5

    def test_bounds_keep_value_inside(self) -> None:
        p = Parameter("d", 1.0, min=0.0, max=2.0).evolve(min=0.5, max=1.5)
        assert p.value == 1.0

    def test_both_bounds_clip(self) -> None:
        p = Parameter("d", 1.0).evolve(min=3.0, max=4.0)
        assert p.value == 3.0

    def test_integer_moved_to_lower_bound(self) -> None:
        p = Parameter("N", 5.0, integer=True).evolve(min=50)
        assert p.value == 50.0

    def test_integer_moved_up_into_range(self) -> None:
        p = Parameter("N", 7.0, integer=True).evolve(min=7.5, max=20)
        assert p.value == 8.0

    def test_integer_moved_down_into_range(self) -> None:
        p = Parameter("N", 7.0, integer=True).evolve(min=0, max=6.5)
        assert p.value == 6.0

    def test_integer_inside_unchanged(self) -> None:
        p = Parameter("N", 7.0, integer=True).evolve(min=1, max=20)
        assert p.value == 7.0

    def test_integer_bounds_without_integer_inside(self) -> None:
        with pytest.raises(ValueError):
            Parameter("N", 7.0, integer=True).evolve(min=7.2, max=7.8)

    def test_other_fields(self) -> None:
        p = Parameter("d", 1.0).evolve(vary=True, unit="nm", description="x")
        assert p.vary
        assert p.unit == "nm"
        assert p.description == "x"

    def test_rename(self) -> None:
        assert Parameter("d", 1.0).evolve(name="e").name == "e"

    def test_frozen(self) -> None:
        with pytest.raises(AttributeError):
            Parameter("d", 1.0).value = 2.0  # type: ignore[misc]


class TestRangeAndFree:
    def test_free_returns_range(self) -> None:
        r = free(5.0, 1.0, 10.0)
        assert r == Range(5.0, 1.0, 10.0)
        assert isinstance(r, Range)


class TestResolve:
    TEMPLATE = Parameter("d", 4.0, min=0.0, max=10.0, unit="nm", description="Period")

    def test_float_fixes(self) -> None:
        p = resolve(self.TEMPLATE, 6.0)
        assert p.value == 6.0
        assert not p.vary
        assert p.expr is None
        assert (p.min, p.max) == (0.0, 10.0)
        assert p.unit == "nm"

    def test_int_fixes(self) -> None:
        assert resolve(self.TEMPLATE, 6).value == 6.0

    def test_float_widens_upper_bound(self) -> None:
        p = resolve(self.TEMPLATE, 20.0)
        assert (p.value, p.min, p.max) == (20.0, 0.0, 20.0)

    def test_float_widens_lower_bound(self) -> None:
        p = resolve(self.TEMPLATE, -2.0)
        assert (p.value, p.min, p.max) == (-2.0, -2.0, 10.0)

    def test_float_clears_expr(self) -> None:
        template = self.TEMPLATE.evolve(expr="2*x")
        assert resolve(template, 3.0).expr is None

    def test_range(self) -> None:
        template = self.TEMPLATE.evolve(expr="2*x")
        p = resolve(template, free(5.0, 2.0, 8.0))
        assert p.vary
        assert p.is_free
        assert (p.value, p.min, p.max) == (5.0, 2.0, 8.0)
        assert p.expr is None
        assert p.name == "d"
        assert p.unit == "nm"

    def test_str(self) -> None:
        p = resolve(self.TEMPLATE.evolve(vary=True), "2*gamma")
        assert p.expr == "2*gamma"
        assert not p.vary
        assert not p.is_free

    def test_parameter_replaces_all_but_name(self) -> None:
        spec = Parameter("other", 2.0, vary=True, min=1.0, max=3.0, unit="um", description="y")
        p = resolve(self.TEMPLATE, spec)
        assert p == spec.evolve(name="d")
        assert p.unit == "um"
        assert p.description == "y"

    @pytest.mark.parametrize("spec", [None, [1.0], (1.0, 2.0), {"value": 1.0}])
    def test_wrong_type(self, spec: object) -> None:
        with pytest.raises(TypeError):
            resolve(self.TEMPLATE, spec)  # type: ignore[arg-type]


def make_set() -> ParameterSet:
    return ParameterSet(
        [
            Parameter("d", 4.0, min=0.0, max=10.0, unit="nm"),
            Parameter("N", 30, min=1, max=100, integer=True, vary=True),
            Parameter("gamma", 0.4, min=0.0, max=1.0, vary=True),
        ]
    )


class TestParameterSetBasics:
    def test_duplicate_names(self) -> None:
        with pytest.raises(ValueError):
            ParameterSet([Parameter("d", 1.0), Parameter("d", 2.0)])

    def test_mapping(self) -> None:
        ps = make_set()
        assert len(ps) == 3
        assert list(ps) == ["d", "N", "gamma"]
        assert "d" in ps
        assert "x" not in ps
        assert ps["d"].value == 4.0
        assert ps.get("x") is None
        assert [p.name for p in ps.values()] == ["d", "N", "gamma"]
        assert dict(ps.items())["N"].value == 30.0

    def test_unknown_name(self) -> None:
        with pytest.raises(KeyError):
            make_set()["x"]

    def test_empty(self) -> None:
        assert len(ParameterSet()) == 0

    def test_equality(self) -> None:
        assert make_set() == make_set()
        assert make_set() != make_set().replace("d", value=5.0)
        assert make_set() != "not a set"

    def test_hash(self) -> None:
        assert hash(make_set()) == hash(make_set())
        assert len({make_set(), make_set()}) == 1

    def test_order_matters_for_equality(self) -> None:
        a, b = Parameter("a", 1.0), Parameter("b", 2.0)
        assert ParameterSet([a, b]) != ParameterSet([b, a])

    def test_repr(self) -> None:
        assert repr(make_set()).startswith("ParameterSet(")

    def test_value_dict(self) -> None:
        assert make_set().value_dict() == {"d": 4.0, "N": 30.0, "gamma": 0.4}

    def test_value_dict_does_not_evaluate(self) -> None:
        ps = ParameterSet([Parameter("a", 1.0), Parameter("b", 7.0, expr="2*a")])
        assert ps.value_dict()["b"] == 7.0

    def test_free_names(self) -> None:
        ps = make_set().replace("gamma", expr="0.5*d")
        assert make_set().free_names() == ("N", "gamma")
        assert ps.free_names() == ("N",)


class TestParameterSetImmutability:
    def test_replace_returns_new(self) -> None:
        ps = make_set()
        new = ps.replace("d", value=5.0, vary=True)
        assert new is not ps
        assert new["d"].value == 5.0
        assert new["d"].vary
        assert ps["d"].value == 4.0
        assert not ps["d"].vary
        assert list(new) == list(ps)

    def test_replace_unknown(self) -> None:
        with pytest.raises(KeyError):
            make_set().replace("x", value=1.0)

    def test_replace_invalid(self) -> None:
        with pytest.raises(ValueError):
            make_set().replace("d", value=100.0)

    def test_replace_bounds_clips_integer(self) -> None:
        assert make_set().replace("N", min=50)["N"].value == 50.0

    def test_with_values_returns_new(self) -> None:
        ps = make_set()
        new = ps.with_values({"d": 5.0})
        assert new is not ps
        assert new["d"].value == 5.0
        assert ps["d"].value == 4.0
        assert new["N"] == ps["N"]

    def test_no_mutation_api(self) -> None:
        with pytest.raises(TypeError):
            make_set()["d"] = Parameter("d", 1.0)  # type: ignore[index]


class TestWithValues:
    def test_unknown(self) -> None:
        with pytest.raises(KeyError):
            make_set().with_values({"x": 1.0})

    def test_integer_rounding(self) -> None:
        assert make_set().with_values({"N": 10.4})["N"].value == 10.0
        assert make_set().with_values({"N": 10.6})["N"].value == 11.0

    def test_out_of_bounds(self) -> None:
        with pytest.raises(ValueError):
            make_set().with_values({"d": 11.0})

    def test_integer_out_of_bounds(self) -> None:
        with pytest.raises(ValueError):
            make_set().with_values({"N": 1000})

    def test_bounds_kept(self) -> None:
        new = make_set().with_values({"d": 5.0})
        assert (new["d"].min, new["d"].max) == (0.0, 10.0)

    def test_empty(self) -> None:
        assert make_set().with_values({}) == make_set()


class TestPrefixed:
    def test_names(self) -> None:
        ps = make_set().prefixed("a")
        assert list(ps) == ["a.d", "a.N", "a.gamma"]
        assert ps["a.d"].value == 4.0
        assert ps["a.N"].integer

    def test_expressions_renamed_only_for_set_names(self) -> None:
        ps = ParameterSet([Parameter("d", 1.0), Parameter("e", 2.0, expr="2*d + x")])
        assert ps.prefixed("a")["a.e"].expr == "2*a.d + x"

    def test_expression_function_names_untouched(self) -> None:
        ps = ParameterSet([Parameter("d", 1.0), Parameter("e", 2.0, expr="sqrt(d) + sin(1)")])
        assert ps.prefixed("a")["a.e"].expr == "sqrt(a.d) + sin(1)"

    def test_expression_names_not_partially_matched(self) -> None:
        ps = ParameterSet([Parameter("d", 1.0), Parameter("e", 2.0, expr="d_max + d")])
        assert ps.prefixed("a")["a.e"].expr == "d_max + a.d"

    def test_no_expr_stays_none(self) -> None:
        assert make_set().prefixed("a")["a.d"].expr is None

    def test_original_unchanged(self) -> None:
        ps = make_set()
        ps.prefixed("a")
        assert list(ps) == ["d", "N", "gamma"]

    def test_prefixed_set_evaluates(self) -> None:
        ps = ParameterSet([Parameter("d", 3.0), Parameter("e", 0.0, expr="2*d")])
        assert ps.prefixed("a").evaluate() == {"a.d": 3.0, "a.e": 6.0}


class TestMerge:
    def test_union_in_order(self) -> None:
        a = ParameterSet([Parameter("a", 1.0)])
        b = ParameterSet([Parameter("b", 2.0)])
        c = ParameterSet([Parameter("c", 3.0)])
        merged = a.merge(b, c)
        assert list(merged) == ["a", "b", "c"]
        assert list(a) == ["a"]

    def test_collision(self) -> None:
        a = ParameterSet([Parameter("a", 1.0)])
        with pytest.raises(ValueError):
            a.merge(a)

    def test_no_others(self) -> None:
        assert make_set().merge() == make_set()


class TestLmfit:
    def test_names_and_values(self) -> None:
        ps = ParameterSet([Parameter("cap.d", 2.0, min=0.0, max=5.0), Parameter("N", 3.0)])
        lm = ps.to_lmfit()
        assert isinstance(lm, lmfit.Parameters)
        assert list(lm) == ["cap__d", "N"]
        assert lm["cap__d"].value == 2.0
        assert (lm["cap__d"].min, lm["cap__d"].max) == (0.0, 5.0)

    def test_vary_only_for_free(self) -> None:
        ps = ParameterSet(
            [
                Parameter("a", 1.0, vary=True),
                Parameter("b", 1.0),
                Parameter("c", 2.0, vary=True, expr="2*a"),
            ]
        )
        lm = ps.to_lmfit()
        assert lm["a"].vary
        assert not lm["b"].vary
        assert not lm["c"].vary

    def test_expr_translated(self) -> None:
        ps = ParameterSet(
            [
                Parameter("graded.d_max", 10.0),
                Parameter("cap.d", 1.0, expr="0.5*graded.d_max + 1"),
            ]
        )
        lm = ps.to_lmfit()
        assert lm["cap__d"].expr == "0.5*graded__d_max + 1"
        assert lm["cap__d"].value == pytest.approx(6.0)

    def test_round_trip_unchanged(self) -> None:
        ps = make_set()
        assert ps.update_from_lmfit(ps.to_lmfit()) == ps

    def test_update_from_modified_values(self) -> None:
        ps = make_set().prefixed("blk")
        lm = ps.to_lmfit()
        lm["blk__d"].value = 6.0
        lm["blk__N"].value = 41.4
        new = ps.update_from_lmfit(lm)
        assert new["blk.d"].value == 6.0
        assert new["blk.N"].value == 41.0
        assert ps["blk.d"].value == 4.0

    def test_update_clips_to_bounds(self) -> None:
        ps = make_set()
        lm = ps.to_lmfit()
        lm["d"].min = -math.inf
        lm["d"].max = math.inf
        lm["d"].value = 50.0
        lm["gamma"].min = -math.inf
        lm["gamma"].max = math.inf
        lm["gamma"].value = -3.0
        new = ps.update_from_lmfit(lm)
        assert new["d"].value == 10.0
        assert new["gamma"].value == 0.0

    def test_update_takes_evaluated_expressions(self) -> None:
        ps = ParameterSet(
            [Parameter("a", 1.0, vary=True, max=100.0), Parameter("b", 2.0, expr="2*a")]
        )
        lm = ps.to_lmfit()
        lm["a"].value = 5.0
        new = ps.update_from_lmfit(lm)
        assert new["a"].value == 5.0
        assert new["b"].value == 10.0
        assert new["b"].expr == "2*a"


class TestEvaluate:
    def test_without_expressions(self) -> None:
        assert make_set().evaluate() == make_set().value_dict()

    def test_simple_expression(self) -> None:
        ps = ParameterSet([Parameter("a", 3.0), Parameter("b", 0.0, expr="2*a + 1")])
        assert ps.evaluate() == {"a": 3.0, "b": 7.0}

    def test_dotted_names(self) -> None:
        ps = ParameterSet(
            [
                Parameter("graded.d_max", 10.0),
                Parameter("cap.d", 0.0, expr="graded.d_max / 4"),
            ]
        )
        assert ps.evaluate()["cap.d"] == pytest.approx(2.5)

    def test_expression_follows_new_values(self) -> None:
        ps = ParameterSet([Parameter("a", 3.0), Parameter("b", 0.0, expr="2*a")])
        assert ps.with_values({"a": 4.0}).evaluate()["b"] == 8.0

    def test_chained_expressions(self) -> None:
        ps = ParameterSet(
            [
                Parameter("a", 1.0),
                Parameter("b", 0.0, expr="a + 1"),
                Parameter("c", 0.0, expr="b * 3"),
            ]
        )
        assert ps.evaluate()["c"] == 6.0

    def test_forward_reference(self) -> None:
        ps = ParameterSet([Parameter("b", 0.0, expr="2*a"), Parameter("a", 3.0)])
        assert ps.evaluate() == {"b": 6.0, "a": 3.0}

    def test_unknown_name_in_expression(self) -> None:
        ps = ParameterSet([Parameter("a", 3.0), Parameter("b", 0.0, expr="2*missing")])
        with pytest.raises(ValueError):
            ps.evaluate()

    def test_syntax_error_in_expression(self) -> None:
        ps = ParameterSet([Parameter("a", 3.0), Parameter("b", 0.0, expr="2*(a")])
        with pytest.raises(ValueError):
            ps.evaluate()
