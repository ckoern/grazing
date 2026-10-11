"""Tests for the thickness tolerance analysis."""

import numpy as np
import pytest

from tests.optim.test_problem import make_problem
from xstack.manufacturing import MaxTotalThickness, perturb_thicknesses, thickness_tolerance
from xstack.materials import get
from xstack.optim import CancellationToken
from xstack.structure import Layer, Stack

W = get("W")
SI = get("Si")


def make_stack(pairs: int = 200) -> Stack:
    layers = []
    for i in range(pairs):
        layers.append(Layer(W, 2.0 + 0.001 * i, 0.3, f"w{i}"))
        layers.append(Layer(SI, 3.0 + 0.001 * i, 0.4, f"s{i}"))
    return Stack(layers, SI, 0.25)


class TestPerturbThicknesses:
    def test_zero_errors_identical(self) -> None:
        stack = make_stack(5)
        out = perturb_thicknesses(stack, np.random.default_rng(0), 0.0, 0.0)
        np.testing.assert_allclose(out.thicknesses(), stack.thicknesses())

    def test_random_error_statistics(self) -> None:
        stack = make_stack()
        out = perturb_thicknesses(stack, np.random.default_rng(1), 0.05, 0.0)
        ratio = out.thicknesses() / stack.thicknesses()
        assert ratio.mean() == pytest.approx(1.0, abs=0.01)
        assert ratio.std() == pytest.approx(0.05, rel=0.15)

    def test_systematic_error_same_factor_per_material(self) -> None:
        stack = make_stack(20)
        out = perturb_thicknesses(stack, np.random.default_rng(2), 0.0, 0.1)
        ratio = out.thicknesses() / stack.thicknesses()
        w_ratio, si_ratio = ratio[0::2], ratio[1::2]
        np.testing.assert_allclose(w_ratio, w_ratio[0])
        np.testing.assert_allclose(si_ratio, si_ratio[0])
        assert w_ratio[0] != pytest.approx(si_ratio[0])
        assert w_ratio[0] != pytest.approx(1.0)

    def test_systematic_error_statistics(self) -> None:
        stack = make_stack(2)
        rng = np.random.default_rng(3)
        factors = [
            perturb_thicknesses(stack, rng, 0.0, 0.05).thicknesses()[0] / stack.thicknesses()[0]
            for _ in range(1000)
        ]
        assert np.mean(factors) == pytest.approx(1.0, abs=0.01)
        assert np.std(factors) == pytest.approx(0.05, rel=0.15)

    def test_negative_thicknesses_clipped(self) -> None:
        stack = make_stack(50)
        out = perturb_thicknesses(stack, np.random.default_rng(4), 3.0, 0.0)
        thick = out.thicknesses()
        assert (thick >= 0).all()
        assert (thick == 0).any()

    def test_substrate_ambient_and_other_fields_preserved(self) -> None:
        stack = make_stack(5)
        out = perturb_thicknesses(stack, np.random.default_rng(5), 0.1, 0.1)
        assert out.substrate == stack.substrate
        assert out.substrate_roughness == stack.substrate_roughness
        assert out.ambient == stack.ambient
        assert out.num_layers == stack.num_layers
        assert [layer.material for layer in out.layers] == [
            layer.material for layer in stack.layers
        ]
        assert [layer.roughness for layer in out.layers] == [
            layer.roughness for layer in stack.layers
        ]
        assert [layer.tag for layer in out.layers] == [layer.tag for layer in stack.layers]

    def test_reproducible_with_seed(self) -> None:
        stack = make_stack(10)
        a = perturb_thicknesses(stack, np.random.default_rng(7), 0.05, 0.02)
        b = perturb_thicknesses(stack, np.random.default_rng(7), 0.05, 0.02)
        c = perturb_thicknesses(stack, np.random.default_rng(8), 0.05, 0.02)
        np.testing.assert_array_equal(a.thicknesses(), b.thicknesses())
        assert not np.array_equal(a.thicknesses(), c.thicknesses())

    def test_input_not_modified(self) -> None:
        stack = make_stack(3)
        before = stack.thicknesses().copy()
        perturb_thicknesses(stack, np.random.default_rng(0), 0.2, 0.2)
        np.testing.assert_array_equal(stack.thicknesses(), before)


class TestThicknessTolerance:
    def test_sample_count_and_nominal(self) -> None:
        problem = make_problem(n=10)
        result = thickness_tolerance(problem, samples=7, seed=0)
        assert len(result.samples) == 7
        assert result.merits.shape == (7,)
        assert result.objectives.shape == (7,)
        assert result.feasible.shape == (7,)
        assert result.feasible.dtype == bool
        assert result.nominal.merit == pytest.approx(problem.evaluate().merit)

    def test_nominal_uses_given_values(self) -> None:
        problem = make_problem(n=10)
        values = {**problem.parameters.value_dict(), "d": 3.8}
        result = thickness_tolerance(problem, values, samples=2, seed=0)
        assert result.nominal.merit == pytest.approx(problem.evaluate(values).merit)
        assert result.nominal.values["d"] == pytest.approx(3.8)

    def test_zero_error_all_nominal(self) -> None:
        problem = make_problem(n=10)
        result = thickness_tolerance(problem, random_error=0.0, samples=4, seed=0)
        np.testing.assert_allclose(result.merits, result.nominal.merit)
        assert result.percentile(50) == pytest.approx(result.nominal.merit)

    def test_larger_error_larger_spread(self) -> None:
        problem = make_problem(n=15)
        small = thickness_tolerance(problem, random_error=0.01, samples=30, seed=1)
        large = thickness_tolerance(problem, random_error=0.1, samples=30, seed=1)
        assert large.merits.std() > small.merits.std()

    def test_systematic_error_changes_merit(self) -> None:
        problem = make_problem(n=15)
        result = thickness_tolerance(
            problem, random_error=0.0, systematic_error=0.05, samples=10, seed=1
        )
        assert result.merits.std() > 0

    def test_percentile(self) -> None:
        problem = make_problem(n=10)
        result = thickness_tolerance(problem, random_error=0.05, samples=20, seed=2)
        assert result.percentile(0) == pytest.approx(result.merits.min())
        assert result.percentile(100) == pytest.approx(result.merits.max())
        assert result.percentile(0) <= result.percentile(50) <= result.percentile(90)

    def test_seed_reproducible(self) -> None:
        problem = make_problem(n=10)
        a = thickness_tolerance(problem, samples=5, seed=3)
        b = thickness_tolerance(problem, samples=5, seed=3)
        np.testing.assert_array_equal(a.merits, b.merits)

    def test_feasible_flags(self) -> None:
        problem = make_problem(n=10)
        nominal = problem.evaluate()
        assert nominal.stack is not None
        limit = nominal.stack.total_thickness
        constrained = problem.evolve(constraints=(MaxTotalThickness(limit),))
        result = thickness_tolerance(constrained, random_error=0.1, samples=40, seed=0)
        assert result.nominal.feasible
        assert result.feasible.any()
        assert not result.feasible.all()
        np.testing.assert_array_equal(result.feasible, [e.feasible for e in result.samples])

    def test_unbuildable_nominal_raises(self) -> None:
        problem = make_problem(n=10)
        with pytest.raises(ValueError, match="cannot be built"):
            thickness_tolerance(problem, {"d": 1e6}, samples=2)

    def test_cancellation_stops_early(self) -> None:
        problem = make_problem(n=10)
        token = CancellationToken()
        token.cancel()
        result = thickness_tolerance(problem, samples=10, cancel=token)
        assert len(result.samples) == 0
        assert result.merits.shape == (0,)
