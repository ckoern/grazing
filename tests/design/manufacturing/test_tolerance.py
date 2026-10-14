"""Tests for the thickness tolerance analysis."""

from dataclasses import fields

import numpy as np
import pytest

from tests.design.optim.test_problem import make_problem
from xstack.core.materials import VACUUM, Material, delta_beta, get
from xstack.core.structure import Layer, Stack
from xstack.design.manufacturing import (
    ErrorModel,
    MaxTotalThickness,
    mixed_material,
    perturb_stack,
    perturb_thicknesses,
    thickness_tolerance,
    tolerance_analysis,
)
from xstack.design.optim import CancellationToken, ProgressEvent

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


ENERGY = 8048.0


def make_rough_stack(pairs: int, roughness: float) -> Stack:
    layers = []
    for i in range(pairs):
        layers.append(Layer(W, 2.0 + 0.001 * i, roughness, f"w{i}"))
        layers.append(Layer(SI, 3.0 + 0.001 * i, roughness, f"s{i}"))
    return Stack(layers, SI, roughness)


class TestErrorModel:
    def test_defaults_are_zero(self) -> None:
        assert ErrorModel().is_zero

    @pytest.mark.parametrize("name", [f.name for f in fields(ErrorModel)])
    def test_any_nonzero_field_is_not_zero(self, name: str) -> None:
        assert not ErrorModel(**{name: 0.1}).is_zero

    @pytest.mark.parametrize("name", [f.name for f in fields(ErrorModel)])
    def test_negative_values_raise(self, name: str) -> None:
        with pytest.raises(ValueError):
            ErrorModel(**{name: -0.1})


class TestMixedMaterial:
    def test_name_and_density(self) -> None:
        mix = mixed_material(W, SI)
        assert mix.name == "W/Si mix"
        assert mix.density == pytest.approx(0.5 * (W.density + SI.density))

    def test_delta_beta_is_mean_of_components(self) -> None:
        mix = mixed_material(W, SI)
        mixed = delta_beta(mix, ENERGY, provider="henke")
        expected = 0.5 * (
            delta_beta(W, ENERGY, provider="henke") + delta_beta(SI, ENERGY, provider="henke")
        )
        np.testing.assert_allclose(mixed, expected, rtol=1e-6)

    def test_vacuum_raises(self) -> None:
        with pytest.raises(ValueError, match="vacuum"):
            mixed_material(VACUUM, W)
        with pytest.raises(ValueError, match="vacuum"):
            mixed_material(W, VACUUM)


class TestPerturbStack:
    def test_zero_errors_identical(self) -> None:
        stack = make_stack(4)
        out = perturb_stack(stack, np.random.default_rng(0), ErrorModel())
        assert out.num_layers == stack.num_layers
        np.testing.assert_array_equal(out.thicknesses(), stack.thicknesses())
        np.testing.assert_array_equal(out.roughnesses(), stack.roughnesses())
        assert out.substrate_roughness == stack.substrate_roughness

    def test_thickness_drift_affects_top_layer_only(self) -> None:
        stack = make_stack(2)
        n = stack.num_layers
        errors = ErrorModel(thickness_drift=0.05)
        changes = []
        for seed in range(400):
            out = perturb_stack(stack, np.random.default_rng(seed), errors)
            rel = out.thicknesses() / stack.thicknesses() - 1.0
            assert rel[-1] == pytest.approx(0.0, abs=1e-12)
            # The drift grows linearly from the substrate (last layer) to the top layer.
            np.testing.assert_allclose(rel, rel[0] * np.arange(n)[::-1] / (n - 1), atol=1e-12)
            changes.append(rel[0])
        changes_arr = np.array(changes)
        assert changes_arr.std() == pytest.approx(0.05, rel=0.15)
        assert changes_arr.mean() == pytest.approx(0.0, abs=0.01)

    def test_systematic_roughness_shifts_all_interfaces_equally(self) -> None:
        stack = make_rough_stack(3, 1.0)
        out = perturb_stack(stack, np.random.default_rng(0), ErrorModel(roughness_systematic=0.2))
        shifts = np.concatenate([out.roughnesses() - 1.0, [out.substrate_roughness - 1.0]])
        assert abs(shifts[0]) > 0
        np.testing.assert_allclose(shifts, shifts[0], atol=1e-12)

    def test_random_roughness_shifts_differ_per_interface(self) -> None:
        stack = make_rough_stack(200, 1.0)
        out = perturb_stack(stack, np.random.default_rng(1), ErrorModel(roughness_random=0.1))
        shifts = np.concatenate([out.roughnesses() - 1.0, [out.substrate_roughness - 1.0]])
        assert len(np.unique(shifts)) > 1
        assert shifts.std() == pytest.approx(0.1, rel=0.15)

    def test_roughness_never_negative(self) -> None:
        stack = make_rough_stack(50, 0.0)
        out = perturb_stack(stack, np.random.default_rng(2), ErrorModel(roughness_random=1.0))
        assert (out.roughnesses() >= 0).all()
        assert out.substrate_roughness >= 0
        assert (out.roughnesses() == 0).any()

    def test_interlayers_insert_mixed_layers(self) -> None:
        pairs = 5
        stack = make_stack(pairs)
        errors = ErrorModel(interlayer_thickness=0.6, interlayer_random=0.0)
        out = perturb_stack(stack, np.random.default_rng(3), errors)
        assert out.num_layers == 2 * pairs + (2 * pairs - 1)
        inserted = [
            (i, layer) for i, layer in enumerate(out.layers) if layer.material.name.endswith(" mix")
        ]
        assert len(inserted) == 2 * pairs - 1
        for _, layer in inserted:
            assert layer.material.name in {"W/Si mix", "Si/W mix"}
            assert layer.thickness == pytest.approx(0.6, abs=1e-12)
        assert out.total_thickness == pytest.approx(stack.total_thickness, rel=1e-12)

        # Original layers in order, with their loss of thickness per mixed interface.
        originals = [layer for layer in out.layers if not layer.material.name.endswith(" mix")]
        assert len(originals) == stack.num_layers
        last = stack.num_layers - 1
        for k, (orig, new) in enumerate(zip(stack.layers, originals, strict=True)):
            touching = (1 if k > 0 else 0) + (1 if k < last else 0)
            assert orig.thickness - new.thickness == pytest.approx(0.3 * touching, abs=1e-12)

    def test_no_mix_between_equal_materials(self) -> None:
        stack = Stack([Layer(W, 2.0), Layer(W, 2.0), Layer(SI, 3.0)], SI)
        errors = ErrorModel(interlayer_thickness=0.6)
        out = perturb_stack(stack, np.random.default_rng(0), errors)
        names = [layer.material.name for layer in out.layers]
        assert names == ["W", "W", "W/Si mix", "Si"]
        assert out.total_thickness == pytest.approx(stack.total_thickness, rel=1e-12)

    def test_mixing_capped_by_thin_layer(self) -> None:
        stack = Stack([Layer(W, 2.0), Layer(SI, 0.2), Layer(W, 2.0)], SI)
        errors = ErrorModel(interlayer_thickness=0.6)
        out = perturb_stack(stack, np.random.default_rng(0), errors)
        inserted = [layer for layer in out.layers if layer.material.name.endswith(" mix")]
        assert inserted
        for layer in inserted:
            assert layer.thickness == pytest.approx(0.2)
        assert out.total_thickness == pytest.approx(stack.total_thickness, rel=1e-12)

    def test_mixture_cache_filled_and_reused(self) -> None:
        stack = make_stack(3)
        errors = ErrorModel(interlayer_thickness=0.6)
        cache: dict[tuple[str, str], Material] = {}
        first = perturb_stack(stack, np.random.default_rng(0), errors, cache)
        assert set(cache) == {("W", "Si"), ("Si", "W")}
        second = perturb_stack(stack, np.random.default_rng(1), errors, cache)
        mix_first = {layer.material.name: layer.material for layer in first.layers}
        mix_second = {layer.material.name: layer.material for layer in second.layers}
        assert mix_first["W/Si mix"] is cache[("W", "Si")]
        assert mix_second["W/Si mix"] is cache[("W", "Si")]
        assert mix_second["Si/W mix"] is cache[("Si", "W")]


class TestToleranceAnalysis:
    def test_sample_count_and_nominal(self) -> None:
        problem = make_problem(n=10, points=20)
        result = tolerance_analysis(problem, samples=6, seed=0)
        assert len(result.samples) == 6
        assert result.merits.shape == (6,)
        assert result.objectives.shape == (6,)
        assert result.feasible.shape == (6,)
        assert result.nominal.merit == pytest.approx(problem.evaluate().merit)

    def test_errors_stored_in_result(self) -> None:
        problem = make_problem(n=10, points=20)
        model = ErrorModel(thickness_random=0.02, roughness_random=0.05)
        result = tolerance_analysis(problem, errors=model, samples=2, seed=0)
        assert result.errors is model

    def test_samples_carry_reflectivity(self) -> None:
        problem = make_problem(n=10, points=20)
        result = tolerance_analysis(problem, samples=3, seed=0)
        for sample in result.samples:
            assert len(sample.results) == len(problem.targets.targets)

    def test_zero_error_all_nominal(self) -> None:
        problem = make_problem(n=10, points=20)
        result = tolerance_analysis(problem, errors=ErrorModel(), samples=3, seed=0)
        np.testing.assert_allclose(result.merits, result.nominal.merit)

    def test_progress_events(self) -> None:
        problem = make_problem(n=10, points=20)
        events: list[ProgressEvent] = []
        tolerance_analysis(problem, samples=4, seed=0, events=events.append)
        assert len(events) == 4
        assert all(event.total == 4 for event in events)
        assert all(event.best_values == {} for event in events)
        assert [event.evaluations for event in events] == [1, 2, 3, 4]

    def test_cancellation_stops_early(self) -> None:
        problem = make_problem(n=10, points=20)
        token = CancellationToken()
        seen: list[ProgressEvent] = []

        def stop_after_three(event: ProgressEvent) -> None:
            seen.append(event)
            if len(seen) == 3:
                token.cancel()

        result = tolerance_analysis(
            problem, samples=10, seed=0, cancel=token, events=stop_after_three
        )
        assert len(result.samples) == 3

    def test_unbuildable_nominal_raises(self) -> None:
        problem = make_problem(n=10, points=20)
        with pytest.raises(ValueError, match="cannot be built"):
            tolerance_analysis(problem, {"d": 1e6}, samples=2)

    def test_thickness_tolerance_matches_error_model(self) -> None:
        problem = make_problem(n=10, points=20)
        via_wrapper = thickness_tolerance(
            problem, random_error=0.05, systematic_error=0.03, samples=5, seed=4
        )
        via_model = tolerance_analysis(
            problem,
            errors=ErrorModel(thickness_random=0.05, thickness_systematic=0.03),
            samples=5,
            seed=4,
        )
        np.testing.assert_array_equal(via_wrapper.merits, via_model.merits)
        assert via_wrapper.errors == via_model.errors


def test_constraints_use_deposited_stack_with_mixing() -> None:
    from xstack.design.manufacturing import ErrorModel, tolerance_analysis
    from xstack.design.project.templates import new_state

    problem = new_state("periodic").problem().evolve(points=20)
    errors = ErrorModel(interlayer_thickness=0.5)
    result = tolerance_analysis(problem, errors=errors, samples=3, seed=0)
    # Mixed layers (0.5 nm) are thinner than the minimum layer thickness (0.8 nm) of the
    # template, but they are not deposited, so the samples stay feasible.
    assert result.feasible.all()
    nominal_layers = result.nominal.stack.num_layers  # type: ignore[union-attr]
    assert all(s.stack.num_layers > nominal_layers for s in result.samples)  # type: ignore[union-attr]
