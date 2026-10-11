"""Tests for `DepositionProcess`."""

import dataclasses

import numpy as np
import pytest

from xstack import materials
from xstack.manufacturing.process import DepositionProcess, DepositionStep
from xstack.structure import Layer, Stack

W = materials.get("W")
SI = materials.get("Si")


def example_stack() -> Stack:
    return Stack([Layer(W, 2.0), Layer(SI, 3.0)] * 10, SI)


def example_process() -> DepositionProcess:
    return DepositionProcess(
        {"W": 0.1, "Si": 0.2},
        overhead_per_layer=10.0,
        overhead_per_material={"W": 5.0},
        fixed_overhead=100.0,
    )


class TestValidation:
    @pytest.mark.parametrize("rate", [0.0, -1.0])
    def test_non_positive_rate(self, rate: float) -> None:
        with pytest.raises(ValueError):
            DepositionProcess({"W": rate})

    def test_negative_layer_overhead(self) -> None:
        with pytest.raises(ValueError):
            DepositionProcess({"W": 1.0}, overhead_per_layer=-1.0)

    def test_negative_material_overhead(self) -> None:
        with pytest.raises(ValueError):
            DepositionProcess({"W": 1.0}, overhead_per_material={"W": -0.1})

    def test_negative_fixed_overhead(self) -> None:
        with pytest.raises(ValueError):
            DepositionProcess({"W": 1.0}, fixed_overhead=-5.0)

    def test_zero_overheads_allowed(self) -> None:
        DepositionProcess({"W": 1.0}, 0.0, {"W": 0.0}, 0.0)


class TestRate:
    def test_rate(self) -> None:
        assert example_process().rate("W") == 0.1
        assert example_process().rate("Si") == 0.2

    def test_missing_material(self) -> None:
        with pytest.raises(KeyError, match="B4C"):
            example_process().rate("B4C")

    def test_missing_material_in_time(self) -> None:
        stack = Stack([Layer(materials.get("B4C"), 1.0)], SI)
        with pytest.raises(KeyError, match="B4C"):
            example_process().time(stack)


class TestLayerTimes:
    def test_top_first_order(self) -> None:
        process = DepositionProcess({"W": 0.1, "Si": 0.2})
        stack = Stack([Layer(W, 1.0), Layer(SI, 1.0)], SI)
        np.testing.assert_allclose(process.layer_times(stack), [10.0, 5.0])

    def test_zero_thickness_costs_nothing(self) -> None:
        process = DepositionProcess({"W": 0.1, "Si": 0.2}, overhead_per_layer=10.0)
        stack = Stack([Layer(W, 1.0), Layer(SI, 0.0), Layer(W, 2.0)], SI)
        np.testing.assert_allclose(process.layer_times(stack), [20.0, 0.0, 30.0])

    def test_material_overhead_only_for_that_material(self) -> None:
        process = DepositionProcess({"W": 1.0, "Si": 1.0}, overhead_per_material={"W": 7.0})
        stack = Stack([Layer(W, 2.0), Layer(SI, 2.0)], SI)
        np.testing.assert_allclose(process.layer_times(stack), [9.0, 2.0])

    def test_empty_stack(self) -> None:
        assert example_process().layer_times(Stack([], SI)).shape == (0,)


class TestTime:
    def test_hand_computed(self) -> None:
        # W: 2/0.1 + 10 + 5 = 35 s, Si: 3/0.2 + 10 = 25 s, per bilayer 60 s.
        assert example_process().time(example_stack()) == pytest.approx(100.0 + 10 * 60.0)

    def test_docstring_example(self) -> None:
        process = DepositionProcess({"W": 0.1, "Si": 0.2}, overhead_per_layer=10.0)
        assert process.time(example_stack()) == pytest.approx(550.0)

    def test_empty_stack_is_fixed_overhead(self) -> None:
        assert example_process().time(Stack([], SI)) == 100.0

    def test_zero_thickness_layers_skip_overheads(self) -> None:
        stack = Stack([Layer(W, 0.0), Layer(SI, 0.0)], SI)
        assert example_process().time(stack) == 100.0


class TestSchedule:
    def test_deposition_order(self) -> None:
        stack = Stack([Layer(W, 2.0, tag="top"), Layer(SI, 3.0, tag="bottom")], SI)
        steps = example_process().schedule(stack)
        assert [s.layer_index for s in steps] == [1, 0]
        assert [s.step for s in steps] == [1, 2]
        assert [s.tag for s in steps] == ["bottom", "top"]
        assert [s.material for s in steps] == ["Si", "W"]
        assert [s.thickness for s in steps] == [3.0, 2.0]

    def test_zero_thickness_omitted(self) -> None:
        stack = Stack([Layer(W, 2.0), Layer(SI, 0.0), Layer(W, 1.0)], SI)
        steps = example_process().schedule(stack)
        assert [s.layer_index for s in steps] == [2, 0]
        assert [s.step for s in steps] == [1, 2]

    def test_start_times_cumulative(self) -> None:
        stack = Stack([Layer(W, 2.0), Layer(SI, 3.0)], SI)
        first, second = example_process().schedule(stack)
        assert first.start_time == 100.0
        assert first.duration == pytest.approx(25.0)
        assert second.start_time == pytest.approx(125.0)
        assert second.duration == pytest.approx(35.0)

    def test_consistent_with_time(self) -> None:
        stack = example_stack()
        process = example_process()
        steps = process.schedule(stack)
        assert len(steps) == 20
        assert steps[-1].start_time + steps[-1].duration == pytest.approx(process.time(stack))
        assert steps[0].layer_index == 19
        assert steps[-1].layer_index == 0

    def test_empty(self) -> None:
        assert example_process().schedule(Stack([], SI)) == []

    def test_missing_rate(self) -> None:
        stack = Stack([Layer(materials.get("B4C"), 1.0)], SI)
        with pytest.raises(KeyError, match="B4C"):
            example_process().schedule(stack)


class TestImmutability:
    def test_frozen(self) -> None:
        process = example_process()
        with pytest.raises(dataclasses.FrozenInstanceError):
            process.fixed_overhead = 1.0  # type: ignore[misc]

    def test_hashable_and_equal(self) -> None:
        assert hash(example_process()) == hash(example_process())
        assert example_process() == example_process()
        assert example_process() != DepositionProcess({"W": 0.1})

    def test_step_frozen(self) -> None:
        step = DepositionStep(1, 0, "W", 1.0, 2.0, 0.0, "")
        with pytest.raises(dataclasses.FrozenInstanceError):
            step.step = 2  # type: ignore[misc]
