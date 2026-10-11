"""Tests for `Layer` and `Stack`."""

import dataclasses

import numpy as np
import pytest

from xstack import materials
from xstack.materials import VACUUM, MaterialLibrary
from xstack.structure.stack import Layer, Stack

SI = materials.get("Si")
W = materials.get("W")
B4C = materials.get("B4C")


def example_stack() -> Stack:
    layers = [Layer(W, 2.0, 0.1), Layer(SI, 3.0, 0.2), Layer(W, 1.5, 0.3)]
    return Stack(layers, B4C, substrate_roughness=0.4)


def test_layer_defaults() -> None:
    layer = Layer(W, 2.0)
    assert layer.roughness == 0.0
    assert layer.tag == ""


def test_layer_accepts_zero() -> None:
    Layer(W, 0.0, 0.0)


def test_layer_negative_thickness() -> None:
    with pytest.raises(ValueError, match="thickness"):
        Layer(W, -1.0)


def test_layer_negative_roughness() -> None:
    with pytest.raises(ValueError, match="roughness"):
        Layer(W, 1.0, -0.1)


def test_layer_is_frozen() -> None:
    layer = Layer(W, 1.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        layer.thickness = 2.0  # type: ignore[misc]


def test_layers_stored_as_tuple() -> None:
    stack = example_stack()
    assert isinstance(stack.layers, tuple)
    assert len(stack.layers) == 3
    assert isinstance(Stack(iter([Layer(W, 1.0)]), SI).layers, tuple)


def test_counts_and_total_thickness() -> None:
    stack = example_stack()
    assert stack.num_layers == 3
    assert stack.total_thickness == pytest.approx(6.5)
    empty = Stack([], SI)
    assert empty.num_layers == 0
    assert empty.total_thickness == 0.0


def test_media_order() -> None:
    stack = example_stack()
    assert stack.media == (VACUUM, W, SI, W, B4C)
    assert stack.ambient is VACUUM
    custom = Stack([Layer(W, 1.0)], SI, ambient=B4C)
    assert custom.media == (B4C, W, SI)


def test_thicknesses_and_roughnesses() -> None:
    stack = example_stack()
    np.testing.assert_array_equal(stack.thicknesses(), [2.0, 3.0, 1.5])
    np.testing.assert_array_equal(stack.roughnesses(), [0.1, 0.2, 0.3, 0.4])
    assert stack.thicknesses().dtype == np.float64
    assert stack.roughnesses().dtype == np.float64


def test_empty_stack_arrays() -> None:
    stack = Stack([], SI, substrate_roughness=0.5)
    assert stack.thicknesses().shape == (0,)
    np.testing.assert_array_equal(stack.roughnesses(), [0.5])


def test_negative_substrate_roughness() -> None:
    with pytest.raises(ValueError, match="roughness"):
        Stack([], SI, substrate_roughness=-0.1)


def test_refractive_indices_shape_and_vacuum() -> None:
    stack = example_stack()
    energy = np.array([5000.0, 8048.0, 12000.0])
    n = stack.refractive_indices(energy)
    assert n.shape == (3, 5)
    assert n.dtype == np.complex128
    np.testing.assert_array_equal(n[:, 0], 1.0)
    assert np.all(n[:, 1:].real < 1.0)
    assert np.all(n[:, 1:].imag > 0.0)


def test_identical_materials_have_identical_columns() -> None:
    n = example_stack().refractive_indices(np.array([5000.0, 8048.0]))
    np.testing.assert_array_equal(n[:, 1], n[:, 3])
    assert not np.array_equal(n[:, 1], n[:, 2])


def test_refractive_indices_scalar_energy() -> None:
    assert example_stack().refractive_indices(8048.0).shape == (1, 5)


def test_refractive_indices_match_library() -> None:
    stack = example_stack()
    energy = np.array([6000.0, 9000.0])
    n = stack.refractive_indices(energy)
    np.testing.assert_allclose(n[:, 2], materials.refractive_index(SI, energy))
    np.testing.assert_allclose(n[:, 4], materials.refractive_index(B4C, energy))


def test_refractive_indices_custom_library() -> None:
    library = MaterialLibrary([SI.with_density(1.0)])
    energy = np.array([8048.0])
    default = Stack([], SI).refractive_indices(energy)
    custom = Stack([], SI.with_density(1.0)).refractive_indices(energy, library)
    assert custom[0, 1].real > default[0, 1].real


def test_two_dimensional_energy_rejected() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        example_stack().refractive_indices(np.ones((2, 2)) * 8048.0)


def test_stack_is_frozen_and_hashable() -> None:
    stack = example_stack()
    with pytest.raises(dataclasses.FrozenInstanceError):
        stack.substrate = SI  # type: ignore[misc]
    assert hash(stack) == hash(example_stack())
    assert stack == example_stack()
    assert len({stack, example_stack()}) == 1
    assert stack != Stack([], SI)
