"""Performance benchmarks of the reflectivity kernel.

Skipped by default. Run with ``pytest tests/physics/test_benchmark.py -o addopts=""``.
"""

from typing import Any

import numpy as np
import pytest

from xstack.physics.kernel import parratt


def periodic_inputs(points: int, bilayers: int) -> dict[str, Any]:
    pair = [1 - 3.0e-5 + 2.5e-6j, 1 - 7.6e-6 + 1.8e-7j]
    n_row = np.array([1.0, *(pair * bilayers), 1 - 7.6e-6 + 1.8e-7j])
    return {
        "n": np.broadcast_to(n_row, (points, n_row.size)),
        "thickness": np.tile([1.6, 2.4], bilayers),
        "roughness": np.full(n_row.size - 1, 0.3),
        "theta": np.linspace(1e-3, 0.05, points),
        "wavelength": 0.154,
    }


@pytest.mark.parametrize(
    ("points", "bilayers"),
    [(1000, 300), (2000, 1500)],
    ids=["1000pts-300bl", "2000pts-1500bl"],
)
def test_parratt_speed(benchmark: Any, points: int, bilayers: int) -> None:
    inputs = periodic_inputs(points, bilayers)
    result = benchmark(parratt, **inputs)
    assert np.all(np.isfinite(result.reflectance))
