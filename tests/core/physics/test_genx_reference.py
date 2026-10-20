"""Comparison of the Parratt kernel with GenX (independent Parratt implementation).

The reference files in ``tests/data/genx`` store the refractive indices, thicknesses and
roughnesses of each stack together with GenX's reflectivity; see their PROVENANCE.md.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from xstack.core.physics.kernel import reflectance

DATA = Path(__file__).parents[2] / "data" / "genx"
CASES = sorted(p.stem for p in DATA.glob("*.json") if p.stem != "footprint")


def load(name: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((DATA / f"{name}.json").read_text())
    assert data["provenance"]["source"].startswith("genx")
    return data


def test_reference_files_exist() -> None:
    assert len(CASES) >= 8
    assert {"wsi_energy", "wb4c_graded_energy"} <= set(CASES)


@pytest.mark.parametrize("name", CASES)
def test_matches_genx(name: str) -> None:
    data = load(name)
    distinct = np.array(data["n_real"]) + 1j * np.array(data["n_imag"])
    n = distinct[:, data["medium"]]
    theta = np.radians(data["theta_deg"])
    wavelength = np.array(data["wavelength_nm"])
    points = theta.size
    result = reflectance(
        np.broadcast_to(n, (points, n.shape[1])),
        data["thickness_nm"],
        data["roughness_nm"],
        theta,
        np.broadcast_to(wavelength, (points,)),
        data["polarization"],
        "nevot_croce",
    )
    expected = np.array(data["reflectivity"])
    np.testing.assert_allclose(result, expected, rtol=1e-8, atol=1e-15)
