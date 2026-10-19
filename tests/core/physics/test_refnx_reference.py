"""Comparison of the Parratt kernel with refnx (independent Abelès implementation).

The reference files in ``tests/data/refnx`` store the refractive indices, thicknesses and
roughnesses of each stack together with refnx's reflectivity; see their PROVENANCE.md.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from xstack.core.physics.kernel import reflectance

DATA = Path(__file__).parents[2] / "data" / "refnx"
CASES = sorted(p.stem for p in DATA.glob("*.json"))


def test_reference_files_exist() -> None:
    assert len(CASES) >= 6


@pytest.mark.parametrize("name", CASES)
def test_matches_refnx(name: str) -> None:
    data = json.loads((DATA / f"{name}.json").read_text())
    assert data["provenance"]["source"].startswith("refnx")
    n = np.array(data["n_real"]) + 1j * np.array(data["n_imag"])
    theta = np.radians(data["theta_deg"])
    points = theta.size
    result = reflectance(
        np.tile(n, (points, 1)),
        data["thickness_nm"],
        data["roughness_nm"],
        theta,
        np.full(points, data["wavelength_nm"]),
        data["polarization"],
        "nevot_croce",
    )
    expected = np.array(data["reflectivity"])
    np.testing.assert_allclose(result, expected, rtol=1e-8, atol=1e-15)
