"""Comparison of the footprint factor with GenX (see tests/data/genx/PROVENANCE.md)."""

import json
from pathlib import Path

import numpy as np

from xstack.fit.instrument import footprint_factor

DATA = Path(__file__).parents[1] / "data" / "genx" / "footprint.json"
FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))


def test_footprint_matches_genx() -> None:
    data = json.loads(DATA.read_text())
    assert data["provenance"]["source"] == "genx.models.lib.instrument.GaussIntensity"
    # The factor depends only on the ratio of the lengths, so both stay in mm.
    factor = footprint_factor(
        np.radians(data["theta_deg"]),
        data["beam_sigma_mm"] * FWHM_PER_SIGMA,
        data["sample_length_mm"],
    )
    np.testing.assert_allclose(factor, data["factor"], rtol=1e-12)
