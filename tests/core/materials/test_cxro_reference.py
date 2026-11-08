"""Comparison of computed optical constants with the CXRO index-of-refraction calculator."""

from pathlib import Path

import numpy as np
import pytest

from grazing.core.materials import Material, delta_beta

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "cxro"
FILES = sorted(DATA_DIR.glob("*.dat"))

# Residual differences come from atomic weights (xraylib vs CXRO, up to ~4e-4 for W) and, at
# high energies, from the Compton cross sections.
RTOL_DELTA = 1e-3
RTOL_BETA = 5e-3


def load_reference(path: Path) -> tuple[Material, np.ndarray, np.ndarray, np.ndarray]:
    header = path.read_text().splitlines()[0].split()
    formula = header[0]
    density = float(header[1].split("=")[1])
    data = np.loadtxt(path, skiprows=2)
    return Material(formula, formula, density), data[:, 0], data[:, 1], data[:, 2]


def test_reference_files_present() -> None:
    assert len(FILES) >= 8


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.stem)
def test_delta_beta_matches_cxro(path: Path) -> None:
    material, energy, delta, beta = load_reference(path)
    result = delta_beta(material, energy, provider="henke")
    np.testing.assert_allclose(result.real, delta, rtol=RTOL_DELTA)
    np.testing.assert_allclose(result.imag, beta, rtol=RTOL_BETA)


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.stem)
def test_default_provider_matches_cxro_below_30kev(path: Path) -> None:
    material, energy, delta, beta = load_reference(path)
    result = delta_beta(material, energy)
    np.testing.assert_allclose(result.real, delta, rtol=RTOL_DELTA)
    np.testing.assert_allclose(result.imag, beta, rtol=RTOL_BETA)
