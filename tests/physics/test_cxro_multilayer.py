"""Tier 6: comparison with the CXRO multilayer reflectivity calculator.

Remaining differences come from the optical constants as processed by CXRO (atomic weights,
interpolation), not from the kernel, which tiers 1–3 verify to machine precision. They show up
as small shifts of steep features. CXRO's "interdiffusion" parameter is not identical to a
Névot–Croce factor on every interface, so the comparison with roughness is looser.
"""

from pathlib import Path

import numpy as np
import pytest

from xstack.materials import Material, MaterialLibrary
from xstack.physics import reflectivity
from xstack.structure import Mirror, Periodic
from xstack.units import HC

DATA = Path(__file__).parent.parent / "data" / "cxro_multilayer"
LIBRARY = MaterialLibrary(default_provider="henke")
W, SI, MO = Material("W", "W", 19.25), Material("Si", "Si", 2.33), Material("Mo", "Mo", 10.22)
B4C, SIO2 = Material("B4C", "B4C", 2.52), Material("SiO2", "SiO2", 2.2)

# name: (H, L, top, d, gamma_H, N, substrate, scan, fixed, polarization, sigma, max|ΔR|, rms ΔR)
CASES = {
    "MoSi_euv": (MO, SI, "L", 6.9, 0.4, 40, SIO2, "wavelength", 90.0, "s", 0.0, 1e-3, 2e-4),
    "MoSi_p45": (MO, SI, "L", 9.5, 0.4, 40, SIO2, "wavelength", 45.0, "p", 0.0, 1e-4, 2e-5),
    "WSi_s": (W, SI, "H", 4.0, 0.4, 30, SI, "angle", 8048.0, "s", 0.0, 6e-3, 6e-4),
    "WSi_p": (W, SI, "H", 4.0, 0.4, 30, SI, "angle", 8048.0, "p", 0.0, 6e-3, 6e-4),
    "WSi_unpol": (W, SI, "H", 4.0, 0.4, 30, SI, "angle", 8048.0, "unpolarized", 0.0, 6e-3, 6e-4),
    "WSi_s_sigma05": (W, SI, "H", 4.0, 0.4, 30, SI, "angle", 8048.0, "s", 0.5, 3e-2, 4e-3),
    "WB4C_energy": (W, B4C, "H", 2.5, 0.4, 50, SI, "energy", 1.0, "s", 0.0, 6e-3, 6e-4),
}


@pytest.mark.parametrize("name", CASES)
def test_matches_cxro(name: str) -> None:
    h, low, top, d, gamma, n, substrate, scan, fixed, pol, sigma, max_tol, rms_tol = CASES[name]
    data = np.loadtxt(DATA / f"{name}.dat", skiprows=2)
    x, expected = data[:, 0], data[:, 1]
    if scan == "wavelength":
        energy, theta = HC / x, np.radians(fixed)
    elif scan == "angle":
        energy, theta = np.full_like(x, fixed), np.radians(x)
    else:
        energy, theta = x, np.radians(fixed)
    coating = Periodic.create(h, low, N=n, d=d, gamma=gamma, top=top, sigma_H=sigma, sigma_L=sigma)
    stack = Mirror(coating, substrate, sigma).stack()
    r = reflectivity(stack, energy, theta, pol, library=LIBRARY)  # type: ignore[arg-type]
    diff = r - expected
    assert np.abs(diff).max() < max_tol
    assert np.sqrt(np.mean(diff**2)) < rms_tol
    assert r.max() == pytest.approx(expected.max(), abs=2e-3)
