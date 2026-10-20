"""Generate the refnx reference reflectivity curves in this directory.

Run from the repository root with refnx installed (``pip install refnx``)::

    python tests/data/refnx/generate.py

Each case is a stack built with xstack (materials, optical constants of the default library). Its
complex refractive indices n are converted exactly to refnx scattering length densities,
SLD = k²(1 − n²)/(4π), and refnx's Abelès implementation with Névot–Croce roughness computes the
reflectivity. The files store the refractive indices, thicknesses and roughnesses used, so the
comparison test does not depend on xstack's optical constants.
"""

import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import refnx
from refnx.reflect import abeles

from xstack.core.units import energy_to_wavelength

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))

from reference_cases import angle_cases  # noqa: E402


def refnx_reflectivity(
    n: np.ndarray,
    thickness: np.ndarray,
    roughness: np.ndarray,
    theta: np.ndarray,
    wavelength: float,
) -> np.ndarray:
    """s-polarized reflectivity from refnx for refractive indices of the media (ambient first)."""
    wavelength_a = wavelength * 10.0
    k = 2.0 * math.pi / wavelength_a
    sld = k**2 * (1.0 - n**2) / (4.0 * math.pi) * 1e6
    layers = np.zeros((n.size, 4))
    layers[:, 1] = sld.real
    layers[:, 2] = -sld.imag
    layers[1:-1, 0] = thickness * 10.0
    layers[1:, 3] = roughness * 10.0
    q = 4.0 * math.pi * np.sin(theta) / wavelength_a
    return np.asarray(abeles(q, layers, threads=1))


def main() -> int:
    """Write one JSON file per case."""
    for name, (stack, energy, angles_deg, description) in angle_cases().items():
        n = stack.refractive_indices(np.array([energy]))[0]
        theta = np.radians(angles_deg)
        wavelength = float(energy_to_wavelength(energy))
        reflectivity = refnx_reflectivity(
            n, stack.thicknesses(), stack.roughnesses(), theta, wavelength
        )
        if np.any(reflectivity > 1.0 + 1e-12):
            raise RuntimeError(f"{name}: reflectivity above 1, wrong SLD sign convention")
        data = {
            "provenance": {
                "source": "refnx.reflect.abeles (Abelès matrices, Névot–Croce roughness)",
                "refnx_version": refnx.version.version,
                "generated": datetime.now(UTC).isoformat(timespec="seconds"),
                "script": "tests/data/refnx/generate.py",
                "python": sys.version.split()[0],
            },
            "description": description,
            "polarization": "s",
            "energy_eV": energy,
            "wavelength_nm": wavelength,
            "n_real": n.real.tolist(),
            "n_imag": n.imag.tolist(),
            "thickness_nm": stack.thicknesses().tolist(),
            "roughness_nm": stack.roughnesses().tolist(),
            "theta_deg": angles_deg.tolist(),
            "reflectivity": reflectivity.tolist(),
        }
        (HERE / f"{name}.json").write_text(json.dumps(data, indent=1) + "\n")
        print(f"{name}: {len(angles_deg)} points, {n.size - 2} layers")
    return 0


if __name__ == "__main__":
    sys.exit(main())
