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

from xstack.core.materials import get
from xstack.core.structure import Mirror, Periodic, SingleLayer
from xstack.core.structure.coatings import DepthGraded
from xstack.core.structure.stack import Stack
from xstack.core.units import energy_to_wavelength

HERE = Path(__file__).parent


def cases() -> dict[str, tuple[Stack, float, np.ndarray, str]]:
    """Stacks, photon energy (eV), grazing angles (deg) and a description of every case."""
    w, si, mo, pt, c, b4c = (get(n) for n in ("W", "Si", "Mo", "Pt", "C", "B4C"))
    angles_cu = np.linspace(0.05, 5.0, 700)
    return {
        "si_rough": (
            Stack((), si, 0.5),
            8048.0,
            angles_cu,
            "Si substrate, roughness 0.5 nm, Cu Kα",
        ),
        "wsi_periodic": (
            Mirror(
                Periodic.create(w, si, N=20, d=4.0, gamma=0.4, sigma_H=0.3, sigma_L=0.4),
                si,
                substrate_roughness=0.3,
            ).stack(),
            8048.0,
            angles_cu,
            "W/Si, N = 20, d = 4.0 nm, Γ = 0.4, σ_H 0.3, σ_L 0.4 nm, on Si (σ 0.3 nm), Cu Kα",
        ),
        "pt_c_rough": (
            Mirror(
                Periodic.create(pt, c, N=10, d=8.0, gamma=0.3, sigma_H=1.0, sigma_L=0.8),
                si,
                substrate_roughness=0.5,
            ).stack(),
            8048.0,
            angles_cu,
            "Pt/C, N = 10, d = 8.0 nm, Γ = 0.3, σ_H 1.0, σ_L 0.8 nm, on Si (σ 0.5 nm), Cu Kα",
        ),
        "wb4c_graded": (
            Mirror(
                DepthGraded.create(
                    w,
                    b4c,
                    N=60,
                    d_min=2.5,
                    d_max=6.0,
                    c=0.25,
                    gamma_top=0.4,
                    sigma_H=0.3,
                    sigma_L=0.3,
                ),
                si,
                substrate_roughness=0.3,
            ).stack(),
            17479.0,
            np.linspace(0.02, 1.5, 700),
            "W/B4C depth graded (power law), N = 60, d 2.5–6.0 nm, σ 0.3 nm, on Si, Mo Kα",
        ),
        "mosi_euv": (
            Mirror(
                Periodic.create(mo, si, N=40, d=6.9, gamma=0.4, sigma_H=0.5, sigma_L=0.5, top="L"),
                si,
            ).stack(),
            92.0,
            np.linspace(60.0, 90.0, 400),
            "Mo/Si, N = 40, d = 6.9 nm, Γ = 0.4, σ 0.5 nm, Si on top, on Si, 92 eV near normal",
        ),
        "pt_single": (
            Mirror(SingleLayer.create(pt, t=30.0, sigma=0.4), si, substrate_roughness=0.3).stack(),
            8048.0,
            angles_cu,
            "Pt 30 nm (σ 0.4 nm) on Si (σ 0.3 nm), Cu Kα",
        ),
    }


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
    for name, (stack, energy, angles_deg, description) in cases().items():
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
