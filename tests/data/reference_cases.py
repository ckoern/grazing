"""Stacks shared by the generators of reference curves from external programs.

The generators in the subdirectories import this module; it is not used by the tests.
"""

import numpy as np

from xstack.core.materials import get
from xstack.core.structure import Mirror, Periodic, SingleLayer
from xstack.core.structure.coatings import DepthGraded
from xstack.core.structure.stack import Stack


def angle_cases() -> dict[str, tuple[Stack, float, np.ndarray, str]]:
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


def energy_cases() -> dict[str, tuple[Stack, float, np.ndarray, str]]:
    """Stacks, grazing angle (deg), photon energies (eV) and a description of every energy scan."""
    w, si, b4c = (get(n) for n in ("W", "Si", "B4C"))
    return {
        "wsi_energy": (
            Mirror(
                Periodic.create(w, si, N=20, d=4.0, gamma=0.4, sigma_H=0.3, sigma_L=0.4),
                si,
                substrate_roughness=0.3,
            ).stack(),
            1.0,
            np.linspace(5000.0, 20000.0, 600),
            "W/Si, N = 20, d = 4.0 nm, Γ = 0.4, σ_H 0.3, σ_L 0.4 nm, on Si (σ 0.3 nm), "
            "energy scan across the W L edges at 1°",
        ),
        "wb4c_graded_energy": (
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
            0.3,
            np.linspace(8000.0, 30000.0, 600),
            "W/B4C depth graded (power law), N = 60, d 2.5–6.0 nm, σ 0.3 nm, on Si, "
            "energy scan at 0.3°",
        ),
    }
