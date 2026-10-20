"""Generate the GenX reference curves in this directory.

Run from the repository root with GenX installed. Only GenX's model library is used, so the GUI
dependencies are not needed (``pip install --no-deps genx3``)::

    python tests/data/genx/generate.py

Each case is a stack built with xstack (materials, optical constants of the default library). Its
complex refractive indices go directly into GenX's Parratt recursion with Névot–Croce roughness
(``genx.models.lib.paratt``), which orders the media from the substrate to the ambient medium.
Angle scans use ``Refl``, energy scans ``Refl_nvary2`` with refractive indices per energy. The
files store the refractive indices, thicknesses and roughnesses used, so the comparison test does
not depend on xstack's optical constants. ``footprint.json`` stores GenX's Gaussian-beam footprint
factor (``genx.models.lib.instrument.GaussIntensity``).
"""

import json
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
from genx.models.lib.instrument import GaussIntensity
from genx.models.lib.paratt import Refl, Refl_nvary2

from xstack.core.structure.stack import Stack
from xstack.core.units import energy_to_wavelength

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))

from reference_cases import angle_cases, energy_cases  # noqa: E402


def genx_arrays(stack: Stack) -> tuple[np.ndarray, np.ndarray]:
    """Thicknesses and roughnesses of all media in GenX order (substrate first), in nm."""
    d = np.concatenate([[0.0], stack.thicknesses()[::-1], [0.0]])
    sigma = np.concatenate([stack.roughnesses()[::-1], [0.0]])
    return d, sigma


def distinct_media(n: np.ndarray) -> dict[str, Any]:
    """Refractive indices of the distinct media and the medium index of every layer position.

    Args:
        n: Refractive indices, shape (P, M), ambient first.

    Returns:
        ``n_real`` and ``n_imag`` of shape (P, K) for the K distinct media and ``medium`` of
        length M, so that ``n[:, j] == n_real[:, medium[j]] + 1j * n_imag[:, medium[j]]``.
    """
    distinct, medium = np.unique(n, axis=1, return_inverse=True)
    return {
        "n_real": distinct.real.tolist(),
        "n_imag": distinct.imag.tolist(),
        "medium": medium.ravel().tolist(),
    }


def provenance(source: str) -> dict[str, str]:
    """Provenance header of a reference file."""
    return {
        "source": source,
        "genx_version": version("genx3"),
        "generated": datetime.now(UTC).isoformat(timespec="seconds"),
        "script": "tests/data/genx/generate.py",
        "python": sys.version.split()[0],
    }


def write(name: str, data: dict[str, Any]) -> None:
    """Write one reference file."""
    (HERE / f"{name}.json").write_text(json.dumps(data, indent=1) + "\n")


def angle_scans() -> None:
    """Write the angle scans, computed with ``Refl`` at a single energy."""
    for name, (stack, energy, angles_deg, description) in angle_cases().items():
        n = stack.refractive_indices(np.array([energy]))[0]
        wavelength = float(energy_to_wavelength(energy))
        d, sigma = genx_arrays(stack)
        reflectivity = Refl(angles_deg, wavelength, n[::-1], d, sigma)
        write(
            name,
            {
                "provenance": provenance(
                    "genx.models.lib.paratt.Refl (Parratt recursion, Névot–Croce roughness)"
                ),
                "description": description,
                "polarization": "s",
                "axis": "angle",
                "energy_eV": [energy],
                "wavelength_nm": [wavelength],
                **distinct_media(n[np.newaxis, :]),
                "thickness_nm": stack.thicknesses().tolist(),
                "roughness_nm": stack.roughnesses().tolist(),
                "theta_deg": angles_deg.tolist(),
                "reflectivity": np.asarray(reflectivity).tolist(),
            },
        )
        print(f"{name}: {len(angles_deg)} angles, {n.size - 2} layers")


def energy_scans() -> None:
    """Write the energy scans, computed with ``Refl_nvary2`` and indices per energy."""
    for name, (stack, angle_deg, energies, description) in energy_cases().items():
        n = stack.refractive_indices(energies)
        wavelengths = np.asarray(energy_to_wavelength(energies))
        d, sigma = genx_arrays(stack)
        theta = np.full(energies.size, angle_deg)
        reflectivity = Refl_nvary2(theta, wavelengths, n[:, ::-1].T, d, sigma)
        write(
            name,
            {
                "provenance": provenance(
                    "genx.models.lib.paratt.Refl_nvary2 (Parratt recursion, Névot–Croce roughness)"
                ),
                "description": description,
                "polarization": "s",
                "axis": "energy",
                "energy_eV": energies.tolist(),
                "wavelength_nm": wavelengths.tolist(),
                **distinct_media(n),
                "thickness_nm": stack.thicknesses().tolist(),
                "roughness_nm": stack.roughnesses().tolist(),
                "theta_deg": theta.tolist(),
                "reflectivity": np.asarray(reflectivity).tolist(),
            },
        )
        print(f"{name}: {energies.size} energies, {n.shape[1] - 2} layers")


def footprint() -> None:
    """Write GenX's footprint factor of a Gaussian beam on a centred sample."""
    angles_deg = np.linspace(0.01, 2.0, 200)
    sample_length_mm, beam_sigma_mm = 20.0, 0.05
    factor = GaussIntensity(angles_deg, sample_length_mm / 2, sample_length_mm / 2, beam_sigma_mm)
    write(
        "footprint",
        {
            "provenance": provenance("genx.models.lib.instrument.GaussIntensity"),
            "description": "Fraction of a Gaussian beam (standard deviation 0.05 mm) that hits a "
            "centred sample of length 20 mm",
            "sample_length_mm": sample_length_mm,
            "beam_sigma_mm": beam_sigma_mm,
            "theta_deg": angles_deg.tolist(),
            "factor": np.asarray(factor).tolist(),
        },
    )
    print(f"footprint: {angles_deg.size} angles")


def main() -> int:
    """Write all reference files."""
    angle_scans()
    energy_scans()
    footprint()
    return 0


if __name__ == "__main__":
    sys.exit(main())
