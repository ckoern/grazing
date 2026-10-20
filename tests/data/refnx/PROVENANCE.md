# refnx reference reflectivity curves

Source: refnx 0.1.67, `refnx.reflect.abeles` (Abelès characteristic matrices with
Névot–Croce interface roughness), https://refnx.readthedocs.io. Generated 2026-10-18 with
`python tests/data/refnx/generate.py`, which documents the conversion.

The stacks are defined in `tests/data/reference_cases.py`, shared with the GenX set. Each JSON file stores the complex refractive index n of every medium (ambient first, substrate last;
from xstack's default material library at the given energy), the layer thicknesses and interface
roughnesses in nm, the grazing angles in degrees and the s-polarized reflectivity computed by refnx.
The refractive indices are converted exactly to refnx scattering length densities,
SLD = k²(1 − n²)/(4π) with k = 2π/λ; the imaginary SLD is passed as refnx's (positive) absorptive part.
The comparison test feeds the stored indices to xstack's kernel, so it checks the reflectivity
calculation independently of the optical constants (those are checked against CXRO data).

| File | Structure | Energy | Grazing angles |
|---|---|---|---|
| si_rough.json | Si substrate, σ 0.5 nm | 8048 eV | 0.05–5.0°, 700 points |
| pt_single.json | Pt 30 nm (σ 0.4 nm) on Si (σ 0.3 nm) | 8048 eV | 0.05–5.0°, 700 points |
| wsi_periodic.json | W/Si, N 20, d 4.0 nm, Γ 0.4, σ_H 0.3, σ_L 0.4 nm, on Si (σ 0.3 nm) | 8048 eV | 0.05–5.0°, 700 points |
| pt_c_rough.json | Pt/C, N 10, d 8.0 nm, Γ 0.3, σ_H 1.0, σ_L 0.8 nm, on Si (σ 0.5 nm) | 8048 eV | 0.05–5.0°, 700 points |
| wb4c_graded.json | W/B4C power-law depth graded, N 60, d 2.5–6.0 nm, σ 0.3 nm, on Si | 17479 eV | 0.02–1.5°, 700 points |
| mosi_euv.json | Mo/Si, N 40, d 6.9 nm, Γ 0.4, σ 0.5 nm, Si on top, on Si | 92 eV | 60–90°, 400 points |
