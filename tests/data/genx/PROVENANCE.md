# GenX reference curves

Source: GenX 3.8.11 (`genx3` on PyPI, https://aglavic.github.io/genx/), model library only:
`genx.models.lib.paratt.Refl` and `Refl_nvary2` (Parratt recursion with Névot–Croce interface
roughness) and `genx.models.lib.instrument.GaussIntensity` (footprint of a Gaussian beam). Generated
2026-10-20 with `python tests/data/genx/generate.py` (GenX installed with `pip install --no-deps
genx3`; the GUI dependencies are not needed), which documents the conversion.

The stacks are those of the refnx set (`tests/data/reference_cases.py`) plus two energy scans. Each
JSON file stores the complex refractive index of every distinct medium (`n_real`, `n_imag`, one row
per energy) and the index of the medium at every position of the stack (`medium`, ambient first,
substrate last), the layer thicknesses and interface roughnesses in nm, the grazing angles in degrees,
the energies and wavelengths, and the s-polarized reflectivity computed by GenX. The refractive
indices come from xstack's default material library and go into GenX unchanged; GenX orders the
media from the substrate to the ambient medium, so the generator reverses them. The comparison test
feeds the stored indices to xstack's kernel, so it checks the reflectivity calculation independently
of the optical constants.

| File | Structure | Scan |
|---|---|---|
| si_rough.json | Si substrate, σ 0.5 nm | 8048 eV, 0.05–5.0°, 700 points |
| pt_single.json | Pt 30 nm (σ 0.4 nm) on Si (σ 0.3 nm) | 8048 eV, 0.05–5.0°, 700 points |
| wsi_periodic.json | W/Si, N 20, d 4.0 nm, Γ 0.4, σ_H 0.3, σ_L 0.4 nm, on Si (σ 0.3 nm) | 8048 eV, 0.05–5.0°, 700 points |
| pt_c_rough.json | Pt/C, N 10, d 8.0 nm, Γ 0.3, σ_H 1.0, σ_L 0.8 nm, on Si (σ 0.5 nm) | 8048 eV, 0.05–5.0°, 700 points |
| wb4c_graded.json | W/B4C power-law depth graded, N 60, d 2.5–6.0 nm, σ 0.3 nm, on Si | 17479 eV, 0.02–1.5°, 700 points |
| mosi_euv.json | Mo/Si, N 40, d 6.9 nm, Γ 0.4, σ 0.5 nm, Si on top, on Si | 92 eV, 60–90°, 400 points |
| wsi_energy.json | W/Si as above | 1°, 5–20 keV (across the W L edges), 600 points |
| wb4c_graded_energy.json | W/B4C depth graded as above | 0.3°, 8–30 keV, 600 points |
| footprint.json | Gaussian beam, σ 0.05 mm, centred sample 20 mm long | 0.01–2.0°, 200 points |
