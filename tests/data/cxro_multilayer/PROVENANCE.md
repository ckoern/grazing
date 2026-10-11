# CXRO multilayer reflectivity reference curves

Source: CXRO "Multilayer Reflectivity" calculator, https://henke.lbl.gov/optical_constants/multi2.html
(form handler https://henke.lbl.gov/cgi-bin/multi.pl), retrieved 2026-10-11. The calculator uses
Henke scattering factors and the analytic formula of V. G. Kohn, Phys. Stat. Sol. (b) 187, 61 (1995).

Columns: scan variable, reflectivity, phase (deg). CXRO conventions: "Top" material is the layer facing
vacuum, Gamma is the thickness fraction of the *bottom* layer of each period, angles are grazing angles,
polarization 1 = s, -1 = p, 0 = unpolarized, Sigma is the "interdiffusion thickness".

| File | Top/Bottom (density g/cm³) | d (nm) | Gamma (bottom) | N | Sigma (nm) | Substrate | Pol | Scan | Fixed |
|---|---|---|---|---|---|---|---|---|---|
| MoSi_euv.dat | Si 2.33 / Mo 10.22 | 6.9 | 0.4 | 40 | 0 | SiO2 2.2 | 1 | wavelength 12.5–14.5 nm, 200 steps | 90° |
| MoSi_p45.dat | Si 2.33 / Mo 10.22 | 9.5 | 0.4 | 40 | 0 | SiO2 2.2 | -1 | wavelength 12.5–14.5 nm, 200 steps | 45° |
| WSi_s.dat | W 19.25 / Si 2.33 | 4.0 | 0.6 | 30 | 0 | Si 2.33 | 1 | angle 0.05–3.0°, 600 steps | 8048 eV |
| WSi_p.dat | same as WSi_s | | | | 0 | | -1 | | |
| WSi_unpol.dat | same as WSi_s | | | | 0 | | 0 | | |
| WSi_s_sigma05.dat | same as WSi_s | | | | 0.5 | | 1 | | |
| WB4C_energy.dat | W 19.25 / B4C 2.52 | 2.5 | 0.6 | 50 | 0 | Si 2.33 | 1 | energy 5000–30000 eV, 500 steps | 1.0° |
