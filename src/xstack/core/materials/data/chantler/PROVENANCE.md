# Chantler (NIST FFAST) atomic scattering factors

Files: `<element>.dat.gz` for Z = 1 (H) to 92 (U), gzip-compressed text. Columns: photon energy (eV),
f1, f2 (electrons per atom), on the energy grid of the FFAST tables (from 10.69 eV, or lower for
some elements, to 432.945 keV). f2 contains photoabsorption only; xstack adds Compton scattering for
the `chantler` provider.

- Source: NIST X-Ray Form Factor, Attenuation and Scattering Tables (FFAST),
  https://physics.nist.gov/PhysRefData/FFast/. The tables of each element, `gtype=4` (form factors,
  attenuation and scattering cross sections).
- References: C. T. Chantler, "Theoretical form factor, attenuation and scattering tabulation for
  Z = 1–92 from E = 1–10 eV to E = 0.4–1.0 MeV", J. Phys. Chem. Ref. Data 24, 71–643 (1995), and
  "Detailed tabulation of atomic form factors, photoelectric absorption and scattering cross
  section, and mass attenuation coefficients in the vicinity of absorption edges in the soft X-ray
  (Z = 30–36, Z = 60–89, E = 0.1 keV–10 keV), addressing convergence issues of earlier work",
  J. Phys. Chem. Ref. Data 29, 597–1048 (2000).
- f1 is the tabulated f1 plus the relativistic correction f_rel (3/5 CL estimate) and the nuclear
  Thomson correction f_NT that FFAST states in the header of each element's table. f2 is the
  tabulated f2.
- Retrieved: 2026-10-21. The FFAST web service was not reachable then; the tables were taken from a
  copy downloaded from FFAST on 2020-07-02 (files `nk_data/nist/sf/<Element>.sf` of
  https://github.com/xastprl/DarpanX, commit 1e0925f), which hold the FFAST energy grid, f1 with
  the corrections above, and f2. Their grid and f2 agree exactly with the tables currently
  offered by FFAST for download.
- Conversion: `python src/xstack/core/materials/data/chantler/convert.py <directory of the .sf
  files>` converts keV to eV and keeps all values without rounding. The gzip files carry no time
  stamp, so repeating the conversion gives identical files.

The tables are calculated for isolated atoms. Near absorption edges in the soft X-ray and EUV range
(below about 1 keV) they can differ strongly from the measured Henke/CXRO tables, which are the
better choice there. H and He have f2 = 0 below their ionization energy.
