# Chantler (NIST FFAST) atomic scattering factors

Files: `<element>.dat.gz` for Z = 1 (H) to 92 (U), gzip-compressed text. Columns: photon energy (eV),
f1, f2 (electrons per atom). f2 contains photoabsorption only; xstack adds Compton scattering for the
`chantler` provider.

- Data: C. T. Chantler, "Theoretical form factor, attenuation and scattering tabulation for
  Z = 1–92 from E = 1–10 eV to E = 0.4–1.0 MeV", J. Phys. Chem. Ref. Data 24, 71–643 (1995), and
  "Detailed tabulation of atomic form factors, photoelectric absorption and scattering cross
  section, and mass attenuation coefficients in the vicinity of absorption edges in the soft X-ray
  (Z = 30–36, Z = 60–89, E = 0.1 keV–10 keV), addressing convergence issues of earlier work",
  J. Phys. Chem. Ref. Data 29, 597–1048 (2000). Distributed by NIST as the FFAST database,
  https://physics.nist.gov/PhysRefData/FFast/.
- Retrieved from: the `Chantler` table of `xraydb.sqlite` in xraydb 4.5.8 (database version 9.2,
  "Chantler data on finer energy grid"), https://github.com/xraypy/XrayDB, on 2026-10-21. The NIST
  FFAST web service was not reachable at that time.
- Conversion: `python src/xstack/core/materials/data/chantler/convert.py path/to/xraydb.sqlite`.
  It keeps the tabulated energies from 10 eV to 433 keV plus one point beyond each end and writes
  f1 = Z + f′ (xraydb stores the anomalous part f′ = f1 − Z) and f2 without rounding. The gzip
  files carry no time stamp, so repeating the conversion gives identical files.

The tables are calculated for isolated atoms. Near absorption edges in the soft X-ray and EUV range
(below about 1 keV) they can differ strongly from the measured Henke/CXRO tables, which are the
better choice there. H and He have f2 = 0 below their ionization energy.
