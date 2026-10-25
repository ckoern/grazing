# Chantler (NIST FFAST) atomic scattering factors

Files: `<element>.dat.gz` for Z = 1 (H) to 92 (U), gzip-compressed text. Columns: photon energy (eV),
f1, f2 (electrons per atom), on the energy grid of the FFAST tables (from 10.69 eV, or lower for
some elements, to 432.945 keV). f2 contains photoabsorption only; xstack adds Compton scattering for
the `chantler` provider.

- Data: NIST X-Ray Form Factor, Attenuation and Scattering Tables (FFAST),
  https://physics.nist.gov/PhysRefData/FFast/. C. T. Chantler, J. Phys. Chem. Ref. Data 24, 71–643
  (1995), and J. Phys. Chem. Ref. Data 29, 597–1048 (2000).
- Energies, f1 and f2: `FFAST_data.sqlite`, https://physics.nist.gov/PhysRefData/FFast/FFAST_data.sqlite,
  retrieved 2026-10-21 (SHA-256
  `b3daf92362eb7465ce369615849340705fb0860cf76bf034c1a0c390fcee2957`). One table per element with the
  energy in keV, f1 and f2.
- Corrections: table `FFAST_EXTRA` of N. Ritchie, X-ray Properties Database in SQLite Format, version
  1.0.0, NIST (2022), https://doi.org/10.18434/mds2-2819, file `atomic_database.db`, retrieved
  2026-10-25 (SHA-256 `4a5268060987ce76117c30a5db3a9aaa7f9db8d4622b2d29e9eb61f0344e585d`). One row per
  element in the order of Z with the relativistic correction estimates of the FFAST tables, `rc1` (H82)
  and `rc2` (3/5 CL), and the nuclear Thomson correction `nt`.
- f1 in the files is the FFAST f1 plus `rc2` + `nt`; f2 is the FFAST f2. The FFAST table of
  `atomic_database.db` is not used: in 662 rows below 1 keV, its f1 lacks the minus sign of the FFAST
  value.
- Conversion: `python src/xstack/core/materials/data/chantler/convert.py FFAST_data.sqlite
  atomic_database.db` converts keV to eV and keeps all values without rounding. The gzip files carry
  no time stamp, so repeating the conversion gives identical files.

The tables are calculated for isolated atoms. Near absorption edges in the soft X-ray and EUV range
(below about 1 keV) they can differ strongly from the measured Henke/CXRO tables, which are the
better choice there. H and He have f2 = 0 below their ionization energy.
