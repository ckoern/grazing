# Henke / CXRO atomic scattering factors

Files: `<element>.nff` for Z = 1 (H) to 92 (U). Columns: photon energy (eV), f1, f2.
An f1 value of -9999 marks energies where f1 is not tabulated.

- Source: Center for X-Ray Optics (CXRO), Lawrence Berkeley National Laboratory,
  https://henke.lbl.gov/optical_constants/sf/<element>.nff
- Retrieved: 2026-10-11
- Reference: B. L. Henke, E. M. Gullikson, J. C. Davis, "X-ray interactions: photoabsorption,
  scattering, transmission, and reflection at E = 50-30000 eV, Z = 1-92",
  Atomic Data and Nuclear Data Tables 54, 181-342 (1993), with later CXRO updates.

The files are stored unmodified. Some tables extend below 10 eV, and a few contain repeated
energies, which the loader merges by averaging.
