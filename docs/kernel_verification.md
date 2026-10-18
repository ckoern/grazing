# Verification strategy for the reflectivity kernel

Goal: high confidence that `xstack.core.physics.kernel.parratt` (and the optical constants feeding it)
are correct, and that they stay correct as the code evolves. All checks run as automated pytest tests
under `tests/core/physics/` and `tests/core/materials/`. Tiers 1–4 need no external data. Tiers 5–6 use
stored reference datasets. Run them with `pytest tests/core`.

## Tier 1: analytic limits (exact, tolerance 1e-12)

`tests/core/physics/test_kernel_limits.py`, with synthetic refractive indices and both polarizations.

| Test | Expectation |
|---|---|
| Bare substrate | equals the single-interface Fresnel formula for s and p |
| Total external reflection | β = 0, θ < θ_c → R = 1, T = 0. Critical angle θ_c = √(2δ) (to first order) |
| Zero-thickness layer | inserting a t = 0 layer of any material leaves r unchanged |
| Ambient-material layer | a top layer made of the ambient material changes only the phase of r, not R |
| Substrate-material layer | a bottom layer made of the substrate material (σ = 0) leaves R unchanged |
| Normal incidence | R_s = R_p at θ = 90° |
| Energy conservation | for β = 0 everywhere: R + T = 1, below and above the critical angle |
| Absorption | for β > 0: R + T < 1 |
| Single thin film | Airy formula r = (r01 + r12 e^{2iφ}) / (1 + r01 r12 e^{2iφ}) |
| Interface models | `sharp`, `nevot_croce` and `debye_waller` are identical for σ = 0; `sharp` ignores σ |
| Névot–Croce | a single rough interface gives r = r_Fresnel · exp(−2·k_z,0·k_z,1·σ²) |
| Unpolarized | R is the mean of R_s and R_p |
| Validation | inconsistent shapes, out-of-range angles, and unknown polarization or interface model raise |

## Tier 2: invariance and property-based tests (hypothesis)

`tests/core/physics/test_kernel_properties.py`

- Splitting any layer into two layers of the same material and total thickness gives the same R.
- Results are independent of point ordering and of which subset of points is evaluated together.
- The result arrays have shape (n_points,), dtype float, and contain no NaN or inf; R ∈ [0, 1].
- **Stress cases**, run with floating-point warnings turned into errors: N = 5000 bilayers, θ down to
  1e-5 rad for all interface models, θ within 1e-9 of θ_c, E = 100 keV, strongly absorbing
  layers (n = 0.95 + 0.05i, 1 µm thick), very thick lossless and lossy layers (10 µm).

The coatings are tested in `tests/core/structure/test_coatings.py`: `Periodic`, `DepthGraded` and
`Composite` build the expected layer sequences, so equal stacks give equal R by the invariances above.

## Tier 3: independent implementations (cross-check)

`tests/core/physics/test_reference_comparison.py`, with the implementations in
`tests/reference/matrix_methods.py` (never imported by the library):
- **Abelès 2×2 characteristic-matrix** method, written independently in plain numpy, for stacks without
  roughness (rtol 1e-8, atol 1e-10).
- **Interface and propagation matrices** with roughness factors, for r and t (same tolerance).
- Randomized comparison: hypothesis generates random stacks (0–20 layers, δ ≈ c·λ² with β/δ between 0 and
  2, thicknesses 0.5–50 nm, σ 0–1 nm, θ from 1e-4 rad to 90°, s and p) and compares kernel and
  references in 300 examples each.
- **Arbitrary precision** (mpmath, 50 digits) matrix implementation for hard cases: grazing incidence at
  1e-5 rad, exactly at the critical angle, evanescent layers, a Bragg peak, strongly absorbing soft X-ray
  layers and 100 keV, each with σ = 0 and 0.4 nm (rtol 1e-11). This is the ground truth for floating-point
  accuracy.

## Tier 4: semi-analytic physics checks (tolerance by physics)

`tests/core/physics/test_multilayer_physics.py`, `test_bragg.py`, `test_models.py`

- **Bragg peak positions** of periodic W/Si and W/C multilayers match the refraction-corrected Bragg law
  for orders m = 1…3 (within 0.3 of the peak width) and lie above the plain Bragg angle.
- **Γ = 1/2** suppresses the second order (R₂ < 1 % of R₁).
- **Bandwidth** ∝ 1/N in the unsaturated regime (Si/C, N = 20, 40, 80), and **saturation** of R vs. N
  for absorbing systems.
- **Optimal Γ**: the Γ of maximum peak reflectivity of W/Si lies near the Vinogradov–Zeldovich value.
- **Kinematic limit**: `KinematicModel` agrees with `ParrattModel` within 5 % for a weakly reflecting
  Si/C stack. Without refraction its peak sits at the plain Bragg angle, with refraction at the
  corrected one.
- **Depth-graded**: R(E) of a power-law stack stays above 5 % across the band predicted by the d_min/d_max
  Bragg energies and is low outside it.
- **Analytic estimates** (`bragg.py`): Bragg law round trips, optimal Γ equation, extinction and
  absorption bilayer counts, relative bandwidth.
- **Resolution** (`test_resolution.py`, `test_models.py`): Gaussian smearing preserves constants and
  peak area, broadens with the expected variance, and a model result with resolution stays close to the
  unsmeared one away from features.

## Tier 5: optical constants

`tests/core/materials/test_cxro_reference.py`, `test_providers.py`, `test_tabulated.py`, `test_material.py`

- δ, β of Si, W, Mo, B₄C, C, Pt, SiO₂ and MoSi₂ at 60 energies between 30 eV and 30 keV vs. the CXRO
  "index of refraction" calculator (`tests/data/cxro/`, rtol 1e-3 for δ and 5e-3 for β), which covers
  interpolation differences. Both the Henke provider and the default
  provider are compared.
- Critical angle of Si and Au at Cu Kα (8.048 keV) vs. tabulated values (3 %).
- Formula parsing and density → atom-density conversion (B₄C, SiO₂, Si, parentheses in formulas).
- Provider switching at 30 keV (Henke → xraylib) is continuous (δ within 0.3 %, β within 5 %), and the
  combined provider uses the expected side of the switching energy.
- `TabulatedProvider` reproduces its nodes and interpolates as documented.

## Tier 6: reference software and literature benchmarks

Reference curves are stored in `tests/data/cxro_multilayer/` as CXRO text files; `PROVENANCE.md` in that
directory records the source, the retrieval date and all inputs of each file so that it can be
regenerated.

| Reference | Systems |
|---|---|
| CXRO multilayer reflectivity calculator (henke.lbl.gov/optical_constants/multi2.html) | Mo/Si periodic at 13.5 nm (s at 90° and p at 45°); W/Si periodic at 8.048 keV (s, p, unpolarized, and with σ = 0.5 nm); W/B₄C energy scan |

`tests/core/physics/test_cxro_multilayer.py` compares the curves with the maximum and RMS deviation
of R per case (2e-5 to 6e-4 RMS and 1e-4 to 6e-3 maximum without roughness, 4e-3 and 3e-2 with roughness,
whose "interdiffusion" convention differs from Névot–Croce). The comparison uses the Henke provider with
the densities of the reference run, so remaining differences come from the optical constants as processed by CXRO and
from the roughness convention. The kernel itself is verified to machine precision by tiers 1–3.

Not covered: comparisons with IMD, GenX, xrt or refnx, and with published literature curves.

## Tier 7: performance regression

`tests/core/physics/test_benchmark.py` times `parratt` for 1000 energy points × 300 bilayers and for
2000 points × 1500 bilayers (3000 layers) with pytest-benchmark. The benchmarks are skipped by default
(`--benchmark-skip` in the pytest options). Run them with
`pytest tests/core/physics/test_benchmark.py -o addopts=""`.

## Process

- Every bug found in the kernel gets a regression test.
- Reference datasets are versioned with the repository. A dataset is regenerated by repeating the inputs
  in its `PROVENANCE.md` in the web calculator.
