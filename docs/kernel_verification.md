# Verification strategy for the reflectivity kernel

Goal: high confidence that `physics.kernel.parratt` (and the optical constants feeding it) are
correct, and that they stay correct as the code evolves. All checks run as automated pytest tests.
Tiers 1–4 need no external data. Tiers 5–6 use stored reference datasets.

## Tier 1: analytic limits (exact, tolerance ~1e-12)

| Test | Expectation |
|---|---|
| Bare substrate | equals the single-interface Fresnel formula for s and p |
| Total external reflection | β = 0, θ < θ_c → R = 1. Critical angle θ_c = √(2δ) (to first order) |
| Zero-thickness layer | inserting a t = 0 layer of any material leaves r unchanged |
| Ambient-material layer | a top layer made of the ambient material changes only the phase of r, not R |
| Substrate-material layer | a bottom layer made of the substrate material (σ = 0) leaves R unchanged |
| Normal incidence | R_s = R_p at θ = 90° |
| Energy conservation | for β = 0 everywhere: R + T = 1 (the kernel can return T) |
| Absorption | for β > 0: R + T < 1 |
| Single thin film | Airy formula r = (r01 + r12 e^{2iφ}) / (1 + r01 r12 e^{2iφ}) |
| Zero roughness | Névot–Croce with σ = 0 is identical to the sharp interface |

## Tier 2: invariance and property-based tests (hypothesis)

- Splitting any layer into two layers of the same material and total thickness gives the same R.
- `Periodic(N=…)` versus an explicitly built list of layers: identical.
- `Composite` built from blocks versus the flattened stack: identical.
- Results are independent of point ordering and of the vectorization batch size.
- The result arrays always have shape (n_points,), dtype float, and contain no NaN or inf.
- **Stress cases**: N = 5000 bilayers, θ → 0, θ near θ_c, E = 100 keV, strongly absorbing layers
  (e.g. Au at 1 keV), very thick layers (µm). No overflow, R ∈ [0, 1].

## Tier 3: independent implementations (cross-check, tolerance ~1e-10)

Implemented in `tests/reference/` and never imported by the library:
- **Abelès 2×2 characteristic-matrix** method, written independently in plain numpy, for
  small to moderate stacks where it is stable.
- **Arbitrary precision** (mpmath, 50 digits) Parratt/matrix implementation for a few points.
  This is the ground truth for checking floating-point accuracy, including near-critical and evanescent cases.
- Randomized comparison: hypothesis generates random stacks (1–20 layers, random materials with
  δ, β in realistic ranges, thicknesses 0.5–50 nm, σ 0–1 nm) and compares kernel vs. references.

## Tier 4: semi-analytic physics checks (tolerance by physics, e.g. 1 %)

- **Bragg peak positions** of periodic MLs match the refraction-corrected Bragg law for orders
  m = 1…3 (peaks found by a local maximum search).
- **Γ = 1/2** suppresses the second order (R₂ ≪ R₁, R₃).
- **Bandwidth** ∝ 1/N in the unsaturated regime, and **saturation** of R vs. N for absorbing systems.
- **Kinematic (Born) limit**: for weak contrast and few layers, R agrees with
  |FT of the electron density gradient|²-type kinematic expressions.
- **Roughness**: Névot–Croce vs. finely sliced erf interface profile (e.g. 200 slices). They
  converge for small σ·k_z, and the deviation trend for larger σ matches published behaviour.
  The Debye–Waller factor matches Névot–Croce in the expected limit.
- **Depth-graded**: R(E) of a power-law stack covers the band predicted by the d_min/d_max Bragg energies.

## Tier 5: optical constants

- δ, β of Si, W, Mo, B₄C, C, Pt at several energies vs. the CXRO "index of refraction" calculator
  (stored values, relative tolerance ~1e-3, which covers interpolation differences).
- Critical angle of Si and Au at 8.048 keV vs. tabulated values.
- Formula parsing and density → atom-density conversion (H₂O, B₄C, SiO₂, MoSi₂).
- Provider switching at 30 keV (Henke → Chantler/xraylib) is continuous within tolerance.

## Tier 6: reference software and literature benchmarks

Reference curves are stored in `tests/data/reference/` as CSV. Each file has a provenance header
(tool, version, date, all inputs, optical-constant source) so it can be regenerated.

| Reference | Systems |
|---|---|
| CXRO multilayer reflectivity calculator (henke.lbl.gov/optical_constants/multi2.html) | Mo/Si periodic at 13.5 nm; W/Si periodic at 8.048 keV; bare Si/Au mirrors |
| IMD (D. Windt) | depth-graded W/Si power-law stack, roughness included |
| GenX and/or xrt (open source, Python) | random small stacks, s/p polarization |
| refnx (neutron reflectometry, same math via SLD mapping) | kernel math with roughness |
| Literature | Mo/Si ≈ 70 % at 13.5 nm near normal incidence (σ ≈ 0.3 nm); NuSTAR-type W/Si or Pt/C designs |

Comparisons use the *same optical constants* as the reference (load the n, k actually used by the
reference via `TabulatedProvider`). This separates kernel errors from database or interpolation
differences, which tier 5 covers separately.

Tolerance: ~1e-6 absolute in R for identical inputs. Looser (~1e-3) for tools with different
roughness conventions, with the differing conventions documented.

## Tier 7: performance regression

pytest-benchmark for representative sizes: 1000 energy points × 300 bilayers, 2000 points × 3000
layers, and a full DE stage on a coarse grid. Track timings in CI so that refactors cannot silently
slow down the optimizer's inner loop.

## Process

- Tiers 1–4 are written together with the kernel (test-first where possible), before coatings and
  optimizers build on it.
- Every bug found in the kernel gets a regression test.
- Reference datasets are versioned. Regenerating them requires an explicit script
  (`tests/data/reference/regenerate.md` documents manual steps for web or GUI tools).
