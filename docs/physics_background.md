# Physics background for xstack

Reference notes on what the software needs under the hood to design X-ray multilayer (ML) mirrors.
Conventions: grazing angle θ (measured from the surface), photon energy E, wavelength λ = hc/E
(hc ≈ 1239.842 eV·nm), wavenumber k = 2π/λ. Lengths are in nm, energies in eV, and angles in rad internally.

---

## 1. Optical constants

The complex refractive index in the X-ray range is

    n = 1 − δ + iβ,     δ, β ~ 1e-3 … 1e-9 (positive)

It comes from the atomic scattering factors f = f1 + i·f2 of the constituent atoms:

    δ + iβ = (r_e λ² / 2π) · Σ_a n_a (f1_a + i f2_a)

- r_e = 2.8179403e-6 nm (classical electron radius)
- n_a = atomic number density of species a = ρ·N_A·x_a / Σ(x_b·M_b), from the chemical formula
  (stoichiometry x_a, molar masses M_a) and the mass density ρ.

**What a material needs:** chemical formula, density, and a data source for f1, f2 per element.

**Data sources**

| Source | Range | Notes |
|---|---|---|
| Henke / CXRO (`*.nff`) | 10 eV – 30 keV | Standard for soft and tender X-rays. Freely available, so it can be bundled. |
| Chantler / NIST FFAST | 1 eV – 433 keV | Needed for hard X-ray MLs (e.g. W/Si, Pt/C up to 100+ keV) |
| xraylib | broad | Python bindings. Optional dependency. |
| User-tabulated n, k | any | For measured or ellipsometry data, or reactive or interdiffused compounds |

- Interpolate f1, f2 on a log-log scale (or linearly in log E) between table points. Absorption edges need dense sampling.
- Cache per (material, energy grid). Optical constants depend only on E, not on θ.
- Thin films are often less dense than bulk (e.g. sputtered B₄C, C). Density must therefore be a user parameter that can optionally be fitted.

**Critical angle** of total external reflection: θ_c ≈ √(2δ). Rule of thumb (assuming Z/A ≈ 0.5): θ_c[mrad] ≈ 16·√ρ[g/cm³]·λ[nm]. For example, Si at 8 keV gives ≈ 3.8 mrad.

---

## 2. Bragg's law and its refraction correction

Kinematic Bragg condition for a periodic ML with period d (bilayer thickness):

    m·λ = 2·d·sin θ                    (m = diffraction order)

Refraction inside the stack shifts the peak to slightly larger angles (or energies):

    m·λ = 2·d·sin θ · √(1 − 2δ̄ / sin²θ)  ≈  2·d·sin θ · (1 − δ̄ / sin²θ)
    δ̄ = Γ·δ_H + (1 − Γ)·δ_L            (period-averaged decrement)

The correction is largest at small θ and low E. It is negligible at large angles and high energies.
This is why a first design stage can use plain Bragg's law: it is cheap and analytic. The refined
stages then use the full Fresnel or Parratt calculation.

Useful inversions for the design stages:
- fixed θ, target E-band [E1, E2] → d range [d(E2), d(E1)]
- fixed E, target θ-band [θ1, θ2] → d range [d(θ2), d(θ1)]

---

## 3. The Γ (gamma) ratio

    Γ = t_H / d       (thickness of the high-Z "absorber" layer / period)

- Optimum Γ for the first order (Vinogradov & Zeldovich 1977):

      tan(πΓ) = π·[Γ + β_L / (β_H − β_L)]

  The optimum is typically Γ ≈ 0.3–0.45, and smaller for strongly absorbing H materials.
- Γ = 1/m suppresses the m-th order (Γ = 0.5 suppresses all even orders).
- A minimum layer thickness bounds Γ: Γ·d ≥ t_min,H and (1−Γ)·d ≥ t_min,L. For small d this
  narrows or even closes the allowed Γ range. This is the main interaction between manufacturing
  constraints and the design space.
- The "rough design-space scan" stage scans mainly over Γ, N, and the d-spacing parameters.

---

## 4. Periodic multilayers: scaling estimates

- The peak reflectivity rises with the number of bilayers N until absorption saturates it.
  N_sat scales roughly like 1/(extinction per period), i.e. it depends on β_H, β_L, Γ, and θ.
  Bilayers beyond about 2·N_sat add nothing but time and stress.
- Relative bandwidth: Δλ/λ = Δθ/tanθ ≈ 1/(m·N) while not saturated. With saturation the
  bandwidth becomes ≈ 1/(m·N_eff).
- These estimates bound the N search range before any expensive calculation.

---

## 5. Depth-graded multilayers (supermirrors)

Broadband reflectivity comes from varying d with depth. Large d at the top reflects low energies
or small angles. Small d deeper down reflects high energies or large angles, which penetrate deeper.

**Power law** (Joensen et al. 1993; Christensen; Windt, used e.g. for NuSTAR):

    d_i = a / (b + i)^c,     i = 1 … N  (i = 1 is the top bilayer)

- c is the power-law exponent (NuSTAR-type designs use c ≈ 0.2–0.25). It is a fixed-or-ranged
  parameter. The concept's example "fix the power-law coefficient to 1" maps to fixing c here.
- The user-facing parametrization is (d_min, d_max, c, N). From d_1 = d_max and d_N = d_min:

      b = (N·k − 1)/(1 − k)  with  k = (d_min/d_max)^(1/c),   a = d_max·(b + 1)^c

- Γ can be constant or graded, e.g. linear between Γ_top and Γ_bottom, or as its own power law.
  Grading Γ helps because absorption matters more for the deep layers.
- **Cap layer pair:** an optional top bilayer with its own d_cap and Γ_cap. It is often thicker,
  to reflect the lowest energies and to protect the stack. It also serves as an oxidation barrier.
- Other grading strategies for later: block or stepped designs (Yamashita), and free-form per-layer
  optimization starting from a power law.

---

## 6. Reflectivity calculation (dynamical theory)

Stack: ambient (j = 0), layers j = 1 … L with thicknesses t_j, substrate (j = L+1).

Perpendicular wavevector in layer j (with the ambient as reference, n_0 = 1 for vacuum):

    k_z,j = k·√(n_j² − cos²θ)          choose the branch with Im(k_z,j) ≥ 0

**Fresnel coefficients** at interface j | j+1:

    s-pol:  r_j = (k_z,j − k_z,j+1) / (k_z,j + k_z,j+1)
    p-pol:  r_j = (n_{j+1}²·k_z,j − n_j²·k_z,j+1) / (n_{j+1}²·k_z,j + n_j²·k_z,j+1)

**Parratt recursion** (Parratt 1954), from the substrate upward:

    X_{L+1} = 0
    X_j = (r_j + X_{j+1}·φ_{j+1}) / (1 + r_j·X_{j+1}·φ_{j+1}),    φ_{j+1} = exp(2i·k_z,j+1·t_{j+1})
    R = |X_0|²

- Parratt (or an equivalent normalized-ratio formulation) is numerically stable. The naive
  2×2 characteristic-matrix product (Abelès) can overflow or lose precision for thick,
  absorbing stacks or evanescent waves. It remains useful as an independent cross-check for small stacks.
- Transmission (needed for energy-conservation tests and for field intensities) follows from the
  amplitude recursion or from the matrix formalism.
- Unpolarized light: R = ½(R_s + R_p). At hard X-ray grazing angles R_s ≈ R_p. In EUV and soft X-rays
  near 45° (Brewster angle ~45° for n ≈ 1), R_p → 0.
- **Vectorization:** evaluate all (E, θ) points at once as numpy complex128 arrays and loop over
  layers in Python. For N ≤ a few thousand this is fast enough. Optional numba/compiled kernels can
  be added later behind the same signature.

---

## 7. Interface imperfections

Real interfaces have roughness and interdiffusion with width σ (typically 0.2–0.5 nm).

- **Névot–Croce** factor (the standard choice for X-ray MLs):

      r_j → r_j · exp(−2·k_z,j·k_z,j+1·σ_j²)

- **Debye–Waller** factor: r_j → r_j·exp(−2·k_z,j²·σ_j²). Valid for small roughness and close to the
  kinematic limit.
- General interface profile functions (Stearns 1989): error function, linear, exponential,
  sinusoidal. Each gives a different k_z-dependent reduction factor. Alternatively, slice the
  graded profile finely, which is exact but slow.
- σ often depends on the material pair and the direction: "H on L" ≠ "L on H"
  (e.g. Mo-on-Si vs. Si-on-Mo).
- Compound interlayers (e.g. MoSi₂ in Mo/Si, WSi₂ in W/Si) are modelled as extra thin layers. They
  reduce the effective optical contrast and consume deposited thickness.
- Roughness can grow with depth (roughness replication or accumulation). Model this optionally as
  σ_j = f(j).

---

## 8. Instrumental and system effects

- Angular divergence and energy bandwidth: convolve R with a resolution kernel (Gaussian, box).
- Source spectrum × detector efficiency weighting for integrated throughput.
- Multiple reflections (e.g. a double-reflection telescope): throughput ∝ R(θ)².
- Several mirror shells or angles (coupled designs): weighted average over the angles. See the concept's coupled optimization.

---

## 9. Manufacturing aspects

| Aspect | Typical numbers / modelling |
|---|---|
| Minimum continuous layer thickness | ≈ 0.7–2 nm, material dependent (islanding below that) |
| Maximum total thickness | Film stress, delamination, coating time. Several µm at most for graded MLs |
| Deposition rate | Per material, e.g. 0.05–1 nm/s (magnetron sputtering) |
| Per-layer overhead | Pre-sputtering, shutter or target switching, substrate rotation: seconds to minutes per layer |
| Fixed overhead | Pump-down, warm-up, calibration runs |
| Thickness errors | Random per layer (≈1 %), systematic drift, rate calibration error |
| Thickness quantization | Machine resolution of time or speed (e.g. substrate velocity steps) |

Coating time model:

    T = Σ_j t_j / rate(material_j)  +  N_layers · t_overhead,layer  +  t_fixed

Depth-graded designs with many thin layers are dominated by per-layer overhead. This is the reason
for the concept's time penalty in the figure of merit.

**Tolerance analysis:** Monte Carlo perturbation of t_j (random and systematic) shows how robust a
design is. It is an optional criterion in the late stages.

---

## 10. Figures of merit (FOM)

Typical optical criteria over a target band (E or θ):
- mean reflectivity / integrated reflectivity ∫R dE (or dθ)
- minimum reflectivity in the band (flatness). Use a soft-min for differentiability.
- RMS deviation from a target profile R_target(E or θ). This also covers "flat at X %" or "zero
  outside the band" (out-of-band suppression).
- spectrum-weighted throughput ∫R(E)·S(E)·D(E) dE
- peak reflectivity, or reflectivity at specific lines (e.g. Cu Kα)

Combined with manufacturing time (concept requirement, weight w ∈ [0, 1]):

    F = (1 − w)·F̂_optics + w·T / T_ref

F̂_optics must be normalized and oriented (lower = better) so that the two terms are comparable.
T_ref can be the maximum allowed time or the time of the initial design.

---

## 11. Common material systems (presets)

| Pair | Typical use |
|---|---|
| Mo/Si | EUV 13.5 nm, normal incidence (~70 % R) |
| W/Si, W/B₄C | Hard X-ray periodic and depth-graded (telescopes, synchrotron optics) |
| Pt/C, Pt/SiC | Hard X-ray graded MLs (NuSTAR uses Pt/C and W/Si) |
| Ni/C, Ni/B₄C | Tender/hard X-rays, neutrons |
| Cr/Sc | Water window (~3.1 nm) |
| La/B₄C | 6.x nm (B K-edge region) |
| Ru, Rh, Pt, Au, Ir single layers | Grazing incidence total-reflection mirrors |

---

## 12. Key references

- L. G. Parratt, Phys. Rev. 95, 359 (1954): recursion formalism
- L. Névot, P. Croce, Rev. Phys. Appl. 15, 761 (1980): roughness factor
- B. L. Henke, E. M. Gullikson, J. C. Davis, At. Data Nucl. Data Tables 54, 181 (1993): scattering factors (CXRO)
- C. T. Chantler, J. Phys. Chem. Ref. Data 24, 71 (1995) / 29, 597 (2000): FFAST tables
- A. V. Vinogradov, B. Ya. Zeldovich, Appl. Opt. 16, 89 (1977): optimum Γ
- K. D. Joensen et al., Proc. SPIE 1736 (1993); F. E. Christensen et al.: power-law supermirrors
- D. L. Windt, Computers in Physics 12, 360 (1998): IMD software
- D. G. Stearns, J. Appl. Phys. 65, 491 (1989): interface profiles
- E. Spiller, *Soft X-ray Optics*, SPIE Press (1994): textbook covering all of the above
- CXRO online tools: https://henke.lbl.gov/optical_constants/
