# Physics background for grazing

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

Photons that are Compton (incoherently) scattered no longer contribute to the specular beam. By default
grazing therefore adds the Compton cross section σ_C of each element to the absorption part,
f2 → f2 + σ_C/(2 r_e λ), as the CXRO index-of-refraction calculator does. This matters mainly for light
elements above ~10 keV. The `*_photo` providers leave it out (photoabsorption only).

**What a material needs:** chemical formula, density, and a data source for f1, f2 per element.

**Data sources**

| Source | Range | Notes |
|---|---|---|
| Henke / CXRO (`*.nff`) | 10 eV – 30 keV | Standard for soft and tender X-rays. Freely available, so it is bundled. |
| Chantler / NIST FFAST | 10.7 eV – 433 keV | Alternative for hard X-ray MLs (provider `chantler`). Calculated for isolated atoms, so less reliable than Henke near soft X-ray and EUV edges. Bundled. |
| xraylib | broad | **Used by grazing for E > 30 keV** (required dependency) |
| User-tabulated δ, β | any | For measured or ellipsometry data, or reactive or interdiffused compounds (`TabulatedProvider`) |

- The default provider uses the Henke tables up to 30 keV and xraylib above. A project can choose another provider
  (Henke, Chantler, xraylib, each also without Compton scattering) for all materials that don't specify their own.
- Between table points, f1 is interpolated linearly in ln E and f2 log-log. Absorption edges need dense sampling.
- Cache per (material, energy grid). Optical constants depend only on E, not on θ.
- Thin films are often less dense than bulk (e.g. sputtered B₄C, C). The density of a material is therefore set by
  the user; the preset materials carry bulk densities. The density of a material role (and of the substrate) can also
  be a parameter, `rho_<role>`, which a fit determines from the measurement (§9) or an optimization varies.

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
- The design-space scan stage scans over the free Γ parameters and the integer parameters (N).

---

## 4. Periodic multilayers: scaling estimates

- The peak reflectivity rises with the number of bilayers N until absorption saturates it.
  N_sat scales roughly like 1/(extinction per period), i.e. it depends on β_H, β_L, Γ, and θ.
  Bilayers beyond about 2·N_sat add nothing but time and stress.
- Estimates used by the analytic outline stage, with index contrast Δn = (δ_H − δ_L) − i(β_H − β_L),
  β̄ = Γ·β_H + (1 − Γ)·β_L and period d:

      N_ext ≈ sin²θ / (|Δn|·|sin(πmΓ)|)          (extinction: interaction with the beam)
      N_abs ≈ λ·sinθ / (8π·β̄·d)                   (absorption: 1/e loss in and out)
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

- Γ is interpolated linearly from Γ_top to Γ_bottom (constant if they are equal). Grading Γ helps
  because absorption matters more for the deep layers.
- Besides the power law, the periods can follow a linear law (d from d_max at the top to d_min at the
  bottom). Further laws are added by registering them.
- **Cap layer pair:** an optional top bilayer with its own d_cap and Γ_cap. It is often thicker,
  to reflect the lowest energies and to protect the stack. It also serves as an oxidation barrier.
- Block or stepped designs (Yamashita) and free-form per-layer optimization are not implemented.

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
- The kernel also returns the transmission amplitude into the substrate and the transmitted energy flux,
  which the energy-conservation tests use.
- Unpolarized light: R = ½(R_s + R_p). At hard X-ray grazing angles R_s ≈ R_p. In EUV and soft X-rays
  near 45° (Brewster angle ~45° for n ≈ 1), R_p → 0.
- **Vectorization:** all (E, θ) points are evaluated at once as numpy complex128 arrays, with a Python
  loop over layers. Quantities that depend only on the medium, interface or layer type (k_z, Fresnel
  coefficients, phase factors) are computed once per distinct one, so stacks with many repeated layers
  are fast.
- **Reflectivity models.** `ParrattModel` evaluates the full recursion. `KinematicModel` is the
  single-scattering approximation: the amplitudes of all interfaces are summed with their phases,
  r = Σ_j r_j·exp(2i·Σ_{i<j} k_z,i·t_i), and multiple reflections are neglected. The intensity is
  taken as tanh²|r|, which equals |r|² for weak reflection and reproduces the peak reflectivity of a
  non-absorbing periodic ML. With `refraction=False` the waves propagate with the vacuum wavevector
  (absorption kept) and the peaks follow the plain Bragg law. Both models take the roughness model
  of §7 and the polarization of §6.

---

## 7. Interface imperfections

Real interfaces have roughness and interdiffusion with width σ (typically 0.2–0.5 nm). The
roughness model is selected by name (`interface_model`) and multiplies the Fresnel coefficients of an
interface between media j and j+1:

| Model | Reflection factor | Transmission factor |
|---|---|---|
| `sharp` | 1 (roughness ignored) | 1 |
| `nevot_croce` (default) | exp(−2·k_z,j·k_z,j+1·σ_j²) | exp((k_z,j − k_z,j+1)²·σ_j²/2) |
| `debye_waller` | exp(−2·k_z,j²·σ_j²) | exp((k_z,j − k_z,j+1)²·σ_j²/2) |

- **Névot–Croce** (Névot and Croce 1980) is the standard choice for X-ray MLs. **Debye–Waller** is valid for
  small roughness and close to the kinematic limit.
- The models assume a Gaussian (error-function) interface profile. Other profile functions (Stearns 1989:
  linear, exponential, sinusoidal) are not implemented. Slicing the graded profile finely would be exact
  but slow.
- σ can depend on the material pair and the direction: "H on L" ≠ "L on H"
  (e.g. Mo-on-Si vs. Si-on-Mo). Bilayer coatings therefore have `sigma_H` and `sigma_L`.
- Compound interlayers (e.g. MoSi₂ in Mo/Si, WSi₂ in W/Si) are modelled as extra thin layers. They
  reduce the effective optical contrast and consume deposited thickness. Each interlayer replaces the
  roughness of the interface it sits in.
- Roughness is a parameter of the coating (per material role). No depth-dependent growth of the
  roughness is modelled.

---

## 8. Instrumental and system effects

- Angular divergence and energy bandwidth: R is convolved with a Gaussian of standard deviation σ
  (rad for angle scans, eV for energy scans), evaluated on a grid finer than σ/3 that extends beyond the
  scanned range.
- Source spectrum × detector efficiency weighting for integrated throughput.
- Multiple reflections (e.g. a double-reflection telescope): throughput ∝ R(θ)².
- Several mirror shells or angles (coupled designs): weighted average over the angles. See the concept's coupled optimization.

---

## 9. Measured reflectivity and fitting

The fitting tool compares the calculated reflectivity R(x) of a structure model with a measured curve
versus the scanned variable x (grazing angle in rad at fixed energy, or energy in eV at fixed angle).
The model is evaluated with a single reflection and the polarization of the curve. A resolution set
in a design target is not used.

**Instrument model.** The measured reflectivity at the nominal value x is

    R_meas(x) = scale · F(θ) · (R ⊛ G_σ)(x + offset) + background

- `scale` and `background` are an intensity factor and a constant background.
- `offset` = true − nominal value of x (rad or eV), for example a misaligned sample angle.
- G_σ is a normalized Gaussian of standard deviation σ = `resolution` (rad or eV). The convolution is
  evaluated at the measured points on a grid of spacing σ/10 per cluster of points, or with
  Gauss–Hermite quadrature if that needs fewer evaluations.
- F(θ) is the footprint factor, the fraction of a Gaussian beam of full width at half maximum
  `beam_width` (perpendicular to the beam) that hits a sample of length `sample_length` along the beam
  (both in nm; reports show mm):

      F(θ) = erf( L·sin θ / (2√2·σ_beam) ),     σ_beam = beam_width / (2√(2 ln 2))

  with the true grazing angle θ (for energy scans the fixed angle). F = 1 if `beam_width` or
  `sample_length` is zero.

All six instrument quantities are parameters of a dataset and can be fixed or fitted, together with the
structure parameters. Parameters shared by several datasets are tied with parameter expressions.

**Film density.** δ and β of a material are proportional to its density ρ, so the density changes the
critical angle (θ_c ∝ √ρ for the top film) and the optical contrast of the layers, i.e. the height of the
Bragg peaks. A density parameter (`rho_<role>`, §1) lets the fit determine it. It is determined best by
data that include the critical-angle region; above it, the density of a multilayer's absorber correlates
strongly with Γ and with the interface roughness, because all of them change the height of the Bragg
peaks.

**Selection of points.** Fit ranges and excluded ranges of x define which measured points take part in a
fit. Excluded points remain in the data and in plots.

**Weighting.** The residual of a fitted point depends on the weighting; the objective is the sum of
the squared residuals, and the residuals of a dataset are multiplied by √(dataset weight):

| Weighting | Residual | Not fitted |
|---|---|---|
| `log` | log₁₀ R_model − log₁₀ R_meas | R_meas ≤ 0 |
| `linear` | R_model − R_meas | |
| `chi2` | (R_model − R_meas)/σ_meas | σ_meas ≤ 0 |

`log` weights all decades of a reflectivity curve equally. `chi2` needs measured uncertainties for every
fitted dataset.

**Uncertainties.** At the best fit the Jacobian J of the residuals with respect to the free parameters
is computed by finite differences. The covariance matrix of the parameters is

    C = s²·(JᵀJ)⁻¹,     s² = 1 for `chi2`, s² = objective/(points − free parameters) otherwise

The standard uncertainties are the square roots of the diagonal of C, and the correlation matrix follows
from C. For `chi2` weighting, objective/(points − free parameters) is the reduced χ². Integer parameters and
parameters with equal bounds are left out. If JᵀJ is singular (parameters the data do not determine) or
there are no degrees of freedom, no uncertainties are given. The analysis does not depend on the
optimizer used to find the minimum.

---

## 10. Manufacturing aspects

| Aspect | Typical numbers / modelling |
|---|---|
| Minimum continuous layer thickness | ≈ 0.7–2 nm, material dependent (islanding below that) |
| Maximum total thickness | Film stress, delamination, coating time. Several µm at most for graded MLs |
| Deposition rate | Per material, e.g. 0.05–1 nm/s (magnetron sputtering) |
| Per-layer overhead | Pre-sputtering, shutter or target switching, substrate rotation: seconds to minutes per layer |
| Fixed overhead | Pump-down, warm-up, calibration runs |
| Thickness errors | Random per layer (≈1 %), systematic drift, rate calibration error |

Coating time model:

    T = t_fixed + Σ_j [ t_j / rate(material_j) + t_overhead,layer + t_overhead(material_j) ]

with the sum over the layers of nonzero thickness.

Depth-graded designs with many thin layers are dominated by per-layer overhead. This is the reason
for the concept's time penalty in the figure of merit.

**Tolerance analysis:** Monte Carlo perturbation shows how robust a design is. The error model covers
random per-layer thickness errors, a rate calibration error per material, a linear rate drift during the
run, changes of the roughness, and mixed interlayers of given thickness at the interfaces between
different materials. The figure of merit is evaluated on the perturbed stack, the coating time and the
constraints on the stack as deposited.

---

## 11. Figures of merit (FOM)

Optical criteria over a target band (E or θ), with the names of the implemented figures of merit.
The value is oriented so that lower is better:
- mean reflectivity, optionally spectrum-weighted (`MeanReflectivity`, value 1 − ⟨R⟩). With a spectrum
  S(E)·D(E) as weight this is the throughput ∫R·S·D dE / ∫S·D dE.
- minimum reflectivity in the band, as a soft-min for differentiability (`SoftMinReflectivity`)
- relative variation of R, std(R)/⟨R⟩ (`Flatness`)
- weighted RMS deviation from a target profile R_target(E or θ), two-sided or only below the profile
  (`ProfileDeviation`). This also covers "flat at X %" or "zero outside the band" (out-of-band
  suppression).
- highest reflectivity in the band (`PeakReflectivity`)

Combined with manufacturing time (concept requirement, weight w ∈ [0, 1]):

    F = (1 − w)·F̂_optics + w·T / T_ref

F̂_optics must be normalized and oriented (lower = better) so that the two terms are comparable.
T_ref can be the maximum allowed time or the time of the initial design.

---

## 12. Common material systems (presets)

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

## 13. Key references

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
