# xstack development milestones

Each milestone ends in a usable, tested state and is merged to `main` on its own. Design
references: `architecture_core.md`, `gui_design.md`, `kernel_verification.md`, `physics_background.md`.
Environment: conda env `xstack` (Python 3.12). Optical constants come from Henke/CXRO for E ≤ 30 keV
and from **xraylib** for E > 30 keV.

"Done when" lists the acceptance criteria for each milestone.

---

## M0: Project scaffolding
- `pyproject.toml` (src layout, hatchling). Extras: `gui` (PySide6, pyqtgraph, matplotlib) and
  `dev` (pytest, hypothesis, mpmath, pytest-benchmark, ruff, mypy).
- `src/xstack/` package skeleton, `tests/`, `.gitignore`, ruff and mypy config (strict for core), README.
- `environment.yml` mirroring the `xstack` conda env.
- `units.py` (eV↔keV, E↔λ, deg/mrad↔rad, Å↔nm) with tests.

**Done when:** `pip install -e .[dev]` works in the env, and `pytest`, `ruff check`, and `mypy` pass on the skeleton.

## M1: Materials and optical constants
- `Material` (formula, density, name), formula parser (stoichiometry, nested groups optional).
- `OpticalConstantsProvider` protocol and implementations:
  - `HenkeProvider`: bundled CXRO `.nff` tables, log-log interpolation, 10 eV–30 keV.
  - `XraylibProvider`: E > 30 keV (via `xraylib.Refractive_Index_Re/Im` or f1/f2 per element).
  - `AutoProvider`: picks by energy and splits a grid that crosses 30 keV.
  - `TabulatedProvider`: user n,k files.
- `MaterialLibrary` with presets (W, Si, B₄C, C, Mo, Pt, Ni, Cr, Sc, La, Au, Ir, Ru, MoSi₂, WSi₂,
  SiO₂, vacuum) and caching per (material, grid).

**Done when:** verification tier 5 passes. δ, β agree with CXRO reference values. Henke and xraylib
agree near 30 keV, with the remaining step documented. Critical angles check out.

## M2: Reflectivity kernel ⟵ *the critical foundation*
- `physics/fresnel.py`: s/p interface coefficients, Névot–Croce and Debye–Waller factors.
- `physics/kernel.py`: vectorized Parratt recursion over (points × layers), returning complex r,
  R_s, R_p, unpolarized R, and T.
- `structure/stack.py`: `Layer` and `Stack`, conversion to kernel arrays.
- `tests/reference/`: independent Abelès matrix and mpmath implementations (test-only).
- First performance benchmark.

**Done when:** verification tiers 1–3 pass, including the stress cases (N = 5000, 100 keV, θ → 0),
and the benchmark baseline is recorded.

## M3: Coatings and analytic physics
- `parameters.py`: `Parameter`/`ParameterSet` (fixed/range, integer, expr), lmfit conversion.
- `plugins.py`: generic `Registry`.
- Coatings: `SingleLayer`, `NLayer`, `Periodic` (incl. interlayers, asymmetric σ),
  `DepthGraded` (power-law d with (d_min, d_max, c, N) parametrization, Γ grading laws, cap pair),
  `Composite`.
- `physics/bragg.py`: Bragg with and without refraction, Γ_opt, N_sat, bandwidth estimates.

**Done when:** verification tier 4 passes (Bragg peaks, Γ = ½ order suppression, bandwidth ∝ 1/N,
graded band coverage), and each coating is checked against an explicitly built stack (tier 2).

## M4: Targets, reflectivity models, figures of merit
- `targets.py`: `EnergyScan`, `AngleScan`, `TargetSet` (weights, two combine modes), target
  profiles, spectrum weighting, grid density control.
- `physics/models.py`: `BraggModel` (fast estimate) and `ParrattModel`. `physics/resolution.py`.
- `merit/`: Mean, Integrated, SoftMin, Flatness, ProfileResidual, Peak, SpectrumWeighted,
  WeightedSum, TimePenalized, CallableMerit.

**Done when:** a script can compute and plot R for a periodic and a graded W/Si design for single
and coupled targets, and FOM values match hand-computed values in tests.

## M5: Manufacturing model and constraints
- `DepositionProcess`: rates, per-layer, per-material, and fixed overheads, `time()` and `schedule()`.
- Constraints: MinLayerThickness (global or per material), MaxTotalThickness, MaxLayers,
  MaxCoatingTime, ThicknessQuantization, with `ConstraintResult` messages and `bounds_hint()`.
- `io/recipe.py`: layer table / deposition recipe export (CSV, JSON).

**Done when:** time and violation tests pass, the bounds hints (e.g. Γ range from t_min) are
verified, and a recipe is exported for a graded design.

## M6: Optimization engine
- `DesignProblem` (objective in residual or scalar form, constraint enforcement: bounds, penalty,
  rejection; evaluation cache).
- `LmfitOptimizer` (all relevant lmfit methods, `iter_cb` progress and cancellation), `GridScan`.
- `events.py`: ProgressEvent, CancellationToken, IterationLog.
- `scan.py`: design-space maps (Γ × N, …) with constraint masks.

**Done when:** a periodic ML is optimized to a known optimum (Γ_opt from theory, peak at the target
energy), DE and local methods both converge, cancellation works, and a scan map is produced.

## M7: Staged pipeline (end-to-end design workflow, library only)
- `Stage` and `Pipeline`, default pipeline: Outline (Bragg) → Scan (Γ/N) → Global (DE, coarse
  Parratt) → Refine (local, fine grid with roughness) → optional Tolerance.
- `manufacturing/tolerance.py`: Monte Carlo thickness errors.
- Hand-over of best parameters and tightened bounds between stages. Per-stage FOM and constraint overrides.
- Example notebooks and scripts: (a) periodic W/B₄C at fixed angle, (b) graded W/Si 15–70 keV with
  time penalty, (c) coupled design with 5 angles.

**Done when:** all three examples run headless from start to finish, the time penalty weight
measurably trades R against time, and the results are physically plausible (reviewed against literature
designs such as NuSTAR-type W/Si).

## M8: Project, persistence, history
- `Project` aggregate, `Candidate` snapshots.
- Versioned JSON serialization driven by the registries (type tags), round-trip tests for every registered type.
- `Command` protocol and `History` (undo/redo, merge of consecutive edits), `Journal`.
- CLI: `python -m xstack run project.xstack` (headless pipeline).

**Done when:** save → load → identical results, undo/redo property tests pass (random command
sequences return to the original state), and the journal is persisted.

## M9: Reference validation campaign
- Generate and store reference datasets (CXRO multilayer calculator, IMD and/or GenX/xrt, refnx)
  with provenance (verification tier 6).
- Literature benchmarks (Mo/Si @ 13.5 nm, W/Si @ 8 keV, graded W/Si).
- Performance tuning if needed (e.g. optional numba kernel behind the same interface, verified
  against the numpy kernel).

**Done when:** all tier 6 comparisons are within the documented tolerances, and the benchmarks meet
targets (to be set in M2).

*M9 can run in parallel with M5–M8. It depends only on M1–M3.*

## M10: GUI foundation
- `gui/app.py`, `main_window.py` with dock layout, `ProjectController` (Project + QUndoStack +
  command adapter), file new/open/save/recent/autosave, undo/redo, journal and history docks.
- Generated `ParameterEditor` forms from the registries.

**Done when:** a project can be created, edited (all edits undoable), saved, and reopened in the GUI.
Smoke tests run with pytest-qt.

## M11: GUI workflow steps and plots
- Workflow stepper pages: Goal (targets table), Materials (δ/β preview), Coating, Constraints, Process.
- pyqtgraph plots: Reflectivity (live preview, debounced worker), Layer profile, Optical constants.
- Live constraint status panel with highlighting of offending layers.

**Done when:** the user can define the full design problem in the GUI and see live R curves and constraint status.

## M12: GUI optimization and analysis
- Scan, Optimize, and Analyze pages: FOM editor (time weight slider), stage/pipeline editor.
- `QThreadPool` workers with progress and cancel, Convergence and Design-map tabs (click to set start parameters).
- Candidates list (overlay, compare, promote, export), Tolerance tab, recipe export.

**Done when:** examples (a)–(c) from M7 can be done entirely in the GUI with results identical to the library.

## M13: Release 1.0
- User documentation (tutorials mirroring the examples, physics notes, API reference via mkdocs or Sphinx).
- Packaging (conda recipe / wheel), CI (tests, lint, type-check, benchmarks).
- Plugin example package (custom coating and FOM via entry points) to prove extensibility.

---

## Dependency overview

```
M0 → M1 → M2 → M3 → M4 → M5 → M6 → M7 → M8 → M10 → M11 → M12 → M13
                 └──────→ M9 (parallel, validation) ─────────────┘
```

## Later (post-1.0) ideas
Free-form per-layer refinement, block/stepped grading, stress model, roughness growth with
depth, fitting measured reflectivity curves (reverse engineering of deposited stacks), GPU kernel,
batch/parameter-study mode.
