# Fitting tool: plan (issue #14)

Goal: a second application, `grazing-fit`, that fits measured reflectivity curves with the
existing physics, structure and parameter machinery. The current GUI becomes `grazing-design`.

Status: phases 1–4 are done (version 0.2.0). Phase 5 is open.

## Decisions
- **Branching:** all work happens on `feature/fitting`. Every phase is a PR into that branch.
  When all phases are done, `feature/fitting` is merged into `main` together with the version
  update to 0.2.0.
- **Package split:** `grazing.core`, `grazing.design`, `grazing.fit`, `grazing.gui.common`,
  `grazing.gui.design`, `grazing.gui.fit`. Old import paths are not kept. Project files keep
  loading, because serialization stores class names, not module paths.
- **Data formats:** CSV only for now. Loaders sit behind a parser registry, so that further
  formats are added by registering a new parser.
- **Fit files:** fit projects use their own file type (`.gfit`), separate from `.grazing` design
  projects. A design file can be imported as the starting model.
- **Independence of design and fit:** `grazing.fit` never imports `grazing.design` (and vice
  versa), so that core + fit can later be installed without design. Optional cooperation goes
  through registries in core that are filled from entry points (group `grazing.plugins`): design
  registers a model reader for `.grazing` files; the data parsers use the same mechanism, so new
  file formats can also come from third-party packages.
- **Fit parameters:** the existing coating parameters are sufficient; no per-layer deviations
  in the first version.

## Phases
Phases 1–4 are implemented.
1. **Package restructure (no behavior change).** Move modules into `core`, `design` and
   `gui.{common,design}`, update imports, docs and tests. Entry points `grazing-design` and
   `grazing-fit`; CLI subcommands grouped under `grazing design …` and `grazing fit …`.
2. **Shared machinery in core.** Optimizers work on a small `Problem` protocol instead of
   `DesignProblem`; undo/redo history and commands become generic so a fit project can reuse
   them.
3. **Fitting core (no GUI).** Measured curves with optional uncertainties and the CSV parser;
   instrument model (scale, background, angle offset, resolution, footprint); fit problem with
   log, linear and χ² weighting; simultaneous fits of several datasets with shared parameters;
   results with uncertainties and correlations; `.gfit` files; `grazing fit` CLI. Registries for
   data parsers and model readers in core, filled from the `grazing.plugins` entry points; design
   registers the `.grazing` model reader. An example script in `examples/` shows a complete fit
   with the library (load data, import or build the model, set up instrument and parameters,
   fit, report and plot the results). Verified by round-trip tests (simulate, add noise, fit,
   recover the parameters).
4. **`grazing-fit` GUI.** Reuses `gui.common`. Steps: Data → Structure → Materials → Parameters →
   Fit → Results, with residual and correlation views. The fit tool works without the design tool:
   coating type and materials are edited in the fit GUI itself (the coating and materials pages
   move to `gui.common`); importing a design from a `.grazing` file is optional. Every fit is
   saved in the project (“fit N”) and can be renamed, deleted and applied again. Fit ranges are
   drawn in the plot (Shift-drag: fit range, Ctrl-drag: excluded range).
5. **Later.** Per-layer deviations, roughness growth with depth, MCMC uncertainties, batch fits,
   further file formats.
