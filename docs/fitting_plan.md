# Fitting tool: plan (issue #14)

Goal: a second application, `xstack-fit`, that fits measured reflectivity curves with the
existing physics, structure and parameter machinery. The current GUI becomes `xstack-design`.

## Decisions
- **Branching:** all work happens on `feature/fitting`. Every phase is a PR into that branch.
  When all phases are done, `feature/fitting` is merged into `main` together with the version
  update to 0.2.0.
- **Package split:** `xstack.core`, `xstack.design`, `xstack.fit`, `xstack.gui.common`,
  `xstack.gui.design`, `xstack.gui.fit`. Old import paths are not kept. Project files keep
  loading, because serialization stores class names, not module paths.
- **Data formats:** CSV only for now. Loaders sit behind a parser registry, so that further
  formats are added by registering a new parser.
- **Fit files:** fit projects use their own file type (`.xfit`), separate from `.xstack` design
  projects. A design file can be imported as the starting model.
- **Independence of design and fit:** `xstack.fit` never imports `xstack.design` (and vice
  versa), so that core + fit can later be installed without design. Optional cooperation goes
  through registries in core that are filled from entry points (group `xstack.plugins`): design
  registers a model reader for `.xstack` files; the data parsers use the same mechanism, so new
  file formats can also come from third-party packages.
- **Fit parameters:** the existing coating parameters are sufficient; no per-layer deviations
  in the first version.

## Phases
1. **Package restructure (no behavior change).** Move modules into `core`, `design` and
   `gui.{common,design}`, update imports, docs and tests. Entry points `xstack-design` and
   `xstack-fit`; CLI subcommands grouped under `xstack design …` and `xstack fit …`.
2. **Shared machinery in core.** Optimizers work on a small `Problem` protocol instead of
   `DesignProblem`; undo/redo history and commands become generic so a fit project can reuse
   them.
3. **Fitting core (no GUI).** Measured curves with optional uncertainties and the CSV parser;
   instrument model (scale, background, angle offset, resolution, footprint); fit problem with
   log, linear and χ² weighting; simultaneous fits of several datasets with shared parameters;
   results with uncertainties and correlations; `.xfit` files; `xstack fit` CLI. Registries for
   data parsers and model readers in core, filled from the `xstack.plugins` entry points; design
   registers the `.xstack` model reader. An example script in `examples/` shows a complete fit
   with the library (load data, import or build the model, set up instrument and parameters,
   fit, report and plot the results). Verified by round-trip tests (simulate, add noise, fit,
   recover the parameters).
4. **`xstack-fit` GUI.** Reuses `gui.common`. Steps: Data → Structure → Materials → Parameters →
   Fit → Results, with residual and correlation views. The fit tool works without the design tool:
   coating type and materials are edited in the fit GUI itself (the coating and materials pages
   move to `gui.common`); importing a design from an `.xstack` file is optional. Every fit is
   saved in the project (“fit N”) and can be renamed, deleted and applied again. Fit ranges are
   drawn in the plot (Shift-drag: fit range, Ctrl-drag: excluded range).
5. **Later.** Per-layer deviations, roughness growth with depth, MCMC uncertainties, batch fits,
   further file formats.
