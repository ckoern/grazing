# xstack GUI: design and layout

Stack: PySide6, pyqtgraph for interactive plots. The GUI is the package `xstack.gui`
(`python -m xstack.gui`).

## 1. Principles

- **Thin GUI.** No physics or optimization logic in the GUI. It edits the core `Project` and displays
  results. Anything the GUI does is a call to the library (`architecture_core.md`).
- **All edits are commands.** Pages never mutate the project directly. They emit `commandRequested`, and
  `ProjectController.execute()` wraps the core `Command` in a `CommandAdapter` (a `QUndoCommand`) and
  pushes it onto a `QUndoStack`. Undo/redo and the history view stay in sync with the core journal.
- **Input design and results are separate.** The Coating page edits the input design. Optimization runs,
  scans and estimates never change it; their results are stored as candidates (see section 2.1).
- **Generated forms.** `forms.DataclassForm` builds editors from the dataclass fields of a value (label
  and unit metadata, `Literal` fields as choices). The parameter table (`widgets/parameter_table.py`)
  shows the parameters of the coating.
- **Responsive.** Long calculations run in a worker thread and are cancellable. Results stream in through
  progress events.

## 2. Main window layout

```
 Menu bar    File   Edit   Run   View
 Toolbar     [New] [Open] [Save] | (undo) (redo) | [Run pipeline] [Stop]

 +----------------+-------------------------------------------------------------------+
 | Candidates     | Design header: project name, input design, substrate              |
 | (left dock)    | [1 Goal][2 Materials][3 Coating][4 Estimate][5 Constraints]       |
 |                | [6 Process][7 Scan][8 Optimize][9 Analyze]                        |
 |  [x] run 2     | +-- help card -----------------------------------------------+    |
 |  [ ] input 1   | | page (scroll area)                                        |    |
 |  details       | |                                                           |    |
 |                | +-----------------------------------------------------------+    |
 | [Store input…] |                                     [← Back] [Next: <step> →]     |
 | [Use as start] | ---- workflow panel (left of the splitter) ----                   |
 | [Delete]       | ---- plot tabs (right of the splitter) ----                       |
 |                | [Reflectivity][Layer profile][Parameters]                         |
 |                | [Design map][Convergence][Tolerance]                              |
 |                |                                                                   |
 +----------------+-------------------------------------------------------------------+
 Bottom docks (tabbed):  [Journal] [History]
 Status bar:  FOM · time · R̄ · layers, thickness · constraints ✓/✗          [progress]
```

The window has a left dock (Candidates), a bottom dock area (Journal and History) and a central
horizontal splitter. The splitter holds the workflow panel on the left and the plot tabs on the right.
All docks are `QDockWidget`s that can be moved; their geometry and state are stored with `QSettings`.
The window opens on the Coating step.

The toolbar holds New, Open, Save, Undo, Redo, Run pipeline and Stop. Undo and Redo are icon-only
buttons; Run pipeline is the highlighted primary action. The toolbar icons are line icons drawn from SVG.
The menus are File (New project from templates, Open, Open recent, Save, Save as, Quit), Edit (Undo,
Redo), Run (Run pipeline, Stop) and View (Theme: Light, Dark, System; toggles for the Candidates,
Journal and History docks).

### 2.1 Left dock: Candidates

The list shows the input design (first entry, in italics) and the stored designs.

- Click the input design entry to deselect all candidates.
- Click a candidate to select it. The selected candidate is shown and analyzed next to the input design
  (selection is a command, so it is undoable).
- Check a candidate to overlay it in the reflectivity plot.
- Double-click to rename it.
- The context menu (right-click) offers *Use as start design*, *Rename*, *Show parameters…*, *Export
  reflectivity…* and *Delete*. *Export reflectivity…* also works for the input design entry. It writes a
  CSV file with the reflectivity of that design over the plot range of every target. The file has one row
  per grid point and the columns `target`, `axis`, `reflections` (N), `energy_keV`, `angle_deg`,
  `reflectivity` (one reflection) and `reflectivity^N` (N reflections).
- **Store input…** stores the input design under a name you enter.
- **Use as start** copies the selected candidate into the input design.
- **Delete** removes the selected candidate.
- A details label below the buttons shows the name, source, figure of merit, objective, coating time and
  creation time of the selected candidate.

An optimization run stores only its final design as `run N`: for a full pipeline run the result of the last
stage, for a single stage run the result of that stage. The stored candidate is selected. In the candidate
list, Shift-click selects a range and Ctrl-click toggles single entries; selecting a candidate keeps the
scroll position and the multi-selection. Estimates are stored as `estimate (plain)` or `estimate (refraction)`.

### 2.2 Workflow panel

The design header shows the project name and a one-line description of the input design (coating type,
materials, substrate). Below it are nine step tabs, `1 Goal` to `9 Analyze`. Each tab shows a help card
with the explanation of the step, the page in a scroll area, and **← Back** and **Next: <step> →**
buttons (the first step has no Back, the last has no Next). The primary action of each step is the
primary button variant.

- **Goal**: table of targets with the columns Name, Scan, Fixed, From, To, Points, Weight,
  Polarization, Reflections. Name is optional; unnamed targets are labeled “target 1”, “target 2”, …
  in plots and exported files. Scan is either *Fixed angle (energy range)* or *Fixed energy (angle range)*. Energies are
  in keV and angles in degrees. Polarization is unpolarized, s, or p. Reflections is n in Rⁿ. Buttons:
  *Add fixed-angle target*, *Add fixed-energy target*, *Remove*. **Combine targets** selects
  *Average reflectivity* (curves averaged, one figure of merit) or *Average figure of merit* (one figure
  of merit per target, averaged); the page explains the choice.
- **Materials**: table of the material of each coating role and of the substrate, with density (editable),
  a dialog to create a material (name, chemical formula, density), and a plot of δ (solid) and β (dashed)
  for the materials in use.
- **Coating**: coating type (Periodic multilayer, Depth-graded multilayer, Single layer, Layer stack (N
  layers)); top layer of each bilayer (absorber H or spacer L); grading law for depth-graded coatings; a
  cap bilayer with its own period and Γ; *Same Γ for all graded bilayers* (ties `gamma_bottom` to
  `gamma_top` by an expression; unchecked, Γ changes linearly with depth); substrate roughness. The N-layer stack has a layer table with
  *Add layer* and *Remove layer*; its parameters are named after the layer materials (`t_Mo`, `sigma_Mo`; `t_Si_1`, `t_Si_2` if a material repeats), and changing a material renames them (expressions between layers follow). Below is the parameter table: columns Parameter, Value, Unit, Vary, Min,
  Max, Expression, Description. A parameter with *Vary* checked is explored between Min and Max; an
  expression computes it from other parameters.
- **Estimate**: analytic start values from the Bragg law for periodic and depth-graded multilayers.
  *Estimate* computes two estimates side by side with the current input values: plain Bragg and
  refraction-corrected (`BraggOutline(refraction=…)`). With several targets, the *Target* selector next to
  the button chooses the target the estimate is based on (`BraggOutline(target=…)`); *All targets* pools
  them. The selector is hidden with a single target. The table lists the free parameters with Input,
  Plain Bragg and With refraction values; the bounds of each estimate are shown as tooltips. Each estimate
  shows its figure of merit and whether it is feasible. **Use estimate** chooses which estimate applies.
  *Use as start values* sets the start values of the input design; *Also narrow the parameter bounds to
  the estimate* sets the bounds to the estimate's bounds, otherwise the bounds are only widened where
  the estimate lies outside them. *Store as candidate* stores the estimate without changing the input
  design. Other coating types show a notice and no estimate.
- **Constraints**: list of constraints with *Add* and *Remove* and a type selector (min layer thickness,
  max total thickness, max layers, max coating time). The selected constraint's settings are edited in a
  generated form. Each constraint shows live whether the input design satisfies it. Hard constraints are
  enforced by the optimizer; soft constraints add a penalty.
- **Process** (optional): enable checkbox; table with deposition rate (nm/s) and extra time per layer (s)
  for each material; overhead per layer (s); fixed overhead per run (h). The page shows the resulting
  coating time.
- **Scan**: first and second free parameter with number of points each, and the number of target grid
  points. *Run scan* evaluates the figure of merit on the grid. The result goes to the Design map tab.
  *Use best grid point* sets the best values as start values. The scan itself optimizes nothing.
- **Optimize**:
  - *Figure of merit*: the optical merit (mean reflectivity, soft-minimum reflectivity, profile
    deviation, flatness, peak reflectivity), with a description of each. Flatness alone does not reward
    high reflectivity, and the page says so.
  - *Combine with second figure of merit*: a second optical merit and a fraction f; the figure of merit
    is (1 − f)·primary + f·secondary.
  - *Penalize coating time* (needs a deposition process): time weight w in [0, 1] and reference time.
    The figure of merit is (1 − w)·F + w·T/T_ref.
  - *Pipeline settings*: random seed, global generations, population factor, coarse grid points, scan
    points per parameter, and *Use the pipeline stored in the project*.
  - *Stages* table: Enabled, Stage (outline, scan, global, refine), Optimizer (Bragg law, grid scan, or
    an lmfit method for the local stages), Max evaluations, Grid points.
  - Buttons *Run pipeline* (all enabled stages in order) and *Run stage* (one enabled stage from the
    current input design).
  - *Last run* table: Stage, Evaluations, Best objective (on the stage's grid), Figure of merit on the
    full problem, Time (s).
- **Analyze**: Monte Carlo tolerance analysis and recipe export.
  - *Design to analyze*: input design or selected candidate.
  - *Error model* with presets (Thickness errors only, Roughness only, Interlayer mixing only, All
    errors, Custom; editing a value switches to Custom). Three sections, each with a checkbox:
    - *Thickness*: random error per layer (%, 1σ), rate error per material (%, systematic), linear rate
      drift over the run (%).
    - *Roughness*: random change per interface (nm), common change of all interfaces (nm).
    - *Interlayer mixing*: mean thickness of the mixed layer at each interface between different
      materials (nm), random variation (%).
  - *Samples* and *Random seed* (empty = random).
  - *Run tolerance analysis* runs in the background and shows the Tolerance tab. The summary gives the
    nominal figure of merit, the median, the 10–90 % range and the worst sample, and the number of feasible
    samples. The figure of merit is evaluated on the physical stack including mixed layers; coating time
    and constraints are evaluated on the stack as deposited.
  - *Export layer table (CSV)…* and *Export recipe (JSON)…* write the deposition recipe of the analyzed
    design.

### 2.3 Plot tabs

| Tab | Content |
|---|---|
| Reflectivity | Reflectivity R(E) or R(θ) of the input design, one panel per scan axis, with the Rⁿ exponent in the axis label for n > 1. Target scan ranges are shaded. Every curve has its own color from the categorical palette in order. Checked and selected candidates are dashed. During a run, the best design so far is drawn live. *Plot range* fields (energy in keV, angle in deg; empty = target range) and a *log scale* switch. The figure of merit always uses the target range. |
| Layer profile | *Show:* input design or selected candidate. Thickness of each layer by index, bars colored by material. For designs with bilayers, the period d and Γ per bilayer. |
| Parameters | Coating parameters of the selected candidate compared with the input design. Columns Parameter, Unit, Input, Selected, Change; free parameters are marked, differing values are bold. A summary gives the figure of merit, objective, coating time and structure of both designs. *Use as start design* sets the input design to the candidate; *Copy selected value to input* sets one parameter of the input design (bounds are widened if needed). |
| Design map | Result of the design-space scan: the figure of merit over one or two parameters. Infeasible points are grey. Clicking a cell sets the start values of the input design. |
| Convergence | Two panels for the current or last run: best objective (FOM plus penalties, log scale), and the quantities the figure of merit rates for the best design (e.g. mean reflectivity; both quantities for a combination of two figures of merit, distinguished by line style; the figure of merit itself if it defines none), against evaluations, one color per stage. |
| Tolerance | Histogram of the figure of merit over all samples, with the nominal value and dashed 10 % and 90 % lines. Below it, the reflectivity of all Monte Carlo samples is overlaid as semi-transparent lines (at most 200), with the nominal curve and a 10–90 % band. |

### 2.4 Bottom docks

- **Journal**: list of the journal entries (every executed command, undo and redo, and every optimization
  run with its settings and result summary).
- **History**: `QUndoView` on the undo stack. Clicking an entry jumps to that state.

### 2.5 Status bar, background jobs, live preview

- **Summary**: the status bar shows the figure of merit, coating time, mean reflectivity, number of layers
  and total thickness, and whether the constraints are satisfied (✓ or ✗) for the input design. A preview
  error is shown with ⚠.
- **Live preview**: after each change, the input design and the shown candidates are recalculated with the
  full model in a worker, and the plots update. Only the latest request is kept.
- **Background jobs**: runs (pipeline, stage, scan, tolerance analysis) run one at a time in a worker
  thread. A progress bar in the status bar shows the evaluations of the running job, and the status
  message shows its current best objective and figure of merit. The live best-so-far reflectivity is
  redrawn while the run is active. *Stop* (Run menu, toolbar) cancels the job. The result of a finished run
  is stored as candidates.
- **Undo**: each edit is an undoable step with a description. Consecutive edits of the same parameter, the
  same notes, or the same candidate selection are merged into one step. Undo and redo are in the Edit menu,
  the toolbar, and the History dock.

### 2.6 Design language

Defined in `xstack.gui.style`.

- **Themes**: `light` and `dark`, plus `system`, which follows the operating system. The choice is set
  under View > Theme and saved in the settings. `apply_style` is called once at start-up by the application
  entry point (`app.py`) and again when the theme is switched.
- **Color tokens**: `THEMES` holds the tokens of each theme; `COLORS` holds the active set. Surfaces are
  neutral (background, surface, border); there is a single indigo accent for primary actions, selected tabs
  and selections.
- **Components**: tabs are pill-shaped; design header, help cards and group boxes are cards with rounded
  corners. Help cards have an accent border on the left. Buttons come in two variants set with
  `set_variant`: `primary` for the main action of an area (Run pipeline, Estimate, Run scan, Run pipeline
  and Run stage, Run tolerance analysis, Next), `secondary` for all others.
- **Icons**: `icon()` draws the line icons (new, open, save, undo, redo, run, stop) as SVG.
- **Plots**: `PLOT_PALETTE` is the categorical palette for curves and markers, the same in both themes.
  `restyle_plots` updates existing pyqtgraph plots after a theme switch; the tolerance view reads the
  colors on every redraw.

## 3. Architecture inside the GUI package

```
src/xstack/gui/
  __init__.py
  __main__.py            # python -m xstack.gui
  app.py                 # QApplication setup, theme from settings, main()
  controller.py          # ProjectController (owns Project and QUndoStack), CommandAdapter
  main_window.py         # MainWindow: docks, splitter, workflow and plot tabs, actions, menus, settings
  forms.py               # DataclassForm: editor generated from dataclass fields
  files.py               # ask_save_path, with_suffix: save dialogs that append the file extension
  plots.py               # pyqtgraph widgets: ReflectivityPlot, LayerProfilePlot, DesignMapPlot,
                         #   ConvergencePlot; scan_axis, bilayer_profile
  preview.py             # compute_preview, compute_curves: calculations for the live views
  style.py               # design language: THEMES, COLORS, PLOT_PALETTE, apply_style, set_variant,
                         #   restyle_plots, icon
  workers.py             # Task, LatestRunner (latest-request-wins), Job and JobRunner (one at a time)
  steps/
    __init__.py          # exports the pages
    base.py              # StepPage: base of all pages, signals and set_state/set_preview
    goal.py              # GoalPage
    materials.py         # MaterialsPage, MaterialDialog
    coating.py           # CoatingPage, convert_coating, has_constant_gamma, set_constant_gamma
    estimate.py          # EstimatePage: Bragg-law estimates (plain and refraction-corrected)
    constraints.py       # ConstraintsPage
    process.py           # ProcessPage
    scan.py              # ScanPage
    optimize.py          # OptimizePage
    analyze.py           # AnalyzePage: error model, tolerance analysis, recipe export
  widgets/
    __init__.py
    design_comparison.py # DesignComparisonView: Parameters tab (selected candidate vs input design)
    journal_view.py      # JournalView
    parameter_table.py   # ParameterTableModel, ParameterTable
    tolerance_view.py    # ToleranceView: Tolerance tab (histogram and sample reflectivity)
```

- `ProjectController` is the single point of truth for the GUI. It emits `stateChanged`, `journalChanged`,
  `fileChanged`, and `errorOccurred`. `MainWindow` subscribes to these and refreshes the pages, the
  candidate list, the plots, and the title.
- Each page is a `StepPage` with the signals `commandRequested` (connected to `controller.execute`),
  `jobRequested` (connected to `MainWindow.start_job`), and `errorOccurred`.
- **Workers**: `Job(name, function, on_finished)` runs in `JobRunner` on a `QThreadPool`. Core
  `ProgressEvent`s are forwarded as Qt signals, and the cancel action sets the core `CancellationToken`.
  `LatestRunner` runs the preview and live curve calculations and drops superseded requests.

## 4. Project files

- Single `.xstack` JSON file written by the core serializer. It contains the project state (input design,
  targets, figure of merit, constraints, process, model, pipeline, candidates, notes) and the journal.
- Opening, saving, and saving as use file dialogs. Closing or opening another project asks to save
  unsaved changes. The window title shows the project name, the file name, and a modified marker
  (`QUndoStack` clean state).
- The project name is edited in the name field above the design summary (applied on Enter or focus loss)
  or with File > Rename project…. Both use the undoable `rename_project` command. Empty names are rejected.
- The Open recent menu lists the last 8 files.
- The undo stack is not saved. The journal is.
