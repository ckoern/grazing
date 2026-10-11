# xstack GUI: design and layout

Status: implemented. Stack: PySide6, pyqtgraph for interactive plots. The GUI is the package `xstack.gui`
(`python -m xstack.gui`).

## 1. Principles

- **Thin GUI.** No physics or optimization logic in the GUI. It edits the core `Project` and displays
  results. Anything the GUI does is a call to the library (`architecture_core.md`).
- **All edits are commands.** Pages never mutate the project directly. They emit `commandRequested`, and
  `ProjectController.execute()` wraps the core `Command` in a `CommandAdapter` (a `QUndoCommand`) and
  pushes it onto a `QUndoStack`. Undo/redo and the history view stay in sync with the core journal.
- **Input design and results are separate.** The Coating page edits the input design. Optimization runs
  and scans never change it; their results are stored as candidates (see section 2.1).
- **Generated forms.** `forms.DataclassForm` builds editors from the dataclass fields of a value (label
  and unit metadata, `Literal` fields as choices). The parameter table (`widgets/parameter_table.py`)
  shows the parameters of the coating.
- **Responsive.** Long calculations run in a worker thread and are cancellable. Results stream in through
  progress events.

## 2. Main window layout

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ File   Edit   Run   View                                                               │
│ [New][Open][Save] │ [Undo][Redo] │ [Run pipeline][Stop]                                │
├───────────────┬────────────────────────────────────────────────┬───────────────────────┤
│ PROJECT       │ [Reflectivity][Layer profile][Design map]      │ PROPERTIES            │
│ Workflow      │ [Convergence][Tolerance]                       │ Periodic (H: W, L: Si)│
│  Goal         │ Plot range: [min] [max] keV  [min] [max] deg   │ ──────────────────── │
│  Materials    │                                 [x] log scale  │ Coating               │
│  Coating   ◀  │                                                │ Help text             │
│  Constraints  │  R                                             │ Coating type [▼]      │
│  Process      │  │      ____                                   │ Parameters table      │
│  Scan         │  │ ____/    \___   input design (solid)        │   …                   │
│  Optimize     │  │/              \  selected / checked: dashed  │                       │
│  Analyze      │  └──────────────────────── E (keV)             │                       │
│ Candidates    │                                                │                       │
│  [x] run 2    │                                                │  [← Back] [Next →]    │
│  [ ] input 1  │                                                │                       │
│ [Store input…] [Use as start] [Delete]                         │                       │
├───────────────┴────────────────────────────────────────────────┴───────────────────────┤
│ [Journal] [History]   14:02 Set parameter N · 14:03 Run pipeline · …                  │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ message                 FOM 0.12 · time 31.4 h · R̄ 0.412 · 180 layers, 1.21 µm · ✓  [▮▮▮] │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

The window has three dock areas around a central tab widget. All docks are `QDockWidget`s, can be moved,
and their geometry and state are stored with `QSettings`. The toolbar holds New, Open, Save, Undo, Redo,
Run pipeline, and Stop. The menus are File (New project from templates, Open, Open recent, Save, Save as,
Quit), Edit (Undo, Redo), Run (Run pipeline, Stop), and View (toggles for the docks).

### 2.1 Left dock: workflow and candidates

- **Workflow list**: the steps Goal, Materials, Coating, Constraints, Process, Scan, Optimize, Analyze.
  Selecting a step shows its page in the right dock. The window opens on Coating.
- **Candidates list**: the stored designs.
  - Click a candidate to select it. The selected candidate is shown and analyzed next to the input design
    (selection is a command, so it is undoable).
  - Check a candidate to overlay it in the reflectivity plot.
  - Double-click to rename it.
  - **Store input…** stores the input design under a name you enter.
  - **Use as start** copies the selected candidate into the input design.
  - **Delete** removes the selected candidate.
  - A details label shows the name, source, figure of merit, objective, coating time, and creation time.
- An optimization run stores its final design as `run N` and each stage result as `run N · <stage>`. The
  final design is selected.

### 2.2 Central tabs

| Tab | Content |
|---|---|
| Reflectivity | Reflectivity R(E) or R(θ) of the input design (solid line), one subplot per target when there are several targets, with the R^n exponent in the axis label for n > 1. Checked and selected candidates are dashed. During a run, the best design so far is drawn live. Plot range fields (energy in keV, angle in deg; empty = target range) and a log-scale switch. The figure of merit always uses the target range. |
| Layer profile | Thickness of each layer by index, bars colored by material. For designs with bilayers, the period d and Γ per bilayer. |
| Design map | Result of the design-space scan: the figure of merit over one or two parameters. Infeasible points are grey. Clicking a cell sets the start values of the input design. |
| Convergence | Two panels for the current or last run: best objective (FOM plus penalties, log scale) and figure of merit of the best design, against evaluations, one trace per stage. |
| Tolerance | Histogram of the figure of merit over the perturbed samples, with the nominal value. |

### 2.3 Right dock: properties and workflow pages

The top of the dock shows a one-line description of the input design (coating type, materials, substrate).
Below it is the page of the selected workflow step. Each page starts with a help text and ends with
**← Back** and **Next: <step> →** buttons (the first page has no Back, the last has no Next).

- **Goal**: table of targets with the columns Scan, Fixed, From, To, Points, Weight, Polarization,
  Reflections. Scan is either *Fixed angle (energy range)* or *Fixed energy (angle range)*. Energies are
  in keV and angles in degrees. Polarization is unpolarized, s, or p. Reflections is n in Rⁿ. Buttons:
  *Add fixed-angle target*, *Add fixed-energy target*, *Remove*. **Combine targets** selects
  *Average reflectivity* (curves averaged, one figure of merit) or *Average figure of merit* (one figure
  of merit per target, averaged); the page explains the choice.
- **Materials**: table of the material of each coating role and of the substrate, with density (editable),
  a *New material…* dialog (name, chemical formula, density), and a plot of δ (solid) and β (dashed) for
  the materials in use.
- **Coating**: coating type (Periodic multilayer, Depth-graded multilayer, Single layer, Layer stack (N
  layers)); top layer of each bilayer (absorber H or spacer L); grading law for depth-graded coatings; a
  cap bilayer with its own period and Γ; substrate roughness. The N-layer stack has a layer table with
  *Add layer* and *Remove layer*. Below is the parameter table: columns Parameter, Value, Unit, Vary, Min,
  Max, Expression, Description. A parameter with *Vary* checked is explored between Min and Max; an
  expression computes it from other parameters.
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
- **Analyze**: design to analyze (input design or selected candidate); random thickness error per layer
  (1σ, relative); rate error per material (systematic); number of samples; random seed (empty = random).
  *Run tolerance analysis* runs in the background and shows the Tolerance tab. *Export layer table (CSV)…*
  and *Export recipe (JSON)…* write the deposition recipe of the analyzed design.

### 2.4 Bottom dock

- **Journal**: list of the journal entries (every executed command, undo and redo, and every optimization
  run with its settings and result summary).
- **History**: `QUndoView` on the undo stack. Clicking an entry jumps to that state.

### 2.5 Status bar, background jobs, live preview

- **Summary**: the status bar shows the figure of merit, coating time, mean reflectivity, number of layers
  and total thickness, and whether the constraints are satisfied (✓ or ✗) for the input design.
- **Live preview**: after each change, the input design and the shown candidates are recalculated with the
  full model in a worker, and the plots update. Only the latest request is kept.
- **Background jobs**: runs (pipeline, stage, scan, tolerance analysis) run one at a time in a worker
  thread. A progress bar in the status bar shows the evaluations of the running job, and the status
  message shows its current best objective and figure of merit. The live best-so-far reflectivity is
  redrawn while the run is active. *Stop* (Run menu, toolbar) cancels the job. The result of a finished run
  is stored as candidates.
- **Undo**: each edit is an undoable step with a description. Consecutive edits of the same parameter, the
  same notes, or the same candidate selection are merged into one step. Undo and redo are in the Edit menu, the toolbar, and the History dock.

## 3. Architecture inside the GUI package

```
src/xstack/gui/
  __main__.py            # python -m xstack.gui
  app.py                 # QApplication setup, main()
  controller.py          # ProjectController (owns Project and QUndoStack), CommandAdapter
  main_window.py         # MainWindow: docks, central tabs, actions, menus, settings
  forms.py               # DataclassForm: editor generated from dataclass fields
  plots.py               # pyqtgraph widgets: ReflectivityPlot, LayerProfilePlot, DesignMapPlot,
                         #   ConvergencePlot, HistogramPlot
  preview.py             # compute_preview, compute_curves: calculations for the live views
  style.py               # application style and help text style
  workers.py             # Task, LatestRunner (latest-request-wins), Job and JobRunner (one at a time)
  steps/                 # one page per workflow step: goal, materials, coating, constraints,
                         #   process, scan, optimize, analyze; base.py (StepPage)
  widgets/               # parameter_table.py (ParameterTableModel, ParameterTable), journal_view.py
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
- The Open recent menu lists the last 8 files.
- The undo stack is not saved. The journal is.
