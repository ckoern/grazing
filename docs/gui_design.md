# xstack GUI: design and layout concept

Status: proposal. Stack: PySide6, pyqtgraph for interactive plots, and matplotlib only for figure export.

## 1. Principles

- **Thin GUI.** No physics or optimization logic in the GUI. It edits the core `Project` and
  displays results. Anything the GUI can do, the library can do (`architecture_core.md`).
- **All edits are commands.** Widgets never mutate the project directly. They create core
  `Command`s, wrapped in a `QUndoCommand` adapter and pushed onto a `QUndoStack`. Undo/redo,
  the `QUndoView` history, and the core journal stay in sync automatically.
- **Generated forms.** Parameter editors are built from `Parameter` declarations of the registered
  class (coating, constraint, FOM, optimizer). New plugins appear in the GUI without GUI code.
  Custom hand-written editors can be registered per type where needed.
- **Responsive.** Long calculations run off the UI thread and are cancellable. Results stream in
  through progress events.

## 2. Main window layout

```
┌───────────────────────────────────────────────────────────────────────────────┐
│ File Edit View Run Tools Help                                                 │
│ [New][Open][Save] │ [↶ Undo][↷ Redo] │ [▶ Run stage][▶▶ Run pipeline][■ Stop]  │
│ │ [⧉ Compare]                                                                 │
├────────────────┬──────────────────────────────────────────┬───────────────────┤
│ WORKFLOW       │ ┌Reflectivity┬Layer profile┬Design map┬  │ PROPERTIES        │
│ ① Goal       ✓ │ │Convergence┬Tolerance┐                 │ ───────────────── │
│ ② Materials  ✓ │ │                                       │ DepthGraded       │
│ ③ Coating    ✓ │ │   R                                   │  d_min [2.50] nm  │
│ ④ Constraints! │ │   │   ___________                     │   ☑ vary 2.0–3.5  │
│ ⑤ Process      │ │   │  /           \__  target ┄┄┄      │  d_max [12.0] nm  │
│ ⑥ Scan         │ │   │_/               \_____            │   ☑ vary 8 – 20   │
│ ⑦ Optimize     │ │   └──────────────────────── E (keV)   │  c     [0.25]     │
│ ⑧ Analyze      │ │                                       │   ☐ vary          │
│ ────────────── │ │  overlay: current │ candidate A │ B   │  N     [150]      │
│ CANDIDATES     │ │                                       │   ☑ vary 50 – 300 │
│ ★ current      │ └───────────────────────────────────────┘ │  …                │
│   A  stage 3   │  status: R̄=41.2 %  T=31.4 h  Σt=1.21 µm   │ ───────────────── │
│   B  stage 4   │                                          │ CONSTRAINTS       │
│                │                                          │ ✓ t_min ≥ 0.8 nm  │
│                │                                          │ ✗ T ≤ 30 h (31.4) │
├────────────────┴──────────────────────────────────────────┴───────────────────┤
│ [Journal] [History] [Log] [Progress]                                          │
│ 14:02 Set coating.N range 50–300 · 14:03 Ran stage "Scan" (12 s) · …          │
└───────────────────────────────────────────────────────────────────────────────┘
```

All panels are `QDockWidget`s (rearrangeable, state saved with `QSettings`). The central widget
is the plot tab area.

### 2.1 Left dock: workflow and candidates
- **Workflow stepper** (list with status icons: ✓ complete, ! issue, ○ open), following the concept's stages:
  1. *Goal*: single or coupled targets (energy scan at fixed angle / angle scan at fixed energy;
     table of targets with weights, ranges, target profile, polarization, resolution)
  2. *Materials*: pick from the library, edit density, add custom materials, preview δ/β plots
  3. *Coating*: choose the coating type (registry), set parameters fixed/ranged, cap pair, composite blocks
  4. *Constraints*: minimum layer thickness and maximum total thickness (preset from the start), plus addable constraints
  5. *Process* (optional): deposition rates per material, per-layer and fixed overheads, max time
  6. *Scan*: choose scan axes and resolution, run the design-space scan
  7. *Optimize*: FOM editor (including time-penalty weight slider 0…1), stage/pipeline editor (model, optimizer, free parameters, options)
  8. *Analyze*: tolerance Monte Carlo, layer table, recipe export
  The steps are not strictly sequential. The user can go back at any time, and later results are
  marked "stale" when inputs change.
- **Candidates list**: snapshots of designs (from any stage or saved manually). Actions: show in
  plots (overlay), compare, promote to current, rename, annotate, delete, export.

### 2.2 Central plot tabs
| Tab | Content |
|---|---|
| Reflectivity | R(E) or R(θ) per target (subplots or overlay for coupled targets), target profile and band shading, candidate overlays, s/p/unpolarized toggle, log/linear y |
| Layer profile | thickness vs. layer index (bar chart colored by material), d and Γ vs. bilayer index for graded designs, cumulative deposition time |
| Design map | heatmap of the FOM over the scan axes (e.g. Γ × N). Constraint-violating regions are hatched. Clicking a cell sets the starting parameters (as a command). |
| Convergence | best FOM vs. iteration/evaluation per stage, parameter traces |
| Tolerance | distribution of the FOM under thickness errors, band of R curves |
| Optical constants | δ, β vs. E for the materials in use (from the Materials step) |

A status line under the plots shows the key numbers of the current design: FOM terms, coating
time, total thickness, number of layers, and constraint state.

### 2.3 Right dock: properties and constraint status
- Context-sensitive form for whatever is selected (workflow step, target row, coating block,
  constraint, stage).
- The **parameter editor row** is the central widget of the GUI:
  `name | value (spinbox with unit) | ☐ vary | min | max | (expr)`. Integer parameters use integer
  spinboxes. Invalid bounds are flagged inline.
- **Constraint panel**: live ✓/✗ per constraint with a message (the `ConstraintResult.message`).
  Clicking it highlights the offending layers in the layer-profile plot.

### 2.4 Bottom dock
- **Journal**: chronological, filterable, exportable (Markdown/CSV). Entries link to candidates and results.
- **History**: `QUndoView` on the `QUndoStack`. Clicking an entry jumps to that state.
- **Log**: Python `logging` output.
- **Progress**: running jobs with progress bar, current best FOM, elapsed time, and cancel button.

## 3. Architecture inside the GUI package

```
src/xstack/gui/
  app.py                 # QApplication setup, main()
  main_window.py         # docks, actions, menus, settings
  project_controller.py  # owns Project + QUndoStack; command adapter; emits change signals
  models/                # Qt item models: targets table, constraints list, candidates, journal
  forms/                 # ParameterEditor, generated form builder, per-type custom editors
  steps/                 # one page widget per workflow step
  plots/                 # pyqtgraph widgets per tab; matplotlib export
  workers.py             # QRunnable job wrapper, progress/cancel bridge to core events
```

- `ProjectController` is the single point of truth for the GUI. It emits fine-grained signals
  (`coatingChanged`, `targetsChanged`, `candidateAdded`, `resultReady`, …). Views subscribe to these.
- **Live preview**: on parameter edits, the current design is recalculated with the full model on a
  modest grid (debounced ~150 ms, in a worker) so that the reflectivity plot follows edits.
- **Workers**: `QThreadPool` + `QRunnable` running the core `Pipeline`/`Stage`. Core `ProgressEvent`s
  are forwarded as Qt signals (queued connection), and the cancel button sets the core
  `CancellationToken`. Differential evolution can additionally use process-level parallelism (lmfit `workers`).
- **Commands from results**: finishing a run pushes a `RunStage` command (undoable, journaled)
  that adds the result and candidate.

## 4. Project files
- Single `.xstack` JSON file (core serializer) containing all inputs, candidates, results summary,
  and journal. Large arrays are optionally stored compressed.
- Recent-files list, autosave plus crash-recovery file, dirty indicator in the title bar
  (`QUndoStack.cleanChanged`).
- The undo stack is not persisted. The journal is.

## 5. Later extensions (kept possible by the design)
- Side-by-side compare window for 2–4 candidates.
- Python console dock with `project` in scope (for power users, shows the library-first approach).
- Plugin manager dialog listing registered coatings, constraints, merits, and optimizers.
- Batch mode: run a project file headless from the CLI (`python -m xstack run project.xstack`).
