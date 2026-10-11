# xstack core library: architecture proposal

Status: proposal (nothing implemented yet). Target: Python 3.12, numpy, lmfit, fully type-hinted.
See `physics_background.md` for the physics and `kernel_verification.md` for the test strategy.

## 1. Design principles

1. **Library first.** Every GUI feature is a library call. The core has **no Qt dependency**.
2. **Layered data flow.** Each layer only knows the one below:

   ```
   ParameterSet ──Coating.build()──▶ Stack ──ReflectivityModel──▶ ReflectivityResult
        ▲                              │                                  │
        │                              ├──▶ DepositionProcess.time()      │
        │                              └──▶ Constraints.evaluate()        ▼
   Optimizer ◀──────────────── DesignProblem.objective() ◀──── FigureOfMerit
   ```
   Optimizers never see layers. The kernel never sees parameters.
3. **Extension through protocols + registries.** Each extension axis is a `typing.Protocol` plus a
   registry: coatings, constraints, figures of merit, optimizers, reflectivity models, and
   optical-constant providers. A new kind is a new class with a decorator. No existing code changes.
4. **Declarative parameters.** Every configurable class describes its parameters with
   `Parameter` objects (name, unit, default, bounds, description). Serialization, GUI forms, and
   lmfit conversion are derived from that description automatically.
5. **Immutable physics, mutable project.** `Material`, `Layer`, `Stack`, and results are frozen
   dataclasses, which makes them cheap to cache and safe to pass to worker threads or processes.
   Only `Project` is mutable, and only through commands (undo/redo, journal).
6. **Staged fidelity.** The same `DesignProblem` can be evaluated with a cheap model (Bragg
   estimate) or the full model (Parratt). A stage selects the model, optimizer, free parameters, and grid density.

## 2. Package layout

```
src/xstack/
  __init__.py
  units.py                  # conversion helpers (keV↔eV, deg/mrad↔rad, Å↔nm, E↔λ)
  plugins.py                # Registry[T], decorators, entry-point loading ("xstack.plugins")
  parameters.py             # Parameter, ParameterSet, lmfit conversion
  materials/
    material.py             # Material
    providers.py            # OpticalConstantsProvider protocol + implementations
    library.py              # preset materials, user material DB (JSON), caching
    data/henke/*.nff        # bundled CXRO tables
  physics/
    kernel.py               # parratt() — the numerical core
    fresnel.py              # interface coefficients, roughness factors
    bragg.py                # analytic estimates (Bragg, Γ_opt, N_sat, bandwidth)
    resolution.py           # convolution with angular / energy resolution
    models.py               # ReflectivityModel protocol: BraggModel, ParrattModel
  structure/
    stack.py                # Layer, Stack, (optional) InterfaceModel per interface
    coating.py              # Coating protocol / base class
    coatings/
      single.py             # SingleLayer
      nlayer.py             # NLayer: N arbitrary materials, no periodicity
      periodic.py           # Periodic: N × (H, L [, interlayers])
      graded.py             # DepthGraded: power-law d and Γ, optional cap pair
      composite.py          # Composite: ordered blocks (cap + graded + adhesion layer…)
  targets.py                # EnergyScan, AngleScan, Target profile, TargetSet
  merit/
    base.py                 # FigureOfMerit protocol, EvaluationContext
    optical.py              # Mean, Integrated, SoftMin, Flatness, ProfileResidual, Peak
    composite.py            # WeightedSum, TimePenalized
  manufacturing/
    process.py              # DepositionProcess (rates, overheads) → time, thickness
    constraints.py          # Constraint protocol + MinLayerThickness, MaxTotalThickness, …
    tolerance.py            # Monte Carlo error analysis
  optim/
    problem.py              # DesignProblem (+ objective, evaluation cache)
    optimizers.py           # Optimizer protocol, LmfitOptimizer, GridScan
    stages.py               # Stage definitions incl. built-in stage types
    pipeline.py             # Pipeline: ordered stages, hand-over of best parameters
    events.py               # ProgressEvent, CancellationToken, IterationLog
  scan.py                   # design-space maps (Γ × N, …)
  project/
    project.py              # Project aggregate (everything the user defined + results)
    serialization.py        # versioned JSON, type tags from registries, migrations
    commands.py             # Command protocol, History (undo/redo)
    journal.py              # Journal (append-only, timestamped entries)
  io/
    recipe.py               # layer table / deposition recipe export (CSV, JSON)
    measured.py             # import measured R(E)/R(θ) for comparison or fitting
tests/ …                    # see kernel_verification.md
```

## 3. Core concepts and interfaces

The sketches below show intent. Names and signatures are not final.

### 3.1 Parameters: fixed or ranged

```python
@dataclass
class Parameter:
    name: str                     # unique within its owner, e.g. "N", "gamma", "d_min"
    value: float
    unit: str = ""                # "nm", "deg", "" …
    vary: bool = False            # False → fixed; True → optimized within [min, max]
    min: float = -inf
    max: float = inf
    expr: str | None = None       # lmfit tie, e.g. "d_max = 2.5 * d_min"
    integer: bool = False         # e.g. N bilayers → rounded / handled by scan stages
    description: str = ""

class ParameterSet(Mapping[str, Parameter]):
    def to_lmfit(self, prefix: str = "") -> lmfit.Parameters: ...
    def update_from_lmfit(self, p: lmfit.Parameters) -> "ParameterSet": ...
```

- This covers the concept's "fix c to 1" (`vary=False`) and "N between X and Y" (`vary=True, min=X, max=Y, integer=True`).
- **Integer parameters** (N, number of materials) don't suit gradient optimizers. Scan stages
  handle them (grid over N), DE can handle them natively (rounding in `build()`), and local stages
  keep them fixed.
- Parameter names are namespaced by owner (`coating.d_min`, `cap.d`, `process.rate_W`) when
  aggregated into the `DesignProblem`.

### 3.2 Materials and optical constants

```python
@dataclass(frozen=True)
class Material:
    name: str                     # "W", "B4C", "MoSi2"
    formula: str                  # parsed into {element: stoichiometry}
    density: float                # g/cm³
    provider: str = "auto"        # which OpticalConstantsProvider to use

class OpticalConstantsProvider(Protocol):
    name: str
    energy_range: tuple[float, float]
    def delta_beta(self, material: Material, energy_ev: np.ndarray) -> np.ndarray: ...  # complex δ+iβ
```

Implementations: `HenkeProvider` (bundled, 10 eV–30 keV), `ChantlerProvider` / `XraylibProvider`
(optional, hard X-ray), and `TabulatedProvider` (user n,k file). An `"auto"` provider picks one by
energy range. A `MaterialLibrary` holds presets and user materials and memoizes (material, energy grid) → n.

### 3.3 Stack: the physical description

```python
@dataclass(frozen=True)
class Layer:
    material: Material
    thickness: float              # nm
    roughness: float = 0.0        # σ of the interface *below* this layer (nm)
    tag: str = ""                 # provenance, e.g. "graded[17].H", "cap.L" — used for plots & recipes

@dataclass(frozen=True)
class Stack:
    ambient: Material             # usually vacuum
    layers: tuple[Layer, ...]     # top → bottom
    substrate: Material
    substrate_roughness: float = 0.0
    interface_model: str = "nevot_croce"
```

`Stack` is the only input to the reflectivity model, the time model, and the constraints. It is
also what gets exported as a deposition recipe.

### 3.4 Coatings: parameterized stack generators

```python
class Coating(Protocol):
    type_name: ClassVar[str]                    # registry key, e.g. "depth_graded"
    def parameters(self) -> ParameterSet: ...   # declarative, defaults + bounds
    def materials(self) -> Mapping[str, Material]: ...  # roles: "H", "L", "cap_H", …
    def build(self, values: Mapping[str, float]) -> tuple[Layer, ...]: ...
```

Built-in coatings:

| Coating | Parameters (examples) |
|---|---|
| `SingleLayer` | t, σ |
| `NLayer` | t_1…t_N, σ_1…σ_N, materials M_1…M_N |
| `Periodic` | d, Γ, N, σ_HL, σ_LH, optional interlayer thicknesses |
| `DepthGraded` | d_min, d_max, c, N, Γ_top, Γ_bottom (Γ grading law), σ's |
| `Composite` | ordered list of child coatings (each with its own namespace) |

A depth-graded ML with a cap pair is
`Composite([Periodic(N=1, name="cap"), DepthGraded(name="graded")])`, or a `cap` option inside
`DepthGraded` as a convenience. Both produce the same `Stack`. `Composite` also covers adhesion,
barrier, and overcoat layers without any new class.

Grading laws (power law, linear, user table) are themselves small registered strategies used by
`DepthGraded`. Adding e.g. a "block design" means registering a new grading law.

### 3.5 Targets and coupled optimization

```python
@dataclass(frozen=True)
class EnergyScan:            # fixed angle, energy range
    angle: float
    energies: np.ndarray     # evaluation grid (density chosen per stage)
    weight: float = 1.0
    profile: TargetProfile | None = None     # desired R(E), default "as high as possible"
    spectrum: Spectrum | None = None         # source × detector weighting
    polarization: Literal["s", "p", "unpolarized"] = "unpolarized"
    resolution: Resolution | None = None

@dataclass(frozen=True)
class AngleScan: ...          # fixed energy, angle range (same fields)

@dataclass(frozen=True)
class TargetSet:
    targets: tuple[EnergyScan | AngleScan, ...]
    combine: Literal["average_reflectivity", "average_merit"] = "average_reflectivity"
```

The single-goal case is a `TargetSet` with one target. The concept's example "fixed energy range,
5 angles with relative weights" is 5 `EnergyScan`s with weights. With `average_reflectivity`, R is
averaged (weighted) before the FOM, as described in the concept. With `average_merit`, the FOM is
computed per target and then averaged.

Grids are defined by range + density. Stages can ask for a coarse or fine version
(`target.with_grid(n_points)`).

### 3.6 Reflectivity models (fidelity levels)

```python
class ReflectivityModel(Protocol):
    name: str
    def reflectivity(self, stack: Stack, target: Target) -> ReflectivityResult: ...

@dataclass(frozen=True)
class ReflectivityResult:
    x: np.ndarray             # energies or angles
    R: np.ndarray             # (and optionally R_s, R_p, T, complex r)
```

- `BraggModel`: analytic, kinematic. It reports which part of the band each bilayer covers (with or
  without refraction correction) and a crude per-bilayer reflectivity estimate. Very cheap. Used for
  the first rough outline and to seed later stages.
- `ParrattModel`: full Fresnel/Parratt with roughness, polarization, and resolution.
- Later options: a compiled/numba Parratt or a GPU kernel, behind the same protocol.

### 3.7 Figures of merit

```python
class FigureOfMerit(Protocol):
    type_name: ClassVar[str]
    def parameters(self) -> ParameterSet: ...               # e.g. weights, softmin temperature
    def residuals(self, ctx: EvaluationContext) -> np.ndarray: ...  # for least squares
    def scalar(self, ctx: EvaluationContext) -> float: ...          # lower = better

@dataclass(frozen=True)
class EvaluationContext:
    stack: Stack
    results: tuple[ReflectivityResult, ...]      # one per target
    targets: TargetSet
    coating_time: float
    total_thickness: float
```

- Optical FOMs: `MeanReflectivity`, `IntegratedReflectivity`, `SoftMinReflectivity`,
  `Flatness`, `ProfileResidual`, `PeakReflectivity`, `SpectrumWeightedThroughput`.
- Combinators: `WeightedSum([(fom, w), …])` and `TimePenalized(fom, w, t_ref)` implementing
  `F = (1−w)·F̂ + w·T/T_ref` (concept requirement).
- The context carries time and thickness, so a FOM can use manufacturing quantities without knowing
  the process model. A default `scalar()` derived from `residuals()` (sum of squares) keeps simple
  FOMs short.
- User-defined FOM: subclass, or wrap a plain callable `(ctx) -> float` with `CallableMerit`. This
  can be serialized by reference (`module:function`).

### 3.8 Manufacturing: process model and constraints

```python
@dataclass(frozen=True)
class DepositionProcess:
    rates: Mapping[str, float]          # material name → nm/s
    overhead_per_layer: float = 0.0     # s, independent of thickness (pre-sputter, shutter …)
    overhead_per_material: Mapping[str, float] = {}   # optional material-specific overhead
    fixed_overhead: float = 0.0         # s per run
    def time(self, stack: Stack) -> float: ...
    def schedule(self, stack: Stack) -> list[DepositionStep]: ...   # for recipe export

class Constraint(Protocol):
    type_name: ClassVar[str]
    severity: Literal["hard", "soft"]
    def parameters(self) -> ParameterSet: ...
    def evaluate(self, stack: Stack, process: DepositionProcess | None) -> ConstraintResult: ...
    def bounds_hint(self, coating: Coating) -> Mapping[str, tuple[float, float]] | None: ...

@dataclass(frozen=True)
class ConstraintResult:
    satisfied: bool
    violation: float                    # ≥ 0, normalized (0 = satisfied)
    message: str                        # "layer 37 (W) is 0.62 nm < 0.8 nm"
    offending_layers: tuple[int, ...] = ()
```

Built-ins: `MinLayerThickness` (global or per material), `MaxTotalThickness`, `MaxLayers`,
`MaxCoatingTime`, `ThicknessQuantization`, `AllowedMaterials`. Later: `MaxStress` (per-material
intrinsic stress × thickness).

**Enforcement paths**:
1. *Bounds / reparameterization* (preferred, exact). `bounds_hint()` lets a constraint tighten
   parameter bounds before optimizing. Example: min thickness with Periodic gives
   Γ ∈ [t_min/d, 1 − t_min/d].
2. *Penalty* (soft, or hard constraints that can't be expressed as bounds):
   objective += λ·violation². λ is configurable per stage.
3. *Rejection* (hard, for global optimizers): return a large constant objective value.

The basic constraints (minimum layer thickness, maximum total thickness) are active from the very
first stage, as required by the concept. Time constraints are added once a `DepositionProcess` is defined.

### 3.9 Design problem, optimizers, stages, pipeline

```python
@dataclass
class DesignProblem:
    coating: Coating
    targets: TargetSet
    merit: FigureOfMerit
    constraints: list[Constraint]
    process: DepositionProcess | None
    model: ReflectivityModel
    def objective(self, params: lmfit.Parameters) -> np.ndarray | float: ...
    def evaluate(self, values: Mapping[str, float]) -> Evaluation: ...   # full diagnostic result

class Optimizer(Protocol):
    type_name: ClassVar[str]
    def parameters(self) -> ParameterSet: ...     # method options (popsize, tol, maxiter …)
    def run(self, problem: DesignProblem, start: ParameterSet,
            events: EventSink, cancel: CancellationToken) -> OptimizationResult: ...
```

- `LmfitOptimizer(method=...)` covers `leastsq`, `least_squares`, `nelder`, `lbfgsb`,
  `powell`, `differential_evolution`, `basinhopping`, `ampgo`, `dual_annealing`, `shgo`, and
  `brute` through `lmfit.Minimizer`. It uses `iter_cb` for progress and cancellation.
  Residual vs. scalar objective is chosen automatically from the method.
- `GridScan`: deterministic sweep over chosen parameters (including integers) that returns a map.
  It backs the "rough scan over Γ".
- Third-party optimizers (e.g. CMA-ES, Optuna) plug in via the registry.

**Stages** bundle *how much effort* is spent:

```python
@dataclass
class Stage:
    name: str
    model: ReflectivityModel           # BraggModel / ParrattModel
    optimizer: Optimizer
    free: set[str] | None              # override which parameters vary (None = as defined)
    grid_points: int                   # target grid density
    merit: FigureOfMerit | None        # override (e.g. no time penalty in early stages)
    constraints: list[Constraint] | None
```

**Default pipeline** (mirrors the concept and can be edited by the user):

| # | Stage | Model | Optimizer | Purpose |
|---|---|---|---|---|
| 1 | Outline | Bragg (no refraction) | analytic | d range from the target band, N bounds from N_sat, Γ_opt; basic constraints |
| 2 | Design-space scan | Bragg + refraction, or coarse Parratt | GridScan over Γ (× N × c) | maps, exclude bad regions, tighten bounds |
| 3 | Global optimization | Parratt, coarse grid | differential evolution | find the best basin with all constraints and the time penalty |
| 4 | Refinement | Parratt, fine grid, roughness and resolution | leastsq / L-BFGS-B / Nelder | polish |
| 5 | (optional) Tolerance | Parratt | Monte Carlo | robustness report |

The `Pipeline` passes each stage's best `ParameterSet` (and tightened bounds) to the next and
records an `OptimizationResult` per stage. Each stage can also be run on its own from the library or the GUI.

**Events and cancellation**: `EventSink` callbacks receive `ProgressEvent(stage, iteration,
best_value, best_params, elapsed)`. `CancellationToken` is checked in `iter_cb`. This is enough for a
GUI progress bar and live plot without any Qt dependency.

**Performance**: memoize optical constants per (material, grid). Cache the last evaluations in
`DesignProblem`. DE uses lmfit `workers` (multiprocessing). Frozen dataclasses make this safe to
pickle.

### 3.10 Project, serialization, commands, journal

```python
class Project:
    materials: MaterialLibrary
    coating: Coating
    targets: TargetSet
    merit: FigureOfMerit
    constraints: list[Constraint]
    process: DepositionProcess | None
    pipeline: Pipeline
    candidates: list[Candidate]        # stored designs (params + stack + results + provenance)
    history: History                   # undo/redo
    journal: Journal                   # human-readable log
```

- **Serialization**: versioned JSON (`"schema_version": 1`). Every registered object serializes
  as `{"type": "<type_name>", "params": {...}}`, so plugins serialize without extra code. Migrations
  per schema version. Numeric arrays are stored as lists, or as base64 for large result arrays.
- **Commands**: every mutation is a `Command` with `do(project)`, `undo(project)`, and `description`.
  Examples: `SetParameter`, `ReplaceCoating`, `AddConstraint`, `RunStage` (its undo removes the
  result), `PromoteCandidate`. `History` holds the undo/redo stacks and merges consecutive edits of
  the same field.
- **Journal**: append-only list of `JournalEntry(timestamp, kind, description, data)`. Every executed
  command and every optimization run (settings, duration, result summary) is logged and saved with
  the project. Undo adds a journal entry and never deletes one.
- Scripted use: `project.execute(SetParameter("coating.N", vary=True, min=20, max=200))`. Direct
  attribute mutation is possible but bypasses history (documented).

### 3.11 Plugins

```python
coatings: Registry[Coating] = Registry("coating")

@coatings.register
class MyBlockCoating(CoatingBase):
    type_name = "block_graded"
    ...
```

There are registries for coatings, grading laws, constraints, merits, optimizers, models, and
providers. Third-party packages are discovered through the entry-point group `xstack.plugins`.
The GUI builds its "Add…" menus and parameter forms from the registries.

## 4. Typical library usage (target API)

```python
import xstack as xs

W, Si = xs.materials.get("W"), xs.materials.get("Si")
coating = xs.coatings.DepthGraded(H=W, L=Si,
    d_min=xs.Param(2.5, vary=True, min=2.0, max=3.5),
    d_max=xs.Param(12.0, vary=True, min=8.0, max=20.0),
    c=xs.Param(0.25, vary=False),
    N=xs.Param(150, vary=True, min=50, max=300, integer=True),
    gamma=xs.Param(0.4, vary=True, min=0.2, max=0.6),
    cap=xs.coatings.CapPair(d=xs.Param(15.0, vary=True, min=8, max=30)))

targets = xs.TargetSet([xs.EnergyScan(angle=xs.deg(0.2), energy=(15e3, 70e3))])
process = xs.DepositionProcess(rates={"W": 0.1, "Si": 0.2}, overhead_per_layer=30)
problem = xs.DesignProblem(coating, targets,
    merit=xs.merit.TimePenalized(xs.merit.MeanReflectivity(), w=0.2, t_ref=36_000),
    constraints=[xs.MinLayerThickness(0.8), xs.MaxTotalThickness(1500), xs.MaxCoatingTime(48*3600)],
    process=process)

result = xs.Pipeline.default().run(problem, progress=print)
xs.io.export_recipe(result.best.stack, process, "recipe.csv")
```

## 5. Implementation order

1. `units`, `materials` (Henke provider, formula parser), `physics.kernel` + verification tests
2. `structure` (Stack, Periodic, DepthGraded, Composite, NLayer, SingleLayer)
3. `targets`, `physics.models`, `merit`
4. `manufacturing` (process, constraints)
5. `optim` (problem, lmfit wrapper, GridScan, stages, pipeline) and `scan`
6. `project` (serialization, commands, journal), `io`
7. GUI (see `gui_design.md`)

Tooling: `pyproject.toml` (src layout, hatchling), ruff, mypy (strict for core), pytest,
hypothesis, pytest-benchmark. Optional extras: `xstack[gui]` (PySide6, pyqtgraph) and
`xstack[hard]` (xraylib).
