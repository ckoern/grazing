# xstack core library: architecture

Status: implemented. Target: Python 3.12, numpy, lmfit, fully type-hinted (`mypy --strict` for the core).
See `physics_background.md` for the physics and `kernel_verification.md` for the test strategy.

## 1. Design principles

1. **Library first.** Every GUI feature is a library call. The core has **no Qt dependency**.
2. **Layered data flow.** Each layer only knows the one below:

   ```
   Mirror (Coating + substrate) ──Mirror.stack()──▶ Stack ──ReflectivityModel──▶ ReflectionResult
        ▲                              │                                  │
        │                              ├──▶ DepositionProcess.time()      │
        │                              └──▶ Constraint.evaluate()         ▼
   Optimizer ◀──────────────── DesignProblem.evaluate() ◀──── FigureOfMerit
   ```
   Optimizers never see layers. The kernel never sees parameters.
3. **Extension through protocols + registries.** Each extension axis has a base class or protocol and a
   registry: coatings, grading laws, constraints, figures of merit, optimizers, reflectivity models,
   interface roughness models, and optical-constant providers. A kind is added by registering a class with a
   decorator (`register_coating`, `register_constraint`, `register_merit`, `register_optimizer`) or a
   `Registry.register` call.
4. **Declarative parameters.** Every configurable class describes its parameters with `Parameter` objects
   (name, value, unit, bounds, description). Serialization, GUI forms, and lmfit conversion are derived
   from that description.
5. **Immutable physics, mutable project.** `Material`, `Layer`, `Stack`, `Mirror`, targets, constraints,
   and results are frozen dataclasses, which makes them cheap to cache and safe to pass to worker threads
   or processes. Only `Project` is mutable, and only through commands (undo/redo, journal).
6. **Staged fidelity.** The same `DesignProblem` can be evaluated with a cheap model (`KinematicModel`)
   or the full model (`ParrattModel`). A pipeline stage selects the model, optimizer, free parameters, and
   grid density.

## 2. Package layout

Modules under `src/xstack/` (all files; `__pycache__` directories omitted):

```
src/xstack/
  __init__.py               # package version
  __main__.py               # command-line interface
  units.py                  # conversion helpers (keV↔eV, deg/mrad↔rad, Å↔nm, µm, E↔λ)
  plugins.py                # Registry[T]: named registry with register/get/names
  parameters.py             # Parameter, Range, free(), ParameterSet, lmfit conversion
  materials/
    __init__.py             # get(name) for preset materials
    material.py             # Material, VACUUM
    providers.py            # OpticalConstantsProvider protocol, HenkeProvider, XraylibProvider,
                            #   CombinedProvider (switches providers at an energy)
    library.py              # MaterialLibrary, default_library(), presets
    data/henke/             # bundled CXRO (Henke) tables
  physics/
    kernel.py               # reflectance(): Parratt recursion on arrays
    fresnel.py              # InterfaceModel protocol; SharpInterface, NevotCroce, DebyeWaller
    bragg.py                # analytic estimates for periodic and depth-graded multilayers
    resolution.py           # convolution with instrumental resolution
    reflectivity.py         # reflectivity of a Stack on arbitrary energies and angles
    models.py               # ReflectivityModel protocol: ParrattModel, KinematicModel
  structure/
    stack.py                # Layer, Stack
    coating.py              # Coating base class, Mirror, register_coating
    coatings/
      __init__.py
      bilayer.py            # shared bilayer roles, parameters and layer construction
      simple.py             # SingleLayer, NLayer
      periodic.py           # Periodic: N × (H, L [, interlayers])
      graded.py             # DepthGraded, grading laws (power_law, linear)
      composite.py          # Composite: ordered blocks with name prefixes
  targets.py                # Target, EnergyScan, AngleScan, Profile, TargetSet
  merit/
    __init__.py
    base.py                 # FigureOfMerit, CurveMerit, EvaluationContext, Curve, CallableMerit
    optical.py              # MeanReflectivity, SoftMinReflectivity, ProfileDeviation, Flatness,
                            #   PeakReflectivity
    composite.py            # WeightedSum, TimePenalized
  manufacturing/
    __init__.py
    process.py              # DepositionProcess, DepositionStep
    constraints.py          # Constraint base class, ConstraintResult, MinLayerThickness,
                            #   MaxTotalThickness, MaxLayers, MaxCoatingTime
    tolerance.py            # Monte Carlo thickness-error analysis
  optim/
    __init__.py
    problem.py              # DesignProblem, Evaluation
    optimizers.py           # Optimizer base class, LmfitOptimizer, OptimizationResult
    scan.py                 # GridScan, scan(), ScanResult
    stages.py               # Stage, BraggOutline
    pipeline.py             # Pipeline, PipelineResult, StageResult
    events.py               # ProgressEvent, CancellationToken, EventSink
  project/
    __init__.py
    project.py              # ProjectState, Candidate, Command, History, Project
    commands.py             # command functions (set_mirror, store_results, run_pipeline, …)
    journal.py              # Journal, JournalEntry
    serialization.py        # versioned JSON: to_data, from_data, register_serializable
    templates.py            # ready-made project states
  io/
    __init__.py
    recipe.py               # layer_table, write_recipe_csv, write_recipe_json
  gui/                      # PySide6 application (not imported by the core)
    __main__.py, app.py, main_window.py, controller.py, forms.py, plots.py, preview.py,
    style.py, workers.py
    steps/                  # one page per workflow step (goal, materials, coating, constraints,
                            #   process, scan, optimize, analyze), base.py
    widgets/                # parameter_table.py, journal_view.py
```

## 3. Core concepts and interfaces

The sketches below show the implemented names and the main fields. Signatures are abbreviated.

### 3.1 Parameters: fixed or ranged

```python
@dataclass(frozen=True)
class Parameter:
    name: str                     # unique within its owner, e.g. "N", "gamma", "cap.d"
    value: float
    unit: str = ""                # "nm", "deg", "" …
    vary: bool = False            # False → fixed; True → optimized within [min, max]
    min: float = -inf
    max: float = inf
    expr: str | None = None       # tie to other parameters, e.g. "2*d_min"
    integer: bool = False         # e.g. number of bilayers
    description: str = ""

class ParameterSet(Mapping[str, Parameter]):
    def value_dict(self) -> dict[str, float]: ...
    def with_values(self, values: Mapping[str, float]) -> "ParameterSet": ...
    def to_lmfit(self, *, expression_bounds: bool = True) -> lmfit.Parameters: ...
    def update_from_lmfit(self, params: lmfit.Parameters) -> "ParameterSet": ...
```

- `free(value, min, max)` returns a `Range`. Passed as a value to a coating factory, it makes the
  parameter variable within the bounds. A plain number makes it fixed.
- `Parameter.is_free` is true when `vary` is set and there is no expression.
- **Integer parameters** (N, number of materials) are varied by global optimizers, which round them when
  building the coating. Local optimizers keep them fixed (`prepare_parameters`). Grid scans round them too.
- Parameter names are namespaced by owner (`cap.d`, `cap.gamma`) when a coating is composed.

### 3.2 Materials and optical constants

```python
@dataclass(frozen=True)
class Material:
    name: str                     # "W", "B4C", "MoSi2"
    formula: str                  # parsed into element composition
    density: float                # g/cm³
    provider: str | None = None   # optional OpticalConstantsProvider name

class OpticalConstantsProvider(Protocol):
    name: str
    energy_range: tuple[float, float]           # eV
    def delta_beta(self, material: Material, energy: NDArray[np.float64]) -> NDArray[np.complex128]: ...
```

Implementations: `HenkeProvider` (bundled tables, per-element scattering factors),
`XraylibProvider`, and `CombinedProvider`, which uses one provider below a switching energy and another
one above. `ScatteringFactorProvider` is the base class for providers built on per-element scattering
factors. `MaterialLibrary` holds the presets and user materials and selects the provider per material
(`provider_for`). `default_library()` returns the library used when none is given. `VACUUM` is the default ambient.

### 3.3 Stack: the physical description

```python
@dataclass(frozen=True)
class Layer:
    material: Material
    thickness: float              # nm
    roughness: float = 0.0        # σ of the interface at the top of this layer, nm
    tag: str = ""                 # provenance, e.g. "graded[17].H" — used for plots and recipes

@dataclass(frozen=True)
class Stack:
    layers: tuple[Layer, ...]     # top → bottom
    substrate: Material
    substrate_roughness: float = 0.0
    ambient: Material = VACUUM
```

`Stack` is the input to the reflectivity models and the constraints. It is also what the deposition
recipe is written from. Interface roughness is applied by the reflectivity model (see 3.6).

### 3.4 Coatings and the mirror

```python
class Coating(ABC):
    type_name: ClassVar[str]                    # registry key, e.g. "depth_graded"
    @property
    def parameters(self) -> ParameterSet: ...   # defaults, bounds, descriptions
    @property
    def materials(self) -> dict[str, Material]: ...     # roles: "H", "L", "interlayer_H_on_L", …
    def layers(self, values: Mapping[str, float] | None = None) -> tuple[Layer, ...]: ...
    def with_parameter(self, name: str, **changes: Any) -> "Coating": ...
    def with_parameters(self, parameters: ParameterSet) -> "Coating": ...

@dataclass(frozen=True)
class Mirror:
    coating: Coating
    substrate: Material
    substrate_roughness: float = 0.0
    ambient: Material = VACUUM
    def stack(self, values: Mapping[str, float] | None = None) -> Stack: ...
```

`Mirror` combines the coating, the substrate, the substrate roughness and the ambient medium. Its
`stack()` calls `coating.layers(values)` and builds the `Stack`. `values` overrides parameter values by
name without changing the coating.

Built-in coatings (`create` classmethods take `ParameterValue`s, i.e. numbers, `free(...)` ranges or
`Parameter`s):

| Coating | Parameters (examples) |
|---|---|
| `SingleLayer.create(material, t, sigma)` | `t`, `sigma` |
| `NLayer.create(materials, thicknesses, roughnesses)` | `t_1…t_N`, `sigma_1…sigma_N`; one material per layer |
| `Periodic.create(H, L, N, d, gamma, sigma_H, sigma_L, …)` | `N`, `d`, `gamma` (Γ = t_H/d), `sigma_H`, `sigma_L`, interlayer thicknesses |
| `DepthGraded.create(H, L, N, gamma_top, gamma_bottom, law, cap, …, **law_parameters)` | `N`, grading-law parameters (`d_min`, `d_max`, `c` for `power_law`), `gamma_top`, `gamma_bottom`, `cap_d`, `cap_gamma` (if `cap`), `sigma_H`, `sigma_L` |
| `Composite.create(blocks)` | the parameters of each block, prefixed with the block name, e.g. `cap.d` |

Bilayer roles are `H` (absorber), `L` (spacer), and the optional interlayers `interlayer_H_on_L` (where H
is deposited on L; thickness `t_interlayer_H_on_L`) and `interlayer_L_on_H` (where L is deposited on H;
thickness `t_interlayer_L_on_H`). Interlayers count towards the period:
d = t_H + t_L + t_interlayer_H_on_L + t_interlayer_L_on_H. Each interlayer replaces the roughness of the
interface it sits in. The bilayer module enforces bounds: thickness and period ≤ 100 nm, roughness ≤ 5 nm,
N ≤ 1000.

Grading laws (`grading_laws` registry: `power_law`, `linear`) define the period of each bilayer of a
depth-graded coating. Γ is interpolated linearly from `gamma_top` to `gamma_bottom`.

Composite coatings cover adhesion, barrier, and overcoat layers without a dedicated class.
`DepthGraded` with `cap=True` adds a cap bilayer with its own period and Γ.

### 3.5 Targets

```python
@dataclass(frozen=True)
class EnergyScan(Target):        # fixed angle, energy range
    angle: float                 # grazing angle, rad
    energy_min: float            # eV
    energy_max: float            # eV

@dataclass(frozen=True)
class AngleScan(Target):         # fixed energy, angle range
    energy: float                # eV
    angle_min: float             # rad
    angle_max: float             # rad

# Common fields of Target (keyword arguments of the subclasses):
#   weight: float = 1.0
#   points: int = 200                       # evaluation grid points
#   polarization: "s" | "p" | "unpolarized" = "unpolarized"
#   profile: Profile | None                 # desired R(x), default "as high as possible"
#   spectrum: Profile | None                # source × detector weighting
#   resolution: float = 0.0                 # instrumental resolution
#   reflections: int = 1                    # number of reflections n
#   name: str = ""

class TargetSet:
    targets: tuple[Target, ...]
    combine: Literal["average_reflectivity", "average_merit"] = "average_reflectivity"
```

- `Target.reflections` is n: the target reflectivity is Rⁿ (for example 2 for a double reflection). The
  plots show the same n in the axis label.
- The single-goal case is a `TargetSet` with one target. Several targets combine as follows.
  With `average_reflectivity`, the weighted reflectivity curves are averaged and the figure of merit is
  evaluated once on the average; all targets must share the same axis, range and points. With
  `average_merit`, the figure of merit is computed per target and the results are averaged with the
  weights.
- `Target.with_points(n)` returns the target on a different grid density; stages use it for coarse grids.

### 3.6 Reflectivity models (fidelity levels)

```python
class ReflectivityModel(Protocol):
    name: str
    def evaluate(self, stack: Stack, target: Target, points: int | None = None,
                 library: MaterialLibrary | None = None) -> ReflectivityResult: ...

@dataclass(frozen=True)
class ReflectivityResult:
    target: Target
    x: NDArray[np.float64]        # scanned variable: eV or rad
    energy: NDArray[np.float64]   # eV
    theta: NDArray[np.float64]    # rad
    reflectivity: NDArray[np.float64]   # Rⁿ after n = target.reflections reflections
```

- `ParrattModel(interface_model="nevot_croce")`: exact dynamical theory (Parratt recursion) with
  roughness, polarization, and instrumental resolution. It is the default model of a project.
- `KinematicModel(refraction=True, interface_model="nevot_croce")`: single-scattering approximation. It
  sums the amplitudes of all interfaces with their phases and neglects multiple reflections; the
  intensity is tanh²(|r|). With `refraction=False` the waves propagate with the vacuum wavevector and the
  result corresponds to the plain Bragg law. It is much faster than `ParrattModel`.
- `evaluate_on_target()` is the shared driver: it builds the grid of the target, applies the polarization
  and resolution, and returns a `ReflectivityResult`.
- `physics/kernel.py` `reflectance()` implements the Parratt recursion on plain arrays. The stack of M
  media (ambient, M − 2 layers, substrate) is given by the refractive indices `n`, the layer thicknesses
  and the interface roughnesses. The optional `media` index array of shape (M,) assigns each medium to
  one of U distinct columns of `n`, so that quantities depending only on the medium are computed once.
- **Interface roughness models** are objects with `reflection_factor` and `transmission_factor` methods
  (`InterfaceModel` protocol, `physics/fresnel.py`). They are registered in `interface_models`:
  `SharpInterface` (no roughness), `NevotCroce` (default), `DebyeWaller`.
- `physics/bragg.py` provides closed-form estimates (Bragg period, Γ_opt, N_sat, bandwidth) used by the
  outline stage.

### 3.7 Figures of merit

```python
class FigureOfMerit(ABC):
    type_name: ClassVar[str]
    def scalar(self, ctx: EvaluationContext) -> float: ...           # lower = better
    def residuals(self, ctx: EvaluationContext) -> NDArray[np.float64]: ...  # for least squares
    def quantities(self, ctx: EvaluationContext) -> dict[str, float]: ...    # rated values, for display

@dataclass(frozen=True)
class EvaluationContext:
    stack: Stack
    targets: TargetSet
    results: tuple[ReflectivityResult, ...]       # one per target
    coating_time: float | None = None             # s
    # total_thickness: nm (property)
```

- `CurveMerit` is the base for optical figures of merit that work on the reflectivity curves
  (`Curve`, with `curve_value` and `curve_residuals`). Built-in optical figures of merit are
  `MeanReflectivity`, `SoftMinReflectivity`, `ProfileDeviation`, `Flatness`, and `PeakReflectivity`.
- Combinators:
  - `WeightedSum(terms=((fom, a), …))`: Σ a_k·F_k.
  - `TimePenalized(merit, weight, reference_time)`: F = (1 − w)·F̂ + w·T/T_ref, where T is the coating
    time. `weight` is w in [0, 1].
- `CallableMerit` wraps a plain callable `(ctx) -> float`. It serializes by reference (`module:function`).
- The context carries time and thickness, so a figure of merit can use manufacturing quantities without
  depending on the process model.

### 3.8 Manufacturing: process model and constraints

```python
class DepositionProcess:
    def __init__(self, rates: Mapping[str, float] | Iterable[tuple[str, float]],  # nm/s by material
                 overhead_per_layer: float = 0.0,                 # s per layer
                 overhead_per_material: Mapping[str, float] | Iterable[tuple[str, float]] | None = None,
                 fixed_overhead: float = 0.0) -> None: ...        # s per run
    def rate(self, material: str) -> float: ...
    def layer_times(self, stack: Stack) -> NDArray[np.float64]: ...
    def time(self, stack: Stack) -> float: ...           # s
    def schedule(self, stack: Stack) -> list[DepositionStep]: ...   # for recipe export

class Constraint(ABC):
    type_name: ClassVar[str]
    severity: Literal["hard", "soft"] = "hard"           # keyword-only field
    def evaluate(self, stack: Stack, process: DepositionProcess | None = None) -> ConstraintResult: ...
    def bounds_hint(self, coating: Coating) -> dict[str, tuple[float, float]]: ...

@dataclass(frozen=True)
class ConstraintResult:
    satisfied: bool
    violation: float                    # ≥ 0, normalized (0 = satisfied)
    message: str                        # e.g. "layer 37 (W) is 0.62 nm < 0.8 nm"
    offending_layers: tuple[int, ...] = ()
```

Built-in constraints (`constraints` registry): `MinLayerThickness(minimum, materials)` (empty
`materials` means all), `MaxTotalThickness(maximum)`, `MaxLayers(…)`, and `MaxCoatingTime(maximum)`
(seconds, requires a `DepositionProcess`). Layers of zero thickness are ignored by the thickness
constraint.

**Enforcement** in `DesignProblem.evaluate()`:
1. *Bounds.* `bounds_hint()` lets a constraint tighten parameter bounds before optimizing. For example,
   a minimum thickness with `Periodic` restricts Γ to [t_min/d, 1 − t_min/d]. The pipeline applies these
   hints.
2. *Penalties.* The objective is F + λ_soft·Σ v_soft² plus, if any hard constraint is violated,
   λ_hard·(1 + Σ v_hard). The penalty weights are fields of `DesignProblem` (`soft_penalty`,
   `hard_penalty`). `Evaluation.feasible` is true when no constraint is violated.

### 3.9 Design problem, optimizers, stages, pipeline

```python
@dataclass(frozen=True)
class DesignProblem:
    mirror: Mirror
    targets: TargetSet
    merit: FigureOfMerit
    constraints: tuple[Constraint, ...] = ()
    process: DepositionProcess | None = None
    model: ReflectivityModel = ParrattModel()
    points: int | None = None              # grid points per target; None = target defaults
    hard_penalty: float = 10.0             # λ_hard
    soft_penalty: float = 1.0              # λ_soft
    library: MaterialLibrary | None = None # optical constants; None = default library
    @property
    def parameters(self) -> ParameterSet: ...
    def with_parameters(self, parameters: ParameterSet) -> "DesignProblem": ...
    def bounds_hints(self) -> dict[str, tuple[float, float]]: ...
    def evaluate(self, values: Mapping[str, float] | None = None) -> Evaluation: ...   # full diagnostic result
    def residuals(self, values: Mapping[str, float] | None, size: int) -> NDArray[np.float64]: ...

@dataclass(frozen=True)
class Evaluation:
    values: Mapping[str, float]            # evaluated parameter values
    objective: float                       # FOM plus constraint penalties
    merit: float                           # FOM without penalties; NaN if the design cannot be built
    stack: Stack | None                    # None if the design cannot be built
    results: tuple[ReflectivityResult, ...] = ()
    coating_time: float | None = None      # s; None without a process
    constraints: tuple[tuple[Constraint, ConstraintResult], ...] = ()
    error: str | None = None
    # property feasible: no error and all hard constraints satisfied

class Optimizer(ABC):
    type_name: ClassVar[str]
    def run(self, problem: DesignProblem, start: ParameterSet | None = None,
            events: EventSink | None = None, cancel: CancellationToken | None = None) -> OptimizationResult: ...
```

- `LmfitOptimizer(method=…)` wraps `lmfit.Minimizer` for `leastsq`, `least_squares`, `nelder`,
  `lbfgsb`, `powell`, `differential_evolution`, and the other lmfit methods. Global methods
  (`GLOBAL_METHODS`) vary integer parameters, which are rounded when the coating is built; they require
  finite bounds for all free parameters. Local methods keep integer parameters fixed.
- `GridScan` (optim/scan.py) is a deterministic sweep over chosen parameters, including integers. The
  function `scan(problem, axes, start, events, cancel)` evaluates a problem on a regular grid and returns
  a `ScanResult` (`best_values()`, merit and feasibility per grid point). Parameters not in `axes` keep
  their start values.
- `BraggOutline` (optim/stages.py) is the analytic first estimate from the Bragg law. It sets the period
  and bounds from the target band and does not optimize. With several targets it pools them by default;
  `target=i` bases the estimate on the `i`-th target alone (`ValueError` if out of range).
- `Stage(name, optimizer, model=None, points=None, merit=None, free=None)` bundles how much effort a run
  spends. `None` fields keep the problem's settings.
- `Pipeline(stages)`:
  - `run(problem, start=None, events=None, cancel=None, on_stage=None) -> PipelineResult`. Each stage
    starts from the parameters of the stage before it. `PipelineResult` holds `stages` (a `StageResult` each,
    with its `OptimizationResult`), the final `parameters`, the `final` evaluation with the original
    problem, and `cancelled`.
  - `Pipeline.default(problem, *, scan_points=9, coarse_points=60, global_iterations=30, population=10,
    seed=None)`:

    | # | Stage | Model | Optimizer | Purpose |
    |---|---|---|---|---|
    | 1 | `outline` | `KinematicModel(refraction=False)` | `BraggOutline` | period and bounds from the target band |
    | 2 | `scan` (skipped if no free Γ or integer parameters) | coarse grid | `GridScan` over free Γ and integer parameters | map the design space, pick a start |
    | 3 | `global` | coarse grid | differential evolution over all free parameters | find the best basin with constraints and time penalty |
    | 4 | `refine` | full grid | least squares, integers fixed | polish |

    The optional tolerance analysis is separate: `thickness_tolerance(problem, values, random_error,
    systematic_error, samples, seed, cancel)` in `manufacturing/tolerance.py` returns a `ToleranceResult`.

**Events and cancellation.** `ProgressEvent` (optim/events.py) carries `task`, `evaluations`, `total`,
`best_objective` (FOM plus penalties), `current_objective`, `best_values`, `elapsed`, and `best_merit`
(FOM of the best design without penalties; NaN if that design cannot be built). `EventSink` is a callable
receiving events. `CancellationToken` has `cancel()` and `cancelled`; optimizers check it between
evaluations.

**Performance.** `MaterialLibrary` caches computed optical constants (`clear_cache()` empties the cache).
`DesignProblem` and the other value objects are frozen and picklable, so they can be passed to worker
processes.

### 3.10 Project, commands, journal, serialization

```python
@dataclass(frozen=True)
class ProjectState:
    name: str = "untitled"
    mirror: Mirror | None = None       # the input design (start point of the optimization)
    targets: TargetSet | None = None
    merit: FigureOfMerit | None = None
    constraints: tuple[Constraint, ...] = ()
    process: DepositionProcess | None = None
    model: ReflectivityModel = ParrattModel()
    pipeline: Pipeline | None = None
    candidates: tuple[Candidate, ...] = ()
    selected: str | None = None
    notes: str = ""
    def problem(self) -> DesignProblem: ...
    def candidate(self, name: str) -> Candidate: ...
    def selected_candidate(self) -> Candidate | None: ...

@dataclass(frozen=True)
class Candidate:
    name: str
    mirror: Mirror                     # the design (coating with its parameter values, substrate)
    source: str = "manual"             # "manual", "pipeline", or a stage name
    created: str = ""
    objective: float | None = None
    merit: float | None = None
    coating_time: float | None = None
    notes: str = ""

@dataclass(frozen=True)
class Command:
    description: str
    transform: Callable[[ProjectState], ProjectState]
    merge_key: str = ""               # commands with the same non-empty key may be merged in the undo history

class Project:
    def execute(self, command: Command) -> None: ...
    def undo(self) -> None: ...
    def redo(self) -> None: ...
    def subscribe(self, listener: Callable[[ProjectState], None]) -> None: ...
    def save(self, path: Path | str) -> None: ...
    @classmethod
    def load(cls, path: Path | str) -> "Project": ...
```

- **Input design and candidates.** `ProjectState.mirror` is the input design. It is only changed by
  commands such as `set_mirror`, `set_parameter`, and `promote_candidate`. Optimization results never
  change it. They are stored as `Candidate`s by `store_results(problem, [(name, parameters, evaluation,
  source), …])`, which also selects the last one. `store_pipeline_result(problem, result, name,
  stage_evaluations)` stores the final design as `name` and each stage result as `"<name> · <stage>"`.
  `evaluate_stages(problem, result)` evaluates every stage result with the full problem so that they are
  comparable.
- **Running a pipeline.** `run_pipeline(project, pipeline=None, events=None, cancel=None,
  candidate_name="optimized")` runs the pipeline on the project's problem, records the run in the journal,
  and stores the final design as a selected candidate in one undoable step.
- **Commands** are described state transformations: `Command(description, transform, merge_key)` returns
  a `ProjectState` derived from the given one. `commands` provides the constructors: `set_mirror`, `set_coating`,
  `set_parameter`, `set_parameters`, `set_targets`, `set_merit`, `set_process`, `set_model`, `set_pipeline`,
  `add_constraint`, `remove_constraint`, `replace_constraint`, `add_candidate`, `remove_candidate`,
  `rename_candidate`, `select_candidate`, `promote_candidate`, `store_results`, `store_pipeline_result`,
  `rename_project`, `set_notes`. `History` keeps the undo and redo stacks.
- Scripted use: `project.execute(commands.set_parameter("N", vary=True, min=20, max=80))`.
- **Journal**: `Journal` is an append-only list of `JournalEntry(timestamp, kind, description, data)`.
  Executed commands and optimization runs (`record_run`) are logged. Undo adds an entry and does not
  delete one. The journal is saved with the project.
- **Serialization**: versioned JSON (`SCHEMA_VERSION = 1`) through `to_data()` and `from_data()`. Every
  registered class is written with a type tag (`register_serializable(tag, cls)`) and its parameters, and
  plugins serialize without extra code. Callables are stored by reference.

### 3.11 Plugins and registries

```python
class Registry[T]:
    def register(self, name: str, item: T, *, replace: bool = False) -> None: ...
    def get(self, name: str) -> T: ...
    def names(self) -> tuple[str, ...]: ...

coatings: Registry[type[Coating]]           # register_coating
grading_laws: Registry[GradingLaw]
constraints: Registry[type[Constraint]]     # register_constraint
merits: Registry[type[FigureOfMerit]]       # register_merit
optimizers: Registry[type[Optimizer]]       # register_optimizer
models: Registry[ReflectivityModel]
interface_models: Registry[InterfaceModel]
```

A plugin class is registered with its decorator, for example:

```python
@register_coating
@dataclass(frozen=True)
class MyCoating(Coating):
    type_name: ClassVar[str] = "my_coating"
    ...
```

The registries are the single source of names. The serializer and the GUI read the same registries, so
a registered class appears in serialization and in the GUI's type choices without other code.

## 4. Typical library usage

Parameters are given as `free(value, min, max)` to vary them, or as plain numbers to fix them.

```python
import numpy as np

from xstack.io.recipe import write_recipe_csv
from xstack.manufacturing import (
    DepositionProcess, MaxCoatingTime, MaxTotalThickness, MinLayerThickness,
)
from xstack.materials import get
from xstack.merit import MeanReflectivity, TimePenalized
from xstack.optim import Pipeline
from xstack.optim.problem import DesignProblem
from xstack.parameters import free
from xstack.physics.models import ParrattModel
from xstack.project import Project, ProjectState
from xstack.project.commands import run_pipeline
from xstack.structure import Mirror, Periodic
from xstack.targets import EnergyScan, TargetSet
from xstack.units import deg

W, Si = get("W"), get("Si")
coating = Periodic.create(
    W, Si,
    N=free(60, 20, 120),           # integer parameter, varied by global stages
    d=free(4.0, 2.5, 8.0),         # period in nm
    gamma=free(0.4, 0.2, 0.6),     # absorber fraction Γ
    sigma_H=0.2, sigma_L=0.2,      # fixed roughness in nm
)
mirror = Mirror(coating, substrate=Si, substrate_roughness=0.3)

targets = TargetSet(EnergyScan(angle=deg(0.2), energy_min=15e3, energy_max=70e3, points=200))
process = DepositionProcess(rates={"W": 0.1, "Si": 0.2}, overhead_per_layer=30.0)
merit = TimePenalized(MeanReflectivity(), weight=0.2, reference_time=36_000.0)
constraints = (MinLayerThickness(minimum=0.8), MaxTotalThickness(maximum=1500.0),
               MaxCoatingTime(maximum=48 * 3600.0))

problem = DesignProblem(mirror, targets, merit, constraints, process, ParrattModel())

# Direct use of the pipeline (returns a PipelineResult; the input design is not changed).
result = Pipeline.default(problem).run(problem)
print(result.final.merit, result.final.coating_time)

# Stored use: the result becomes a selected candidate of the project.
project = Project(ProjectState(name="W/Si mirror", mirror=mirror, targets=targets, merit=merit,
                               constraints=constraints, process=process))
run_pipeline(project, candidate_name="optimized")
best = project.state.selected_candidate()
assert best is not None
write_recipe_csv("recipe.csv", best.mirror.stack(), process)
```

## 5. Tooling

`pyproject.toml` (src layout, hatchling), ruff (line length 100), mypy (strict for the core), pytest,
hypothesis, pytest-benchmark. xraylib is a core dependency. Optional extra: `xstack[gui]` (PySide6,
pyqtgraph).
