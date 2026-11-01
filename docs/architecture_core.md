# xstack core library: architecture

Status: implemented. Target: Python 3.12, numpy, lmfit, fully type-hinted (`mypy --strict` for everything except `xstack.gui`).
See `physics_background.md` for the physics and `kernel_verification.md` for the test strategy.

## 1. Design principles

1. **Library first.** Every GUI feature is a library call. Only `xstack.gui` imports Qt.
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
   interface roughness models, optical-constant providers, data parsers, and model readers. A kind is added
   by registering a class with a decorator (`register_coating`, `register_constraint`, `register_merit`,
   `register_optimizer`) or a `Registry.register` call.
4. **Declarative parameters.** Every configurable class describes its parameters with `Parameter` objects
   (name, value, unit, bounds, description). Serialization, GUI forms, and lmfit conversion are derived
   from that description.
5. **Immutable physics, mutable project.** `Material`, `Layer`, `Stack`, `Mirror`, targets, constraints,
   datasets, and results are frozen dataclasses, which makes them cheap to cache and safe to pass to worker
   threads or processes. Only the documents `Project` (design) and `FitProject` (fit) are mutable, and only
   through commands (undo/redo, journal).
6. **Staged fidelity.** The same `DesignProblem` can be evaluated with a cheap model (`KinematicModel`)
   or the full model (`ParrattModel`). A pipeline stage selects the model, optimizer, free parameters, and
   grid density.

## 2. Package layout

Modules under `src/xstack/` (all files; `__pycache__` directories omitted). The layers depend only
downward: `core` ← `design` and `fit` ← `gui`. Design and fit never import each other; they cooperate
through registries filled from entry points (see 3.11).

```
src/xstack/
  __init__.py               # package version
  __main__.py               # command dispatcher: `xstack design ...`, `xstack fit ...`
  core/                     # physics, parameters, serialization (no Qt)
    __init__.py
    units.py                # constants and conversion helpers (keV↔eV, deg/mrad↔rad, Å↔nm, µm, E↔λ, θ↔q)
    plugins.py              # Registry[T]: named registry with register/get/names;
                            #   load_plugins() imports the `xstack.plugins` entry points
    parameters.py           # Parameter, Range, free(), ParameterSet, lmfit conversion
    serialization.py        # versioned JSON: to_data, from_data, register_serializable,
                            #   register_component_registry
    document.py             # Command[S], History[S], Document[S]: undoable, saveable state
    journal.py              # Journal, JournalEntry
    optim/
      __init__.py           # exports Problem, Evaluation, optimizers and events
      problem.py            # Problem[E], Evaluation protocols
      optimizers.py         # Optimizer base class, LmfitOptimizer, OptimizationResult, Tracker,
                            #   prepare_parameters, optimizers registry, register_optimizer
      events.py             # ProgressEvent, CancellationToken, EventSink
    calc.py                 # reflectivity_vs_angle, reflectivity_vs_energy: direct calculations
                            #   for simple scripts
    targets.py              # Target, EnergyScan, AngleScan, Profile, TargetSet
    measurement.py          # MeasuredCurve: measured R(θ) at fixed energy or R(E) at fixed angle,
                            #   with optional uncertainties
    materials/
      __init__.py           # get(name) for preset materials
      material.py           # Material, VACUUM
      providers.py          # OpticalConstantsProvider protocol, ScatteringFactorProvider, HenkeProvider,
                            #   ChantlerProvider, XraylibProvider, CombinedProvider (switches
                            #   providers at an energy),
                            #   TabulatedProvider, providers registry, delta_beta(), refractive_index()
      library.py            # MaterialLibrary, default_library(), provider_library(), PRESETS
      data/henke/           # bundled CXRO (Henke) tables
      data/chantler/        # bundled Chantler (NIST FFAST) tables and their conversion script
    physics/
      kernel.py             # reflectance(): Parratt recursion on arrays
      fresnel.py            # InterfaceModel protocol; SharpInterface, NevotCroce, DebyeWaller
      bragg.py              # analytic estimates for periodic and depth-graded multilayers
      resolution.py         # Gaussian convolution on uniform grids (extended_grid, gaussian_smear)
      reflectivity.py       # reflectivity() and stack_response() of a Stack on arbitrary energies and angles
      models.py             # ReflectivityModel protocol: ParrattModel, KinematicModel, models registry
    structure/
      stack.py              # Layer, Stack
      coating.py            # Coating base class, Mirror, coatings registry, register_coating
      coatings/
        __init__.py
        bilayer.py          # shared bilayer roles, parameters and layer construction
        simple.py           # SingleLayer, NLayer
        periodic.py         # Periodic: N × (H, L [, interlayers])
        graded.py           # DepthGraded, grading laws (PowerLaw, LinearLaw), grading_laws registry
        composite.py        # Composite: ordered blocks with name prefixes
      presets.py            # StructureChoice, build_mirror: starting structures by kind and materials,
                            #   COATING_KINDS, MATERIAL_PAIRS
    io/
      __init__.py           # exports readers, parsers and reflectivity export
      files.py              # FileHandler protocol, handler_for(registry, path): choice by suffix
                            #   (loads plugins first)
      data.py               # DataParser protocol, data_parsers registry, read_curves(path, options)
      delimited.py          # CsvLayout, CsvParser: measured curves from CSV/TSV/column files,
                            #   automatic reading of xstack reflectivity exports
      models.py             # ImportedModel, ModelReader protocol, model_readers registry,
                            #   read_models(path): start models from other applications' files
      reflectivity.py       # reflectivity_table(), write_reflectivity_csv(): export of curves (CSV)
  design/                   # mirror design workflow (depends on core)
    __init__.py
    cli.py                  # `xstack design run|info`
    plugin.py               # XstackModelReader: designs of `.xstack` files as fit models
    recipe.py               # layer_table, write_recipe_csv, write_recipe_json
    merit/
      __init__.py
      base.py               # FigureOfMerit, CurveMerit, EvaluationContext, Curve, CallableMerit,
                            #   merits registry, register_merit
      optical.py            # MeanReflectivity, SoftMinReflectivity, ProfileDeviation, Flatness,
                            #   PeakReflectivity
      composite.py          # WeightedSum, TimePenalized
    manufacturing/
      __init__.py
      process.py            # DepositionProcess, DepositionStep
      constraints.py        # Constraint base class, ConstraintResult, MinLayerThickness,
                            #   MaxTotalThickness, MaxLayers, MaxCoatingTime, constraints registry
      tolerance.py          # ErrorModel, tolerance_analysis(), thickness_tolerance(), ToleranceResult:
                            #   Monte Carlo analysis of manufacturing errors
    optim/
      __init__.py
      problem.py            # DesignProblem, Evaluation, require_design_problem
      scan.py               # GridScan, scan(), ScanResult
      stages.py             # Stage, BraggOutline
      pipeline.py           # Pipeline, PipelineResult, StageResult
      tradeoff.py           # time_tradeoff, TradeoffPoint, TradeoffResult, default_limits
    project/
      __init__.py
      project.py            # ProjectState, Candidate, Command, Project (`.xstack` files)
      commands.py           # command functions (set_mirror, store_results, run_pipeline, …)
      templates.py          # ready-made project states (new_state, TEMPLATES)
  fit/                      # fitting of measured reflectivity (depends on core, not on design)
    __init__.py
    instrument.py           # Instrument (scale, background, offset, resolution, beam width,
                            #   sample length), footprint_factor(), smear(): Gaussian resolution
                            #   at arbitrary points
    dataset.py              # Dataset: curve + instrument + fit ranges/excluded ranges (mask)
    problem.py              # FitProblem (Problem protocol), FitEvaluation, weightings
    results.py              # analyze(): uncertainties and correlations; FitStatistics, FitResult
    project.py              # FitState, StoredFit, FitProject (`.xfit` files), command functions,
                            #   run_fit
    templates.py            # start structures of new fit projects (new_state, TEMPLATES)
    report.py               # text reports of parameters and results
    cli.py                  # `xstack fit new|run|info|export`
  gui/                      # PySide6 applications (not imported by core, design or fit)
    __init__.py
    common/                 # components shared by both applications
      style.py, forms.py, files.py, workers.py, controller.py, pages.py, coating.py,
      materials.py, parameter_table.py, journal_view.py, profile.py, wizard.py, garbage.py,
      plot_tabs.py
    design/                 # the design application (`xstack-design`)
      __main__.py, app.py, main_window.py, controller.py, plots.py, preview.py
      steps/                # one page per workflow step (goal, estimate, materials, coating,
                            #   constraints, process, scan, optimize, tradeoff, analyze), base.py
      widgets/              # candidate_dialog.py, design_comparison.py, journal_view.py,
                            #   parameter_table.py, tolerance_view.py
    fit/                    # the fit application (`xstack-fit`)
      __main__.py, app.py, main_window.py, controller.py, plots.py, preview.py
      steps/                # one page per workflow step (data, materials, structure, parameters,
                            #   fit, results), base.py
      widgets/              # import_dialog.py
```

## 3. Core concepts and interfaces

The sketches below show the implemented names and the main fields. Signatures are abbreviated.

### 3.1 Parameters: fixed or ranged

```python
@dataclass(frozen=True)
class Parameter:
    name: str                     # unique within its owner, e.g. "N", "gamma", "cap.d"
    value: float
    vary: bool = False            # False → fixed; True → optimized within [min, max]
    min: float = -inf
    max: float = inf
    expr: str | None = None       # tie to other parameters, e.g. "2*d_min"
    integer: bool = False         # e.g. number of bilayers
    unit: str = ""                # "nm", "rad", "" …
    description: str = ""

class ParameterSet(Mapping[str, Parameter]):
    def value_dict(self) -> dict[str, float]: ...        # values as stored
    def evaluate(self) -> dict[str, float]: ...          # values with expressions resolved
    def free_names(self) -> tuple[str, ...]: ...
    def merge(self, *others: "ParameterSet") -> "ParameterSet": ...
    def prefixed(self, prefix: str) -> "ParameterSet": ...
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
`ChantlerProvider` (bundled Chantler/NIST FFAST tables, 10.7 eV to 433 keV), `XraylibProvider`, `TabulatedProvider` (δ and β of one material from a user table), and
`CombinedProvider`, which uses one provider below a switching energy and another one above.
`ScatteringFactorProvider` is the base class for providers built on per-element scattering factors; they
include Compton scattering in β unless the `*_photo` variant is chosen. The `providers` registry holds
`henke`, `xraylib`, `chantler`, `henke_xraylib` (Henke up to 30 keV, xraylib above; the default) and
their `_photo` variants. `MaterialLibrary` holds the presets and user materials and selects the provider per material
(`provider_for`); its `delta_beta` and `refractive_index` cache the results. `default_library()` returns
the library used when none is given; `provider_library(name)` returns a shared library with another
default provider. The design and fit project states have the field `optical_constants` (a provider
name, default `henke_xraylib`); their `library()` is `provider_library(optical_constants)`, which their
problems and previews use. `VACUUM` is the default ambient.

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
    densities: ParameterSet = ParameterSet()      # rho_<role>, g/cm³; empty by default
    parameters: ParameterSet                      # coating parameters + densities (property)
    def stack(self, values: Mapping[str, float] | None = None) -> Stack: ...
    def with_parameters(self, parameters: ParameterSet) -> "Mirror": ...
    def with_parameter(self, name: str, **changes: Any) -> "Mirror": ...
    def with_density_parameters(self, roles: Sequence[str] | None = None) -> "Mirror": ...
    def without_density_parameters(self, roles: Sequence[str] | None = None) -> "Mirror": ...
```

`Mirror` combines the coating, the substrate, the substrate roughness and the ambient medium. Its
`stack()` calls `coating.layers(values)` and builds the `Stack`. `values` overrides parameter values by
name without changing the coating.

The density of a material role can be a parameter: `with_density_parameters(roles)` adds
`rho_<role>` (for roles of nested coatings `<block>.rho_<role>`, for the substrate `rho_substrate`),
starting at the material's density, fixed, with bounds 0.5–1.1 times it. In `stack()` a density parameter
replaces the density of that role's material (`Coating.with_materials`, roles from
`Coating.material_map`). `Mirror.parameters` lists the coating parameters followed by the density
parameters, so design and fit problems vary densities like any other parameter; parameters are changed
through `Mirror.with_parameters` and `Mirror.with_parameter`.

Built-in coatings (`create` classmethods take `ParameterValue`s, i.e. numbers, `free(...)` ranges or
`Parameter`s):

| Coating | Parameters (examples) |
|---|---|
| `SingleLayer.create(material, t, sigma)` | `t`, `sigma` |
| `NLayer.create(materials, thicknesses, roughnesses)` | `t_<label>`, `sigma_<label>` per layer, labelled by material (`Mo`, or `Si_1`, `Si_2` if a material repeats) |
| `Periodic.create(H, L, N, d, gamma, sigma_H, sigma_L, …, top)` | `N`, `d`, `gamma` (Γ = t_H/d), `sigma_H`, `sigma_L`, interlayer thicknesses |
| `DepthGraded.create(H, L, N, gamma_top, gamma_bottom, law, cap, …, top, **law_parameters)` | `N`, grading-law parameters (`d_min`, `d_max`, `c` for `power_law`), `gamma_top`, `gamma_bottom`, `cap_d`, `cap_gamma` (if `cap`), `sigma_H`, `sigma_L` |
| `Composite.create(blocks)` | the parameters of each block, prefixed with the block name, e.g. `cap.d` |

`top` (`"H"` or `"L"`) selects the material of the topmost layer of a bilayer coating. Bilayer roles are `H` (absorber), `L` (spacer), and the optional interlayers `interlayer_H_on_L` (where H
is deposited on L; thickness `t_interlayer_H_on_L`) and `interlayer_L_on_H` (where L is deposited on H;
thickness `t_interlayer_L_on_H`). Interlayers count towards the period:
d = t_H + t_L + t_interlayer_H_on_L + t_interlayer_L_on_H. Each interlayer replaces the roughness of the
interface it sits in. The bilayer module enforces bounds: thickness and period ≤ 100 nm, roughness ≤ 5 nm,
N ≤ 1000. Interlayers are optional: the field `use_interlayers` (default `False`, set when an
interlayer material is given) switches them on, and `with_interlayers(enabled)` switches them; off,
the interlayer thicknesses act as zero and `Coating.inactive_parameters()` names them so that
applications can hide them. With interlayers on, a thickness above 0 without a material for its
role raises `CoatingBuildError`; `Coating.without_material(role)` removes an optional material.

Grading laws (`grading_laws` registry: `power_law`, `linear`) define the period of each bilayer of a
depth-graded coating. Γ is interpolated linearly from `gamma_top` to `gamma_bottom`.

Composite coatings (*combined coatings* in the GUI) cover adhesion, barrier, and overcoat layers and
sections of different types (e.g. a depth-graded section on a periodic one) without a dedicated
class. Their material roles carry the block prefix as well (`cap.H`), and expressions may refer to
other blocks. The parameters of the composite are authoritative; `block(name)` returns a block with
them. `with_block`, `insert_block`, `remove_block`, `move_block` and `rename_block` edit the block list
and keep the settings of all other parameters (rebuilding a composite keeps the settings of every
parameter name that still exists); `layers_by_block` returns the layers per block.
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
    def reflectance(self, stack: Stack, energy: NDArray, theta: NDArray,
                    polarization="unpolarized", library=None) -> NDArray: ...

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
- `reflectance(stack, energy, theta, polarization, library)` evaluates a model at individual points
  without resolution; the fitting tool uses it at the measured points.
- `evaluate_on_target()` is the shared driver: it builds the grid of the target, applies the polarization
  and resolution, and returns a `ReflectivityResult`.
- `core/physics/kernel.py` implements the Parratt recursion on plain arrays: `parratt()` returns the
  complex amplitudes `r` and `t` and the intensities, `reflectance()` returns R for `"s"`, `"p"` or
  `"unpolarized"`. The stack of M media (ambient, M − 2 layers, substrate) is given by the refractive
  indices `n`, the layer thicknesses and the interface roughnesses. The optional `media` index array of shape (M,) assigns each medium to
  one of U distinct columns of `n`, so that quantities depending only on the medium are computed once.
- **Interface roughness models** are objects with `reflection_factor` and `transmission_factor` methods
  (`InterfaceModel` protocol, `core/physics/fresnel.py`). They are registered in `interface_models`:
  `SharpInterface` (no roughness), `NevotCroce` (default), `DebyeWaller`.
- `core/physics/reflectivity.py` `reflectivity(stack, energy, theta, polarization, interface_model, library)`
  evaluates a `Stack` on broadcast energy and angle arrays without a target.
- `core/physics/bragg.py` provides closed-form estimates (Bragg angle, wavelength and period, Γ_opt, N_sat,
  bandwidth) used by the outline stage.

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
    def run[E: Evaluation](self, problem: Problem[E], start: ParameterSet | None = None,
            events: EventSink | None = None,
            cancel: CancellationToken | None = None) -> OptimizationResult[E]: ...
```

Optimizers are generic core pieces (`xstack.core.optim`) that run on any `Problem`.
`DesignProblem` satisfies that protocol, and design and fitting use the same optimizers.
`BraggOutline` and `GridScan` accept only a `DesignProblem` and raise `TypeError` otherwise.

- `LmfitOptimizer(method=…)` wraps `lmfit.Minimizer` for `leastsq`, `least_squares`, `nelder`,
  `lbfgsb`, `powell`, `differential_evolution`, and the other lmfit methods. Global methods
  (`GLOBAL_METHODS` in `xstack.core.optim.optimizers`) vary integer parameters, which are rounded
  when the coating is built; they require finite bounds for all free parameters. Local methods keep integer parameters fixed. The start
  design is evaluated first, so the result is never worse than it; differential evolution also includes
  it in its initial population (unless an `init` option is given).
- `GridScan` (design/optim/scan.py) is a deterministic sweep over chosen parameters, including integers. The
  function `scan(problem, axes, start, events, cancel)` evaluates a problem on a regular grid and returns
  a `ScanResult` (`best_values()`, merit and feasibility per grid point). Parameters not in `axes` keep
  their start values.
- `BraggOutline` (design/optim/stages.py) is the analytic first estimate from the Bragg law. It sets the period
  and bounds from the target band and does not optimize. With several targets it pools them by default;
  `target=i` bases the estimate on the `i`-th target alone (`ValueError` if out of range).
- `Stage(name, optimizer, model=None, points=None, merit=None, free=None)` bundles how much effort a run
  spends. `None` fields keep the problem's settings.
- `Pipeline(stages)`:
  - `run(problem, start=None, events=None, cancel=None, on_stage=None) -> PipelineResult`. Every stage
    result is evaluated with the original problem, and each stage starts from the best design so far:
    the result of the stage before it if that improved the objective, otherwise the design that stage
    started from. The final design is the best one, so a run never ends worse than its start design
    (e.g. a Bragg outline that is worse than a good start design is discarded). `PipelineResult` holds `stages` (a `StageResult` each,
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
    systematic_error, samples, seed, cancel)` in `design/manufacturing/tolerance.py` returns a
    `ToleranceResult`. `tolerance_analysis(problem, values, errors, samples, seed, cancel, events)` takes an
    `ErrorModel` that also covers rate drift, roughness variations and mixed interlayers.

**Coating-time trade-off.** `time_tradeoff(problem, limits=None, *, points, lowest, mode, pipeline, refine,
start, events, cancel, on_point)` (design/optim/tradeoff.py) is an ε-constraint sweep: the problem is optimized for a
sequence of maximum coating times, each limit replacing the problem's `MaxCoatingTime` constraints by a hard
constraint. It needs a deposition process (`ValueError` otherwise).
- Without explicit `limits`, the problem is first optimized without a limit (a point with `limit=inf`);
  `default_limits(reference_time, points, lowest)` then gives `points` limits on a log scale from `lowest`
  times that design's coating time up to just below it. Limits are processed from the longest to the shortest.
- `mode="continuation"` runs the full pipeline for the longest limit only. Each shorter limit starts from the
  previous design, scans the free integer parameters on a grid and refines locally (Nelder–Mead by default,
  as the hard time penalty is discontinuous); a limit that ends infeasible falls back to the full pipeline.
  `mode="thorough"` runs the full pipeline for every limit, starting from the previous design.
- Dominance filter: a design found for a shorter limit also satisfies every longer limit. If it has a lower
  objective than the design of a longer limit, it replaces it (`TradeoffPoint.replaced`).
- `TradeoffResult.feasible_points()` (sorted by coating time), `marginal_gains()` (decrease of the figure of
  merit per additional hour between neighbouring feasible points) and `knee()` (the feasible point farthest
  from the line between the shortest and the longest design, with time and merit scaled to [0, 1]; `None` for
  fewer than three feasible points or a nearly straight front). A point is infeasible if its limit is below
  the minimum achievable time, e.g. due to the per-layer overhead of a fixed number of layers.
- The last result is stored in `ProjectState.tradeoff` with the command `store_tradeoff(result)` and is part of
  the project file.

**Events and cancellation.** `ProgressEvent` (core/optim/events.py) carries `task`, `evaluations`, `total`,
`best_objective` (FOM plus penalties), `current_objective`, `best_values`, `elapsed`, and `best_merit`
(FOM of the best design without penalties; NaN if that design cannot be built), and `best_quantities`
(the quantities rated by the figure of merit). `EventSink` is a callable
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
    tradeoff: TradeoffResult | None = None
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
class Command(document.Command[ProjectState]):   # fields from xstack.core.document.Command
    description: str
    transform: Callable[[ProjectState], ProjectState]
    merge_key: str = ""               # commands with the same non-empty key may be merged in the undo history

class Project(Document[ProjectState]):
    def execute(self, command: Command) -> None: ...
    def undo(self) -> None: ...
    def redo(self) -> None: ...
    def subscribe(self, listener: Callable[[ProjectState], None]) -> None: ...
    def save(self, path: Path | str) -> None: ...
    @classmethod
    def load(cls, path: Path | str) -> "Project": ...
```

Documents are generic core pieces: `Document[S]` (`xstack.core.document`) provides execute,
undo/redo, subscribe and save/load for any state type `S`, with `History[S]` as its undo stack.
`Project` subclasses it with the file format `"xstack-project"`, and the design `Command` is a
`Command[ProjectState]`.

- **Input design and candidates.** `ProjectState.mirror` is the input design. It is only changed by
  commands such as `set_mirror`, `set_parameter`, and `promote_candidate`. Optimization results never
  change it. They are stored as `Candidate`s by `store_results(problem, [(name, parameters, evaluation,
  source), …])`, which also selects the last one. `store_pipeline_result(problem, result, name)`
  stores only the final design as `name`; the source is `"pipeline"`. `evaluate_stages(problem, result)` evaluates every stage result with the full
  problem so that they are comparable (used for the results table).
- **Running a pipeline.** `run_pipeline(project, pipeline=None, events=None, cancel=None,
  candidate_name="optimized")` runs the pipeline on the project's problem, records the run in the journal,
  and stores the final design as a selected candidate in one undoable step.
- **Commands** are described state transformations: `Command(description, transform, merge_key)` returns
  a `ProjectState` derived from the given one. `commands` provides the constructors: `set_mirror`, `set_coating`,
  `set_parameter`, `set_parameters`, `set_targets`, `set_merit`, `set_process`, `set_model`, `set_pipeline`,
  `add_constraint`, `remove_constraint`, `replace_constraint`, `add_candidate`, `remove_candidate`,
  `rename_candidate`, `select_candidate`, `promote_candidate`, `store_results`, `store_pipeline_result`,
  `remove_candidates`, `rename_project`, `set_notes`. `History` keeps the undo and redo stacks.
- Scripted use: `project.execute(commands.set_parameter("N", vary=True, min=20, max=80))`.
- **Journal** (`xstack.core.journal`): `Journal` is an append-only list of
  `JournalEntry(timestamp, kind, message, data)`.
  Executed commands and optimization runs (`record_run`) are logged. Undo adds an entry and does not
  delete one. The journal is saved with the project.
- **Serialization**: JSON through `to_data()` and `from_data()` in `xstack.core.serialization`. A document
  file carries its `format` name and `schema_version` (`"xstack-project"` for `.xstack`, `"xstack-fit"`
  for `.xfit`), which are checked when reading. Every registered class is written with a type tag
  (`register_serializable(tag, cls)`) and its parameters, and plugins serialize without extra code.
  Registries of serializable components are attached with `register_component_registry(prefix, registry)`.
  Serializable classes and registries register themselves in the module that defines them. Callables are
  stored by reference.

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
providers: Registry[OpticalConstantsProvider]
model_readers: Registry[ModelReader]        # filled by plugins, e.g. design's `.xstack` reader
data_parsers: Registry[DataParser]          # "csv" built in; further formats from plugins
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

Packages contribute to registries they don't own through entry points in the group `xstack.plugins`.
Each entry point names a module that registers its components when imported:

```toml
[project.entry-points."xstack.plugins"]
design = "xstack.design.plugin"
```

`load_plugins()` imports all of them once; a plugin that fails to load is skipped with a
`PluginWarning`. Lookups in registries filled by plugins (`reader_for(path)`, `parser_for(path)`) call it first. This
is how fit reads `.xstack` designs without importing `xstack.design`, and how third-party packages
add components.

### 3.12 Fitting

A fit compares one structure model (a `Mirror`) with one or more measured curves.

- `MeasuredCurve` (`core/measurement.py`): R(θ) at a fixed energy or R(E) at a fixed angle, with
  optional uncertainties. Read with `read_curves(path, options)`, where `options` is the `CsvLayout` of
  the file (columns, quantity of the x column: θ, 2θ, q or energy, and its unit) or `None` to detect it.
- `Instrument` (`fit/instrument.py`): R_meas(x) = scale · F(θ) · (R ⊛ G_σ)(x + offset) + background with
  Gaussian resolution σ and the footprint factor F of a Gaussian beam (`beam_width`, FWHM in nm) on a
  sample of finite length (`sample_length`, nm). `scale`, `background`, `offset`, `resolution`,
  `beam_width` and `sample_length` are parameters and can be fitted. The convolution at the measured points uses a
  uniform grid (step σ/10) per cluster of points, or Gauss–Hermite quadrature when cheaper.
- `Dataset` (`fit/dataset.py`): a curve, its instrument, fit `ranges` and `excluded` ranges, a weight
  and an enabled flag. The selection is a mask over all measured points, so excluded points stay
  available for plots and the selection can be changed and undone.
- `FitProblem` (`fit/problem.py`) implements the core `Problem` protocol. Its parameters are the coating
  parameters followed by `"<dataset>.<instrument parameter>"` for every dataset; parameters shared
  between datasets use expressions (`expr="a.offset"`). Weightings: `"log"` (log₁₀ residuals),
  `"linear"`, `"chi2"` (residuals over the measured σ). Any core optimizer can fit it.
- `analyze(problem)` (`fit/results.py`) computes uncertainties and correlations (`FitStatistics`) from a
  finite-difference Jacobian at the best fit, C = s²(JᵀJ)⁻¹ with s² = 1 for χ² weighting and the reduced
  objective otherwise. It does not depend on the optimizer. `FitResult` holds the best parameters, the
  statistics and the run information.
- `FitProject` (`fit/project.py`) stores `FitState` in `.xfit` files with undo history and journal.
  The structure is defined in the fit project itself (`fit/templates.py` provides start
  structures); `import_model(path)` optionally reads one through the model-reader registry, e.g. a
  design from an `.xstack` file. `run_fit` runs the `optimizers` of the state as stages, each from
  the best values of the previous one. Every stored result is a `StoredFit` (name, result, fitted
  structure, model, notes) in `FitState.results`; `apply_fit` restores one.

`examples/fit_reflectivity.py` shows a complete fit.

## 4. Typical library usage

Parameters are given as `free(value, min, max)` to vary them, or as plain numbers to fix them.

```python
from xstack.core.materials import get
from xstack.core.parameters import free
from xstack.core.physics.models import ParrattModel
from xstack.core.structure import Mirror, Periodic
from xstack.core.targets import EnergyScan, TargetSet
from xstack.core.units import deg
from xstack.design.manufacturing import (
    DepositionProcess, MaxCoatingTime, MaxTotalThickness, MinLayerThickness,
)
from xstack.design.merit import MeanReflectivity, TimePenalized
from xstack.design.optim import Pipeline
from xstack.design.optim.problem import DesignProblem
from xstack.design.project import Project, ProjectState
from xstack.design.project.commands import run_pipeline
from xstack.design.recipe import write_recipe_csv

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

A fit of a measured curve (`examples/fit_reflectivity.py` shows the complete workflow with fit ranges, a
global search and a project file). Instrument parameters are given to `Dataset.create` like structure
parameters:

```python
from xstack.core.io import CsvLayout, read_curves
from xstack.core.materials import get
from xstack.core.optim import LmfitOptimizer
from xstack.core.parameters import free
from xstack.core.structure import Mirror, Periodic
from xstack.core.units import deg
from xstack.fit.dataset import Dataset
from xstack.fit.problem import FitProblem
from xstack.fit.report import format_result
from xstack.fit.results import FitResult

layout = CsvLayout(x_column="2theta_deg", reflectivity_column="R", uncertainty_column="dR",
                   quantity="two_theta", energy=8048.0)
(curve,) = read_curves("examples/data/WSi_multilayer_cu_ka.csv", layout)

coating = Periodic.create(get("W"), get("Si"), N=20, d=free(4.0, 3.8, 4.2),
                          gamma=free(0.4, 0.25, 0.55),
                          sigma_H=free(0.3, 0.0, 0.8), sigma_L=free(0.3, 0.0, 0.8))
dataset = Dataset.create("cu_ka", curve, scale=free(1.0, 0.5, 1.5), background=free(1e-7, 0.0, 1e-5),
                         offset=free(0.0, -deg(0.02), deg(0.02)), resolution=deg(0.004))
dataset = dataset.with_ranges([(deg(0.4), deg(3.0))], excluded=[(deg(2.05), deg(2.15))])
problem = FitProblem(Mirror(coating, get("Si")), (dataset,), weighting="chi2")

result = FitResult.from_optimization(problem, LmfitOptimizer("least_squares").run(problem))
print(format_result(result))     # best values, uncertainties and strong correlations
```

## 5. Tooling

`pyproject.toml` (src layout, hatchling), ruff (line length 100), mypy (strict except for `xstack.gui`), pytest,
hypothesis, pytest-benchmark. numpy, scipy, lmfit and xraylib are core dependencies. Optional extras:
`xstack[gui]` (PySide6, pyqtgraph, matplotlib) and `xstack[dev]`. The scripts `xstack`, `xstack-design`
and `xstack-fit` start the command line and the two applications.
