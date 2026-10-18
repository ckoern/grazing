# Design a mirror with Python

This page designs the same mirror as [the GUI quick start](quickstart_design_gui.md), now with
the Python library: a periodic W/B4C multilayer that reflects the Cu Kα line (8048 eV) at a fixed
grazing angle of 1.0°. The code is also available as the script `examples/periodic_fixed_angle.py`.
Every snippet on this page runs in order in one Python session; the optimization budgets are small
so that the whole page runs in a few seconds. For a production design, raise them as described in
[Run the pipeline](#run-the-pipeline).

The page covers the object model of a design problem (what each object means), the staged
optimization, the project with its undoable commands, the tolerance analysis and the export of the
deposition recipe.

## The design problem

A {py:class}`~xstack.design.optim.problem.DesignProblem` bundles everything needed to rate a
design: the mirror (coating and substrate), the targets (where it shall reflect), the figure of
merit (how reflectivity is rated), the constraints (what can be manufactured) and, optionally, a
deposition process (how long coating takes). The optimizer only varies the *free* parameters of
the coating.

```python
from xstack.core.materials import get
from xstack.core.parameters import free
from xstack.core.structure import Mirror, Periodic
from xstack.core.targets import EnergyScan, TargetSet
from xstack.core.units import deg
from xstack.design.manufacturing import DepositionProcess, MinLayerThickness
from xstack.design.merit import MeanReflectivity
from xstack.design.optim import DesignProblem

ENERGY = 8048.0  # Cu K-alpha in eV
tungsten, carbide, silicon = get("W"), get("B4C"), get("Si")

coating = Periodic.create(
    tungsten,
    carbide,
    N=free(60, 20, 150),
    d=free(4.4, 2.0, 8.0),
    gamma=free(0.4, 0.1, 0.8),
    sigma_H=0.3,
    sigma_L=0.3,
)
mirror = Mirror(coating, silicon, substrate_roughness=0.3)

targets = TargetSet(
    EnergyScan(
        angle=deg(1.0),
        energy_min=0.99 * ENERGY,
        energy_max=1.01 * ENERGY,
        points=61,
        polarization="s",
    )
)
process = DepositionProcess({"W": 0.1, "B4C": 0.05}, overhead_per_layer=15.0, fixed_overhead=1800.0)
problem = DesignProblem(
    mirror,
    targets,
    MeanReflectivity(),
    constraints=(MinLayerThickness(0.8),),
    process=process,
)
```

The choices and their implications:

- **Parameters.** `free(value, min, max)` makes a parameter free: `value` is the start value, and
  `min` and `max` are the bounds that the optimizer searches within. A plain number fixes the
  parameter, here the roughness `sigma_H` and `sigma_L` of the interfaces (in nm), which you
  would take from a measurement or from {doc}`fitting a reflectivity curve <quickstart_fit_library>`.
  Narrow bounds make the search faster and more reliable; bounds that exclude the optimum make it
  impossible. The number of bilayers `N` is an integer parameter and is searched by the grid-scan and
  global stages. The parameters of {py:class}`~xstack.core.structure.coatings.periodic.Periodic` are the
  number of bilayers `N`, the period `d` (nm) and the absorber fraction
  `gamma` = t<sub>W</sub> / d.
- **Units.** The library works in nm, eV and rad. {py:func}`~xstack.core.units.deg` and
  {py:func}`~xstack.core.units.kev` convert at the boundary.
- **Target.** An {py:class}`~xstack.core.targets.EnergyScan` rates the mirror on an energy range at one
  grazing angle. A ±1 % band is narrow, so the optimizer builds a Bragg peak that covers it.
  `polarization="s"` is appropriate for a horizontally reflecting synchrotron mirror; the
  default is unpolarized.
- **Figure of merit.** {py:class}`~xstack.design.merit.optical.MeanReflectivity` has the value 1 − ⟨R⟩,
  so lower is better and a perfect mirror scores 0. It rewards a high *average* over the whole band:
  a narrow peak that misses part of the band scores worse than a broad, lower one. See the
  [figures of merit](../physics_background.md) for the alternatives.
- **Constraint.** {py:class}`~xstack.design.manufacturing.constraints.MinLayerThickness` of 0.8 nm
  is a hard constraint: designs with thinner layers are infeasible. Layers thinner than about 1 nm
  do not form continuous films, so the optimizer must not use them. The optimizers also use it
  to narrow the bounds of `gamma` for a given period.
- **Process.** The deposition rates (nm/s) and overheads (s) give each design a coating time.
  The process is not needed for the optimization itself, but for the recipe export and
  for time-aware figures of merit (see {doc}`topics/design_depth_graded`).

Evaluate the start design to check the setup:

```python
start = problem.evaluate()
print(f"start: mean R = {1 - start.merit:.3f}, feasible = {start.feasible}")
print(f"coating time = {start.coating_time / 3600:.1f} h")
```

{py:meth}`~xstack.design.optim.problem.DesignProblem.evaluate` returns an
{py:class}`~xstack.design.optim.problem.Evaluation` with the figure of merit (`merit`), the
`objective` (the figure of merit plus penalties for violated constraints), the reflectivity curves
(`results`), the stack and the constraint results. The problem object is immutable; the free
parameters keep their start values.

## An analytic first estimate

Before searching numerically, the Bragg law provides start values. The first stage of the pipeline
is this analytic estimate; you can also run it alone:

```python
from xstack.core.materials import default_library
from xstack.core.physics import bragg
from xstack.design.optim.stages import BraggOutline

outline = BraggOutline(refraction=True).run(problem)
estimated = outline.parameters.value_dict()
print(f"estimated period d = {estimated['d']:.3f} nm, gamma = {estimated['gamma']:.3f}")
print(f"N searched between {outline.parameters['N'].min:.0f} and {outline.parameters['N'].max:.0f}")

library = default_library()
beta_w = float(library.delta_beta(tungsten, ENERGY).imag)
beta_c = float(library.delta_beta(carbide, ENERGY).imag)
print(f"Vinogradov-Zeldovich optimum of gamma: {bragg.optimal_gamma(beta_w, beta_c):.3f}")
```

`refraction=True` includes the shift of the Bragg peak caused by the mean refractive index, which
matters at small angles (see the [Bragg law](../physics_background.md)). The estimate fixes the
period from the band centre and sets Γ to the Vinogradov–Zeldovich optimum, which maximizes the
peak reflectivity for given absorption. That optimum is a very thin tungsten layer, thinner than the
0.8 nm limit allows, so the estimate moves Γ into the window that the constraint permits. It also
narrows the bounds of `d` to ±20 % and restricts `N` around the saturation estimate. The optimized
design starts from these values.

## Run the pipeline

{py:meth}`~xstack.design.optim.pipeline.Pipeline.default` builds the standard staged
optimization:

1. `outline`: the analytic estimate above.
2. `scan`: a grid scan over Γ and the integer parameters (here `N`) on a coarse target grid.
3. `global`: differential evolution over all free parameters, again on the coarse grid.
4. `refine`: a local least-squares refinement on the full grid.

A global search finds the right basin of attraction among the many local maxima of a multilayer
response, and the local refinement then converges to the exact optimum within it. The coarse grid
makes the first two stages cheap. Each stage starts from the best design so far, so the final
design is never worse than the start design.

```python
from xstack.core.optim.optimizers import LmfitOptimizer
from xstack.design.optim import Pipeline
from xstack.design.optim.pipeline import StageResult
from xstack.design.optim.stages import Stage

default = Pipeline.default(problem, seed=0, global_iterations=6, population=8, coarse_points=21)
pipeline = Pipeline(
    tuple(
        Stage("refine", LmfitOptimizer("least_squares", 400)) if s.name == "refine" else s
        for s in default.stages
    )
)


def report(stage: StageResult) -> None:
    r = stage.result
    print(f"{stage.stage.name:8s} objective {r.best.objective:.4f}  evaluations {r.evaluations}")


result = pipeline.run(problem, on_stage=report)
```

The arguments control the cost of the run:

- `seed` makes the global search reproducible. Omit it for a different run each time.
- `global_iterations` and `population` set the budget of differential evolution. The defaults
  (30 generations, population factor 10) are suitable for real designs; the values here are small
  to keep this page fast.
- `coarse_points` is the number of energy points per target in the scan and global stages. The
  refinement always uses the full grid of the target (61 points here).
- The cap of 400 evaluations on `refine` replaces the default of an unlimited local search.

The stage report lists the best objective of each stage on *its own* grid (and the `outline`
stage with an approximate kinematic model), so the values of different stages are not directly
comparable. Check the final design on the full problem:

```python
final = result.final
values = result.parameters.value_dict()
print(f"period d = {values['d']:.3f} nm, gamma = {values['gamma']:.3f}, N = {values['N']:.0f}")
print(f"mean R in the band = {float(final.results[0].reflectivity.mean()):.3f}")
print(f"W layer = {values['gamma'] * values['d']:.2f} nm (minimum allowed 0.8 nm)")
print(f"feasible = {final.feasible}, coating time = {final.coating_time / 3600:.1f} h")
```

`result.parameters` carries the optimized values with the bounds and free/fixed settings of the
start parameters; `result.final` is the evaluation of that design on the full problem. The
optimized Γ is above the analytic optimum because of the minimum layer thickness (and the finite
roughness, which makes thin layers less effective). The input `problem` is unchanged: to evaluate
the optimized design as a problem, use `problem.with_parameters(result.parameters)`.

## Keep results in a project

For anything you want to keep, store the result in a
{py:class}`~xstack.design.project.project.Project`. A project holds the input design, the targets, the
figure of merit, constraints, process, stored designs (*candidates*) and a journal. All changes
go through commands, so that they can be undone and are recorded, exactly as in the GUI.

```python
from xstack.design.project import Project, ProjectState
from xstack.design.project.commands import store_pipeline_result

project = Project(
    ProjectState(
        name="W/B4C Cu K-alpha monochromator",
        mirror=mirror,
        targets=targets,
        merit=MeanReflectivity(),
        constraints=(MinLayerThickness(0.8),),
        process=process,
    )
)
project.execute(store_pipeline_result(problem, result, "optimized"))

best = project.state.selected_candidate()
assert best is not None
print(f"candidates: {[c.name for c in project.state.candidates]}")
print(f"selected '{best.name}' from the {best.source}: figure of merit {best.merit:.4f}")
print(f"input design still has N = {project.state.mirror.coating.parameters['N'].value:.0f}")
```

The optimization result becomes a selected candidate; **the input design is never overwritten**. To
continue from a candidate, promote it with
{py:func}`~xstack.design.project.commands.promote_candidate`. Commands can be undone:

```python
project.undo()
print(f"after undo: {len(project.state.candidates)} candidates")
project.redo()
print(f"after redo: {len(project.state.candidates)} candidate")
```

{py:func}`~xstack.design.project.commands.run_pipeline` combines both steps: it runs a pipeline on the
project's problem, records the run in the journal and stores the final design. The GUI does the same.

```{code-block} python
from xstack.design.project.commands import run_pipeline

run_pipeline(project, pipeline, candidate_name="optimized")
```

The project is saved as a JSON file with the suffix `.xstack` and can be opened in
`xstack-design`:

```python
project.save("monochromator.xstack")
reloaded = Project.load("monochromator.xstack")
print(reloaded.state.selected, len(reloaded.journal), "journal entries")
```

## Tolerance analysis

A design that is optimal on paper may be fragile. The Monte Carlo
{py:func}`~xstack.design.manufacturing.tolerance.tolerance_analysis` draws random manufacturing errors
from an {py:class}`~xstack.design.manufacturing.tolerance.ErrorModel`, builds the perturbed stacks and
evaluates the figure of merit for each. Compare two error models: random thickness errors of 2 %
per layer, and the same plus a systematic 1 % rate calibration error per material.

```python
from xstack.design.manufacturing import ErrorModel, tolerance_analysis

optimized = problem.with_parameters(result.parameters)
models = {
    "2 % random": ErrorModel(thickness_random=0.02),
    "2 % random + 1 % rate": ErrorModel(thickness_random=0.02, thickness_systematic=0.01),
}
for label, errors in models.items():
    tol = tolerance_analysis(optimized, errors=errors, samples=40, seed=1)
    print(
        f"{label:22s} nominal R = {1 - tol.nominal.merit:.3f}, "
        f"median R = {1 - tol.percentile(50):.3f}, "
        f"R at the 90th percentile of the merit = {1 - tol.percentile(90):.3f}, "
        f"feasible {int(tol.feasible.sum())} of {len(tol.feasible)}"
    )
```

Because the figure of merit is lower for better mirrors, the 90th percentile of the figure of merit
is the *pessimistic* end: 90 % of the samples are at least that good, so `1 - tol.percentile(90)`
is the reflectivity that all but the worst 10 % of the coatings exceed. Random errors average out over
the bilayers, but a systematic rate error shifts the period of the whole stack and with it
the Bragg peak. For a ±1 % band, a rate error of 1 % moves the peak by about the half-width of the band.
The consequence for the manufacturing is that this mirror needs a well-calibrated deposition
rate (or a witness-sample calibration), which random layer errors alone would not suggest. See
{doc}`topics/design_tolerances` for all error types and their interpretation.

## Export the recipe

The deposition recipe is the layer table of the stack in deposition order, with the thickness,
roughness and, given a process, the duration and start time of every layer:

```python
from xstack.design.recipe import layer_table, write_recipe_csv, write_recipe_json

stack = final.stack
assert stack is not None
rows = layer_table(stack, process)
print(f"{len(rows)} layers; first: {rows[0]['material']} {rows[0]['thickness_nm']:.3f} nm")
write_recipe_csv("recipe.csv", stack, process)
write_recipe_json("recipe.json", stack, process)
```

Row 1 is the layer on the substrate (`order="deposition"`); use `order="top_down"` to list the
surface first. The JSON file adds the substrate, the total thickness and the total coating time.
Use the *nominal* thicknesses with the calibration of your own deposition system: the recipe
assumes the rates given to the {py:class}`~xstack.design.manufacturing.process.DepositionProcess`.

## Next steps

- {doc}`topics/design_depth_graded`: broadband supermirrors and the trade-off between reflectivity and
  coating time.
- {doc}`topics/design_multiple_targets`: one coating for several angles or energy bands.
- {doc}`topics/design_tolerances`: error models, constraints and the process in detail.
- {doc}`quickstart_design_gui`: the same task in the graphical application.
