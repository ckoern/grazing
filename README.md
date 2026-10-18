# xstack

xstack is a Python library with two PySide6 applications: `xstack-design` for designing X-ray
multilayer mirrors and `xstack-fit` for fitting measured reflectivity curves. It models periodic
and depth-graded coatings and supports staged optimization under manufacturing constraints.

## Setup

```bash
conda env create -f environment.yml
```

or, in an existing environment:

```bash
conda activate xstack && pip install -e ".[dev,gui]"
```

## Usage

The package has four parts: `xstack.core` (physics, parameters, serialization, data files),
`xstack.design` (mirror design workflow), `xstack.fit` (fitting of measured reflectivity) and
`xstack.gui` (PySide6 applications).

Run the design pipeline of a project file from the command line:

```bash
xstack design run project.xstack [--output OUT] [--seed N] [-v]
xstack design info project.xstack
```

Set up and run a fit of measured reflectivity from the command line:

```bash
xstack fit new fit.xfit --model design.xstack --data scan.csv --quantity two_theta --energy 8.048
xstack fit run fit.xfit [--method least_squares] [--output OUT] [-v]
xstack fit info fit.xfit
xstack fit export fit.xfit curves.csv
```

`xstack fit new` also takes `--unit`, `--columns`, `--percent`, `--range LO:HI` and `--weighting`
(see `xstack fit new --help`). `examples/fit_reflectivity.py` shows a fit with the library.

(`python -m xstack design ...` is equivalent.) Start the design GUI:

```bash
xstack-design [project.xstack]
```

or `python -m xstack.gui.design`. Start the fit GUI:

```bash
xstack-fit [project.xfit]
```

In both applications, *File > New…* opens a wizard for the project name, location and starting
structure (coating type, materials, top layer, substrate).

or `python -m xstack.gui.fit`. To try it, import the synthetic measurements in `examples/data/`
(a W/Si multilayer measured at Cu Kα and Mo Kα; true parameters in the file headers) with
*Add data…*.

## Examples

`examples/` holds scripts for the library (run with `python examples/<name>.py`; `--quick` reduces
the optimization budget, `--plot` or `--save PATH` shows or saves the plot): `reflectivity_demo.py`,
`periodic_fixed_angle.py`, `graded_time_penalty.py`, `coupled_angles.py` and `fit_reflectivity.py`.

## Checks

```bash
pytest
ruff check src tests
ruff format --check src tests
mypy
```

## Documentation

- [Physics background](docs/physics_background.md)
- [Core architecture](docs/architecture_core.md)
- [GUI design](docs/gui_design.md)
- [Kernel verification](docs/kernel_verification.md)
- [Milestones](docs/milestones.md)
- [Fitting tool plan](docs/fitting_plan.md)
- [Product concept](xstack_concept.md)
