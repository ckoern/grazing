# xstack

xstack is a Python library and PySide6 GUI for designing X-ray multilayer mirrors. It models
periodic and depth-graded coatings and supports staged optimization under manufacturing
constraints.

## Setup

```bash
conda env create -f environment.yml
```

or, in an existing environment:

```bash
conda activate xstack && pip install -e ".[dev,gui]"
```

## Usage

The package has three layers: `xstack.core` (physics, parameters, serialization), `xstack.design`
(mirror design workflow) and `xstack.gui` (PySide6 applications).

Run the design pipeline of a project file from the command line:

```bash
xstack design run project.xstack [--output OUT] [--seed N] [-v]
xstack design info project.xstack
```

(`python -m xstack design ...` is equivalent.) Start the design GUI:

```bash
xstack-design [project.xstack]
```

or `python -m xstack.gui.design`.

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
