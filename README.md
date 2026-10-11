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
