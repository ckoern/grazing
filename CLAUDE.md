# CLAUDE.md

Guidance for working on xstack, a Python library and PySide6 GUI for designing X-ray multilayer mirrors.

## Key documents
- `xstack_concept.md`: product concept and requirements
- `docs/physics_background.md`: physics reference
- `docs/architecture_core.md`: core library architecture
- `docs/gui_design.md`: GUI design and layout
- `docs/kernel_verification.md`: reflectivity kernel test strategy
- `docs/milestones.md`: development milestones
- `docs/fitting_plan.md`: plan and decisions for the fitting tool (`feature/fitting`)

## Environment
- Use the conda environment `xstack` (Python 3.12): `conda run -n xstack <cmd>` or `conda activate xstack`.
- Install the package in editable mode: `pip install -e ".[dev,gui]"`. Install additional packages with pip.
- Run checks before committing:
  - `pytest`
  - `ruff check src tests` and `ruff format --check src tests`
  - `mypy`

## Code style
- Target Python 3.12. Use type hints everywhere. The core package (`xstack` without `xstack.gui`) passes `mypy --strict`.
- Formatting and linting with ruff (line length 100). Configuration lives in `pyproject.toml`.
- Use numpy for numerics and lmfit for optimization. The core library must not import Qt. Only `xstack.gui` may.
- Prefer frozen dataclasses for physical value objects. Mutable state lives in the `Project`.
- Internal units: lengths in nm, energies in eV, angles in rad, time in s, density in g/cm³.
  Convert at the boundaries with `xstack.units`.
- Extension points (coatings, constraints, figures of merit, optimizers, models, providers) use
  protocols and registries. Add new kinds by registering new classes, not by editing existing dispatch code.

## Documentation
- Every public module, function, class and method has a Google-style docstring (`Args:`,
  `Returns:`, `Raises:`, `Attributes:`, and `Examples:` where helpful). Include units in argument descriptions.
- Docstrings describe the interface: what the function does, its inputs, its outputs and its
  errors. Don't describe internal implementation details that aren't needed to use it.
- Comments and docstrings describe only the current state of the software. Never add comments about
  design changes, earlier versions, removed code, or previous mistakes ("changed from…", "previously…",
  "fixed bug where…"). History belongs in git commit messages.
- Inline comments only where the code is not self-explanatory (e.g. a non-obvious physics formula
  with its reference).

## Testing
- pytest with hypothesis for property-based tests. Tests mirror the package structure under `tests/`.
- Reference implementations used only for verification live in `tests/reference/` and are never
  imported by the library.
- Reference datasets in `tests/data/` carry a provenance header (source, version, inputs).
- Every kernel bug gets a regression test.

## Working with subagents
- Delegate work that doesn't need the main model: Sonnet (`model: "sonnet"`) for well-specified
  moderate tasks (tests from a spec, self-contained GUI pages, scaffolding), Haiku
  (`model: "haiku"`) for trivial mechanical tasks (renames, small text and doc edits, boilerplate).
  Keep design decisions, physics and numerical kernel work, cross-cutting integration and review
  in the main session.
- Split delegated work by file so agents don't collide, and give precise prompts that point to
  the conventions in this file.
- Review every agent's result: check `git show --stat`, rerun the checks yourself and make sure the
  reported tests exist. Agents that find bugs in `src/` mark the test `xfail` and report the bug
  instead of fixing it.
- Parallel agents in worktrees must not run `pip install -e` (shared conda env). Use
  `PYTHONPATH=<worktree>/src` and `conda run --cwd <worktree>` with absolute paths.
