"""Fit project: its state, file format and commands.

All changes go through `FitProject.execute` with a `Command` from this module, so that they can
be undone and are recorded in the journal. Fit projects are stored in ``.xfit`` files.

Examples:
    >>> from xstack.core.materials import get
    >>> from xstack.core.measurement import MeasuredCurve
    >>> from xstack.core.structure import Mirror, SingleLayer
    >>> project = FitProject()
    >>> project.execute(set_mirror(Mirror(SingleLayer.create(get("Pt"), t=30.0), get("Si"))))
    >>> curve = MeasuredCurve("angle", [0.01, 0.02, 0.03], [0.9, 0.1, 0.01], energy=8048.0)
    >>> project.execute(add_dataset(Dataset.create("scan", curve)))
    >>> project.execute(set_parameter("t", vary=True, min=20.0, max=40.0))
    >>> project.execute(set_parameter("scan.scale", vary=True, min=0.5, max=2.0))
    >>> project.state.problem().parameters.free_names()
    ('t', 'scan.scale')
"""

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from xstack.core import document
from xstack.core.document import Document
from xstack.core.io.models import read_models
from xstack.core.optim.events import CancellationToken, EventSink
from xstack.core.optim.optimizers import LmfitOptimizer, Optimizer
from xstack.core.parameters import ParameterSet
from xstack.core.physics.models import ParrattModel, ReflectivityModel
from xstack.core.serialization import register_serializable
from xstack.core.structure.coating import Coating, Mirror
from xstack.fit.dataset import Dataset, Interval
from xstack.fit.problem import FitProblem, Weighting
from xstack.fit.results import FitResult

__all__ = [
    "FILE_SUFFIX",
    "SCHEMA_VERSION",
    "Command",
    "FitProject",
    "FitState",
    "StoredFit",
    "add_dataset",
    "apply_fit",
    "import_model",
    "remove_dataset",
    "remove_fit",
    "rename_dataset",
    "rename_fit",
    "rename_project",
    "replace_dataset",
    "run_fit",
    "set_coating",
    "set_mirror",
    "set_model",
    "set_notes",
    "set_optimizer",
    "set_optimizers",
    "set_parameter",
    "set_parameters",
    "set_ranges",
    "set_weighting",
    "store_result",
]

SCHEMA_VERSION = 1
"""Version of the fit project file format."""

FILE_SUFFIX = ".xfit"
"""Conventional suffix of fit project files."""


@dataclass(frozen=True)
class StoredFit:
    """A saved fit: its result together with the structure it was obtained with.

    Attributes:
        name: Unique name within the project, e.g. ``"fit 3"``.
        result: The fit result (all fit parameters with best values, statistics).
        mirror: The structure with the best-fit values.
        model: The reflectivity model of the fit.
        notes: Free-form notes.
    """

    name: str
    result: FitResult
    mirror: Mirror
    model: ReflectivityModel = field(default_factory=ParrattModel)
    notes: str = ""


register_serializable("StoredFit", StoredFit)


@dataclass(frozen=True)
class FitState:
    """Complete, immutable description of a fit project.

    Attributes:
        name: Project name.
        mirror: The structure model with its parameters, or ``None`` if not yet defined.
        model: Reflectivity model.
        datasets: The measured curves with instruments and point selections.
        weighting: Residual weighting, see `xstack.fit.problem`.
        optimizers: Stages of `run_fit`, run in order; each starts from the best values of the
            previous one (e.g. a global search followed by a least-squares refinement).
        results: Saved fits, oldest first.
        model_source: Where the structure model comes from, e.g. ``"mirror.xstack: input
            design"``.
        notes: Free-form notes.
    """

    name: str = "untitled"
    mirror: Mirror | None = None
    model: ReflectivityModel = field(default_factory=ParrattModel)
    datasets: tuple[Dataset, ...] = ()
    weighting: Weighting = "log"
    optimizers: tuple[Optimizer, ...] = (LmfitOptimizer("least_squares"),)
    results: tuple[StoredFit, ...] = ()
    model_source: str = ""
    notes: str = ""

    def evolve(self, **changes: Any) -> "FitState":
        """Return a copy with some fields changed.

        Args:
            **changes: New values for fields of the state.

        Returns:
            The modified state.
        """
        return replace(self, **changes)

    @property
    def result(self) -> FitResult | None:
        """The result of the most recently saved fit, or ``None``."""
        return self.results[-1].result if self.results else None

    def stored(self, name: str) -> StoredFit:
        """Return a saved fit by name.

        Args:
            name: Name of the saved fit.

        Returns:
            The saved fit.

        Raises:
            KeyError: If there is no saved fit with this name.
        """
        for fit in self.results:
            if fit.name == name:
                return fit
        raise KeyError(f"unknown saved fit {name!r}")

    def problem(self) -> FitProblem:
        """Return the fit problem described by this state.

        Returns:
            The fit problem.

        Raises:
            ValueError: If the mirror or the datasets are missing or inconsistent.
        """
        if self.mirror is None:
            raise ValueError("the project has no structure model")
        if not self.datasets:
            raise ValueError("the project has no measured data")
        return FitProblem(self.mirror, self.datasets, self.model, self.weighting)

    def dataset(self, name: str) -> Dataset:
        """Return a dataset by name.

        Args:
            name: Dataset name.

        Returns:
            The dataset.

        Raises:
            KeyError: If there is no dataset with this name.
        """
        for d in self.datasets:
            if d.name == name:
                return d
        raise KeyError(f"unknown dataset {name!r}")


register_serializable("FitState", FitState)


@dataclass(frozen=True)
class Command(document.Command[FitState]):
    """A described change of the fit state.

    Attributes:
        description: Human-readable description, shown in the journal and undo history.
        transform: Function computing the new state from the current one.
        merge_key: Consecutive commands with the same non-empty key are merged into one undo
            step.
    """


class FitProject(Document[FitState]):
    """A fit project: the current state, its undo history and the journal.

    Project files conventionally have the suffix `FILE_SUFFIX`.

    Args:
        state: Initial state.
        journal: Existing journal, e.g. loaded from a file.
    """

    file_format = "xstack-fit"
    file_description = "xstack fit project"
    schema_version = SCHEMA_VERSION
    state_type = FitState


def _require_mirror(state: FitState) -> Mirror:
    if state.mirror is None:
        raise ValueError("the project has no structure model")
    return state.mirror


def rename_project(name: str) -> Command:
    """Rename the project.

    Args:
        name: New name.

    Returns:
        The command.
    """
    return Command(f"Rename project to {name!r}", lambda s: s.evolve(name=name))


def set_notes(notes: str) -> Command:
    """Replace the notes.

    Args:
        notes: The new notes.

    Returns:
        The command.
    """
    return Command("Edit notes", lambda s: s.evolve(notes=notes), merge_key="notes")


def set_mirror(
    mirror: Mirror, model: ReflectivityModel | None = None, source: str | None = None
) -> Command:
    """Replace the structure model (coating, substrate, substrate roughness).

    Args:
        mirror: The new mirror.
        model: Reflectivity model to use with it, or ``None`` to keep the current one.
        source: Where the mirror comes from, or ``None`` to keep the current description.

    Returns:
        The command.
    """

    def transform(state: FitState) -> FitState:
        return state.evolve(
            mirror=mirror,
            model=model or state.model,
            model_source=state.model_source if source is None else source,
        )

    return Command(f"Set structure model{f' from {source}' if source else ''}", transform)


def set_coating(coating: Coating) -> Command:
    """Replace the coating of the structure model.

    Args:
        coating: The new coating.

    Returns:
        The command.
    """

    def transform(state: FitState) -> FitState:
        return state.evolve(mirror=_require_mirror(state).with_coating(coating))

    return Command(f"Set coating {coating.name!r}", transform)


def import_model(path: Path | str, name: str | None = None) -> Command:
    """Replace the structure model by one read from a file (e.g. an ``.xstack`` design).

    The file is read when the command is created.

    Args:
        path: The file.
        name: Name of the model within the file, or ``None`` for the first one.

    Returns:
        The command.

    Raises:
        ValueError: If the file cannot be read, contains no model or none named `name`.
        OSError: If the file cannot be read.
    """
    models = read_models(path)
    if not models:
        raise ValueError(f"{Path(path).name} contains no structure model")
    if name is None:
        chosen = models[0]
    else:
        matches = [m for m in models if m.name == name]
        if not matches:
            available = ", ".join(repr(m.name) for m in models)
            raise ValueError(f"no model {name!r} in {Path(path).name} (available: {available})")
        chosen = matches[0]
    return set_mirror(chosen.mirror, chosen.model, f"{Path(path).name}: {chosen.name}")


def set_model(model: ReflectivityModel) -> Command:
    """Replace the reflectivity model.

    Args:
        model: The new model.

    Returns:
        The command.
    """
    return Command(f"Set model {model.name}", lambda s: s.evolve(model=model))


def set_weighting(weighting: Weighting) -> Command:
    """Replace the residual weighting.

    Args:
        weighting: ``"log"``, ``"linear"`` or ``"chi2"``.

    Returns:
        The command.
    """
    return Command(f"Set weighting {weighting}", lambda s: s.evolve(weighting=weighting))


def set_optimizer(optimizer: Optimizer) -> Command:
    """Use a single optimizer for fits.

    Args:
        optimizer: The optimizer.

    Returns:
        The command.
    """
    return set_optimizers((optimizer,))


def set_optimizers(optimizers: Sequence[Optimizer]) -> Command:
    """Replace the stages of fits.

    Args:
        optimizers: The optimizers, run in order; at least one.

    Returns:
        The command.
    """
    stages = tuple(optimizers)

    def transform(state: FitState) -> FitState:
        if not stages:
            raise ValueError("a fit needs at least one optimizer")
        return state.evolve(optimizers=stages)

    return Command("Set optimizer", transform)


def add_dataset(dataset: Dataset) -> Command:
    """Add a dataset.

    Args:
        dataset: The dataset; its name must not be used yet.

    Returns:
        The command.
    """

    def transform(state: FitState) -> FitState:
        if any(d.name == dataset.name for d in state.datasets):
            raise ValueError(f"a dataset named {dataset.name!r} already exists")
        return state.evolve(datasets=(*state.datasets, dataset))

    return Command(f"Add dataset {dataset.name!r}", transform)


def remove_dataset(name: str) -> Command:
    """Remove a dataset.

    Parameter expressions that refer to its parameters must be removed first.

    Args:
        name: Dataset name.

    Returns:
        The command.
    """

    def transform(state: FitState) -> FitState:
        state.dataset(name)
        return state.evolve(datasets=tuple(d for d in state.datasets if d.name != name))

    return Command(f"Remove dataset {name!r}", transform)


def _rename_in_expressions(parameters: ParameterSet, old: str, new: str) -> ParameterSet:
    pattern = re.compile(rf"(?<![\w.]){re.escape(old)}\.")
    return ParameterSet(
        p.evolve(expr=pattern.sub(f"{new}.", p.expr)) if p.expr is not None else p
        for p in parameters.values()
    )


def rename_dataset(name: str, new_name: str) -> Command:
    """Rename a dataset and its instrument parameters.

    Parameter expressions of all datasets that refer to the renamed parameters are updated.

    Args:
        name: Current dataset name.
        new_name: New name, a Python identifier not used by another dataset.

    Returns:
        The command.
    """

    def transform(state: FitState) -> FitState:
        state.dataset(name)
        if new_name != name and any(d.name == new_name for d in state.datasets):
            raise ValueError(f"a dataset named {new_name!r} already exists")
        datasets = []
        for dataset in state.datasets:
            instrument = dataset.instrument
            parameters = _rename_in_expressions(instrument.parameters, name, new_name)
            dataset = replace(dataset, instrument=instrument.with_parameters(parameters))
            if dataset.name == name:
                dataset = replace(dataset, name=new_name)
            datasets.append(dataset)
        return state.evolve(datasets=tuple(datasets))

    return Command(f"Rename dataset {name!r} to {new_name!r}", transform)


def replace_dataset(dataset: Dataset, description: str = "") -> Command:
    """Replace the dataset with the same name.

    Args:
        dataset: The new dataset.
        description: Description for the journal, or empty for a default.

    Returns:
        The command.
    """

    def transform(state: FitState) -> FitState:
        state.dataset(dataset.name)
        datasets = tuple(dataset if d.name == dataset.name else d for d in state.datasets)
        return state.evolve(datasets=datasets)

    return Command(description or f"Edit dataset {dataset.name!r}", transform)


def set_ranges(
    name: str, ranges: Iterable[Interval], excluded: Iterable[Interval] | None = None
) -> Command:
    """Set the fit ranges of a dataset.

    Args:
        name: Dataset name.
        ranges: Fit ranges of the scanned variable (rad or eV); empty for all points.
        excluded: Excluded ranges, or ``None`` to keep the current ones.

    Returns:
        The command.
    """
    ranges = tuple(ranges)
    excluded = None if excluded is None else tuple(excluded)

    def transform(state: FitState) -> FitState:
        dataset = state.dataset(name).with_ranges(ranges, excluded)
        return state.evolve(
            datasets=tuple(dataset if d.name == name else d for d in state.datasets)
        )

    return Command(f"Set fit ranges of {name!r}", transform, merge_key=f"ranges:{name}")


def set_parameter(name: str, **changes: Any) -> Command:
    """Modify one fit parameter: a coating parameter or ``"<dataset>.<instrument parameter>"``.

    Consecutive edits of the same parameter form a single undo step.

    Args:
        name: Parameter name.
        **changes: New values for fields of the `xstack.core.parameters.Parameter`.

    Returns:
        The command.
    """
    summary = ", ".join(f"{k}={v!r}" for k, v in changes.items())

    def transform(state: FitState) -> FitState:
        mirror = _require_mirror(state)
        if name in mirror.parameters:
            return state.evolve(mirror=mirror.with_parameter(name, **changes))
        dataset_name, _, parameter = name.partition(".")
        dataset = state.dataset(dataset_name)
        instrument = dataset.instrument
        parameters = instrument.parameters.replace(parameter, **changes)
        new = replace(dataset, instrument=instrument.with_parameters(parameters))
        return state.evolve(datasets=tuple(new if d is dataset else d for d in state.datasets))

    return Command(f"Set {name}: {summary}", transform, merge_key=f"parameter:{name}")


def set_parameters(parameters: ParameterSet, description: str = "Set parameters") -> Command:
    """Replace all fit parameters.

    Args:
        parameters: The new parameter set, with the names of the fit parameters.
        description: Description for the journal.

    Returns:
        The command.
    """

    def transform(state: FitState) -> FitState:
        problem = state.problem().with_parameters(parameters)
        return state.evolve(mirror=problem.mirror, datasets=problem.datasets)

    return Command(description, transform)


def _unique_fit_name(state: FitState) -> str:
    used = {fit.name for fit in state.results}
    n = len(state.results) + 1
    while f"fit {n}" in used:
        n += 1
    return f"fit {n}"


def store_result(result: FitResult, apply: bool = True, name: str | None = None) -> Command:
    """Save a fit result in the project.

    Args:
        result: The result.
        apply: Also set the parameter values to the best fit.
        name: Name of the saved fit, or ``None`` for ``"fit N"``.

    Returns:
        The command.
    """

    def transform(state: FitState) -> FitState:
        if name is not None and any(fit.name == name for fit in state.results):
            raise ValueError(f"a saved fit named {name!r} already exists")
        fitted = state.problem().with_parameters(result.parameters)
        stored = StoredFit(name or _unique_fit_name(state), result, fitted.mirror, state.model)
        if apply:
            state = state.evolve(mirror=fitted.mirror, datasets=fitted.datasets)
        return state.evolve(results=(*state.results, stored))

    return Command(f"Save fit result ({result.method})", transform)


def apply_fit(name: str) -> Command:
    """Restore the structure, model and parameters of a saved fit.

    Instrument parameters are restored for the datasets that still exist.

    Args:
        name: Name of the saved fit.

    Returns:
        The command.
    """

    def transform(state: FitState) -> FitState:
        fit = state.stored(name)
        saved = fit.result.parameters
        datasets = []
        for dataset in state.datasets:
            names = [n for n in dataset.parameters if n in saved]
            if names:
                current = dataset.parameters
                parameters = ParameterSet(saved[n] if n in saved else current[n] for n in current)
                dataset = dataset.with_parameters(parameters)
            datasets.append(dataset)
        return state.evolve(mirror=fit.mirror, model=fit.model, datasets=tuple(datasets))

    return Command(f"Apply saved fit {name!r}", transform)


def rename_fit(name: str, new_name: str) -> Command:
    """Rename a saved fit.

    Args:
        name: Current name.
        new_name: New, unique, non-empty name.

    Returns:
        The command.
    """

    def transform(state: FitState) -> FitState:
        fit = state.stored(name)
        if not new_name.strip():
            raise ValueError("the name must not be empty")
        if new_name != name and any(f.name == new_name for f in state.results):
            raise ValueError(f"a saved fit named {new_name!r} already exists")
        renamed = replace(fit, name=new_name)
        return state.evolve(results=tuple(renamed if f is fit else f for f in state.results))

    return Command(f"Rename saved fit {name!r} to {new_name!r}", transform)


def remove_fit(name: str) -> Command:
    """Delete a saved fit.

    Args:
        name: Name of the saved fit.

    Returns:
        The command.
    """

    def transform(state: FitState) -> FitState:
        state.stored(name)
        return state.evolve(results=tuple(f for f in state.results if f.name != name))

    return Command(f"Delete saved fit {name!r}", transform)


def run_fit(
    state: FitState,
    events: EventSink | None = None,
    cancel: CancellationToken | None = None,
) -> FitResult:
    """Run the optimizers of a state in order and analyze the result.

    Each stage starts from the best values of the previous one. The result's evaluations and
    time are summed over the stages, its method lists the stages, and its message is the last
    stage's. A cancelled stage ends the fit. The state is not changed; pass the result to
    `store_result`.

    Args:
        state: The fit state.
        events: Receiver of progress events.
        cancel: Token to cancel the fit; the best values so far are returned.

    Returns:
        The fit result with uncertainties and correlations.

    Raises:
        ValueError: If the state does not describe a fit, or nothing is free.
    """
    problem = state.problem()
    parameters = problem.parameters
    evaluations, elapsed, methods = 0, 0.0, []
    for optimizer in state.optimizers:
        stage = optimizer.run(problem, parameters, events, cancel)
        parameters = stage.parameters
        evaluations += stage.evaluations
        elapsed += stage.elapsed
        methods.append(stage.method)
        if stage.cancelled:
            break
    result = FitResult.from_optimization(problem, stage)
    return replace(result, method=" + ".join(methods), evaluations=evaluations, elapsed=elapsed)
