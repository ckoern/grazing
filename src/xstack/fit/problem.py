"""Fit problem: one structure model compared with one or more measured curves.

The parameters of a fit are the coating parameters of the mirror and the instrument
parameters of every dataset (``"<dataset>.scale"``, ...). Parameters shared between datasets
are expressed with parameter expressions, e.g. ``"b.offset"`` with ``expr="a.offset"``.

The residual of a fitted point depends on the weighting:

* ``"log"``: log₁₀ R_model − log₁₀ R_meas (points with R_meas ≤ 0 are not fitted),
* ``"linear"``: R_model − R_meas,
* ``"chi2"``: (R_model − R_meas) / σ with the measured uncertainty σ (points with σ ≤ 0 are
  not fitted; every fitted dataset needs uncertainties).

The residuals of each dataset are multiplied by √weight; the objective is the sum of squares.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from xstack.core.materials import MaterialLibrary
from xstack.core.parameters import ParameterSet
from xstack.core.physics.models import ParrattModel, ReflectivityModel
from xstack.core.structure.coating import CoatingBuildError, Mirror
from xstack.core.structure.stack import Stack
from xstack.fit.dataset import Dataset

__all__ = ["FAILED_OBJECTIVE", "WEIGHTINGS", "FitEvaluation", "FitProblem", "Weighting"]

type Weighting = Literal["log", "linear", "chi2"]
"""How residuals are computed, see the module documentation."""

WEIGHTINGS: tuple[Weighting, ...] = ("log", "linear", "chi2")
"""All weightings."""

FAILED_OBJECTIVE = 1e12
"""Objective of parameter values for which the model cannot be built."""

_LOG_FLOOR = 1e-30


@dataclass(frozen=True, eq=False)
class FitEvaluation:
    """Result of evaluating one set of parameter values of a fit.

    Attributes:
        values: Evaluated parameter values, by name.
        objective: Sum of squared residuals.
        residuals: Residuals of all fitted points, dataset by dataset.
        models: Modeled measurement at the fitted points, per dataset in the problem (empty
            for disabled datasets).
        stack: The stack, or ``None`` if the model could not be built.
        error: Description of why the model could not be built, or ``None``.
    """

    values: Mapping[str, float]
    objective: float
    residuals: NDArray[np.float64]
    models: tuple[NDArray[np.float64], ...] = ()
    stack: Stack | None = None
    error: str | None = None

    @property
    def merit(self) -> float:
        """The objective, ``nan`` if the model could not be built."""
        return math.nan if self.error is not None else self.objective

    @property
    def points(self) -> int:
        """Number of fitted points."""
        return int(self.residuals.size)


@dataclass(frozen=True)
class FitProblem:
    """A fit of a mirror model to measured curves.

    Attributes:
        mirror: The structure model; its coating parameters are fit parameters.
        datasets: The measured curves with instrument models and point selections. Names are
            unique.
        model: Reflectivity model.
        weighting: Residual weighting, see the module documentation.
        library: Material library for optical constants, or ``None`` for the default library.

    Raises:
        ValueError: If dataset names are not unique, parameter names collide, the weighting is
            unknown, ``"chi2"`` is used with an enabled dataset without uncertainties, an
            enabled dataset lacks its fixed energy or angle, or no enabled dataset has a fitted
            point.
    """

    mirror: Mirror
    datasets: tuple[Dataset, ...]
    model: ReflectivityModel = field(default_factory=ParrattModel)
    weighting: Weighting = "log"
    library: MaterialLibrary | None = None
    _masks: tuple[NDArray[np.bool_], ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "datasets", tuple(self.datasets))
        names = [d.name for d in self.datasets]
        if len(set(names)) != len(names):
            raise ValueError("dataset names must be unique")
        if self.weighting not in WEIGHTINGS:
            raise ValueError(f"unknown weighting {self.weighting!r}")
        if self.weighting == "chi2":
            missing = [d.name for d in self.datasets if d.enabled and d.curve.uncertainty is None]
            if missing:
                raise ValueError(f"chi2 weighting needs uncertainties; missing for {missing}")
        unset = [d.name for d in self.datasets if d.enabled and d.curve.fixed is None]
        if unset:
            raise ValueError(f"the fixed energy or angle is not set for datasets {unset}")
        coating = set(self.mirror.parameters)
        clashes = sorted({n for d in self.datasets for n in d.parameters if n in coating})
        if clashes:
            raise ValueError(f"dataset parameters {clashes} collide with coating parameters")
        masks = tuple(self.fit_mask(d) for d in self.datasets)
        if not any(m.any() for m in masks):
            raise ValueError("no measured point is selected for the fit")
        object.__setattr__(self, "_masks", masks)

    def fit_mask(self, dataset: Dataset) -> NDArray[np.bool_]:
        """Return which points of a dataset are fitted with this problem's weighting.

        Args:
            dataset: A dataset.

        Returns:
            The dataset mask, restricted to points valid for the weighting; all ``False`` for a
            disabled dataset.
        """
        if not dataset.enabled:
            return np.zeros(dataset.curve.size, dtype=bool)
        mask = dataset.mask()
        _, r, u = dataset.curve.arrays()
        if self.weighting == "log":
            mask &= r > 0.0
        elif self.weighting == "chi2" and u is not None:
            mask &= u > 0.0
        return mask

    @property
    def parameters(self) -> ParameterSet:
        """Coating parameters followed by the instrument parameters of every dataset."""
        return self.mirror.parameters.merge(*(d.parameters for d in self.datasets))

    def with_parameters(self, parameters: ParameterSet) -> "FitProblem":
        """Return a copy with different parameters.

        Args:
            parameters: The new parameter set (same names).

        Returns:
            The modified problem.
        """
        structure_names = set(self.mirror.parameters)
        structure = ParameterSet(p for n, p in parameters.items() if n in structure_names)
        datasets = tuple(
            d.with_parameters(
                ParameterSet(p for n, p in parameters.items() if n.startswith(f"{d.name}."))
            )
            for d in self.datasets
        )
        mirror = self.mirror.with_parameters(structure)
        return replace(self, mirror=mirror, datasets=datasets)

    def evolve(self, **changes: Any) -> "FitProblem":
        """Return a copy with some fields changed.

        Args:
            **changes: New values for fields of the problem.

        Returns:
            The modified problem.
        """
        return replace(self, **changes)

    def bounds_hints(self) -> dict[str, tuple[float, float]]:
        """Return parameter bounds implied by the problem: none for fits."""
        return {}

    def resolve(self, values: Mapping[str, float] | None = None) -> dict[str, float]:
        """Return all parameter values with expressions evaluated.

        Args:
            values: Parameter values overriding the current ones, by name.

        Returns:
            Value of every parameter by name.

        Raises:
            ValueError: If a value is out of bounds or an expression cannot be evaluated.
            KeyError: If `values` contains unknown names.
        """
        params = self.parameters
        if values is not None:
            params = params.with_values(values)
        return params.evaluate()

    def evaluate(self, values: Mapping[str, float] | None = None) -> FitEvaluation:
        """Evaluate a set of parameter values.

        Args:
            values: Parameter values overriding the current ones, by name.

        Returns:
            The evaluation. Models that cannot be built get the objective `FAILED_OBJECTIVE`
            and an error message instead of raising.
        """
        try:
            resolved = self.resolve(values)
            stack = self.stack(resolved)
        except (CoatingBuildError, ValueError) as exc:
            given = self.parameters.value_dict() | dict(values or {})
            return FitEvaluation(given, FAILED_OBJECTIVE, np.zeros(0), error=str(exc))
        models, residuals = [], []
        for dataset, mask in zip(self.datasets, self._masks, strict=True):
            if not mask.any():
                models.append(np.zeros(0))
                continue
            x, r, u = dataset.curve.arrays()
            modeled = self.simulate(dataset, x[mask], resolved, stack)
            models.append(modeled)
            residuals.append(math.sqrt(dataset.weight) * self._residuals(modeled, r[mask], u, mask))
        res = np.concatenate(residuals)
        if not np.all(np.isfinite(res)):
            return FitEvaluation(resolved, FAILED_OBJECTIVE, np.zeros(0), error="model not finite")
        return FitEvaluation(resolved, float(res @ res), res, tuple(models), stack)

    def _residuals(
        self,
        modeled: NDArray[np.float64],
        measured: NDArray[np.float64],
        uncertainty: NDArray[np.float64] | None,
        mask: NDArray[np.bool_],
    ) -> NDArray[np.float64]:
        if self.weighting == "log":
            return np.log10(np.maximum(modeled, _LOG_FLOOR)) - np.log10(measured)
        if self.weighting == "chi2":
            assert uncertainty is not None
            return (modeled - measured) / uncertainty[mask]
        return modeled - measured

    def stack(self, values: Mapping[str, float] | None = None) -> Stack:
        """Build the stack of the mirror.

        Args:
            values: Evaluated parameter values of the fit, by name, or ``None`` for the
                current values.

        Returns:
            The stack.

        Raises:
            CoatingBuildError: If the values do not describe a valid layer sequence.
        """
        resolved = values if values is not None else self.resolve()
        return self.mirror.stack({n: resolved[n] for n in self.mirror.parameters})

    def simulate(
        self,
        dataset: Dataset,
        x: NDArray[np.float64] | None = None,
        values: Mapping[str, float] | None = None,
        stack: Stack | None = None,
    ) -> NDArray[np.float64]:
        """Return the modeled measurement of a dataset, including its instrument.

        Args:
            dataset: The dataset (need not be part of the problem, but its instrument
                parameters are read from `values` if present there).
            x: Nominal angles in rad or energies in eV, increasing. Defaults to all measured
                points of the dataset.
            values: Evaluated parameter values of the fit, or ``None`` for the current values.
            stack: The stack, if already built for `values`.

        Returns:
            Modeled measured reflectivity at `x`.

        Raises:
            CoatingBuildError: If the stack cannot be built.
            ValueError: If the fixed energy or angle of the curve is not set.
        """
        resolved = values if values is not None else self.resolve()
        if stack is None:
            stack = self.stack(resolved)
        if x is None:
            x = np.array(dataset.curve.x)
        curve, fixed = dataset.curve, dataset.curve.fixed
        if fixed is None:
            missing = "energy" if curve.axis == "angle" else "angle"
            raise ValueError(f"the {missing} of dataset {dataset.name!r} is not set")
        prefix = f"{dataset.name}."
        instrument = {
            n.removeprefix(prefix): v for n, v in resolved.items() if n.startswith(prefix)
        }

        def reflectivity(points: NDArray[np.float64]) -> NDArray[np.float64]:
            constant = np.full_like(points, fixed)
            energy, theta = (constant, points) if curve.axis == "angle" else (points, constant)
            return self.model.reflectance(stack, energy, theta, curve.polarization, self.library)

        fixed_angle = fixed if curve.axis == "energy" else None
        return dataset.instrument.apply(x, reflectivity, fixed_angle, instrument)

    def residuals(self, evaluation: FitEvaluation, size: int) -> NDArray[np.float64]:
        """Return the residual vector of an evaluation for least-squares optimizers.

        Args:
            evaluation: An evaluation of this problem.
            size: Length of the vector for models that could not be built.

        Returns:
            The residuals of the evaluation, or `size` equal entries whose sum of squares is
            `FAILED_OBJECTIVE` if the model could not be built.
        """
        if evaluation.error is not None:
            return np.full(size, math.sqrt(FAILED_OBJECTIVE / size))
        return evaluation.residuals

    def quantities(self, evaluation: FitEvaluation) -> dict[str, float]:
        """Return quantities of an evaluation for progress reports.

        Args:
            evaluation: An evaluation of this problem.

        Returns:
            ``"χ²_red"`` for χ² weighting, otherwise the ``"rms residual"``; empty if the model
            could not be built.
        """
        if evaluation.error is not None or evaluation.points == 0:
            return {}
        if self.weighting == "chi2":
            dof = max(evaluation.points - len(self.parameters.free_names()), 1)
            return {"χ²_red": evaluation.objective / dof}
        return {"rms residual": math.sqrt(evaluation.objective / evaluation.points)}
