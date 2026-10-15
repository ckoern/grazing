"""Fit results: best parameters, uncertainties and correlations.

Uncertainties follow from the Jacobian J of the residual vector with respect to the free
parameters at the best fit, computed by finite differences. The covariance matrix is

    C = s² · (JᵀJ)⁻¹

with s² = 1 for ``"chi2"`` weighting (the residuals are already normalized by the measured
uncertainties) and s² = objective / (points − free parameters) otherwise, i.e. the scatter of
the residuals sets the scale. The analysis does not depend on the optimizer, so it also applies
to results of global methods.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
from numpy.typing import NDArray

from xstack.core.optim.optimizers import OptimizationResult
from xstack.core.parameters import ParameterSet
from xstack.core.serialization import register_serializable
from xstack.fit.problem import FitEvaluation, FitProblem, Weighting

__all__ = ["FitResult", "FitStatistics", "analyze"]

_RELATIVE_STEP = 1e-5
_SINGULAR = 1e-10


@dataclass(frozen=True)
class FitStatistics:
    """Goodness of fit, uncertainties and correlations of the free parameters.

    Attributes:
        names: Names of the free parameters.
        values: Their best values.
        errors: Standard uncertainties, ``nan`` if not determined by the data.
        correlation: Correlation matrix, rows and columns in the order of `names`; ``nan``
            where not determined.
        objective: Sum of squared residuals.
        points: Number of fitted points.
        weighting: Residual weighting of the fit.
    """

    names: tuple[str, ...]
    values: tuple[float, ...]
    errors: tuple[float, ...]
    correlation: tuple[tuple[float, ...], ...]
    objective: float
    points: int
    weighting: Weighting

    @property
    def degrees_of_freedom(self) -> int:
        """Number of fitted points minus number of free parameters."""
        return self.points - len(self.names)

    @property
    def reduced_objective(self) -> float:
        """Objective per degree of freedom: χ²_red for ``"chi2"`` weighting."""
        dof = self.degrees_of_freedom
        return self.objective / dof if dof > 0 else math.nan

    @property
    def determined(self) -> bool:
        """Whether the data determine all free parameters (no singular covariance)."""
        return all(math.isfinite(e) for e in self.errors)

    def error(self, name: str) -> float:
        """Return the standard uncertainty of a free parameter.

        Args:
            name: Parameter name.

        Returns:
            The uncertainty, ``nan`` if not determined.

        Raises:
            KeyError: If the parameter was not free.
        """
        return self.errors[self._index(name)]

    def correlation_of(self, a: str, b: str) -> float:
        """Return the correlation coefficient of two free parameters.

        Args:
            a: First parameter name.
            b: Second parameter name.

        Returns:
            The coefficient in [-1, 1], ``nan`` if not determined.

        Raises:
            KeyError: If a parameter was not free.
        """
        return self.correlation[self._index(a)][self._index(b)]

    def strong_correlations(self, threshold: float = 0.9) -> list[tuple[str, str, float]]:
        """Return pairs of parameters with a large absolute correlation.

        Args:
            threshold: Minimum absolute correlation coefficient.

        Returns:
            Triples (name, name, coefficient), strongest first.
        """
        pairs = [
            (a, b, self.correlation[i][j])
            for i, a in enumerate(self.names)
            for j, b in enumerate(self.names)
            if i < j and abs(self.correlation[i][j]) >= threshold
        ]
        return sorted(pairs, key=lambda p: -abs(p[2]))

    def _index(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError:
            raise KeyError(f"{name!r} is not a free parameter of the fit") from None


@dataclass(frozen=True)
class FitResult:
    """Outcome of a fit.

    Attributes:
        parameters: All fit parameters with the best values.
        statistics: Uncertainties and correlations, or ``None`` if they could not be computed.
        method: Optimization method.
        message: Message of the optimizer.
        evaluations: Number of model evaluations.
        cancelled: Whether the fit was cancelled.
        elapsed: Wall-clock time in s.
        created: Time of the fit, ISO 8601 in UTC.
        note: Why the statistics are missing, or empty.
    """

    parameters: ParameterSet
    statistics: FitStatistics | None
    method: str = ""
    message: str = ""
    evaluations: int = 0
    cancelled: bool = False
    elapsed: float = 0.0
    created: str = ""
    note: str = ""

    @classmethod
    def from_optimization(
        cls, problem: FitProblem, result: OptimizationResult[FitEvaluation]
    ) -> "FitResult":
        """Create a fit result from an optimization of a fit problem, with its statistics.

        Args:
            problem: The fit problem that was optimized.
            result: The optimization result.

        Returns:
            The fit result.
        """
        fitted = problem.with_parameters(result.parameters)
        statistics, note = None, ""
        try:
            statistics = analyze(fitted)
        except ValueError as exc:
            note = str(exc)
        return cls(
            result.parameters,
            statistics,
            result.method,
            result.message,
            result.evaluations,
            result.cancelled,
            result.elapsed,
            datetime.now(UTC).isoformat(timespec="seconds"),
            note,
        )


def _residuals(problem: FitProblem, values: Mapping[str, float]) -> NDArray[np.float64]:
    evaluation = problem.evaluate(values)
    if evaluation.error is not None:
        raise ValueError(f"model cannot be evaluated: {evaluation.error}")
    return evaluation.residuals


def _jacobian(problem: FitProblem, names: tuple[str, ...]) -> NDArray[np.float64]:
    params = problem.parameters
    base = params.value_dict()
    columns = []
    for name in names:
        p = params[name]
        span = p.max - p.min
        scale = abs(p.value) or (span if math.isfinite(span) else 1.0)
        h = _RELATIVE_STEP * scale
        up = p.value + h <= p.max
        down = p.value - h >= p.min
        hi = base | {name: p.value + h} if up else base
        lo = base | {name: p.value - h} if down else base
        if not (up or down):
            raise ValueError(f"bounds of {name!r} are too narrow for the uncertainty analysis")
        width = h * (int(up) + int(down))
        columns.append((_residuals(problem, hi) - _residuals(problem, lo)) / width)
    return np.column_stack(columns)


def analyze(problem: FitProblem) -> FitStatistics:
    """Compute goodness of fit, uncertainties and correlations at the current parameters.

    Integer parameters and parameters whose bounds coincide are excluded from the free
    parameters.

    Args:
        problem: The fit problem with the best-fit parameters as current values.

    Returns:
        The statistics.

    Raises:
        ValueError: If no parameter is free or the model cannot be evaluated near the
            current values.
    """
    params = problem.parameters
    names = tuple(
        n for n in params.free_names() if not params[n].integer and params[n].min < params[n].max
    )
    if not names:
        raise ValueError("no free parameters")
    residuals = _residuals(problem, params.value_dict())
    objective = float(residuals @ residuals)
    jacobian = _jacobian(problem, names)
    _, singular, vt = np.linalg.svd(jacobian, full_matrices=False)
    p = len(names)
    dof = residuals.size - p
    if singular.size < p or singular[-1] <= _SINGULAR * singular[0] or dof <= 0:
        nan = math.nan
        return FitStatistics(
            names,
            tuple(params[n].value for n in names),
            (nan,) * p,
            tuple(tuple(1.0 if i == j else nan for j in range(p)) for i in range(p)),
            objective,
            int(residuals.size),
            problem.weighting,
        )
    covariance = (vt.T / singular**2) @ vt
    if problem.weighting != "chi2":
        covariance *= objective / dof
    errors = np.sqrt(np.diag(covariance))
    correlation = np.clip(covariance / np.outer(errors, errors), -1.0, 1.0)
    return FitStatistics(
        names,
        tuple(params[n].value for n in names),
        tuple(float(e) for e in errors),
        tuple(tuple(float(c) for c in row) for row in correlation),
        objective,
        int(residuals.size),
        problem.weighting,
    )


register_serializable("FitStatistics", FitStatistics)
register_serializable("FitResult", FitResult)
