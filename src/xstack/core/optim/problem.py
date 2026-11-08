"""Interface between optimizers and the problems they solve.

A problem rates sets of parameter values, e.g. a mirror design (figure of merit and constraint
penalties) or a fit of measured data (weighted residuals). Optimizers only use the interface
defined here, so that they work for every kind of problem.
"""

from collections.abc import Mapping
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from xstack.core.parameters import ParameterSet

__all__ = ["Evaluation", "Problem"]


class Evaluation(Protocol):
    """Result of evaluating one set of parameter values."""

    @property
    def values(self) -> Mapping[str, float]:
        """Evaluated parameter values, by name."""
        ...

    @property
    def objective(self) -> float:
        """Value to minimize, including penalties."""
        ...

    @property
    def merit(self) -> float:
        """The rating without penalties, ``nan`` if the model could not be built."""
        ...

    @property
    def error(self) -> str | None:
        """Description of why the model could not be built, or ``None``."""
        ...


class Problem[E: Evaluation](Protocol):
    """A problem whose parameter values are optimized."""

    @property
    def parameters(self) -> ParameterSet:
        """The parameters with their current values, bounds and fixed/free flags."""
        ...

    def with_parameters(self, parameters: ParameterSet) -> "Problem[E]":
        """Return the problem with other parameters.

        Args:
            parameters: The new parameter set.

        Returns:
            The modified problem.
        """
        ...

    def evaluate(self, values: Mapping[str, float] | None = None) -> E:
        """Evaluate a set of parameter values.

        Args:
            values: Parameter values overriding the current ones, by name.

        Returns:
            The evaluation. Models that cannot be built give an evaluation with an `error`
            instead of raising.
        """
        ...

    def residuals(self, evaluation: E, size: int) -> NDArray[np.float64]:
        """Return the residual vector of an evaluation for least-squares optimizers.

        Args:
            evaluation: An evaluation of this problem.
            size: Length of the vector for evaluations whose model could not be built.

        Returns:
            Residual vector whose sum of squares is the objective of the evaluation.
        """
        ...

    def quantities(self, evaluation: E) -> dict[str, float]:
        """Return named quantities of an evaluation for progress reports.

        Args:
            evaluation: An evaluation of this problem.

        Returns:
            Values by label.
        """
        ...

    def bounds_hints(self) -> dict[str, tuple[float, float]]:
        """Return bounds of parameters implied by hard limits of the problem.

        Returns:
            Lower and upper bound by parameter name; optimizers narrow the bounds of free
            parameters to them.
        """
        ...
