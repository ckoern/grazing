"""Calculations behind the live views of the fit application.

`compute_preview` evaluates the current fit state: the modeled measurement of every dataset over
all its points, the residuals of the fitted points and the goodness of fit. It runs in a worker
thread and never raises; problems are reported in `FitPreview.error`.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from grazing.core.structure.stack import Stack
from grazing.core.units import to_deg, to_kev
from grazing.fit.dataset import Dataset
from grazing.fit.problem import FitProblem, Weighting
from grazing.fit.project import FitState

__all__ = ["DatasetCurves", "FitPreview", "compute_preview", "display_x", "residual_label"]


def display_x(axis: str, x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert values of the scanned variable to display units (deg or keV).

    Args:
        axis: ``"angle"`` or ``"energy"``.
        x: Angles in rad or energies in eV.

    Returns:
        Angles in deg or energies in keV.
    """
    return to_deg(x) if axis == "angle" else to_kev(x)


def residual_label(weighting: Weighting) -> str:
    """Return the axis label of the residuals of a weighting.

    Args:
        weighting: The weighting.

    Returns:
        The label.
    """
    return {
        "chi2": "(fit − meas.)/σ",
        "log": "log₁₀(fit/meas.)",
        "linear": "fit − meas.",
    }[weighting]


@dataclass(frozen=True, eq=False)
class DatasetCurves:
    """Measured and modeled curve of one dataset, in display units.

    Attributes:
        name: Dataset name.
        axis: Scanned variable, ``"angle"`` (x in deg) or ``"energy"`` (x in keV).
        x: Scanned variable of all points, in deg or keV.
        measured: Measured reflectivity.
        uncertainty: Measured uncertainty, or ``None``.
        model: Modeled measurement at all points, or ``None`` if it cannot be computed.
        fitted: Which points take part in the fit.
        residuals: Residuals of all points for the project's weighting (``nan`` where not
            defined), or ``None`` without model.
        enabled: Whether the dataset takes part in fits.
    """

    name: str
    axis: str
    x: NDArray[np.float64]
    measured: NDArray[np.float64]
    uncertainty: NDArray[np.float64] | None
    model: NDArray[np.float64] | None
    fitted: NDArray[np.bool_]
    residuals: NDArray[np.float64] | None
    enabled: bool = True


@dataclass(frozen=True, eq=False)
class FitPreview:
    """Evaluation of a fit state for the views.

    Attributes:
        curves: Curves of all datasets, in project order.
        stack: The stack of the structure model, or ``None``.
        weighting: Residual weighting of the project.
        objective: Sum of squared residuals of the fitted points, ``nan`` if not available.
        points: Number of fitted points.
        free: Number of free parameters.
        error: Why the model or the fit is not available, or ``None``.
    """

    curves: tuple[DatasetCurves, ...] = ()
    stack: Stack | None = None
    weighting: Weighting = "log"
    objective: float = math.nan
    points: int = 0
    free: int = 0
    error: str | None = None
    values: Mapping[str, float] = field(default_factory=dict)

    @property
    def reduced_objective(self) -> float:
        """Objective per degree of freedom (χ²_red for χ² weighting), ``nan`` if undefined."""
        dof = self.points - self.free
        return self.objective / dof if dof > 0 and math.isfinite(self.objective) else math.nan


def _residuals(
    weighting: Weighting,
    model: NDArray[np.float64],
    measured: NDArray[np.float64],
    uncertainty: NDArray[np.float64] | None,
) -> NDArray[np.float64]:
    with np.errstate(divide="ignore", invalid="ignore"):
        if weighting == "log":
            return np.where(measured > 0, np.log10(np.maximum(model, 1e-30) / measured), np.nan)
        if weighting == "chi2":
            if uncertainty is None:
                return np.full_like(model, np.nan)
            return np.where(uncertainty > 0, (model - measured) / uncertainty, np.nan)
        return model - measured


def _measured_only(dataset: Dataset) -> DatasetCurves:
    x, r, u = dataset.curve.arrays()
    return DatasetCurves(
        dataset.name,
        dataset.curve.axis,
        display_x(dataset.curve.axis, x),
        r,
        u,
        None,
        dataset.mask() & dataset.enabled,
        None,
        dataset.enabled,
    )


def compute_preview(state: FitState, values: Mapping[str, float] | None = None) -> FitPreview:
    """Evaluate a fit state for the views.

    Args:
        state: The fit state.
        values: Parameter values overriding the current ones, by name (e.g. the best values of a
            running fit), or ``None``.

    Returns:
        The preview. Without a structure, with invalid parameters or without fitted points, the
        curves contain the measured data only and `error` says why.
    """
    measured = tuple(_measured_only(d) for d in state.datasets)
    if state.mirror is None:
        return FitPreview(measured, None, state.weighting, error="no structure model")
    try:
        stack = state.mirror.stack(
            None
            if values is None
            else {n: values[n] for n in state.mirror.parameters if n in values}
        )
    except ValueError as exc:
        return FitPreview(measured, None, state.weighting, error=str(exc))
    if not state.datasets:
        return FitPreview((), stack, state.weighting, error="no measured data")
    try:
        problem = FitProblem(
            state.mirror, state.datasets, state.model, state.weighting, library=state.library()
        )
        resolved = problem.resolve(values)
    except (ValueError, KeyError) as exc:
        return FitPreview(measured, stack, state.weighting, error=str(exc))
    curves = []
    for dataset in state.datasets:
        x, r, u = dataset.curve.arrays()
        try:
            model = problem.simulate(dataset, None, resolved, stack)
        except ValueError:
            curves.append(_measured_only(dataset))
            continue
        curves.append(
            DatasetCurves(
                dataset.name,
                dataset.curve.axis,
                display_x(dataset.curve.axis, x),
                r,
                u,
                model,
                problem.fit_mask(dataset),
                _residuals(state.weighting, model, r, u),
                dataset.enabled,
            )
        )
    evaluation = problem.evaluate(values)
    objective = math.nan if evaluation.error is not None else evaluation.objective
    return FitPreview(
        tuple(curves),
        stack,
        state.weighting,
        objective,
        evaluation.points,
        len(problem.parameters.free_names()),
        evaluation.error,
        resolved,
    )
