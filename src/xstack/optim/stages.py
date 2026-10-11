"""Design stages: an optimizer together with the effort spent on the problem.

A `Stage` runs an optimizer on a variant of the design problem, e.g. with a faster model, a
coarser grid, another figure of merit, or with only some parameters free. `BraggOutline` is
an analytic stage that derives start values and bounds from the Bragg law.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import ClassVar

import numpy as np

from xstack.materials import default_library
from xstack.merit.base import FigureOfMerit
from xstack.optim.events import CancellationToken, EventSink, ProgressEvent
from xstack.optim.optimizers import (
    OptimizationResult,
    Optimizer,
    Tracker,
    prepare_parameters,
    register_optimizer,
)
from xstack.optim.problem import DesignProblem
from xstack.parameters import ParameterSet
from xstack.physics import bragg
from xstack.physics.models import ReflectivityModel
from xstack.structure.coatings.graded import DepthGraded
from xstack.structure.coatings.periodic import Periodic
from xstack.targets import AngleScan, EnergyScan, TargetSet
from xstack.units import energy_to_wavelength

__all__ = ["BraggOutline", "Stage"]


@dataclass(frozen=True)
class Stage:
    """One step of a design pipeline.

    Attributes:
        name: Name of the stage, used in progress events and results.
        optimizer: The optimizer.
        model: Reflectivity model for this stage, or ``None`` to keep the problem's model.
        points: Grid points per target, or ``None`` to keep the problem's setting.
        merit: Figure of merit for this stage, or ``None`` to keep the problem's.
        free: Names of the parameters that may vary in this stage, or ``None`` for all
            parameters that are free in the start parameters. Parameters not listed are kept
            at their current values.
    """

    name: str
    optimizer: Optimizer
    model: ReflectivityModel | None = None
    points: int | None = None
    merit: FigureOfMerit | None = None
    free: tuple[str, ...] | None = None

    def problem_for(self, problem: DesignProblem) -> DesignProblem:
        """Return the variant of a problem used in this stage.

        Args:
            problem: The design problem.

        Returns:
            The problem with model, grid points and figure of merit of this stage.
        """
        changes: dict[str, object] = {}
        if self.model is not None:
            changes["model"] = self.model
        if self.points is not None:
            changes["points"] = self.points
        if self.merit is not None:
            changes["merit"] = self.merit
        return problem.evolve(**changes) if changes else problem

    def run(
        self,
        problem: DesignProblem,
        start: ParameterSet | None = None,
        events: EventSink | None = None,
        cancel: CancellationToken | None = None,
    ) -> OptimizationResult:
        """Run the stage.

        Args:
            problem: The design problem.
            start: Start parameters. Defaults to the problem's parameters.
            events: Receiver of progress events; their task name is prefixed with the stage
                name.
            cancel: Cancellation token.

        Returns:
            The optimization result. Its parameters keep the free/fixed flags of `start`.
        """
        params = start if start is not None else problem.parameters
        restricted = params
        if self.free is not None:
            for name, p in params.items():
                if p.vary and name not in self.free:
                    restricted = restricted.replace(name, vary=False)

        def forward(event: ProgressEvent) -> None:
            if events is not None:
                events(replace(event, task=f"{self.name}: {event.task}"))

        result = self.optimizer.run(self.problem_for(problem), restricted, forward, cancel)
        restored = result.parameters
        for name, p in params.items():
            if restored[name].vary != p.vary:
                restored = restored.replace(name, vary=p.vary)
        return replace(result, parameters=restored, method=f"{self.name}: {result.method}")


def _target_extremes(
    targets: TargetSet,
) -> list[tuple[float, float, float]]:
    """Return (energy, angle, weight) at both ends and the centre of every target."""
    points = []
    for t in targets.targets:
        if isinstance(t, EnergyScan):
            for e in (t.energy_min, 0.5 * (t.energy_min + t.energy_max), t.energy_max):
                points.append((e, t.angle, t.weight))
        elif isinstance(t, AngleScan):
            for a in (t.angle_min, 0.5 * (t.angle_min + t.angle_max), t.angle_max):
                points.append((t.energy, a, t.weight))
    return points


@register_optimizer
@dataclass(frozen=True)
class BraggOutline(Optimizer):
    """Analytic first estimate of a design from the Bragg law.

    For periodic multilayers, the period is chosen to reflect the centre of the target band,
    Γ is set to the Vinogradov–Zeldovich optimum and the number of bilayers is narrowed to a
    range around the saturation estimate. For depth-graded multilayers, ``d_max`` and ``d_min``
    are chosen to reflect the lower and upper end of the band (in energy or momentum transfer)
    and Γ is set to the optimum. Only free parameters are changed; the bounds of free period
    parameters are narrowed to ±`margin` around the estimates (within the original bounds).
    Other coatings are left unchanged.

    Attributes:
        refraction: Use the refraction-corrected Bragg law.
        margin: Relative half width of the narrowed bounds of period parameters.
    """

    type_name: ClassVar[str] = "bragg_outline"
    refraction: bool = False
    margin: float = 0.2

    def run(
        self,
        problem: DesignProblem,
        start: ParameterSet | None = None,
        events: EventSink | None = None,
        cancel: CancellationToken | None = None,
    ) -> OptimizationResult:
        """Derive start values and bounds and evaluate the resulting design.

        Args:
            problem: The design problem.
            start: Start parameters. Defaults to the problem's parameters.
            events: Receiver of progress events.
            cancel: Not used; the stage is not iterative.

        Returns:
            The result with the estimated parameters.
        """
        params = prepare_parameters(problem, start, vary_integers=True)
        coating = problem.mirror.coating
        if isinstance(coating, Periodic | DepthGraded):
            params = self._estimate(problem, params, coating)
        tracker = Tracker(problem, self.type_name, events, cancel, 1)
        tracker.record(problem.with_parameters(params).evaluate())
        return tracker.result(params, "analytic estimate")

    def _estimate(
        self, problem: DesignProblem, params: ParameterSet, coating: Periodic | DepthGraded
    ) -> ParameterSet:
        library = problem.library or default_library()
        h, low = coating.material("H"), coating.material("L")
        values = params.evaluate()
        gamma_name = "gamma" if isinstance(coating, Periodic) else "gamma_top"
        gamma = values[gamma_name]

        def period(energy: float, angle: float) -> float:
            wavelength = float(energy_to_wavelength(energy))
            delta = 0.0
            if self.refraction:
                db_h = library.delta_beta(h, energy)
                db_l = library.delta_beta(low, energy)
                delta = float(bragg.mean_delta(db_h.real, db_l.real, gamma))
            d = float(bragg.bragg_period(angle, wavelength, 1, delta))
            return d if math.isfinite(d) else float(bragg.bragg_period(angle, wavelength))

        extremes = _target_extremes(problem.targets)
        periods = np.array([period(e, a) for e, a, _ in extremes])
        energies = np.array([e for e, _, _ in extremes])
        e_centre = float(np.average(energies, weights=[w for _, _, w in extremes]))
        updates: dict[str, tuple[float, float | None, float | None]] = {}

        db_h, db_l = library.delta_beta(h, e_centre), library.delta_beta(low, e_centre)
        if float(db_h.imag) > float(db_l.imag):
            g_opt = bragg.optimal_gamma(float(db_h.imag), float(db_l.imag))
            for name in ("gamma", "gamma_top", "gamma_bottom"):
                updates[name] = (g_opt, None, None)

        if isinstance(coating, Periodic):
            d = float(np.median(periods))
            updates["d"] = (d, d * (1 - self.margin), d * (1 + self.margin))
            angle = float(np.median([a for _, a, _ in extremes]))
            g = updates.get("gamma", (gamma, None, None))[0]
            n_sat = min(
                bragg.extinction_bilayers(
                    float(db_h.real), float(db_h.imag), float(db_l.real), float(db_l.imag), g, angle
                ),
                bragg.absorption_bilayers(
                    float(db_h.imag),
                    float(db_l.imag),
                    g,
                    d,
                    angle,
                    float(energy_to_wavelength(e_centre)),
                ),
            )
            if math.isfinite(n_sat):
                updates["N"] = (
                    float(round(2 * n_sat)),
                    float(max(1, math.floor(0.5 * n_sat))),
                    float(math.ceil(4 * n_sat)),
                )
        else:
            d_max, d_min = float(periods.max()), float(periods.min())
            updates["d_max"] = (d_max, d_max * (1 - self.margin), d_max * (1 + self.margin))
            updates["d_min"] = (d_min, d_min * (1 - self.margin), d_min * (1 + self.margin))
        return _apply(params, updates)


def _apply(
    params: ParameterSet, updates: Mapping[str, tuple[float, float | None, float | None]]
) -> ParameterSet:
    """Apply estimated values and narrowed bounds to the free parameters."""
    for name, (value, lo, hi) in updates.items():
        if name not in params or not params[name].is_free:
            continue
        p = params[name]
        new_lo = p.min if lo is None else max(p.min, lo)
        new_hi = p.max if hi is None else min(p.max, hi)
        if new_lo > new_hi:
            new_lo, new_hi = p.min, p.max
        clipped = min(max(value, new_lo), new_hi)
        if p.integer:
            clipped = float(min(max(round(clipped), math.ceil(new_lo)), math.floor(new_hi)))
        params = params.replace(name, value=clipped, min=new_lo, max=new_hi)
    return params
