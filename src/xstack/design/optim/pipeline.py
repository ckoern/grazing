"""Staged design pipeline."""

from collections.abc import Callable
from dataclasses import dataclass, field

from xstack.core.optim.events import CancellationToken, EventSink
from xstack.core.optim.optimizers import LmfitOptimizer, OptimizationResult
from xstack.core.parameters import ParameterSet
from xstack.core.physics.models import KinematicModel
from xstack.core.serialization import register_serializable
from xstack.design.optim.problem import DesignProblem, Evaluation
from xstack.design.optim.scan import GridScan
from xstack.design.optim.stages import BraggOutline, Stage

__all__ = ["Pipeline", "PipelineResult", "StageResult"]


@dataclass(frozen=True)
class StageResult:
    """Result of one stage of a pipeline.

    Attributes:
        stage: The stage.
        result: Its optimization result.
    """

    stage: Stage
    result: OptimizationResult[Evaluation]


@dataclass(frozen=True)
class PipelineResult:
    """Result of a pipeline run.

    Attributes:
        stages: Results of the stages that were run, in order.
        parameters: Final parameters (from the last stage that was run).
        final: Evaluation of the final parameters with the original design problem.
        cancelled: Whether the run was cancelled.
    """

    stages: tuple[StageResult, ...]
    parameters: ParameterSet
    final: Evaluation
    cancelled: bool


@dataclass(frozen=True)
class Pipeline:
    """An ordered sequence of stages.

    Every stage result is evaluated with the original design problem. Each stage starts from the
    best design so far: the result of the previous stage (including bounds that it narrowed) if
    it improved the objective, otherwise the design the previous stage started from. The final
    parameters carry the values of the best design, which is never worse than the start
    design, with the bounds and free/fixed settings of the start parameters.

    Attributes:
        stages: The stages in order.
    """

    stages: tuple[Stage, ...] = field(default=())

    def run(
        self,
        problem: DesignProblem,
        start: ParameterSet | None = None,
        events: EventSink | None = None,
        cancel: CancellationToken | None = None,
        on_stage: Callable[[StageResult], None] | None = None,
    ) -> PipelineResult:
        """Run all stages.

        Args:
            problem: The design problem.
            start: Start parameters. Defaults to the problem's parameters.
            events: Receiver of progress events.
            cancel: Cancellation token; the current stage stops and no further stage runs.
            on_stage: Called after every completed stage.

        Returns:
            The pipeline result.
        """
        initial = start if start is not None else problem.parameters
        params = initial
        best = problem.with_parameters(initial).evaluate().objective
        results: list[StageResult] = []
        for stage in self.stages:
            if cancel is not None and cancel.cancelled:
                break
            outcome = StageResult(stage, stage.run(problem, params, events, cancel))
            results.append(outcome)
            found = outcome.result.parameters
            objective = problem.with_parameters(found).evaluate().objective
            if objective < best:
                params, best = found, objective
            if on_stage is not None:
                on_stage(outcome)
        final_values = {n: v for n, v in params.value_dict().items() if initial[n].expr is None}
        params = initial.with_values(final_values)
        final = problem.with_parameters(params).evaluate()
        cancelled = cancel is not None and cancel.cancelled
        return PipelineResult(tuple(results), params, final, cancelled)

    @classmethod
    def default(
        cls,
        problem: DesignProblem,
        *,
        scan_points: int = 9,
        coarse_points: int = 60,
        global_iterations: int = 30,
        population: int = 10,
        seed: int | None = None,
    ) -> "Pipeline":
        """Return the default staged pipeline for a problem.

        1. ``outline``: analytic Bragg-law estimate without refraction (`BraggOutline`).
        2. ``scan``: grid scan over the free Γ parameters and free integer parameters on a
           coarse target grid (skipped if there are none).
        3. ``global``: differential evolution over all free parameters on the coarse grid.
        4. ``refine``: local least-squares refinement with the full grid (integer parameters
           fixed).

        Args:
            problem: The design problem; used to choose the scan parameters.
            scan_points: Grid points per scanned parameter.
            coarse_points: Target grid points in the scan and global stages (at most the
                number of points of the problem).
            global_iterations: Number of differential-evolution generations that the evaluation
                budget of the global stage allows for.
            population: Differential-evolution population size multiplier.
            seed: Random seed for reproducible runs.

        Returns:
            The pipeline.
        """
        params = problem.parameters
        scan_names = tuple(
            n
            for n, p in params.items()
            if p.is_free and (n.rsplit(".", 1)[-1].startswith(("gamma", "cap_gamma")) or p.integer)
        )
        target_points = min(t.points for t in problem.targets.targets)
        coarse = min(coarse_points, problem.points or target_points)
        stages = [Stage("outline", BraggOutline(), model=KinematicModel(refraction=False))]
        if scan_names:
            stages.append(
                Stage("scan", GridScan(points=scan_points), points=coarse, free=scan_names)
            )
        de_options = (("popsize", population), ("seed", seed), ("polish", False))
        free_count = max(1, len(params.free_names()))
        budget = (global_iterations + 1) * population * free_count
        stages += [
            Stage(
                "global",
                LmfitOptimizer("differential_evolution", budget, de_options),
                points=coarse,
            ),
            Stage("refine", LmfitOptimizer("least_squares")),
        ]
        return cls(tuple(stages))


register_serializable("Pipeline", Pipeline)
