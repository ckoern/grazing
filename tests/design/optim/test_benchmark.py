"""Benchmark of the global (differential-evolution) design stage: runtime and result quality.

Skipped by default. Run with ``pytest tests/design/optim/test_benchmark.py -o addopts=""``.

The problem is the W/B4C Cu Kα monochromator of the user guide: mean reflectivity of s-polarized
light over ±1 % around 8048 eV at 1° grazing angle, with N, d and Γ free and a minimum layer
thickness of 0.8 nm. The global stage of the default pipeline starts far from the optimum; the
benchmark records its runtime, number of evaluations and figure of merit, and checks that a local
refinement from its result does not find a better design.
"""

from typing import Any

import pytest

from xstack.core.materials import get
from xstack.core.optim import LmfitOptimizer
from xstack.core.parameters import free
from xstack.core.structure import Mirror, Periodic
from xstack.core.targets import EnergyScan, TargetSet
from xstack.core.units import deg
from xstack.design.manufacturing import MinLayerThickness
from xstack.design.merit import MeanReflectivity
from xstack.design.optim import DesignProblem, Pipeline

ENERGY = 8048.0


def monochromator() -> DesignProblem:
    coating = Periodic.create(
        get("W"),
        get("B4C"),
        N=free(30, 20, 150),
        d=free(6.0, 2.0, 8.0),
        gamma=free(0.6, 0.1, 0.8),
    )
    target = EnergyScan(
        angle=deg(1.0),
        energy_min=0.99 * ENERGY,
        energy_max=1.01 * ENERGY,
        points=61,
        polarization="s",
    )
    return DesignProblem(
        Mirror(coating, get("Si")),
        TargetSet(target),
        MeanReflectivity(),
        constraints=(MinLayerThickness(0.8),),
    )


def test_global_stage(benchmark: Any) -> None:
    problem = monochromator()
    stage = next(
        s
        for s in Pipeline.default(problem, seed=0, global_iterations=20, population=10).stages
        if s.name == "global"
    )
    result = benchmark.pedantic(stage.run, args=(problem, problem.parameters), rounds=3)
    found = problem.with_parameters(result.parameters).evaluate()
    refined = LmfitOptimizer("least_squares").run(problem.with_parameters(result.parameters))
    benchmark.extra_info.update(
        evaluations=result.evaluations,
        merit=found.merit,
        refined_merit=refined.best.merit,
        start_merit=problem.evaluate().merit,
    )
    assert found.feasible
    assert found.merit < 0.15
    assert found.merit == pytest.approx(refined.best.merit, abs=2e-3)
