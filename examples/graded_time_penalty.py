"""Depth-graded W/Si supermirror with a coating-time penalty.

Scenario: a broadband hard X-ray mirror reflects 15 to 70 keV at a fixed grazing angle of
0.2 degrees. The coating is a depth-graded W/Si multilayer with a power-law period profile
(bilayer number N, shortest and longest period d_min and d_max, power-law constant c, and
separate absorber fractions at the top and the bottom of the stack).

The deposition process is magnetron sputtering: W grows at 0.1 nm/s and Si at 0.2 nm/s, every
layer costs 20 s of overhead and 30 min are needed for set-up. Constraints require layers of at
least 0.8 nm, a total thickness of at most 1.5 micrometres and a coating time of at most 24 h.

The figure of merit is ``TimePenalized(MeanReflectivity())``: (1 - w)(1 - <R>) + w T / T_ref
with the coating time T, the reference time T_ref = 8 h and the weight w given by ``--weight``
(0 optimizes the reflectivity only; larger values trade reflectivity for a shorter process).
After the optimization, the sensitivity to random thickness errors (2 % per layer, 20 samples)
is estimated, and ``--recipe PATH`` writes the layer table as a CSV file.

Run it with ``python examples/graded_time_penalty.py``. ``--plot`` shows the reflectivity,
``--save PATH`` writes the plot to a file and ``--quick`` reduces the optimization budget.
"""

import argparse
from pathlib import Path

from xstack.io.recipe import write_recipe_csv
from xstack.manufacturing import (
    DepositionProcess,
    MaxCoatingTime,
    MaxTotalThickness,
    MinLayerThickness,
    thickness_tolerance,
)
from xstack.materials import get
from xstack.merit import MeanReflectivity, TimePenalized
from xstack.optim import DesignProblem, Pipeline
from xstack.optim.optimizers import LmfitOptimizer
from xstack.optim.pipeline import PipelineResult, StageResult
from xstack.optim.stages import Stage
from xstack.parameters import free
from xstack.structure import DepthGraded, Mirror
from xstack.targets import EnergyScan, TargetSet
from xstack.units import deg, kev, to_kev

ANGLE = deg(0.2)
HOUR = 3600.0


def build_problem(weight: float) -> DesignProblem:
    """Return the design problem.

    Args:
        weight: Weight w of the time penalty in [0, 1].
    """
    tungsten, silicon = get("W"), get("Si")
    coating = DepthGraded.create(
        tungsten,
        silicon,
        N=free(100, 20, 300),
        d_min=free(3.0, 1.5, 6.0),
        d_max=free(10.0, 5.0, 25.0),
        c=0.25,
        gamma_top=free(0.4, 0.15, 0.7),
        gamma_bottom=free(0.4, 0.15, 0.7),
        sigma_H=0.35,
        sigma_L=0.35,
    )
    process = DepositionProcess(
        {"W": 0.1, "Si": 0.2}, overhead_per_layer=20.0, fixed_overhead=1800.0
    )
    targets = TargetSet(EnergyScan(angle=ANGLE, energy_min=kev(15), energy_max=kev(70), points=150))
    return DesignProblem(
        Mirror(coating, silicon, 0.3),
        targets,
        TimePenalized(MeanReflectivity(), weight=weight, reference_time=8 * HOUR),
        constraints=(MinLayerThickness(0.8), MaxTotalThickness(1500.0), MaxCoatingTime(24 * HOUR)),
        process=process,
    )


def report_stage(stage: StageResult) -> None:
    """Print a one-line summary of a finished stage."""
    r = stage.result
    print(
        f"  {stage.stage.name:8s} objective {r.best.objective:.4f}  "
        f"evaluations {r.evaluations:5d}  {r.elapsed:5.1f} s"
    )


def summarize(problem: DesignProblem, result: PipelineResult, quick: bool) -> None:
    """Print the final design and its tolerance to thickness errors."""
    final = result.final
    assert final.stack is not None
    assert final.coating_time is not None
    values = result.parameters.value_dict()
    print("\nFinal design")
    print(f"  N = {values['N']:.0f}")
    print(f"  d_min = {values['d_min']:.2f} nm, d_max = {values['d_max']:.2f} nm")
    print(f"  gamma_top = {values['gamma_top']:.3f}, gamma_bottom = {values['gamma_bottom']:.3f}")
    print(f"  mean reflectivity = {float(final.results[0].reflectivity.mean()):.4f}")
    print(f"  coating time      = {final.coating_time / HOUR:.2f} h")
    print(f"  total thickness   = {final.stack.total_thickness:.0f} nm")
    print(f"  layers            = {final.stack.num_layers}")
    print(f"  feasible          = {final.feasible}")
    tol = thickness_tolerance(
        problem, final.values, random_error=0.02, samples=8 if quick else 20, seed=0
    )
    print("\nTolerance to 2 % random thickness errors")
    print(
        f"  figure of merit: nominal {tol.nominal.merit:.4f}, median {tol.percentile(50):.4f}, "
        f"90th percentile {tol.percentile(90):.4f}"
    )
    print(f"  feasible samples: {int(tol.feasible.sum())} of {len(tol.feasible)}")


def plot(problem: DesignProblem, result: PipelineResult, save: str | None) -> None:
    """Plot the reflectivity of the final design."""
    import matplotlib.pyplot as plt

    curve = result.final.results[0]
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.plot(to_kev(curve.energy), curve.reflectivity)
    ax.set(xlabel="energy (keV)", ylabel="reflectivity", ylim=(0, 1.05))
    ax.set_title("Depth-graded W/Si at 0.2 deg")
    if save:
        fig.savefig(save, dpi=150)
        print(f"saved {save}")
    else:
        plt.show()
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    """Run the optimization and print the results.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else None)
    parser.add_argument("--weight", type=float, default=0.2, help="weight of the time penalty")
    parser.add_argument("--recipe", metavar="PATH", help="write the layer table as CSV")
    parser.add_argument("--plot", action="store_true", help="show the reflectivity")
    parser.add_argument("--save", metavar="PATH", help="write the plot to a file")
    parser.add_argument("--quick", action="store_true", help="use a small optimization budget")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    args = parser.parse_args(argv)

    problem = build_problem(args.weight)
    pipeline = Pipeline.default(
        problem,
        seed=args.seed,
        global_iterations=4 if args.quick else 15,
        scan_points=4 if args.quick else 7,
        coarse_points=40 if args.quick else 80,
    )
    # Limit the evaluations of the final refinement to keep the run time moderate.
    refine_budget = 150 if args.quick else 600
    pipeline = Pipeline(
        tuple(
            Stage("refine", LmfitOptimizer("least_squares", refine_budget))
            if stage.name == "refine"
            else stage
            for stage in pipeline.stages
        )
    )
    print(f"Optimizing with time weight {args.weight:g}")
    print("Stages")
    result = pipeline.run(problem, on_stage=report_stage)
    summarize(problem, result, args.quick)
    if args.recipe:
        assert result.final.stack is not None
        write_recipe_csv(Path(args.recipe), result.final.stack, problem.process)
        print(f"\nwrote recipe to {args.recipe}")
    if args.plot or args.save:
        plot(problem, result, args.save)


if __name__ == "__main__":
    main()
