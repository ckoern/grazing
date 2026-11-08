"""One depth-graded W/Si coating optimized for several fixed grazing angles at once.

Scenario: a single mirror is used at five grazing angles (0.15, 0.18, 0.21, 0.24 and 0.27
degrees) over the energy band 20 to 60 keV, for example on a focusing optic whose incidence
angle varies along its length. The five energy scans are combined in one `TargetSet` with
``combine="average_reflectivity"`` and relative weights 1, 2, 3, 2, 1: the reflectivity curves
are averaged with the weights before the `MeanReflectivity` merit is evaluated, so one coating
is rated for all angles together.

The script runs the default `Pipeline` and prints the mean reflectivity of the final design
for every angle separately.

Run it with ``python examples/coupled_angles.py``. ``--plot`` shows the reflectivity curves,
``--save PATH`` writes the plot to a file and ``--quick`` reduces the optimization budget.
"""

import argparse

from grazing.core.materials import get
from grazing.core.optim.optimizers import LmfitOptimizer
from grazing.core.parameters import free
from grazing.core.structure import DepthGraded, Mirror
from grazing.core.targets import EnergyScan, Target, TargetSet
from grazing.core.units import deg, kev, to_deg, to_kev
from grazing.design.manufacturing import MinLayerThickness
from grazing.design.merit import MeanReflectivity
from grazing.design.optim import DesignProblem, Pipeline
from grazing.design.optim.pipeline import PipelineResult, StageResult
from grazing.design.optim.stages import Stage

ANGLES_DEG = (0.15, 0.18, 0.21, 0.24, 0.27)
WEIGHTS = (1.0, 2.0, 3.0, 2.0, 1.0)


def build_problem() -> DesignProblem:
    """Return the design problem with five coupled energy scans."""
    tungsten, silicon = get("W"), get("Si")
    coating = DepthGraded.create(
        tungsten,
        silicon,
        N=free(80, 20, 250),
        d_min=free(3.0, 1.5, 6.0),
        d_max=free(10.0, 4.0, 25.0),
        c=0.25,
        gamma_top=free(0.4, 0.15, 0.7),
        gamma_bottom=free(0.4, 0.15, 0.7),
        sigma_H=0.35,
        sigma_L=0.35,
    )
    scans: list[Target] = [
        EnergyScan(
            angle=deg(angle),
            energy_min=kev(20),
            energy_max=kev(60),
            points=100,
            weight=weight,
            name=f"{angle} deg",
        )
        for angle, weight in zip(ANGLES_DEG, WEIGHTS, strict=True)
    ]
    return DesignProblem(
        Mirror(coating, silicon, 0.3),
        TargetSet(scans, combine="average_reflectivity"),
        MeanReflectivity(),
        constraints=(MinLayerThickness(0.8),),
    )


def report_stage(stage: StageResult) -> None:
    """Print a one-line summary of a finished stage."""
    r = stage.result
    print(
        f"  {stage.stage.name:8s} merit {r.best.merit:.4f}  "
        f"evaluations {r.evaluations:5d}  {r.elapsed:5.1f} s"
    )


def summarize(problem: DesignProblem, result: PipelineResult) -> None:
    """Print the final design and the mean reflectivity per angle."""
    values = result.parameters.value_dict()
    final = result.final
    assert final.stack is not None
    print("\nFinal design")
    print(f"  N = {values['N']:.0f}")
    print(f"  d_min = {values['d_min']:.2f} nm, d_max = {values['d_max']:.2f} nm")
    print(f"  gamma_top = {values['gamma_top']:.3f}, gamma_bottom = {values['gamma_bottom']:.3f}")
    print(f"  total thickness = {final.stack.total_thickness:.0f} nm, feasible = {final.feasible}")
    print("\nMean reflectivity per angle (20 to 60 keV)")
    for target, res in zip(problem.targets.targets, final.results, strict=True):
        angle = to_deg(float(res.theta[0]))
        print(f"  {angle:.2f} deg (weight {target.weight:g}): {float(res.reflectivity.mean()):.4f}")
    weights = problem.targets.normalized_weights()
    average = sum(
        w * float(r.reflectivity.mean()) for w, r in zip(weights, final.results, strict=True)
    )
    print(f"  weighted average          : {average:.4f}")


def plot(result: PipelineResult, save: str | None) -> None:
    """Plot the reflectivity curve of every angle."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    for res in result.final.results:
        ax.plot(
            to_kev(res.energy), res.reflectivity, label=f"{to_deg(float(res.theta[0])):.2f} deg"
        )
    ax.set(xlabel="energy (keV)", ylabel="reflectivity", ylim=(0, 1.05))
    ax.legend()
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
    parser.add_argument("--plot", action="store_true", help="show the reflectivity curves")
    parser.add_argument("--save", metavar="PATH", help="write the plot to a file")
    parser.add_argument("--quick", action="store_true", help="use a small optimization budget")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    args = parser.parse_args(argv)

    problem = build_problem()
    pipeline = Pipeline.default(
        problem,
        seed=args.seed,
        global_iterations=4 if args.quick else 8,
        scan_points=4 if args.quick else 6,
        coarse_points=30 if args.quick else 40,
    )
    # Limit the evaluations of the final refinement to keep the run time moderate.
    refine_budget = 150 if args.quick else 120
    pipeline = Pipeline(
        tuple(
            Stage("refine", LmfitOptimizer("least_squares", refine_budget))
            if stage.name == "refine"
            else stage
            for stage in pipeline.stages
        )
    )
    print("Stages")
    result = pipeline.run(problem, on_stage=report_stage)
    summarize(problem, result)
    if args.plot or args.save:
        plot(result, args.save)


if __name__ == "__main__":
    main()
