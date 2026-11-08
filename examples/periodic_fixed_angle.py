"""Periodic W/B4C multilayer for Cu K-alpha at a fixed grazing angle.

Scenario: a monochromator mirror reflects the Cu K-alpha line (8048 eV) at a fixed grazing
angle of 1.0 degree. The target is a narrow energy band of +-1 % around the line. The free
parameters are the period d, the absorber fraction gamma (W thickness over period) and the
number of bilayers N (20 to 150). A minimum layer thickness of 0.8 nm keeps the design
manufacturable.

The default `Pipeline` is used: an analytic Bragg-law estimate, a grid scan over gamma and N,
a global differential-evolution search and a local least-squares refinement. The script prints
a summary of every stage and the final design, and compares the optimized gamma with the
Vinogradov-Zeldovich optimum `bragg.optimal_gamma` of the two materials. The optimum is a
very thin W layer; the minimum layer thickness constraint pushes the optimized gamma above it.

Run it with ``python examples/periodic_fixed_angle.py``. ``--plot`` shows the reflectivity
curve, ``--save PATH`` writes it to a file instead, and ``--quick`` reduces the optimization
budget.
"""

import argparse

from grazing.core.materials import default_library, get
from grazing.core.parameters import free
from grazing.core.physics import bragg
from grazing.core.structure import Mirror, Periodic
from grazing.core.targets import EnergyScan, TargetSet
from grazing.core.units import deg, energy_to_wavelength, to_deg
from grazing.design.manufacturing import MinLayerThickness
from grazing.design.merit import MeanReflectivity
from grazing.design.optim import DesignProblem, Pipeline
from grazing.design.optim.pipeline import PipelineResult, StageResult

ENERGY = 8048.0  # Cu K-alpha in eV
ANGLE = deg(1.0)


def build_problem() -> DesignProblem:
    """Return the design problem of this example."""
    tungsten, carbide, silicon = get("W"), get("B4C"), get("Si")
    coating = Periodic.create(
        tungsten,
        carbide,
        N=free(60, 20, 150),
        d=free(4.4, 2.0, 8.0),
        gamma=free(0.4, 0.1, 0.8),
    )
    targets = TargetSet(
        EnergyScan(
            angle=ANGLE,
            energy_min=0.99 * ENERGY,
            energy_max=1.01 * ENERGY,
            points=61,
            polarization="s",
        )
    )
    return DesignProblem(
        Mirror(coating, silicon),
        targets,
        MeanReflectivity(),
        constraints=(MinLayerThickness(0.8),),
    )


def report_stage(stage: StageResult) -> None:
    """Print a one-line summary of a finished stage."""
    r = stage.result
    print(
        f"  {stage.stage.name:8s} merit {r.best.merit:.4f}  objective {r.best.objective:.4f}  "
        f"evaluations {r.evaluations:5d}  {r.elapsed:5.1f} s"
    )


def summarize(problem: DesignProblem, result: PipelineResult) -> None:
    """Print the final design and compare gamma with the analytic optimum."""
    values = result.parameters.value_dict()
    final = result.final
    print("\nFinal design")
    print(f"  period d       = {values['d']:.3f} nm")
    print(f"  gamma          = {values['gamma']:.3f}")
    print(f"  bilayers N     = {values['N']:.0f}")
    print(f"  mean R in band = {float(final.results[0].reflectivity.mean()):.4f}")
    print(f"  feasible       = {final.feasible}")
    library = problem.library or default_library()
    beta_w = float(library.delta_beta(get("W"), ENERGY).imag)
    beta_c = float(library.delta_beta(get("B4C"), ENERGY).imag)
    print(f"  optimal gamma (Vinogradov-Zeldovich) = {bragg.optimal_gamma(beta_w, beta_c):.3f}")
    d_bragg = float(bragg.bragg_period(ANGLE, float(energy_to_wavelength(ENERGY))))
    print(f"  W thickness    = {values['gamma'] * values['d']:.3f} nm (minimum allowed 0.8 nm)")
    print(f"  plain Bragg period at {to_deg(ANGLE):.1f} deg   = {d_bragg:.3f} nm")


def plot(problem: DesignProblem, result: PipelineResult, save: str | None) -> None:
    """Plot the reflectivity of the final design around the Cu K-alpha line."""
    import matplotlib.pyplot as plt

    scan = problem.targets.targets[0]
    wide = EnergyScan(
        angle=ANGLE,
        energy_min=0.95 * ENERGY,
        energy_max=1.05 * ENERGY,
        points=400,
        polarization=scan.polarization,
    )
    wide_problem = problem.evolve(targets=TargetSet(wide)).with_parameters(result.parameters)
    curve = wide_problem.evaluate().results[0]
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.plot(curve.energy, curve.reflectivity)
    ax.axvspan(0.99 * ENERGY, 1.01 * ENERGY, alpha=0.2, label="target band")
    ax.set(xlabel="energy (eV)", ylabel="reflectivity", title="W/B4C at 1.0 deg")
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
    parser.add_argument("--plot", action="store_true", help="show the reflectivity curve")
    parser.add_argument("--save", metavar="PATH", help="write the plot to a file")
    parser.add_argument("--quick", action="store_true", help="use a small optimization budget")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    args = parser.parse_args(argv)

    problem = build_problem()
    start = problem.evaluate()
    print(f"Start design: mean R = {1.0 - start.merit:.4f}")
    pipeline = Pipeline.default(
        problem,
        seed=args.seed,
        global_iterations=4 if args.quick else 30,
        population=6 if args.quick else 10,
        coarse_points=21 if args.quick else 61,
    )
    print("Stages")
    result = pipeline.run(problem, on_stage=report_stage)
    summarize(problem, result)
    if args.plot or args.save:
        plot(problem, result, args.save)


if __name__ == "__main__":
    main()
