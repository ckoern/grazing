"""Reflectivity of periodic and depth-graded W/Si multilayers.

This script shows how to

* build a periodic W/Si multilayer and a depth-graded (power-law) W/Si multilayer with a cap,
* define an angle scan at 8048 eV and a coupled set of three energy scans (15 to 70 keV at
  0.15, 0.2 and 0.25 degrees grazing incidence, weights 1, 2 and 1),
* compute the reflectivity with the exact `ParrattModel`,
* rate the designs with the `MeanReflectivity` figure of merit, and
* plot the curves.

Run it with ``python examples/reflectivity_demo.py``. With ``--save`` the figure is written to
``reflectivity_demo.png`` next to the script instead of being shown.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

from xstack.core.materials import get
from xstack.core.physics.models import ParrattModel
from xstack.core.structure import DepthGraded, Mirror, Periodic
from xstack.core.targets import AngleScan, EnergyScan, TargetSet
from xstack.core.units import deg, kev, mrad, to_deg, to_kev, to_mrad
from xstack.design.merit import EvaluationContext, MeanReflectivity


def main(save: bool = False) -> None:
    """Compute, print and plot the reflectivity of both designs.

    Args:
        save: Write the figure to ``reflectivity_demo.png`` next to this script instead of
            showing it.
    """
    tungsten, silicon = get("W"), get("Si")
    model = ParrattModel()
    merit = MeanReflectivity()

    periodic = Mirror(Periodic.create(tungsten, silicon, N=30, d=4.0, gamma=0.4), silicon).stack()
    graded = Mirror(
        DepthGraded.create(
            tungsten,
            silicon,
            N=150,
            d_min=2.5,
            d_max=8.0,
            c=0.3,
            gamma_top=0.35,
            cap=True,
            cap_d=10.0,
            cap_gamma=0.3,
        ),
        silicon,
    ).stack()

    angle_scan = AngleScan(energy=8048.0, angle_min=mrad(8), angle_max=mrad(40), points=600)
    energy_scans = TargetSet(
        [
            EnergyScan(
                angle=deg(angle),
                energy_min=kev(15),
                energy_max=kev(70),
                points=300,
                weight=weight,
                name=f"{angle} deg",
            )
            for angle, weight in ((0.15, 1.0), (0.2, 2.0), (0.25, 1.0))
        ]
    )

    fig, (ax_angle, ax_energy) = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)

    # (a) Angle scan at 8048 eV.
    targets_a = TargetSet(angle_scan)
    for label, stack in (("periodic", periodic), ("graded", graded)):
        res = model.evaluate(stack, angle_scan)
        ctx = EvaluationContext(stack, targets_a, (res,))
        print(f"{label:9s} angle scan at 8048 eV: 1 - <R> = {merit.scalar(ctx):.4f}")
        ax_angle.semilogy(to_mrad(res.x), res.reflectivity, label=label)
    ax_angle.set(xlabel="grazing angle (mrad)", ylabel="reflectivity", ylim=(1e-5, 1.5))
    ax_angle.set_title("Angle scan, 8048 eV")
    ax_angle.legend()

    # (b) Coupled energy scans: the merit sees the weighted average of the three curves.
    for label, stack in (("periodic", periodic), ("graded", graded)):
        results = tuple(model.evaluate(stack, t) for t in energy_scans.targets)
        ctx = EvaluationContext(stack, energy_scans, results)
        print(f"{label:9s} coupled energy scans:   1 - <R> = {merit.scalar(ctx):.4f}")
        average = ctx.curves()[0].reflectivity
        ax_energy.plot(to_kev(results[0].x), average, label=f"{label} (weighted average)")
        if label == "graded":
            for res, t in zip(results, energy_scans.targets, strict=True):
                ax_energy.plot(
                    to_kev(res.x),
                    res.reflectivity,
                    lw=0.8,
                    ls="--",
                    label=f"graded at {to_deg(t.angle):.2f} deg",  # type: ignore[attr-defined]
                )
    ax_energy.set(xlabel="energy (keV)", ylabel="reflectivity", ylim=(0, 1.05))
    ax_energy.set_title("Energy scans, 0.15 / 0.2 / 0.25 deg")
    ax_energy.legend(fontsize=8)

    if save:
        path = Path(__file__).with_name("reflectivity_demo.png")
        fig.savefig(path, dpi=150)
        print(f"saved {path}")
    else:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main(save="--save" in sys.argv[1:])
