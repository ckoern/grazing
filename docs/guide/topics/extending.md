# Extending xstack

Every extension axis of xstack is a registry: data formats, structure-model readers, coatings,
figures of merit, constraints, optimizers, reflectivity and roughness models. A new component
becomes available everywhere — in the library, in project files and in the GUI's choices — by
registering it. Nothing in xstack has to be edited.

This page shows two examples: a reader for another data format and a new figure of merit. The
last section explains how to ship them as a plugin package that xstack loads automatically.

## A data format

Say your reflectometer writes two-column files with the suffix `.xy`: the grazing angle in
degrees and the reflectivity, with the photon energy in the first line. A parser implements
{py:class}`~xstack.core.io.data.DataParser`: a description, the suffixes it handles, the type of
its options (here none) and `read`, which returns measured curves.

```python
from pathlib import Path

import numpy as np

from xstack.core.io import data_parsers, read_curves
from xstack.core.measurement import MeasuredCurve
from xstack.core.units import deg, kev


class XyParser:
    """Two columns (θ in deg, R); first line: ``energy_keV = <value>``."""

    description = "XY reflectometer file"
    suffixes = (".xy",)
    options_type = None

    def read(self, path: Path, options: object = None) -> tuple[MeasuredCurve, ...]:
        lines = path.read_text().splitlines()
        energy = kev(float(lines[0].split("=")[1]))
        theta, reflectivity = np.loadtxt(lines[1:], unpack=True)
        curve = MeasuredCurve(
            "angle", deg(theta), reflectivity, energy=energy, name=path.stem, source=path.name
        )
        return (curve,)


data_parsers.register("xy", XyParser())
```

Parsers are chosen by the file suffix, so {py:func}`~xstack.core.io.data.read_curves` — and with
it the `xstack fit new` command and the import dialog of xstack-fit — now reads `.xy` files:

```python
Path("scan.xy").write_text("energy_keV = 8.048\n0.2 0.95\n0.4 0.31\n0.6 0.02\n")
(curve,) = read_curves("scan.xy")
print(curve.name, curve.size, curve.energy)
```

## A figure of merit

Figures of merit of the design tool rate the reflectivity curves of a design; the optimizer
minimizes them. Most rate each target's curve separately and derive from
{py:class}`~xstack.design.merit.base.CurveMerit`: `curve_value` returns the value to minimize
for one curve and `curve_residuals` the residual vector used by least-squares optimizers (its sum
of squares should behave like the value). This one maximizes the lowest reflectivity in a band,
smoothed with a soft minimum so that gradient-based optimizers can follow it:

```python
from dataclasses import dataclass
from typing import ClassVar

from xstack.design.merit.base import Curve, CurveMerit, register_merit


@register_merit
@dataclass(frozen=True)
class SoftMinimumReflectivity(CurveMerit):
    """Maximize a soft minimum of the reflectivity: 1 − (−τ·log ⟨exp(−R/τ)⟩).

    Attributes:
        tau: Softness; smaller values approach the true minimum.
    """

    type_name: ClassVar[str] = "soft_minimum_example"
    quantity_label: ClassVar[str | None] = "soft minimum reflectivity"
    tau: float = 0.02

    def curve_quantity(self, curve: Curve) -> float:
        r = curve.reflectivity
        return float(-self.tau * np.log(np.sum(curve.weights * np.exp(-r / self.tau))))

    def curve_value(self, curve: Curve) -> float:
        return 1.0 - self.curve_quantity(curve)

    def curve_residuals(self, curve: Curve) -> np.ndarray:
        return np.array([self.curve_value(curve)])
```

```{note}
xstack already contains a soft-minimum figure of merit; the class above only demonstrates the
mechanism.
```

The decorator registers the class under its `type_name`. It can now be used in a design problem,
is stored in and read from `.xstack` project files, and appears in the figure-of-merit choice of
xstack-design:

```python
from xstack.core.materials import get
from xstack.core.structure import Mirror, Periodic
from xstack.core.targets import EnergyScan, TargetSet
from xstack.core.units import deg
from xstack.design.optim.problem import DesignProblem

mirror = Mirror(Periodic.create(get("W"), get("Si"), N=30, d=3.0, gamma=0.4), get("Si"))
band = EnergyScan(angle=deg(1.5), energy_min=7900.0, energy_max=8200.0, points=50)
problem = DesignProblem(mirror, TargetSet(band), SoftMinimumReflectivity())
print(problem.quantities(problem.evaluate()))
```

Other components follow the same pattern: coatings derive from
{py:class}`~xstack.core.structure.coating.Coating` and use `register_coating`, constraints
`register_constraint`, optimizers {py:class}`~xstack.core.optim.optimizers.Optimizer` and
`register_optimizer`; structure-model readers implement
{py:class}`~xstack.core.io.models.ModelReader` and register in `model_readers`. The registries
and their protocols are listed in [the architecture](../../architecture_core.md).

## Shipping a plugin

Components defined in your own scripts are available as soon as the module is imported. To make
them available to the xstack applications and command line without importing anything, put them
in a package and declare its module in the entry point group `xstack.plugins`:

```{code-block} toml
[project]
name = "xstack-mylab"
dependencies = ["xstack>=0.2"]

[project.entry-points."xstack.plugins"]
mylab = "xstack_mylab.plugin"
```

`xstack_mylab/plugin.py` imports or defines the components; registering happens at import:

```{code-block} python
from xstack.core.io import data_parsers

from xstack_mylab.formats import XyParser

data_parsers.register("xy", XyParser())
```

After `pip install xstack-mylab`, xstack loads the plugin on the first lookup in a registry that
plugins contribute to (for example when reading a data file). A plugin that fails to import is
skipped with a `PluginWarning`, so a broken plugin never stops the applications. The design tool
itself uses this mechanism: it contributes the reader of `.xstack` files to the fitting tool.
