# Quick reflectivity calculations

To calculate or compare reflectivity curves in a short script, define a mirror and call one of the
functions of {py:mod}`grazing.core.calc`. They return the reflectivity at the values you scan:

```python
import numpy as np

from grazing.core.calc import reflectivity_vs_angle, reflectivity_vs_energy
from grazing.core.materials import get
from grazing.core.structure import Mirror, Periodic
from grazing.core.units import deg, kev

W, Si = get("W"), get("Si")
mirror = Mirror(Periodic.create(W, Si, N=30, d=4.0, gamma=0.4, sigma_H=0.3, sigma_L=0.3), Si)

angles = deg(np.linspace(0.2, 3.0, 600))
energies = kev(np.linspace(15.0, 70.0, 300))

r_angle = reflectivity_vs_angle(mirror, angles, energy=8048.0)
r_energy = reflectivity_vs_energy(mirror, energies, angle=deg(0.2))
print(f"highest reflectivity above 1 deg: {r_angle[angles > deg(1.0)].max():.3f}")
```

Angles are in rad and energies in eV, like everywhere in the library; `deg` and `kev` convert them.
The fixed value must be given by keyword (`energy=` or `angle=`). If it is an array too, the
arguments broadcast against each other, which gives a map.

Optional arguments choose the polarization (unpolarized by default), the reflectivity model
(Parratt with Névot–Croce roughness by default) and the optical constants:

```python
from grazing.core.physics.models import ParrattModel

r_s = reflectivity_vs_angle(mirror, angles, energy=8048.0, polarization="s")
r_kinematic = reflectivity_vs_angle(mirror, angles, energy=8048.0, model="kinematic")
r_debye_waller = reflectivity_vs_angle(
    mirror, angles, energy=8048.0, model=ParrattModel(interface_model="debye_waller")
)
r_chantler = reflectivity_vs_angle(mirror, angles, energy=8048.0, provider="chantler")
```

Comparing designs is a loop over mirrors:

```python
for d in (3.8, 4.0, 4.2):
    r = reflectivity_vs_angle(mirror.with_parameter("d", value=d), angles, energy=8048.0)
    print(f"d = {d} nm: highest reflectivity above 1 deg {r[angles > deg(1.0)].max():.3f}")
```

These functions compute a single reflection at exactly the given points. For instrument resolution,
several reflections or a figure of merit, define targets as in the
[library quick start](../quickstart_design_library.md).
