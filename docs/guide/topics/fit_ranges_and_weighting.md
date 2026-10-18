# Fit ranges, weighting and the instrument model

Three choices decide what a fit of a reflectivity curve actually determines: which measured points
take part, how their deviations from the model are weighted, and how the instrument is described.
This page explains each of them and what it implies. The examples use the Cu Kα measurement of the
[quick starts](../quickstart_fit_library.md), the snippets of this page run in about a quarter of a
minute.

```python
from xstack.core.io import CsvLayout, read_curves
from xstack.core.materials import get
from xstack.core.parameters import free
from xstack.core.structure import Mirror, Periodic
from xstack.core.units import deg
from xstack.fit.dataset import Dataset
from xstack.fit.project import FitState, run_fit

layout = CsvLayout(
    x_column="2theta_deg",
    reflectivity_column="R",
    uncertainty_column="dR",
    quantity="two_theta",
    energy=8048.0,
)
(curve,) = read_curves("examples/data/WSi_multilayer_cu_ka.csv", layout)


def mirror() -> Mirror:
    return Mirror(
        Periodic.create(
            get("W"),
            get("Si"),
            N=20,
            d=free(4.0, 3.8, 4.2),
            gamma=free(0.4, 0.25, 0.55),
            sigma_H=free(0.3, 0.0, 0.8),
            sigma_L=free(0.3, 0.0, 0.8),
        ),
        substrate=get("Si"),
    )


def dataset(upper_angle=3.0) -> Dataset:
    instrument = Dataset.create(
        "cu_ka",
        curve,
        scale=free(1.0, 0.5, 1.5),
        background=free(1e-7, 0.0, 1e-5),
        offset=free(0.0, -deg(0.02), deg(0.02)),
        resolution=deg(0.004),
        beam_width=0.05e6,
        sample_length=20e6,
    )
    return instrument.with_ranges(
        [(deg(0.35), deg(1.6)), (deg(1.8), deg(upper_angle))],
        excluded=[(deg(2.05), deg(2.15))],
    )


def fit(weighting, upper_angle=3.0):
    state = FitState(mirror=mirror(), datasets=(dataset(upper_angle),), weighting=weighting)
    return run_fit(state)
```

## Fit ranges and excluded ranges

A dataset keeps every measured point. The points that are fitted are those that lie inside at least
one *fit range* (all points if there is none) and outside every *excluded range*. Ranges are
closed intervals of the scanned variable, in rad of the grazing angle θ or in eV of the energy, and
`xstack-fit` shows them in degrees or keV. Excluded points stay in the data and in the plots, in
grey. With the weighting `"log"`, points with R ≤ 0 are not fitted either, and with `"chi2"`,
points with σ ≤ 0.

In `xstack-fit`, **Shift-drag** in the reflectivity plot adds a fit range and **Ctrl-drag** an
excluded range; the shaded ranges can be moved and resized by their edges, or edited in the table
of the Data page. In Python use
{py:meth}`~xstack.fit.dataset.Dataset.with_ranges` or the command
{py:func}`~xstack.fit.project.set_ranges`.

Choose the ranges by asking which points the model can describe, and which points carry
information.

**The region below the critical angle.** Near total reflection the reflectivity is close to one and
changes slowly with the angle. It depends mostly on the footprint of the beam on the sample, the
intensity scale and the alignment, and hardly on the multilayer structure. If you leave the region
in, the fit has to describe the footprint well, otherwise the error of this simplification is
forced into the structure parameters. Excluding it loses the information on those instrument
parameters but protects the structure. The footprint factor of the example
(beam 0.05 mm FWHM, sample 20 mm, see below) shows how far up the effect reaches:

```python
import numpy as np

from xstack.fit.instrument import footprint_factor

angles = np.array([0.02, 0.05, 0.1, 0.2, 0.35])
factor = footprint_factor(deg(angles), beam_width=0.05e6, sample_length=20e6)
for a, f in zip(angles, factor):
    print(f"theta = {a:.2f} deg: {100 * f:.1f} % of the beam hits the sample")
```

**The noise floor.** Where the reflectivity falls to the background and the counting noise, the points say
little about the structure. With a logarithmic weighting they are even harmful, as shown below, because
the logarithm of a noisy small number is erratic. End the fit range where the signal is clearly above
the noise, or accept it only with a fitted background.

**Artifacts and outliers.** Detector glitches, a spike from a second wavelength, or a
region where the model is known to fail should be excluded. A few outliers are not harmless:
with χ² weighting a point several σ away contributes the square of that deviation. A good
excluded range removes the points of the artifact and not much more; in the example it is
θ = 2.05° to 2.15° for an artifact at θ ≈ 2.06° to 2.14°. Do not exclude points only because
they do not fit; a misfit that you remove by selecting points is a hint that the model is wrong.

**Several fit ranges.** Separate ranges are useful when a region is not described by the model
(for example, the third Bragg peak of a measured sample that the model does not reproduce).
Always compare the fit with and without such a gap: parameters that change much depend on the
selection.

## Weighting

The weighting defines the residual of a fitted point. The fit minimizes the sum of the squared
residuals of all fitted points of all datasets (each multiplied by √weight of its dataset).

| Weighting | Residual | Not fitted | Emphasis |
|---|---|---|---|
| `"chi2"` | (R_model − R_meas)/σ | σ ≤ 0 | points with a small absolute uncertainty |
| `"log"` | log₁₀ R_model − log₁₀ R_meas | R_meas ≤ 0 | all decades alike |
| `"linear"` | R_model − R_meas | | the high reflectivity region |

In `xstack-fit` the weighting is chosen on the Fit page: *χ² (measured uncertainties)*,
*Logarithmic* or *Linear*. In Python it is the `weighting` of the
{py:class}`~xstack.fit.problem.FitProblem` or the state of the project, set with
{py:func}`~xstack.fit.project.set_weighting`.

**χ² weighting** is the statistically correct least-squares fit if the uncertainties describe
the noise of the measurement. Then χ²_red is meaningful (about one for a good fit), and the parameter
uncertainties are absolute; see [Uncertainties](uncertainties.md). For counting statistics the
uncertainty grows with √R, so the weight of a point goes with 1/R: the strong parts of the curve,
with the best relative precision, determine the result. Its weakness is that it trusts the
uncertainties: if they are too small somewhere (for example at points with a few counts) or if the
model has a small systematic error where the signal is strong, the fit adapts to them.

**Logarithmic weighting** treats an error of a factor of two the same everywhere, at R = 1 and
at R = 10⁻⁶. This is how a reflectivity curve over many orders of magnitude is judged by eye, and
it needs no uncertainties. It makes the tails, with low counts, as important as the peaks. This is
a problem near the noise floor: the background must then be modeled right, and the weights do not
reflect that the noise there is large. The objective is also very non-quadratic when the model is
far from the data, so a local optimizer is more easily trapped.

**Linear weighting** is dominated by the largest values: the total reflection region and the
Bragg peaks. It determines the peak positions and heights well and ignores the low-reflectivity
tails and the roughness they encode. Use it for a first rough fit, or when only the peaks matter.

The three weightings on the example (the same ranges, local least squares from the same start):

```python
results = {w: fit(w) for w in ("chi2", "linear", "log")}
results["log, up to 2.5 deg"] = fit("log", upper_angle=2.5)
for label, result in results.items():
    stats = result.statistics
    p = result.parameters
    print(
        f"{label:20s} d = {p['d'].value:.4f} nm  gamma = {p['gamma'].value:.4f}  "
        f"sigma_H = {p['sigma_H'].value:.3f} nm  objective/dof = {stats.reduced_objective:.3g}"
    )
```

The true values are d = 4.06 nm, Γ = 0.38, σ_H = 0.32 nm. The χ² and linear fits recover
them. The logarithmic fit to the full range does not: the tail of the curve, at the level of the
background and with few counts per point, dominates it, and the fit ends in a worse minimum. Limiting the
fit range to θ ≤ 2.5°, where the signal is well above the noise, gives the right structure
again. The numbers of *objective/dof* of the weightings cannot be compared with each other: they
have different units (χ² per point, squared decades, squared reflectivity).

**Dataset weight.** In a simultaneous fit each dataset has a weight (the Data page, or
`Dataset.weight`) that multiplies its squared residuals. Use it to balance datasets with very
different numbers of points or to reduce the influence of a less trusted one; see
[Fitting several datasets together](fit_simultaneous.md#dataset-weights). With `"chi2"`, a weight
other than one changes the statistical meaning of the result.

## The instrument model

What is measured is not the reflectivity R of the sample but

> R_meas(x) = scale · F(θ) · (R ⊛ G_σ)(x + offset) + background

where ⊛ G_σ is a convolution with a Gaussian of standard deviation σ (the *resolution*), and F is the
footprint factor. All six quantities are parameters of the
{py:class}`~xstack.fit.instrument.Instrument` of a dataset, which can be fixed or varied like
the parameters of the structure. They appear in the fit as `<dataset>.<name>`. In `xstack-fit`
angles are shown in degrees and the footprint lengths in mm; in the library, angles are in rad and
lengths in nm.

| Parameter | Meaning | Free it when | Keep it fixed when |
|---|---|---|---|
| `scale` | Factor between the true and the measured intensity | The incident intensity is not normalized away, which is almost always the case. | The curve is normalized to the direct beam with confidence (then also check the plateau below the critical angle). |
| `background` | Constant added to the measurement | The curve reaches the noise floor and the background is not subtracted. It matters where R is of the order of the background (typically 10⁻⁶ or lower). | The background was measured and subtracted, or the fit ranges stay far above it. |
| `offset` | True minus nominal value of the scanned variable (rad of θ or eV) | The zero of the angle scale is not calibrated better than a fraction of a Bragg peak width. | The alignment is calibrated by an independent measurement. |
| `resolution` | Gaussian σ of the angular (or energy) resolution | The fringes and minima are less deep than the model predicts, and the resolution of the instrument is not known. | The resolution is known from the instrument or a direct-beam scan; it is a better-determined quantity that way. |
| `beam_width` | FWHM of the beam, in mm | Rarely: usually it is known. | Always, unless the footprint region is fitted and the width is unknown. |
| `sample_length` | Length of the sample along the beam, in mm | Rarely. | Usually; set 0 (together with `beam_width`) to switch the footprint off. |

The model has the following details.

**Scale and background** act on the whole curve, so they are well determined. The scale
correlates with the structure only weakly, but it is partly redundant with the footprint when the
total reflection region is fitted.

**Offset.** The scanned variable is shifted by `offset` before the sample is evaluated: a measured
θ is a true θ + offset. For files with 2θ the offset is in θ, half the error of the 2θ zero. All
Bragg peaks are shifted in proportion to the angle, as are the peaks of a different period, so the
offset and the period are strongly correlated (see the correlation of d and the offset in the
[quick start](../quickstart_fit_gui.md)). Only the shape of the curve over a wide
range, and its fringes, tell them apart. If the correlation is a problem, fix one of the two
from independent knowledge. For energy scans the offset is in eV.

**Resolution.** The convolution with the Gaussian smooths the curve on the scale σ: it fills the
minima between the fringes and rounds the peaks. The measured step of the points limits what
the data can tell: a resolution well below the point spacing has no visible effect. If you
free the resolution, its correlation with the roughnesses and the background must be watched,
since all three change the depth of the minima. Free the resolution when the minima of the data
are clearly less deep than the fit's, and when the instrument's resolution is not known; keep it
fixed, once you have it from a measurement. A resolution that is too large for the true instrument
shows up as too much smoothing of fine fringes and a roughness that compensates it.

**Footprint.** F(θ) = erf(L sin θ / (2√2 σ_b)) with σ_b = beam width/(2√(2 ln 2)) is the fraction of
a Gaussian beam of full width at half maximum `beam_width` that hits a sample of length `L` =
`sample_length`. It is 1 for angles with L sin θ much larger than the beam width and drops at small
angles (the table printed above for the example). F is 1 if either length is zero. Include it
when the fit range reaches below the angle where F is close to 1.

### What to free: a recipe

1. Free what is unknown and clearly determined by the fit ranges: the structure parameters of
   interest, the `scale` and, for a curve that reaches the noise, the `background`.
2. Free the `offset` unless the zero is calibrated. Expect a strong correlation with the
   period.
3. Fix the `resolution`, the `beam_width` and the `sample_length` at the known values at first. Free
   the resolution only if the fit residuals show systematically deeper minima in the data than in the
   model after the structure is right.
4. After the first fit, look at the correlations and the uncertainties (see
   [Uncertainties](uncertainties.md)) and at the residuals. If a parameter is *undetermined* or
   at a bound, fix it or change its bound.
5. Check how much the structure parameters change when you fix or free an instrument parameter
   and compare the saved fits. A result that depends on such a choice is not well determined.
