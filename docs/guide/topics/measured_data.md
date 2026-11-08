# Measured data

A fit starts from measured reflectivity curves. This page describes which files xstack reads, how
a file's layout is described to the reader, what is skipped, and how the same import is done in
`xstack-fit`, in Python and on the command line.

## What a curve is

A {py:class}`~xstack.core.measurement.MeasuredCurve` is one of two kinds of scans:

- an **angle scan**: reflectivity against the grazing angle θ at a fixed photon energy, or
- an **energy scan**: reflectivity against the photon energy at a fixed grazing angle.

The scanned variable is stored as the grazing angle θ in radians or as the energy in eV, with the
reflectivity and, optionally, its standard uncertainty. The fixed energy or angle is part of the
curve, but it is usually not part of a data file, so you state it when you read the file. Reading
converts everything to these internal values: a file with 2θ in degrees, q in 1/Å or the energy
in keV gives the same kind of curve.

## Describing a file: `CsvLayout`

Delimited text files (`.csv`, `.tsv`, `.txt`, `.dat`) are read with
{py:func}`~xstack.core.io.data.read_curves`. A {py:class}`~xstack.core.io.delimited.CsvLayout`
describes the layout; without one, the reader assumes the grazing angle in degrees in the first
column and the reflectivity in the second column, and no uncertainties.

| Option | Meaning |
|---|---|
| `x_column`, `reflectivity_column`, `uncertainty_column` | Columns of the scanned variable, the reflectivity (or the counts, see `counts`) and its uncertainty (`None`: no uncertainties). Give a 0-based index or the name from the header row, which is compared ignoring case. |
| `quantity` | What the x column holds: `"angle"` (grazing angle θ), `"two_theta"` (scattering angle 2θ), `"q"` (momentum transfer q = 4π sin θ/λ) or `"energy"`. |
| `unit` | Unit of the x column. Angles: `"deg"` (default), `"mrad"`, `"rad"`. q: `"1/A"` (default), `"1/nm"`. Energy: `"eV"` (default), `"keV"`. |
| `reflectivity_scale` | Factor applied to the reflectivity and the uncertainty, `0.01` for values in percent. With `counts`, it is applied after the normalization. |
| `counts` | `True` if the reflectivity column holds detector counts; see [Detector counts](#detector-counts). |
| `full_scale` | With `counts`: the count that corresponds to a reflectivity of 1, for example the direct-beam count. `None` (default) normalizes to the largest count of the curve. |
| `energy` | Photon energy in eV of an angle scan. Required for `quantity="q"`. |
| `angle` | Grazing angle in rad of an energy scan. |
| `polarization` | `"unpolarized"` (default), `"s"` or `"p"`. |
| `delimiter` | The column separator. `None` detects a comma, semicolon or tab from the last data line and otherwise splits at whitespace. |
| `decimal` | `"."` (default) or `","`. |
| `comment` | Prefix of comment lines, `"#"` by default. |
| `name` | Name of the curve; by default the file name without suffix. |

The first row that is not a comment and not entirely numeric is the header. Everything else is
data. Files are read as UTF-8 (a byte-order mark is allowed), or as Latin-1 if that fails.

The energy of an angle scan and the angle of an energy scan must be known when the fit starts. If
you leave them out when reading, set them later with
{py:meth}`~xstack.core.measurement.MeasuredCurve.with_fixed`.

### Examples

The snippets create small files, so that the page runs on its own.

An angle scan with the scattering angle 2θ, a semicolon as separator and the reflectivity in
percent:

```python
from pathlib import Path

from xstack.core.io import CsvLayout, read_curves
from xstack.core.units import to_deg

Path("scan_percent.txt").write_text(
    "# two-theta (deg); reflectivity (%); error (%)\n"
    "two_theta;refl;err\n"
    "0.4;95.0;1.0\n"
    "0.6;60.0;0.8\n"
    "0.8;12.5;0.3\n"
)
layout = CsvLayout(
    x_column="two_theta",
    reflectivity_column="refl",
    uncertainty_column="err",
    quantity="two_theta",
    reflectivity_scale=0.01,
    energy=8048.0,
)
(curve,) = read_curves("scan_percent.txt", layout)
print(curve.name, [round(float(to_deg(t)), 3) for t in curve.x], curve.reflectivity)
print("uncertainties:", curve.uncertainty)
```

A file with a decimal comma needs `decimal=","` and a delimiter other than the comma; here the
columns are separated by tabs. Columns can be selected by index, and the reflectivity is
the third column here:

```python
Path("scan_comma.dat").write_text("0,25\t0,0010\t0,93\n0,50\t0,0020\t0,41\n0,75\t0,0030\t0,06\n")
layout = CsvLayout(
    x_column=0,
    reflectivity_column=2,
    quantity="angle",
    unit="deg",
    decimal=",",
    delimiter="\t",
    energy=8048.0,
)
(curve,) = read_curves("scan_comma.dat", layout)
print(curve.size, "points,", "uncertainties:", curve.uncertainty)
```

The momentum transfer q is converted to the grazing angle with the photon energy, which is why
`energy` is required. q values above 4π/λ (θ above 90°) raise an error:

```python
Path("scan_q.csv").write_text("q,R\n0.10,0.9\n0.20,0.3\n0.30,0.05\n")
layout = CsvLayout(quantity="q", unit="1/A", energy=8048.0, name="q scan")
(curve,) = read_curves("scan_q.csv", layout)
print(curve.name, [round(float(to_deg(t)), 3) for t in curve.x])
```

An energy scan at a fixed grazing angle:

```python
from xstack.core.units import deg

Path("energy_scan.csv").write_text("E_keV,R\n8.0,0.60\n8.5,0.45\n9.0,0.31\n")
layout = CsvLayout(quantity="energy", unit="keV", angle=deg(0.5))
(curve,) = read_curves("energy_scan.csv", layout)
print(curve.axis, curve.x, "at", round(float(to_deg(curve.angle)), 3), "deg")
```

### Detector counts

Raw data of a reflectometer often holds the detector counts N per angle rather than a
reflectivity. With `counts=True` the reflectivity column holds these counts, and the reader
converts them to the reflectivity

R = N / N_ref · `reflectivity_scale`

with the full-scale count N_ref. Give N_ref as `full_scale` if you know the count of the direct
beam (with the same counting time and attenuation). Otherwise the largest count of the curve is
used, which sets the reflectivity of the plateau of total reflection to `reflectivity_scale`
(1 by default). The instrument parameter *scale* of the dataset can take up the remaining
difference in the fit.

Without an uncertainty column, the uncertainty of N counts is √N, normalized in the same way.
Points with zero counts (or fewer, after a background subtraction) get the uncertainty of one
count, so that they stay in a *χ²* fit with a finite weight instead of being left out. An
uncertainty column with counts is normalized like the counts.

```python
Path("scan_counts.csv").write_text(
    "theta_deg,counts\n"
    "0.2,152000\n"
    "0.4,148000\n"
    "0.6,9100\n"
    "0.8,720\n"
)
layout = CsvLayout(counts=True, energy=8048.0)
(curve,) = read_curves("scan_counts.csv", layout)
print([round(r, 4) for r in curve.reflectivity])
print("uncertainties:", [f"{u:.1e}" for u in curve.uncertainty])

direct_beam = CsvLayout(counts=True, full_scale=1.6e5, energy=8048.0)
(curve,) = read_curves("scan_counts.csv", direct_beam)
print([round(r, 4) for r in curve.reflectivity])
```

### Rows that are skipped

- Rows with a non-finite value (`nan`, `inf`) in a used column are skipped. When an uncertainty
  column is used, its value must also be finite.
- Rows with a scanned variable at zero or below are skipped, because angle and energy must be
  positive.
- Comment lines (starting with the comment prefix) and empty lines are ignored.

A field that is not a number (for example an empty cell or a unit text) is an error that names the
line, rather than a skipped row, so that wrong column choices do not go unnoticed. The import
dialog of `xstack-fit` shows how many rows were read and how many were skipped.

Reflectivity values of zero or below are kept in the data. The fit leaves them out where its
weighting cannot use them; see [Fit ranges and weighting](fit_ranges_and_weighting.md#weighting).

## Uncertainties

The uncertainty column holds the standard uncertainty (one σ) of the reflectivity in the units
of the reflectivity column. With uncertainties you can use the statistical *χ²* weighting, the
fit reports an absolute *χ²_red*, and the uncertainties of the parameters are meaningful in an
absolute sense; see [Uncertainties](uncertainties.md). Without them only the logarithmic and
the linear weighting are available. For counting experiments the uncertainty of N counts is √N,
divided by the incident counts; with `counts=True` the reader computes it for you (see
[Detector counts](#detector-counts)). Take care with points of zero or very few counts: an uncertainty
estimated from the observed counts is too small there.

## Files written by xstack

The reflectivity export of `xstack-design` (and
{py:func}`~xstack.core.io.reflectivity.write_reflectivity_csv`) has a header row with the columns
`target`, `axis`, `reflections`, `energy_keV`, `angle_deg`, `reflectivity` and `reflectivity^N`.
Such a file is recognized by its header and read **without a layout**: every target becomes one
curve, named by the label of the target, with its energy or angle taken from the file. This is the
quickest way to use a simulated curve as test data. The export carries no uncertainties, so use
a logarithmic or linear weighting, or add uncertainties yourself.

## The import dialog of xstack-fit

*File > Add data…* (Ctrl+D) or the *Add data…* button of the *Datasets* dock opens the dialog
*Add measured data*. Choose the file with *Browse…*, by typing the path, or by dropping the file
on the dialog.

- **Format** shows what was detected: the delimiter (comma, semicolon, tab or whitespace), whether the
  file has a header row, and the number of columns.
- **Columns** *x*, *R* and *σ* choose the columns by name from the header row (or *column 1*,
  *column 2*, … without one). The dialog proposes the first column for x, the second for R and a
  later column whose name contains `err`, `sigma`, `σ`, `unc` or starts with `d` for σ; *(none)* means
  no uncertainties. The quantity and unit are guessed from the header of the first column, for
  example `2theta_deg` gives *scattering angle 2θ* in *deg*.
- **x is** chooses the quantity (*grazing angle θ*, *scattering angle 2θ*, *momentum transfer q*,
  *energy*) and its unit.
- **Photon energy** (for angle scans) or **Grazing angle** (for energy scans). The dialog searches the
  comment lines of the file for a photon energy in keV or eV after the word “energy”, or a
  wavelength in Å or nm after “wavelength”, “lambda” or “λ”, converts it, and notes it in the
  format line (“energy 8.048 keV from the file header”). Otherwise the energy has the preset
  *Cu Kα* (8.048 keV); the presets *Mo Kα* (17.479 keV) and *Ag Kα* (22.163 keV) and *custom*
  values are available. Check the value: it determines the optical constants of the materials.
- **Polarization** and **reflectivity in %**, as for the layout options.
- **Dataset name**: an identifier (letters, digits, underscore) that is not used yet; it is the
  prefix of the instrument parameters. The dialog derives it from the file name.
- The **preview** shows the first rows of the file next to the converted values (θ in degrees or the
  energy in keV, R, σ), and below it the number of rows read and skipped. The same reader is
  used as in Python, so errors are shown in the dialog and *Add dataset* stays disabled until
  the layout is valid.

The dialog scans the file for delimiters, header and `#` comment lines on its own, and it offers
no entries for the delimiter, the decimal separator or another comment prefix. For files with a
decimal comma or other comment characters, read them with `CsvLayout` and save a project (see
below), or convert the file.

For an xstack export the dialog shows *xstack reflectivity export*, disables the layout entries
and adds one dataset per target.

A project created in Python (see [Fit a measurement with Python](../quickstart_fit_library.md))
is saved with `FitProject.save("name.xfit")` and opens in `xstack-fit` with all data and settings.

## The command line: `xstack fit new`

`xstack fit new` creates a project from a model file and data files:

```{code-block} bash
xstack fit new OUTPUT.xfit --model design.xstack --data scan.csv \
    --energy 8.048 --quantity two_theta --columns 2theta_deg,R,dR \
    --range 0.35:3.0 --weighting chi2 --name "W/Si sample"
```

`--model FILE` is required: the file with the start structure, for example an `.xstack` design
project, with `--candidate NAME` choosing one of its designs (default: the first). `--data FILE` can be
repeated; every curve of every file becomes a dataset named after the file (made an identifier).

The layout is set by the options `--quantity` (`angle`, `two_theta`, `q` or `energy`), `--unit`,
`--columns X,R[,SIGMA]` (0-based indices or header names), `--percent` or `--scale FACTOR`, and
for detector counts `--counts` with the optional `--full-scale COUNTS` (default: the largest
count; σ defaults to √N, at least one count). If none of these is
given, the file layout is detected: an xstack export is recognized, any other file is read with
the default layout (grazing angle in degrees in the first column, reflectivity in the second, no
uncertainties). Give `--columns` to use uncertainties. The photon energy of angle scans is
`--energy` in keV and the grazing angle of energy scans is `--angle` in degrees. The options
cannot set a delimiter, decimal separator or comment prefix.

`--range LO:HI` (repeatable) sets fit ranges for all datasets in degrees of the grazing angle
for angle scans, and in keV for energy scans, whatever the quantity in the file. `--weighting`
is `log` (default), `linear` or `chi2`.

All instrument parameters of the new datasets start fixed at their defaults (scale 1, no
background, no offset, no resolution, no footprint). Open the project in `xstack-fit` to vary
them, or edit it in Python. `xstack fit info`, `xstack fit run` and `xstack fit export` inspect,
fit and export a project.
