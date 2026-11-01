# User guide

xstack has two tools that share one library:

- **Design** (xstack-design): find the layer structure of a mirror that reflects best for given
  energies and angles, within manufacturing limits, and check how robust it is.
- **Fit** (xstack-fit): determine the actual structure of a coated sample from its measured
  reflectivity, with uncertainties.

Each tool can be used through its application or from Python. Start with the quick start that
matches what you want to do; the topics go deeper into individual subjects.

| | Application | Python library |
|---|---|---|
| **Design** a mirror | [Design a mirror in xstack-design](quickstart_design_gui.md) | [Design a mirror with Python](quickstart_design_library.md) |
| **Fit** a measurement | [Fit a measurement in xstack-fit](quickstart_fit_gui.md) | [Fit a measurement with Python](quickstart_fit_library.md) |

The physics behind both tools is summarized in the
[physics background](../physics_background.md).

```{toctree}
:maxdepth: 1
:caption: Getting started

installation
quickstart_design_gui
quickstart_design_library
quickstart_fit_gui
quickstart_fit_library
topics/quick_calculations
```

```{toctree}
:maxdepth: 1
:caption: Design topics

topics/design_multiple_targets
topics/design_depth_graded
topics/combined_coatings
topics/design_tolerances
```

```{toctree}
:maxdepth: 1
:caption: Fit topics

topics/measured_data
topics/fit_ranges_and_weighting
topics/fit_simultaneous
topics/uncertainties
```

```{toctree}
:maxdepth: 1
:caption: Advanced

topics/extending
```
