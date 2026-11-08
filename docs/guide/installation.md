# Installation

grazing needs Python 3.12 or newer. The library depends on numpy, scipy, lmfit and xraylib; the
applications additionally on PySide6, pyqtgraph and matplotlib.

Install from a checkout of the repository, preferably in its own environment, for example with
conda:

```{code-block} bash
conda create -n grazing python=3.12
conda activate grazing
pip install -e ".[gui]"
```

The extras are:

`gui`
: the applications `grazing-design` and `grazing-fit`.

`dev`
: tests, linters and type checking (pytest, ruff, mypy).

`docs`
: this documentation (Sphinx, MyST, Furo).

Check the installation:

```{code-block} bash
grazing --help
grazing-design
grazing-fit
```

`grazing-design [PROJECT.grazing]` and `grazing-fit [PROJECT.gfit]` open the applications, optionally
with a project; `python -m grazing.gui.design` and `python -m grazing.gui.fit` are equivalent. The
command-line tools are `grazing design …` and `grazing fit …`.

## Building this documentation

```{code-block} bash
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
```

The screenshots in this guide are rendered from the applications with
`QT_QPA_PLATFORM=offscreen python -m docs.guide.screenshots`, and the code snippets are run by
the test suite (`pytest tests/test_guide_snippets.py`).
