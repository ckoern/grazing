"""Sphinx configuration of the xstack documentation."""

from xstack import __version__

project = "xstack"
author = "xstack contributors"
release = __version__
version = __version__

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
]

source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
root_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
templates_path = ["_templates"]
html_static_path = ["_static"]

myst_enable_extensions = ["colon_fence", "deflist", "dollarmath", "amsmath", "attrs_inline"]
myst_heading_anchors = 3

autosummary_generate = True
autosummary_imported_members = False
autodoc_default_options = {"members": True, "show-inheritance": True}
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_class_signature = "separated"
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_ivar = True
napoleon_attr_annotations = True
napoleon_custom_sections = [("Signals", "params_style")]
suppress_warnings = ["ref.python"]

intersphinx_timeout = 5
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

html_theme = "furo"
html_title = f"xstack {__version__}"
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True


def _skip_main_modules(app, what, name, obj, skip, options):  # type: ignore[no-untyped-def]
    """Leave ``__main__`` modules (command-line entry points) out of the API reference."""
    return True if name.endswith("__main__") else skip


def setup(app):  # type: ignore[no-untyped-def]
    """Connect the event handlers of this configuration."""
    app.connect("autodoc-skip-member", _skip_main_modules)
