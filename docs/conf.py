"""Sphinx configuration of the xstack documentation."""

import importlib
import pkgutil
from pathlib import Path

from xstack import __version__

project = "xstack"
author = "xstack contributors"
release = __version__
version = __version__

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
]

source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
root_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_static_path = ["_static"]

myst_enable_extensions = ["colon_fence", "deflist", "dollarmath", "amsmath", "attrs_inline"]
myst_heading_anchors = 3

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


_PACKAGES = ("xstack.core", "xstack.design", "xstack.fit", "xstack.gui")
_API = Path(__file__).parent / "api" / "generated"


def _write_api_pages(app):  # type: ignore[no-untyped-def]
    """Write one page per module of the API reference, with a toctree of the submodules.

    Modules are found with pkgutil and documented with ``automodule`` by their full name, so a
    package attribute with the name of a submodule (e.g. a registry) cannot hide the module.
    ``__main__`` modules (command-line entry points) are left out.
    """
    _API.mkdir(parents=True, exist_ok=True)
    for page in _API.glob("*.rst"):
        page.unlink()

    def write(name: str) -> None:
        module = importlib.import_module(name)
        children = []
        if hasattr(module, "__path__"):
            children = [
                f"{name}.{info.name}"
                for info in pkgutil.iter_modules(module.__path__)
                if info.name != "__main__"
            ]
        lines = [name, "=" * len(name), "", f".. automodule:: {name}"]
        if children:
            lines += ["   :no-members:", "", ".. rubric:: Modules", ""]
            lines += [".. toctree::", "   :maxdepth: 1", ""]
            lines += [f"   {child}" for child in children]
        (_API / f"{name}.rst").write_text("\n".join(lines) + "\n")
        for child in children:
            write(child)

    for package in _PACKAGES:
        write(package)


def _skip_main_modules(app, what, name, obj, skip, options):  # type: ignore[no-untyped-def]
    """Leave ``__main__`` modules (command-line entry points) out of the API reference."""
    return True if name.endswith("__main__") else skip


def setup(app):  # type: ignore[no-untyped-def]
    """Connect the event handlers of this configuration."""
    app.connect("builder-inited", _write_api_pages)
    app.connect("autodoc-skip-member", _skip_main_modules)
