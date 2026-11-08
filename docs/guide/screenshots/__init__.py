"""Screenshots of the user guide, rendered from the real applications.

Every module in this package defines functions decorated with `screenshot`; each writes one image
into ``docs/guide/images``. Regenerate all images (or the named ones) with::

    QT_QPA_PLATFORM=offscreen python -m docs.guide.screenshots [NAME ...]

from the repository root.
"""

import importlib
import pkgutil
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QWidget

if TYPE_CHECKING:
    from grazing.gui.design.main_window import MainWindow as DesignWindow
    from grazing.gui.fit.main_window import MainWindow as FitWindow

IMAGES = Path(__file__).parents[1] / "images"
"""Directory of the guide's images."""

SIZE = (1500, 920)
"""Window size of main-window screenshots in pixels."""

_REGISTRY: dict[str, Callable[[], None]] = {}


def screenshot(function: Callable[[], None]) -> Callable[[], None]:
    """Register a function that writes one screenshot, named after the function."""
    _REGISTRY[function.__name__] = function
    return function


def app() -> QApplication:
    """Return the application, created with the light theme on first use."""
    from grazing.gui.common.garbage import GarbageCollector
    from grazing.gui.common.style import apply_style

    instance = QApplication.instance()
    if not isinstance(instance, QApplication):
        instance = QApplication([sys.argv[0]])
        apply_style(instance, "light")
        GarbageCollector(parent=instance)
    return instance


def settings() -> QSettings:
    """Return throw-away settings, so screenshots don't depend on the user's settings."""
    directory = Path(tempfile.mkdtemp(prefix="grazing-guide-"))
    return QSettings(str(directory / "settings.ini"), QSettings.Format.IniFormat)


def pump(
    seconds: float = 0.5, until: Callable[[], bool] | None = None, timeout: float = 120.0
) -> None:
    """Process events for a while, or until a condition holds (at most `timeout` s)."""
    application = app()
    end = time.monotonic() + (timeout if until is not None else seconds)
    while time.monotonic() < end:
        application.processEvents()
        if until is not None and until():
            break
        time.sleep(0.02)
    application.processEvents()


def design_window() -> "DesignWindow":
    """Return a new, shown design main window with throw-away settings and no close prompt."""
    from grazing.gui.design.main_window import MainWindow as DesignWindow

    app()
    window = DesignWindow(settings=settings(), confirm_close=False)
    window.resize(*SIZE)
    window.show()
    pump(1.0)
    return window


def fit_window() -> "FitWindow":
    """Return a new, shown fit main window with throw-away settings and no close prompt."""
    from grazing.gui.fit.main_window import MainWindow as FitWindow

    app()
    window = FitWindow(settings=settings(), confirm_close=False)
    window.resize(*SIZE)
    window.show()
    pump(1.0)
    return window


def save(widget: QWidget, name: str) -> Path:
    """Render a widget into ``images/<name>.png``.

    Args:
        widget: The widget, shown and laid out.
        name: File name without suffix.

    Returns:
        The written file.
    """
    pump(0.2)
    IMAGES.mkdir(parents=True, exist_ok=True)
    path = IMAGES / f"{name}.png"
    widget.grab().save(str(path))
    return path


def main(names: list[str]) -> int:
    """Write the screenshots of all registered functions, or of the named ones."""
    app()
    for module in pkgutil.iter_modules(__path__):
        importlib.import_module(f"{__name__}.{module.name}")
    unknown = set(names) - set(_REGISTRY)
    if unknown:
        print(f"unknown screenshots: {sorted(unknown)}", file=sys.stderr)
        return 1
    for name, function in _REGISTRY.items():
        if not names or name in names:
            print(f"{name} …", flush=True)
            function()
    return 0
