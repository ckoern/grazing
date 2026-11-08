"""Application entry point of the fit GUI."""

import sys
from collections.abc import Sequence

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from grazing import __version__
from grazing.gui.common.garbage import GarbageCollector
from grazing.gui.common.style import Theme, apply_style
from grazing.gui.fit.main_window import MainWindow

__all__ = ["main"]


def main(argv: Sequence[str] | None = None) -> int:
    """Start the grazing-fit GUI.

    Args:
        argv: Command-line arguments without the program name; an optional first argument is
            a fit project (.gfit) to open. Defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    app = QApplication.instance() or QApplication([sys.argv[0], *args])
    app.setApplicationName("grazing-fit")
    app.setApplicationVersion(__version__)
    app.setOrganizationName("grazing")
    saved = QSettings("grazing", "grazing-fit").value("theme", "system")
    theme: Theme = saved if saved in ("light", "dark") else "system"
    apply_style(app, theme)
    collector = GarbageCollector(parent=app)
    window = MainWindow()
    if args:
        window.open_project(args[0])
    window.show()
    status = int(app.exec())
    collector.stop()
    return status


if __name__ == "__main__":
    sys.exit(main())
