"""Application entry point of the graphical user interface."""

import sys
from collections.abc import Sequence

from PySide6.QtWidgets import QApplication

from xstack import __version__
from xstack.gui.main_window import MainWindow
from xstack.gui.style import apply_style

__all__ = ["main"]


def main(argv: Sequence[str] | None = None) -> int:
    """Start the xstack GUI.

    Args:
        argv: Command-line arguments without the program name; an optional first argument is
            a project file to open. Defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    app = QApplication.instance() or QApplication([sys.argv[0], *args])
    app.setApplicationName("xstack")
    app.setApplicationVersion(__version__)
    app.setOrganizationName("xstack")
    apply_style(app)
    window = MainWindow()
    if args:
        window.open_project(args[0])
    window.show()
    return int(app.exec())


if __name__ == "__main__":
    sys.exit(main())
