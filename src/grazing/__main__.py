"""Command-line interface.

Usage::

    grazing design run PROJECT.grazing [--output OUT.grazing] [--seed N]
    grazing design info PROJECT.grazing
    grazing fit new OUTPUT.gfit --model MODEL_FILE --data DATA_FILE [...]
    grazing fit run PROJECT.gfit [--output OUT.gfit] [--method METHOD] [--max-evaluations N]
    grazing fit info PROJECT.gfit
    grazing fit export PROJECT.gfit OUTPUT.csv

``python -m grazing`` is equivalent to the ``grazing`` command.
"""

import argparse
import sys
from collections.abc import Sequence

from grazing.design import cli as design_cli
from grazing.fit import cli as fit_cli

__all__ = ["main"]


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface.

    Args:
        argv: Command-line arguments without the program name. Defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(prog="grazing", description="X-ray multilayer mirrors")
    sub = parser.add_subparsers(dest="tool", required=True)
    design_cli.add_commands(sub.add_parser("design", help="design multilayer mirrors"))
    fit_cli.add_commands(sub.add_parser("fit", help="fit measured reflectivity"))
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (KeyError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
