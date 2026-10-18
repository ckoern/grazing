"""Command-line interface.

Usage::

    xstack design run PROJECT.xstack [--output OUT.xstack] [--seed N]
    xstack design info PROJECT.xstack
    xstack fit new OUTPUT.xfit --model MODEL_FILE --data DATA_FILE [...]
    xstack fit run PROJECT.xfit [--output OUT.xfit] [--method METHOD] [--max-evaluations N]
    xstack fit info PROJECT.xfit
    xstack fit export PROJECT.xfit OUTPUT.csv

``python -m xstack`` is equivalent to the ``xstack`` command.
"""

import argparse
import sys
from collections.abc import Sequence

from xstack.design import cli as design_cli
from xstack.fit import cli as fit_cli

__all__ = ["main"]


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface.

    Args:
        argv: Command-line arguments without the program name. Defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(prog="xstack", description="X-ray multilayer mirrors")
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
