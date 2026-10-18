"""Command-line entry point: ``python -m docs.guide.screenshots [NAME ...]``."""

import sys

from docs.guide.screenshots import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
