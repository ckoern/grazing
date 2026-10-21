"""Convert FFAST scattering-factor tables into the files of this directory.

Run with the directory of the tables, one file ``<Element>.sf`` per element with the columns
photon energy (keV), f1 and f2 (comment lines start with ``#``)::

    python src/xstack/core/materials/data/chantler/convert.py path/to/sf

For every element from H to U, it writes ``<element>.dat.gz`` with the columns photon energy (eV),
f1 and f2, keeping all tabulated energies. The gzip files carry no time stamp, so the conversion is
reproducible.
"""

import gzip
import sys
from itertools import pairwise
from pathlib import Path

HERE = Path(__file__).parent
SYMBOLS = [
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
]


def table(path: Path) -> list[tuple[float, float, float]]:
    """Return the rows (energy in eV, f1, f2) of one table file.

    Args:
        path: Table with the columns energy (keV), f1 and f2.

    Returns:
        The rows, ordered by energy.

    Raises:
        ValueError: If a row is malformed or the energies are not increasing.
    """
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        e_kev, f1, f2 = (float(v) for v in line.split())
        rows.append((e_kev * 1e3, f1, f2))
    if any(b[0] <= a[0] for a, b in pairwise(rows)):
        raise ValueError(f"{path}: energies are not increasing")
    return rows


def main(argv: list[str]) -> int:
    """Write one file per element.

    Args:
        argv: Command-line arguments: the directory of the table files.

    Returns:
        The exit status.
    """
    source = Path(argv[0])
    for symbol in SYMBOLS:
        rows = table(source / f"{symbol}.sf")
        text = "# energy_eV f1 f2\n" + "".join(f"{e:.10g} {a:.10g} {b:.10g}\n" for e, a, b in rows)
        with (
            open(HERE / f"{symbol.lower()}.dat.gz", "wb") as raw,
            gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as file,
        ):
            file.write(text.encode())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
