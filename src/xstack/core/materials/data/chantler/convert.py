"""Convert the Chantler (NIST FFAST) tables of xraydb into the files of this directory.

Run with the path of the ``xraydb.sqlite`` file distributed with xraydb::

    python src/xstack/core/materials/data/chantler/convert.py path/to/xraydb.sqlite

For every element from H to U, it writes ``<element>.dat.gz`` with the columns photon energy (eV),
f1 and f2, keeping the tabulated energies from 10 eV to 433 keV plus one point beyond each end.
xraydb stores the anomalous part f1 − Z; the files contain f1. The gzip files carry no time stamp,
so the conversion is reproducible.
"""

import gzip
import json
import sqlite3
import sys
from pathlib import Path

HERE = Path(__file__).parent
E_MIN = 10.0
E_MAX = 433000.0
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


def table(database: sqlite3.Connection, symbol: str) -> list[tuple[float, float, float]]:
    """Return the rows (energy in eV, f1, f2) of one element within the kept range.

    Args:
        database: Connection to ``xraydb.sqlite``.
        symbol: Element symbol.

    Returns:
        The rows, ordered by energy.
    """
    z = SYMBOLS.index(symbol) + 1
    energy, f1, f2 = (
        json.loads(column)
        for column in database.execute(
            "select energy, f1, f2 from Chantler where element = ?", (symbol,)
        ).fetchone()
    )
    first = max(i for i, e in enumerate(energy) if e <= E_MIN)
    last = min(i for i, e in enumerate(energy) if e >= E_MAX)
    return [(energy[i], z + f1[i], f2[i]) for i in range(first, last + 1)]


def main(argv: list[str]) -> int:
    """Write one file per element.

    Args:
        argv: Command-line arguments: the path of ``xraydb.sqlite``.

    Returns:
        The exit status.
    """
    database = sqlite3.connect(argv[0])
    for symbol in SYMBOLS:
        rows = table(database, symbol)
        text = "# energy_eV f1 f2\n" + "".join(f"{e:.10g} {a:.10g} {b:.10g}\n" for e, a, b in rows)
        with (
            open(HERE / f"{symbol.lower()}.dat.gz", "wb") as raw,
            gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as file,
        ):
            file.write(text.encode())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
