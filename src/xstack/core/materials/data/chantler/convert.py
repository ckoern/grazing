"""Convert the NIST FFAST tables into the files of this directory.

Run with the FFAST database of the FFAST web pages and the X-ray properties database of NIST
(DOI 10.18434/mds2-2819), which provides the correction terms of the FFAST tables::

    python src/xstack/core/materials/data/chantler/convert.py FFAST_data.sqlite atomic_database.db

``FFAST_data.sqlite`` has one table per element (named by Z) with the photon energy (keV), f1 and
f2. The table ``FFAST_EXTRA`` of ``atomic_database.db`` has one row per element, in the order of
Z, with the relativistic correction estimates ``rc1`` (H82) and ``rc2`` (3/5 CL) and the nuclear
Thomson correction ``nt``. f1 and f2 are taken from ``FFAST_data.sqlite`` only.

For every element from H to U, it writes ``<element>.dat.gz`` with the columns photon energy (eV),
f1 + rc2 + nt and f2, keeping all tabulated energies. The gzip files carry no time stamp, so the
conversion is reproducible.
"""

import gzip
import sqlite3
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


def corrections(path: Path) -> list[float]:
    """Return the correction f_rel (3/5 CL) + f_NT of every element.

    Args:
        path: The X-ray properties database (``atomic_database.db``).

    Returns:
        The corrections in electrons per atom, for Z = 1 to 92.

    Raises:
        ValueError: If the table does not have one row per element.
    """
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as database:
        rows = database.execute("select rc2, nt from FFAST_EXTRA order by rowid").fetchall()
    if len(rows) != len(SYMBOLS):
        raise ValueError(f"{path}: FFAST_EXTRA has {len(rows)} rows, expected {len(SYMBOLS)}")
    return [rc2 + nt for rc2, nt in rows]


def table(database: sqlite3.Connection, z: int) -> list[tuple[float, float, float]]:
    """Return the rows (energy in eV, f1, f2) of one element of the FFAST database.

    Args:
        database: Connection to ``FFAST_data.sqlite``.
        z: Atomic number.

    Returns:
        The rows, ordered by energy.

    Raises:
        ValueError: If the energies are not increasing.
    """
    rows = [
        (e_kev * 1e3, f1, f2)
        for e_kev, f1, f2 in database.execute(f'select e, f1, f2 from "{z}" order by e')
    ]
    if any(b[0] <= a[0] for a, b in pairwise(rows)):
        raise ValueError(f"Z = {z}: energies are not increasing")
    return rows


def main(argv: list[str]) -> int:
    """Write one file per element.

    Args:
        argv: Command-line arguments: the paths of ``FFAST_data.sqlite`` and
            ``atomic_database.db``.

    Returns:
        The exit status.
    """
    correction = corrections(Path(argv[1]))
    with sqlite3.connect(f"file:{argv[0]}?mode=ro", uri=True) as database:
        for z, symbol in enumerate(SYMBOLS, start=1):
            c = correction[z - 1]
            rows = table(database, z)
            text = "# energy_eV f1 f2\n" + "".join(
                f"{e:.10g} {f1 + c:.10g} {f2:.10g}\n" for e, f1, f2 in rows
            )
            with (
                open(HERE / f"{symbol.lower()}.dat.gz", "wb") as raw,
                gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as file,
            ):
                file.write(text.encode())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
