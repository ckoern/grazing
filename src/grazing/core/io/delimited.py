r"""Measured curves from delimited text files (CSV, TSV, whitespace-separated columns).

The layout of a file is described by a `CsvLayout`: which columns hold the scanned variable,
the reflectivity and its uncertainty, the quantity and unit of the scanned variable, and the
fixed energy or angle. Files written by `grazing.core.io.write_reflectivity_csv` are recognized
by their header and read without a layout, one curve per target.

Examples:
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     path = Path(tmp) / "scan.csv"
    ...     _ = path.write_text("# 2theta/deg, R\n0.2, 0.9\n0.4, 0.5\n0.6, 0.1\n")
    ...     layout = CsvLayout(quantity="two_theta", energy=8048.0)
    ...     (curve,) = CsvParser().read(path, layout)
    >>> curve.name, curve.size, round(curve.x[0], 6)
    ('scan', 3, 0.001745)
"""

import csv
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from grazing.core.io.data import data_parsers
from grazing.core.io.reflectivity import REFLECTIVITY_COLUMNS
from grazing.core.measurement import MeasuredCurve
from grazing.core.physics.fresnel import Polarization
from grazing.core.units import angstrom, deg, kev, mrad, q_to_angle

__all__ = ["UNITS", "CsvLayout", "CsvParser", "Quantity"]

type Quantity = Literal["angle", "two_theta", "q", "energy"]
"""Quantity in the x column: grazing angle θ, scattering angle 2θ, momentum transfer
q = 4π sin θ / λ, or photon energy."""

type _Conversion = Callable[[NDArray[np.float64]], NDArray[np.float64]]


def _identity(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return x


def _per_angstrom(q: NDArray[np.float64]) -> NDArray[np.float64]:
    return q / angstrom(1.0)


_ANGLE_UNITS: dict[str, _Conversion] = {"deg": deg, "mrad": mrad, "rad": _identity}

UNITS: dict[Quantity, dict[str, _Conversion]] = {
    "angle": _ANGLE_UNITS,
    "two_theta": _ANGLE_UNITS,
    "q": {"1/A": _per_angstrom, "1/nm": _identity},
    "energy": {"eV": _identity, "keV": kev},
}
"""Supported units per quantity, the first being the default, with their conversion to the
internal unit (rad, 1/nm or eV)."""

_DELIMITERS = (",", ";", "\t")


@dataclass(frozen=True)
class CsvLayout:
    """Layout of a delimited text file with one measured curve.

    Columns are given by 0-based index or by name from the header row. A header row is a first
    non-comment row that is not entirely numeric.

    With `counts`, the reflectivity column holds detector counts N. They are converted to the
    reflectivity N / N_ref · `reflectivity_scale`, with the full-scale count N_ref given by
    `full_scale` or, if that is ``None``, the largest count of the curve. Without an uncertainty
    column the uncertainty of the counts is √N, and 1 for counts of zero or below.

    Attributes:
        x_column: Column of the scanned variable.
        reflectivity_column: Column of the reflectivity, or of the counts with `counts`.
        uncertainty_column: Column of the uncertainty of the values in `reflectivity_column`,
            or ``None``.
        quantity: Quantity in `x_column`. ``"q"`` requires `energy`.
        unit: Unit of `x_column`, one of `UNITS` for `quantity`, or ``None`` for the default
            (deg, 1/A or eV).
        reflectivity_scale: Factor applied to reflectivity and uncertainty, e.g. 0.01 for
            values in percent. With `counts`, it is applied after the normalization.
        energy: Photon energy in eV of an angle scan, or ``None`` to set it later.
        angle: Grazing angle in rad of an energy scan, or ``None`` to set it later.
        polarization: ``"s"``, ``"p"`` or ``"unpolarized"``.
        delimiter: Column delimiter, or ``None`` to detect comma, semicolon, tab or whitespace.
        decimal: Decimal separator, ``"."`` or ``","``.
        comment: Prefix of comment lines.
        name: Name of the curve, or empty to use the file name without suffix.
        counts: Whether `reflectivity_column` holds detector counts instead of reflectivity.
        full_scale: Count that corresponds to a reflectivity of 1 (e.g. the direct-beam count),
            or ``None`` to normalize to the largest count. Only allowed with `counts`.

    Raises:
        ValueError: If the unit is unknown for the quantity, the scale or full-scale count is
            not positive, a full-scale count is given without `counts`, or the decimal
            separator equals the delimiter.
    """

    x_column: int | str = 0
    reflectivity_column: int | str = 1
    uncertainty_column: int | str | None = None
    quantity: Quantity = "angle"
    unit: str | None = None
    reflectivity_scale: float = 1.0
    energy: float | None = None
    angle: float | None = None
    polarization: Polarization | Literal["unpolarized"] = "unpolarized"
    delimiter: str | None = None
    decimal: Literal[".", ","] = "."
    comment: str = "#"
    name: str = ""
    counts: bool = False
    full_scale: float | None = None

    def __post_init__(self) -> None:
        if self.quantity not in UNITS:
            raise ValueError(f"unknown quantity {self.quantity!r}")
        if self.unit is not None and self.unit not in UNITS[self.quantity]:
            known = ", ".join(UNITS[self.quantity])
            raise ValueError(f"unknown unit {self.unit!r} for {self.quantity} (known: {known})")
        if not self.reflectivity_scale > 0.0:
            raise ValueError("reflectivity scale must be positive")
        if self.full_scale is not None:
            if not self.counts:
                raise ValueError("a full-scale count requires counts in the data column")
            if not self.full_scale > 0.0:
                raise ValueError("full-scale count must be positive")
        if self.decimal not in (".", ","):
            raise ValueError("decimal separator must be '.' or ','")
        if self.decimal == self.delimiter:
            raise ValueError("decimal separator and delimiter must differ")

    @property
    def axis(self) -> Literal["angle", "energy"]:
        """The scanned variable of the curve."""
        return "energy" if self.quantity == "energy" else "angle"

    def to_internal(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Convert values of the x column to the scanned variable.

        Args:
            x: Values in `unit`.

        Returns:
            Grazing angles in rad or energies in eV; NaN where q exceeds 4π / λ.

        Raises:
            ValueError: If the quantity is ``"q"`` and `energy` is not set.
        """
        units = UNITS[self.quantity]
        value = units[self.unit or next(iter(units))](x)
        if self.quantity == "two_theta":
            return value / 2.0
        if self.quantity == "q":
            if self.energy is None:
                raise ValueError("reading q values requires the photon energy")
            return q_to_angle(value, self.energy)
        return value


@dataclass(frozen=True)
class _Table:
    header: tuple[str, ...] | None
    rows: tuple[tuple[int, tuple[str, ...]], ...]

    def column(self, column: int | str) -> int:
        if isinstance(column, int):
            return column
        if self.header is None:
            raise ValueError(f"column {column!r} given by name, but the file has no header")
        names = [h.casefold() for h in self.header]
        if column.casefold() not in names:
            raise ValueError(f"no column {column!r} (columns: {', '.join(self.header)})")
        return names.index(column.casefold())

    def values(self, column: int | str, decimal: str) -> NDArray[np.float64]:
        index = self.column(column)
        values = []
        for line, fields in self.rows:
            if index >= len(fields):
                raise ValueError(f"line {line}: no column {index}")
            text = fields[index].strip()
            try:
                values.append(float(text.replace(",", ".") if decimal == "," else text))
            except ValueError:
                raise ValueError(f"line {line}: {text!r} is not a number") from None
        return np.array(values, dtype=np.float64)

    def strings(self, column: str) -> list[str]:
        index = self.column(column)
        return [fields[index].strip() for _, fields in self.rows]


def _is_number(text: str, decimal: str) -> bool:
    try:
        float(text.strip().replace(",", ".") if decimal == "," else text)
    except ValueError:
        return False
    return True


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


def _table(text: str, delimiter: str | None, decimal: str, comment: str) -> _Table:
    lines = [
        (number, line)
        for number, line in enumerate(text.splitlines(), start=1)
        if line.strip() and not (comment and line.lstrip().startswith(comment))
    ]
    if not lines:
        raise ValueError("the file contains no data")
    if delimiter is None:
        first = lines[-1][1]
        delimiter = next((d for d in _DELIMITERS if d in first and d != decimal), "")
    if delimiter:
        rows = [
            (number, tuple(next(csv.reader([line], delimiter=delimiter)))) for number, line in lines
        ]
    else:
        rows = [(number, tuple(line.split())) for number, line in lines]
    header = None
    if not all(_is_number(field, decimal) for field in rows[0][1]):
        header = tuple(field.strip() for field in rows[0][1])
        rows = rows[1:]
    if not rows:
        raise ValueError("the file contains no data rows")
    return _Table(header, tuple(rows))


def _curve(
    layout: CsvLayout,
    x: NDArray[np.float64],
    reflectivity: NDArray[np.float64],
    uncertainty: NDArray[np.float64] | None,
    name: str,
    source: str,
) -> MeasuredCurve:
    keep = np.isfinite(x) & np.isfinite(reflectivity) & (x > 0.0)
    if uncertainty is not None:
        keep &= np.isfinite(uncertainty)
    if not np.any(keep):
        raise ValueError("no row with finite values at positive angle or energy")
    x = layout.to_internal(x[keep])
    if np.any(np.isnan(x)):
        raise ValueError("q values exceed 4π/λ at the given photon energy")
    values = reflectivity[keep]
    sigma = None if uncertainty is None else uncertainty[keep]
    scale = layout.reflectivity_scale
    if layout.counts:
        if sigma is None:
            sigma = np.sqrt(np.maximum(values, 1.0))
        full_scale = layout.full_scale if layout.full_scale is not None else float(values.max())
        if not full_scale > 0.0:
            raise ValueError("the counts contain no positive value to normalize to")
        scale /= full_scale
    return MeasuredCurve(
        layout.axis,
        x,
        values * scale,
        None if sigma is None else sigma * scale,
        energy=layout.energy,
        angle=layout.angle,
        polarization=layout.polarization,
        name=name,
        source=source,
    )


class CsvParser:
    """Parser of delimited text files.

    Rows with non-finite values and rows at zero or negative angle or energy are skipped.
    """

    description = "delimited text (CSV, TSV, columns)"
    suffixes: Sequence[str] = (".csv", ".tsv", ".txt", ".dat")
    options_type = CsvLayout

    def read(self, path: Path, options: object = None) -> tuple[MeasuredCurve, ...]:
        """Read the curves of a file.

        Args:
            path: The file.
            options: A `CsvLayout`, or ``None`` to read files written by
                `grazing.core.io.write_reflectivity_csv` and otherwise use the default layout
                (grazing angle in deg in the first column, reflectivity in the second).

        Returns:
            One curve per target for grazing exports, otherwise one curve.

        Raises:
            ValueError: If the file does not match the layout or `options` is not a
                `CsvLayout`.
            OSError: If the file cannot be read.
        """
        if options is not None and not isinstance(options, CsvLayout):
            raise ValueError(f"expected CsvLayout options, got {type(options).__name__}")
        layout = options or CsvLayout()
        table = _table(_read_text(path), layout.delimiter, layout.decimal, layout.comment)
        if options is None and table.header == REFLECTIVITY_COLUMNS:
            return _export_curves(table, path)
        uncertainty = None
        if layout.uncertainty_column is not None:
            uncertainty = table.values(layout.uncertainty_column, layout.decimal)
        curve = _curve(
            layout,
            table.values(layout.x_column, layout.decimal),
            table.values(layout.reflectivity_column, layout.decimal),
            uncertainty,
            layout.name or path.stem,
            path.name,
        )
        return (curve,)


def _export_curves(table: _Table, path: Path) -> tuple[MeasuredCurve, ...]:
    labels, axes = table.strings("target"), table.strings("axis")
    energy = table.values("energy_keV", ".")
    angle = table.values("angle_deg", ".")
    reflectivity = table.values("reflectivity", ".")
    curves = []
    for label in dict.fromkeys(labels):
        rows = np.array([lbl == label for lbl in labels])
        axis = axes[int(np.argmax(rows))]
        if axis == "angle":
            layout = CsvLayout(quantity="angle", unit="deg", energy=kev(float(energy[rows][0])))
            x = angle[rows]
        elif axis == "energy":
            layout = CsvLayout(quantity="energy", unit="keV", angle=deg(float(angle[rows][0])))
            x = energy[rows]
        else:
            raise ValueError(f"unknown axis {axis!r} of target {label!r}")
        curves.append(_curve(layout, x, reflectivity[rows], None, label, path.name))
    return tuple(curves)


data_parsers.register("csv", CsvParser())
