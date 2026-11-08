"""Tests for `xstack.core.io.delimited`."""

import math
from pathlib import Path

import numpy as np
import pytest

from xstack.core import materials
from xstack.core.io import files
from xstack.core.io.data import read_curves
from xstack.core.io.delimited import UNITS, CsvLayout, CsvParser
from xstack.core.io.reflectivity import reflectivity_table, write_reflectivity_csv
from xstack.core.measurement import MeasuredCurve
from xstack.core.physics.models import ParrattModel
from xstack.core.structure import Layer, Stack
from xstack.core.targets import AngleScan, EnergyScan
from xstack.core.units import angle_to_q, angstrom, deg, energy_to_wavelength, kev, mrad

ANGLES_DEG = [0.1, 0.2, 0.3]
R = [0.9, 0.5, 0.2]


@pytest.fixture(autouse=True)
def no_plugins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(files, "load_plugins", lambda: ())


def write(tmp_path: Path, text: str, name: str = "scan.csv") -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


def read_one(path: Path, layout: CsvLayout | None = None) -> MeasuredCurve:
    (curve,) = CsvParser().read(path, layout)
    return curve


def test_default_layout(tmp_path: Path) -> None:
    path = write(tmp_path, "0.1,0.9\n0.2,0.5\n0.3,0.2\n")
    curve = read_one(path)
    assert curve.axis == "angle"
    assert curve.x == pytest.approx(tuple(deg(np.array(ANGLES_DEG))))
    assert curve.reflectivity == tuple(R)
    assert curve.uncertainty is None
    assert curve.energy is None


def test_name_and_source_from_file(tmp_path: Path) -> None:
    curve = read_one(write(tmp_path, "0.1,0.9\n", "my.scan.csv"))
    assert curve.name == "my.scan"
    assert curve.source == "my.scan.csv"


def test_explicit_name(tmp_path: Path) -> None:
    curve = read_one(write(tmp_path, "0.1,0.9\n"), CsvLayout(name="sample A"))
    assert curve.name == "sample A"


def test_header_columns_by_name_case_insensitive(tmp_path: Path) -> None:
    path = write(tmp_path, "Time,Theta,Counts\n1,0.1,0.9\n2,0.2,0.5\n")
    layout = CsvLayout(x_column="THETA", reflectivity_column="counts")
    curve = read_one(path, layout)
    assert curve.reflectivity == (0.9, 0.5)
    assert curve.x == pytest.approx(tuple(deg(np.array([0.1, 0.2]))))


def test_columns_by_index(tmp_path: Path) -> None:
    path = write(tmp_path, "a,b,c\n9,0.1,0.9\n9,0.2,0.5\n")
    curve = read_one(path, CsvLayout(x_column=1, reflectivity_column=2))
    assert curve.reflectivity == (0.9, 0.5)


def test_uncertainty_column(tmp_path: Path) -> None:
    path = write(tmp_path, "t,r,err\n0.1,0.9,0.09\n0.2,0.5,0.05\n")
    curve = read_one(path, CsvLayout(uncertainty_column="err"))
    assert curve.uncertainty == (0.09, 0.05)


@pytest.mark.parametrize(
    ("delimiter", "sep"),
    [(None, ","), (None, ";"), (None, "\t"), (None, "  "), (None, " "), (",", ","), (";", ";")],
)
def test_delimiters(tmp_path: Path, delimiter: str | None, sep: str) -> None:
    text = "\n".join(f"{a}{sep}{r}" for a, r in zip(ANGLES_DEG, R, strict=True))
    curve = read_one(write(tmp_path, text), CsvLayout(delimiter=delimiter))
    assert curve.reflectivity == tuple(R)
    assert curve.size == 3


def test_whitespace_with_explicit_none_and_header(tmp_path: Path) -> None:
    path = write(tmp_path, "theta   R\n0.1   0.9\n0.2   0.5\n")
    assert read_one(path).reflectivity == (0.9, 0.5)


def test_decimal_comma_with_semicolon(tmp_path: Path) -> None:
    path = write(tmp_path, "0,1;0,9\n0,2;0,5\n")
    curve = read_one(path, CsvLayout(decimal=","))
    assert curve.reflectivity == (0.9, 0.5)
    assert curve.x == pytest.approx(tuple(deg(np.array([0.1, 0.2]))))


def test_decimal_comma_with_header(tmp_path: Path) -> None:
    path = write(tmp_path, "theta;R\n0,1;0,9\n0,2;0,5\n")
    assert read_one(path, CsvLayout(decimal=",")).reflectivity == (0.9, 0.5)


def test_comments_and_blank_lines(tmp_path: Path) -> None:
    path = write(tmp_path, "# comment\n\n0.1,0.9\n   # indented\n\n0.2,0.5\n")
    assert read_one(path).size == 2


def test_custom_comment_prefix(tmp_path: Path) -> None:
    path = write(tmp_path, "% comment\n0.1,0.9\n")
    assert read_one(path, CsvLayout(comment="%")).size == 1


@pytest.mark.parametrize(
    ("unit", "factor"), [("deg", deg(1.0)), ("mrad", mrad(1.0)), ("rad", 1.0), (None, deg(1.0))]
)
def test_angle_units(tmp_path: Path, unit: str | None, factor: float) -> None:
    curve = read_one(write(tmp_path, "0.1,0.9\n0.2,0.5\n"), CsvLayout(unit=unit))
    assert curve.x == pytest.approx((0.1 * factor, 0.2 * factor))


def test_two_theta_halves_angle(tmp_path: Path) -> None:
    layout = CsvLayout(quantity="two_theta", unit="deg")
    curve = read_one(write(tmp_path, "0.2,0.9\n0.4,0.5\n"), layout)
    assert curve.x == pytest.approx((deg(0.1), deg(0.2)))


@pytest.mark.parametrize(("unit", "factor"), [("1/A", 10.0), ("1/nm", 1.0), (None, 10.0)])
def test_q_units(tmp_path: Path, unit: str | None, factor: float) -> None:
    energy = 8048.0
    theta = np.array([0.2, 0.5])
    q_nm = angle_to_q(deg(theta), energy)
    text = "".join(f"{float(q) / factor!r},0.5\n" for q in q_nm)
    layout = CsvLayout(quantity="q", unit=unit, energy=energy)
    curve = read_one(write(tmp_path, text), layout)
    assert curve.x == pytest.approx(tuple(deg(theta)), rel=1e-9)
    assert curve.energy == energy


def test_q_requires_energy(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires the photon energy"):
        read_one(write(tmp_path, "0.1,0.9\n"), CsvLayout(quantity="q"))


def test_q_beyond_maximum(tmp_path: Path) -> None:
    q_max = 4.0 * math.pi / energy_to_wavelength(8048.0)
    path = write(tmp_path, f"{q_max * 1.1},0.5\n")
    with pytest.raises(ValueError, match="exceed"):
        read_one(path, CsvLayout(quantity="q", unit="1/nm", energy=8048.0))


@pytest.mark.parametrize(("unit", "factor"), [("eV", 1.0), ("keV", 1000.0), (None, 1.0)])
def test_energy_units(tmp_path: Path, unit: str | None, factor: float) -> None:
    layout = CsvLayout(quantity="energy", unit=unit, angle=deg(0.5))
    curve = read_one(write(tmp_path, "5,0.9\n10,0.5\n"), layout)
    assert curve.axis == "energy"
    assert curve.x == pytest.approx((5.0 * factor, 10.0 * factor))
    assert curve.angle == deg(0.5)


def test_reflectivity_scale(tmp_path: Path) -> None:
    path = write(tmp_path, "0.1,90,9\n0.2,50,5\n")
    curve = read_one(path, CsvLayout(uncertainty_column=2, reflectivity_scale=0.01))
    assert curve.reflectivity == pytest.approx((0.9, 0.5))
    assert curve.uncertainty == pytest.approx((0.09, 0.05))


def test_counts_normalized_to_maximum(tmp_path: Path) -> None:
    path = write(tmp_path, "0.1,400\n0.2,1600\n0.3,100\n0.4,0\n")
    curve = read_one(path, CsvLayout(counts=True))
    assert curve.reflectivity == pytest.approx((0.25, 1.0, 0.0625, 0.0))
    assert curve.uncertainty == pytest.approx((20 / 1600, 40 / 1600, 10 / 1600, 0.0))


def test_counts_with_additional_scale(tmp_path: Path) -> None:
    path = write(tmp_path, "0.1,400\n0.2,1600\n")
    curve = read_one(path, CsvLayout(counts=True, reflectivity_scale=0.8))
    assert curve.reflectivity == pytest.approx((0.2, 0.8))
    assert curve.uncertainty == pytest.approx((0.8 * 20 / 1600, 0.8 * 40 / 1600))


def test_counts_normalized_to_full_scale(tmp_path: Path) -> None:
    path = write(tmp_path, "0.1,400\n0.2,1600\n")
    curve = read_one(path, CsvLayout(counts=True, full_scale=2000.0))
    assert curve.reflectivity == pytest.approx((0.2, 0.8))
    assert curve.uncertainty == pytest.approx((0.01, 0.02))


def test_counts_with_uncertainty_column(tmp_path: Path) -> None:
    path = write(tmp_path, "0.1,400,30\n0.2,1600,50\n")
    curve = read_one(path, CsvLayout(uncertainty_column=2, counts=True))
    assert curve.reflectivity == pytest.approx((0.25, 1.0))
    assert curve.uncertainty == pytest.approx((30 / 1600, 50 / 1600))


def test_counts_maximum_of_kept_rows(tmp_path: Path) -> None:
    path = write(tmp_path, "0.0,5000\n0.1,400\n0.2,800\n")
    curve = read_one(path, CsvLayout(counts=True))
    assert curve.reflectivity == pytest.approx((0.5, 1.0))


def test_negative_counts_get_zero_uncertainty(tmp_path: Path) -> None:
    path = write(tmp_path, "0.1,-4\n0.2,100\n")
    curve = read_one(path, CsvLayout(counts=True))
    assert curve.reflectivity == pytest.approx((-0.04, 1.0))
    assert curve.uncertainty == pytest.approx((0.0, 0.1))


def test_counts_without_positive_value(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no positive value"):
        read_one(write(tmp_path, "0.1,0\n0.2,0\n"), CsvLayout(counts=True))


@pytest.mark.parametrize(
    ("options", "match"),
    [
        ({"full_scale": 100.0}, "requires counts"),
        ({"counts": True, "full_scale": 0.0}, "must be positive"),
        ({"counts": True, "full_scale": -5.0}, "must be positive"),
    ],
)
def test_invalid_full_scale(options: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        CsvLayout(**options)  # type: ignore[arg-type]


def test_polarization_and_fixed_values(tmp_path: Path) -> None:
    layout = CsvLayout(energy=8000.0, polarization="s")
    curve = read_one(write(tmp_path, "0.1,0.9\n"), layout)
    assert curve.polarization == "s"
    assert curve.energy == 8000.0


def test_nonfinite_and_nonpositive_x_rows_skipped(tmp_path: Path) -> None:
    path = write(tmp_path, "0.0,0.9\n-0.1,0.9\n0.1,nan\nnan,0.5\n0.2,0.5\n0.3,0.2\n")
    curve = read_one(path)
    assert curve.reflectivity == (0.5, 0.2)


def test_nan_uncertainty_row_skipped(tmp_path: Path) -> None:
    path = write(tmp_path, "0.1,0.9,nan\n0.2,0.5,0.1\n")
    curve = read_one(path, CsvLayout(uncertainty_column=2))
    assert curve.size == 1


def test_all_rows_skipped_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no row with finite values"):
        read_one(write(tmp_path, "0.0,0.9\n"))


def test_non_numeric_data_reports_line(tmp_path: Path) -> None:
    path = write(tmp_path, "theta,R\n0.1,0.9\n0.2,oops\n")
    with pytest.raises(ValueError, match=r"line 3: 'oops' is not a number"):
        read_one(path)


def test_line_numbers_count_comments_and_blanks(tmp_path: Path) -> None:
    path = write(tmp_path, "# c\n\n0.1,0.9\n0.2,x\n")
    with pytest.raises(ValueError, match="line 4"):
        read_one(path)


def test_missing_column_by_index(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"line 1: no column 5"):
        read_one(write(tmp_path, "0.1,0.9\n"), CsvLayout(reflectivity_column=5))


def test_missing_column_by_name(tmp_path: Path) -> None:
    path = write(tmp_path, "theta,R\n0.1,0.9\n")
    with pytest.raises(ValueError, match=r"no column 'counts'.*theta, R"):
        read_one(path, CsvLayout(reflectivity_column="counts"))


def test_name_without_header(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no header"):
        read_one(write(tmp_path, "0.1,0.9\n"), CsvLayout(x_column="theta"))


@pytest.mark.parametrize("text", ["", "\n\n", "# only comment\n"])
def test_empty_file(tmp_path: Path, text: str) -> None:
    with pytest.raises(ValueError, match="no data"):
        read_one(write(tmp_path, text))


def test_header_only_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no data rows"):
        read_one(write(tmp_path, "theta,R\n"))


def test_wrong_options_type(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="expected CsvLayout options, got dict"):
        CsvParser().read(write(tmp_path, "0.1,0.9\n"), {})


def test_utf8_bom_and_latin1(tmp_path: Path) -> None:
    path = tmp_path / "bom.csv"
    path.write_bytes("theta,R\n0.1,0.9\n".encode("utf-8-sig"))
    assert read_one(path, CsvLayout(x_column="theta")).size == 1
    path = tmp_path / "latin.csv"
    path.write_bytes("# Winkel \xb0\n0.1,0.9\n".encode("latin-1"))
    assert read_one(path).size == 1


def test_layout_unknown_unit() -> None:
    with pytest.raises(ValueError, match="unknown unit 'furlong'"):
        CsvLayout(unit="furlong")
    with pytest.raises(ValueError, match="unknown unit"):
        CsvLayout(quantity="energy", unit="deg")


def test_layout_unknown_quantity() -> None:
    with pytest.raises(ValueError, match="unknown quantity"):
        CsvLayout(quantity="length")  # type: ignore[arg-type]


@pytest.mark.parametrize("scale", [0.0, -1.0, math.nan])
def test_layout_scale_must_be_positive(scale: float) -> None:
    with pytest.raises(ValueError, match="scale must be positive"):
        CsvLayout(reflectivity_scale=scale)


def test_layout_decimal_equals_delimiter() -> None:
    with pytest.raises(ValueError, match="must differ"):
        CsvLayout(decimal=",", delimiter=",")


def test_layout_bad_decimal() -> None:
    with pytest.raises(ValueError, match="decimal separator"):
        CsvLayout(decimal=";")  # type: ignore[arg-type]


def test_layout_axis() -> None:
    assert CsvLayout().axis == "angle"
    assert CsvLayout(quantity="two_theta").axis == "angle"
    assert CsvLayout(quantity="q").axis == "angle"
    assert CsvLayout(quantity="energy").axis == "energy"


def test_units_defaults_first() -> None:
    assert next(iter(UNITS["angle"])) == "deg"
    assert next(iter(UNITS["q"])) == "1/A"
    assert next(iter(UNITS["energy"])) == "eV"
    assert UNITS["q"]["1/A"](np.array([1.0]))[0] == pytest.approx(1.0 / angstrom(1.0))


# --- export round trip ---------------------------------------------------------------------

W = materials.get("W")
SI = materials.get("Si")
MODEL = ParrattModel()


def make_stack() -> Stack:
    return Stack([Layer(W, 2.0, 0.3, "top"), Layer(SI, 3.0, 0.2, "bottom")], SI, 0.5, SI)


def test_export_round_trip(tmp_path: Path) -> None:
    angles = AngleScan(
        energy=8000.0,
        angle_min=math.radians(0.2),
        angle_max=math.radians(1.0),
        name="angles",
        points=5,
    )
    energies = EnergyScan(
        angle=math.radians(0.5),
        energy_min=5000.0,
        energy_max=20000.0,
        name="scan",
        reflections=2,
        points=7,
    )
    stack = make_stack()
    path = tmp_path / "export.csv"
    write_reflectivity_csv(path, stack, [angles, energies], MODEL)
    rows = reflectivity_table(stack, [angles, energies], MODEL)

    a, e = read_curves(path)
    assert (a.name, e.name) == ("angles", "scan")
    assert (a.axis, e.axis) == ("angle", "energy")
    assert a.source == e.source == "export.csv"
    assert a.energy == pytest.approx(8000.0)
    assert e.angle == pytest.approx(math.radians(0.5))
    assert a.size == 5
    assert e.size == 7
    assert (a.x[0], a.x[-1]) == pytest.approx((math.radians(0.2), math.radians(1.0)))
    assert e.x[0] == pytest.approx(kev(5.0))
    assert e.x[-1] == pytest.approx(kev(20.0))
    expected_a = [r["reflectivity"] for r in rows if r["target"] == "angles"]
    expected_e = [r["reflectivity"] for r in rows if r["target"] == "scan"]
    assert a.reflectivity == pytest.approx(tuple(expected_a), rel=1e-12)
    assert e.reflectivity == pytest.approx(tuple(expected_e), rel=1e-12)
    # the two-reflection column differs, so the single-reflection column was read
    expected_n = [r["reflectivity^N"] for r in rows if r["target"] == "scan"]
    assert e.reflectivity != pytest.approx(tuple(expected_n), rel=1e-6)
