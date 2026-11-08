"""Tests for `xstack.fit.cli`."""

import csv
from pathlib import Path

import pytest

from xstack.__main__ import main
from xstack.core import materials
from xstack.core.io.reflectivity import write_reflectivity_csv
from xstack.core.physics.models import ParrattModel
from xstack.core.structure import Layer, Stack
from xstack.core.targets import AngleScan, EnergyScan
from xstack.core.units import deg, kev
from xstack.design.project import Project
from xstack.design.project.templates import new_state
from xstack.fit.project import FitProject, set_parameter

PT = materials.get("Pt")
SIO2 = materials.get("SiO2")


@pytest.fixture
def model_file(tmp_path: Path) -> Path:
    path = tmp_path / "mirror.xstack"
    Project(new_state("single_layer")).save(path)
    return path


@pytest.fixture
def export_file(tmp_path: Path) -> Path:
    """Measured curves of a 28 nm Pt layer, one angle scan and one energy scan."""
    stack = Stack([Layer(PT, 28.0, 0.5, "top")], SIO2, 0.5, SIO2)
    angles = AngleScan(
        energy=8048.0, angle_min=deg(0.2), angle_max=deg(0.6), name="1st scan", points=9
    )
    energies = EnergyScan(
        angle=deg(0.4), energy_min=kev(6), energy_max=kev(9), name="e scan", points=6
    )
    path = tmp_path / "measured.csv"
    write_reflectivity_csv(path, stack, [angles, energies], ParrattModel())
    return path


@pytest.fixture
def plain_file(tmp_path: Path) -> Path:
    path = tmp_path / "plain.csv"
    path.write_text(
        "angle,refl,sigma\n"
        + "".join(f"{a},{r},{r / 10}\n" for a, r in [(0.2, 90), (0.3, 50), (0.4, 20), (0.5, 5)])
    )
    return path


def new(tmp_path: Path, model: Path, *extra: str) -> Path:
    out = tmp_path / "fit.xfit"
    assert main(["fit", "new", str(out), "--model", str(model), *extra]) == 0
    return out


def test_new_autodetect(tmp_path: Path, model_file: Path, export_file: Path, capsys) -> None:
    out = new(tmp_path, model_file, "--data", str(export_file), "--name", "demo")
    state = FitProject.load(out).state
    assert state.name == "demo"
    assert [d.name for d in state.datasets] == ["d_1st_scan", "e_scan"]
    assert [d.curve.axis for d in state.datasets] == ["angle", "energy"]
    assert state.datasets[0].curve.energy == pytest.approx(8048.0)
    assert state.datasets[1].curve.angle == pytest.approx(deg(0.4))
    assert state.weighting == "log"
    text = capsys.readouterr().out
    assert "2 datasets, 15 points" in text
    assert "t" in text


def test_new_explicit_layout(tmp_path: Path, model_file: Path, plain_file: Path) -> None:
    out = new(
        tmp_path,
        model_file,
        "--data",
        str(plain_file),
        "--columns",
        "angle,refl,sigma",
        "--percent",
        "--energy",
        "8.048",
        "--weighting",
        "chi2",
    )
    state = FitProject.load(out).state
    (d,) = state.datasets
    assert d.name == "plain"
    assert d.curve.reflectivity[0] == pytest.approx(0.9)
    assert d.curve.uncertainty is not None
    assert d.curve.uncertainty[0] == pytest.approx(0.09)
    assert d.curve.x[0] == pytest.approx(deg(0.2))
    assert d.curve.energy == pytest.approx(8048.0)
    assert state.weighting == "chi2"


@pytest.fixture
def counts_file(tmp_path: Path) -> Path:
    path = tmp_path / "counts.csv"
    path.write_text("angle,N\n0.2,1600\n0.3,400\n0.4,100\n")
    return path


def test_new_counts_normalized_to_maximum(
    tmp_path: Path, model_file: Path, counts_file: Path
) -> None:
    out = new(tmp_path, model_file, "--data", str(counts_file), "--counts", "--energy", "8.048")
    (d,) = FitProject.load(out).state.datasets
    assert d.curve.reflectivity == pytest.approx((1.0, 0.25, 0.0625))
    assert d.curve.uncertainty == pytest.approx((40 / 1600, 20 / 1600, 10 / 1600))


def test_new_counts_full_scale_and_factor(
    tmp_path: Path, model_file: Path, counts_file: Path
) -> None:
    out = new(
        tmp_path,
        model_file,
        "--data",
        str(counts_file),
        "--counts",
        "--full-scale",
        "3200",
        "--scale",
        "2",
        "--energy",
        "8.048",
    )
    (d,) = FitProject.load(out).state.datasets
    assert d.curve.reflectivity == pytest.approx((1.0, 0.25, 0.0625))


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (("--full-scale", "100"), "requires --counts"),
        (("--percent", "--scale", "2"), "cannot be combined"),
    ],
)
def test_new_invalid_count_options(
    tmp_path: Path,
    model_file: Path,
    counts_file: Path,
    capsys: pytest.CaptureFixture[str],
    options: tuple[str, ...],
    message: str,
) -> None:
    args = ["fit", "new", str(tmp_path / "f.xfit"), "--model", str(model_file)]
    code = main([*args, "--data", str(counts_file), "--energy", "8", *options])
    assert code == 1
    assert message in capsys.readouterr().err


def test_new_two_theta_unit(tmp_path: Path, model_file: Path, plain_file: Path) -> None:
    out = new(
        tmp_path,
        model_file,
        "--data",
        str(plain_file),
        "--quantity",
        "two_theta",
        "--unit",
        "deg",
        "--energy",
        "8.048",
    )
    (d,) = FitProject.load(out).state.datasets
    assert d.curve.x[0] == pytest.approx(deg(0.1))


def test_new_energy_applied_to_plain_file(
    tmp_path: Path, model_file: Path, plain_file: Path
) -> None:
    out = new(tmp_path, model_file, "--data", str(plain_file), "--energy", "8.048")
    (d,) = FitProject.load(out).state.datasets
    assert d.curve.energy == pytest.approx(8048.0)


def test_new_ranges_converted(tmp_path: Path, model_file: Path, export_file: Path) -> None:
    out = new(
        tmp_path, model_file, "--data", str(export_file), "--range", "0.3:0.5", "--range", "7:8"
    )
    angle, energy = FitProject.load(out).state.datasets
    flat = [x for r in angle.ranges for x in r]
    assert flat == pytest.approx([deg(0.3), deg(0.5), deg(7.0), deg(8.0)])
    flat = [x for r in energy.ranges for x in r]
    assert flat == pytest.approx([kev(0.3), kev(0.5), kev(7.0), kev(8.0)])


def test_new_dedups_names(tmp_path: Path, model_file: Path, plain_file: Path) -> None:
    out = new(
        tmp_path,
        model_file,
        "--data",
        str(plain_file),
        "--data",
        str(plain_file),
        "--data",
        str(plain_file),
        "--energy",
        "8",
    )
    names = [d.name for d in FitProject.load(out).state.datasets]
    assert names == ["plain", "plain_2", "plain_3"]


def test_new_unknown_candidate(
    tmp_path: Path, model_file: Path, plain_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "fit.xfit"
    code = main(
        [
            "fit",
            "new",
            str(out),
            "--model",
            str(model_file),
            "--candidate",
            "nope",
            "--data",
            str(plain_file),
            "--energy",
            "8",
        ]
    )
    assert code == 1
    assert "nope" in capsys.readouterr().err
    assert not out.exists()


def test_new_bad_range(
    tmp_path: Path, model_file: Path, plain_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "fit",
            "new",
            str(tmp_path / "f.xfit"),
            "--model",
            str(model_file),
            "--data",
            str(plain_file),
            "--energy",
            "8",
            "--range",
            "abc",
        ]
    )
    assert code == 1
    assert "LO:HI" in capsys.readouterr().err


def test_new_missing_data_file(
    tmp_path: Path, model_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "fit",
            "new",
            str(tmp_path / "f.xfit"),
            "--model",
            str(model_file),
            "--data",
            str(tmp_path / "missing.csv"),
        ]
    )
    assert code == 1
    assert "error" in capsys.readouterr().err


@pytest.fixture
def project_file(tmp_path: Path, model_file: Path, export_file: Path) -> Path:
    out = new(tmp_path, model_file, "--data", str(export_file), "--range", "0.27:0.53")
    project = FitProject.load(out)
    project.execute(set_parameter("t", value=26.0, vary=True, min=20.0, max=35.0))
    project.save(out)
    return out


def test_run(project_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "fitted.xfit"
    before = FitProject.load(project_file).state
    code = main(
        [
            "fit",
            "run",
            str(project_file),
            "--output",
            str(out),
            "--method",
            "least_squares",
            "--max-evaluations",
            "20",
            "-v",
        ]
    )
    assert code == 0
    text = capsys.readouterr().out
    assert "least_squares" in text
    assert "free" in text
    fitted = FitProject.load(out).state
    assert fitted.result is not None
    assert fitted.optimizers[-1].method == "least_squares"  # type: ignore[attr-defined]
    assert fitted.optimizers[-1].max_evaluations == 20  # type: ignore[attr-defined]
    assert FitProject.load(project_file).state.result is None
    assert fitted.mirror != before.mirror


def test_run_overwrites_input(project_file: Path) -> None:
    assert main(["fit", "run", str(project_file), "--max-evaluations", "5"]) == 0
    assert FitProject.load(project_file).state.result is not None


def test_run_nothing_free(
    tmp_path: Path, model_file: Path, export_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = new(tmp_path, model_file, "--data", str(export_file))
    project = FitProject.load(out)
    project.execute(set_parameter("t", vary=False))
    project.save(out)
    assert main(["fit", "run", str(out)]) == 1
    assert "error" in capsys.readouterr().err


def test_info(project_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["fit", "info", str(project_file)]) == 0
    text = capsys.readouterr().out
    assert "mirror.xstack" in text
    assert "weighting: log" in text
    assert "d_1st_scan: angle scan, 9 points" in text
    assert "d_1st_scan: angle scan, 9 points, 5 fitted" in text
    assert "e_scan: energy scan, 6 points, 0 fitted, ranges 0.27:0.53 keV" in text
    assert "0.27:0.53 deg" in text
    assert "8.048 keV" in text
    assert "0.4 deg" in text
    assert "free" in text
    assert "method:" not in text


def test_info_with_result(project_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    main(["fit", "run", str(project_file), "--max-evaluations", "5"])
    capsys.readouterr()
    assert main(["fit", "info", str(project_file)]) == 0
    assert "method: least_squares" in capsys.readouterr().out


def test_export(project_file: Path, tmp_path: Path) -> None:
    out = tmp_path / "curves.csv"
    assert main(["fit", "export", str(project_file), str(out)]) == 0
    with out.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == [
        "dataset",
        "axis",
        "x",
        "x_unit",
        "measured",
        "uncertainty",
        "model",
        "fitted",
    ]
    assert len(rows) == 15
    angle = [r for r in rows if r["dataset"] == "d_1st_scan"]
    energy = [r for r in rows if r["dataset"] == "e_scan"]
    assert {r["x_unit"] for r in angle} == {"deg"}
    assert {r["x_unit"] for r in energy} == {"keV"}
    assert float(angle[0]["x"]) == pytest.approx(0.2)
    assert float(energy[0]["x"]) == pytest.approx(6.0)
    assert all(r["uncertainty"] == "" for r in rows)
    assert all(0.0 < float(r["model"]) <= 1.0 for r in rows)
    assert [r["fitted"] for r in angle] == list("001111100")
    assert {r["fitted"] for r in energy} == {"0"}
