"""Tests for `xstack.gui.fit.widgets.import_dialog`."""

import math
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtWidgets import QFileDialog
from pytestqt.qtbot import QtBot

from xstack.core import materials
from xstack.core.io import files
from xstack.core.io.delimited import CsvLayout
from xstack.core.io.reflectivity import write_reflectivity_csv
from xstack.core.physics.models import ParrattModel
from xstack.core.structure import Layer, Stack
from xstack.core.targets import AngleScan, EnergyScan
from xstack.core.units import deg, kev
from xstack.gui.fit.widgets import ImportDialog
from xstack.gui.fit.widgets.import_dialog import energy_hint, unique_identifier


@pytest.fixture(autouse=True)
def no_plugins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(files, "load_plugins", lambda: ())


def make(qtbot: QtBot, used: tuple[str, ...] = ()) -> ImportDialog:
    dialog = ImportDialog(used)
    qtbot.addWidget(dialog)
    return dialog


def write(tmp_path: Path, text: str, name: str = "scan.csv") -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


HEADER_FILE = "# comment\n2theta_deg,R,dR\n0.2,0.9,0.01\n0.4,0.5,0.02\n0.6,0.1,0.03\n"


def cell(dialog: ImportDialog, row: int, column: int) -> str:
    item = dialog.preview_table.item(row, column)
    assert item is not None
    return str(item.text())


def test_initial_state(qtbot: QtBot) -> None:
    dialog = make(qtbot)
    assert dialog.ok_button.text() == "Add dataset"
    assert not dialog.ok_button.isEnabled()
    assert dialog.layout_options() is None
    assert dialog.datasets() == []


def test_header_file_guesses_columns_and_quantity(qtbot: QtBot, tmp_path: Path) -> None:
    dialog = make(qtbot)
    dialog.set_path(write(tmp_path, HEADER_FILE))
    assert [dialog.x_combo.itemText(i) for i in range(3)] == ["2theta_deg", "R", "dR"]
    assert dialog.x_combo.currentText() == "2theta_deg"
    assert dialog.r_combo.currentText() == "R"
    assert dialog.sigma_combo.currentText() == "dR"
    assert dialog.quantity_combo.currentData() == "two_theta"
    assert dialog.unit_combo.currentText() == "deg"
    assert "header row" in dialog.format_label.text()
    assert "comma" in dialog.format_label.text()
    assert dialog.error_label.text() == ""
    assert dialog.ok_button.isEnabled()
    assert dialog.name_edit.text() == "scan"


def test_layout_and_dataset(qtbot: QtBot, tmp_path: Path) -> None:
    dialog = make(qtbot)
    dialog.set_path(write(tmp_path, HEADER_FILE))
    layout = dialog.layout_options()
    assert layout is not None
    assert layout.quantity == "two_theta"
    assert layout.unit == "deg"
    assert layout.x_column == 0
    assert layout.reflectivity_column == 1
    assert layout.uncertainty_column == 2
    assert layout.energy == pytest.approx(kev(8.048))
    (dataset,) = dialog.datasets()
    assert dataset.name == "scan"
    curve = dataset.curve
    assert curve.x == pytest.approx(tuple(deg(np.array([0.1, 0.2, 0.3]))))
    assert curve.uncertainty == (0.01, 0.02, 0.03)
    assert curve.energy == pytest.approx(kev(8.048))
    assert curve.polarization == "unpolarized"


def test_preview_values(qtbot: QtBot, tmp_path: Path) -> None:
    dialog = make(qtbot)
    dialog.set_path(write(tmp_path, HEADER_FILE))
    table = dialog.preview_table
    assert table.columnCount() == 6
    assert table.rowCount() == 3
    headers = [table.horizontalHeaderItem(i).text() for i in range(6)]
    assert headers == ["2theta_deg", "R", "dR", "→ θ (deg)", "→ R", "→ σ"]
    assert [cell(dialog, 0, c) for c in range(3)] == ["0.2", "0.9", "0.01"]
    assert cell(dialog, 0, 3) == "0.1000"
    assert float(cell(dialog, 1, 4)) == pytest.approx(0.5)
    assert float(cell(dialog, 2, 5)) == pytest.approx(0.03)
    assert dialog.count_label.text() == "3 rows read, 0 skipped"


def test_skipped_rows_counted(qtbot: QtBot, tmp_path: Path) -> None:
    path = write(tmp_path, "0.0,0.9\n0.1,0.8\nnan,0.5\n0.3,0.2\n")
    dialog = make(qtbot)
    dialog.set_path(path)
    assert dialog.count_label.text() == "2 rows read, 2 skipped"


def test_headerless_whitespace_file(qtbot: QtBot, tmp_path: Path) -> None:
    dialog = make(qtbot)
    dialog.set_path(write(tmp_path, "0.1 0.9\n0.2 0.5\n0.3 0.2\n", "plain.dat"))
    assert [dialog.x_combo.itemText(i) for i in range(2)] == ["column 1", "column 2"]
    assert dialog.sigma_combo.currentText() == "(none)"
    assert "no header" in dialog.format_label.text()
    assert "whitespace" in dialog.format_label.text()
    assert dialog.quantity_combo.currentData() == "angle"
    (dataset,) = dialog.datasets()
    assert dataset.curve.uncertainty is None
    assert dataset.curve.reflectivity == (0.9, 0.5, 0.2)
    assert cell(dialog, 0, 2) == "0.1000"
    assert cell(dialog, 0, 4) == "—"


def test_column_choice(qtbot: QtBot, tmp_path: Path) -> None:
    path = write(tmp_path, "t,theta,counts\n1,0.1,0.9\n2,0.2,0.5\n")
    dialog = make(qtbot)
    dialog.set_path(path)
    dialog.x_combo.setCurrentIndex(1)
    dialog.r_combo.setCurrentIndex(2)
    layout = dialog.layout_options()
    assert layout is not None
    assert (layout.x_column, layout.reflectivity_column) == (1, 2)
    (dataset,) = dialog.datasets()
    assert dataset.curve.reflectivity == (0.9, 0.5)


def test_unit_and_quantity(qtbot: QtBot, tmp_path: Path) -> None:
    dialog = make(qtbot)
    dialog.set_path(write(tmp_path, "10,0.9\n20,0.5\n"))
    dialog.unit_combo.setCurrentText("mrad")
    (dataset,) = dialog.datasets()
    assert dataset.curve.x == pytest.approx((0.010, 0.020))
    assert cell(dialog, 0, 2) == f"{math.degrees(0.010):.4f}"
    assert [dialog.unit_combo.itemText(i) for i in range(3)] == ["deg", "mrad", "rad"]
    dialog.quantity_combo.setCurrentIndex(dialog.quantity_combo.findData("two_theta"))
    assert dialog.unit_combo.currentText() == "deg"
    dialog.unit_combo.setCurrentText("mrad")
    (dataset,) = dialog.datasets()
    assert dataset.curve.x == pytest.approx((0.005, 0.010))


def test_two_theta_halves_angle(qtbot: QtBot, tmp_path: Path) -> None:
    dialog = make(qtbot)
    dialog.set_path(write(tmp_path, "0.2,0.9\n0.4,0.5\n"))
    dialog.quantity_combo.setCurrentIndex(dialog.quantity_combo.findData("two_theta"))
    (dataset,) = dialog.datasets()
    assert dataset.curve.x == pytest.approx(tuple(deg(np.array([0.1, 0.2]))))


def test_q_uses_entered_energy(qtbot: QtBot, tmp_path: Path) -> None:
    dialog = make(qtbot)
    dialog.set_path(write(tmp_path, "0.1,0.9\n0.2,0.5\n"))
    dialog.quantity_combo.setCurrentIndex(dialog.quantity_combo.findData("q"))
    assert dialog.unit_combo.currentText() == "1/A"
    assert not dialog.energy_widget.isHidden()
    dialog.preset_combo.setCurrentText("Mo Kα")
    assert dialog.energy_spin.value() == pytest.approx(17.479)
    (dataset,) = dialog.datasets()
    lam = 12.398419843320026 / 17.479 / 10.0  # nm
    assert dataset.curve.x[0] == pytest.approx(math.asin(1.0 * lam / (4 * math.pi)), rel=1e-6)
    assert dataset.curve.energy == pytest.approx(kev(17.479))


def test_energy_presets_and_custom(qtbot: QtBot, tmp_path: Path) -> None:
    dialog = make(qtbot)
    dialog.set_path(write(tmp_path, "0.1,0.9\n"))
    assert dialog.energy_spin.value() == pytest.approx(8.048)
    for name, value in (("Mo Kα", 17.479), ("Ag Kα", 22.163), ("Cu Kα", 8.048)):
        dialog.preset_combo.setCurrentText(name)
        assert dialog.energy_spin.value() == pytest.approx(value)
    dialog.energy_spin.setValue(12.5)
    assert dialog.preset_combo.currentText() == "custom"
    (dataset,) = dialog.datasets()
    assert dataset.curve.energy == pytest.approx(kev(12.5))
    dialog.energy_spin.setValue(17.479)
    assert dialog.preset_combo.currentText() == "Mo Kα"


def test_energy_scan_uses_angle(qtbot: QtBot, tmp_path: Path) -> None:
    dialog = make(qtbot)
    dialog.set_path(write(tmp_path, "E_keV,R\n6.0,0.9\n7.0,0.5\n"))
    assert dialog.quantity_combo.currentData() == "energy"
    assert dialog.energy_widget.isHidden()
    assert not dialog.angle_spin.isHidden()
    dialog.unit_combo.setCurrentText("keV")
    dialog.angle_spin.setValue(2.5)
    assert cell(dialog, 0, 2) == "6.0000"
    (dataset,) = dialog.datasets()
    curve = dataset.curve
    assert curve.axis == "energy"
    assert curve.x == pytest.approx((kev(6.0), kev(7.0)))
    assert curve.angle == pytest.approx(deg(2.5))
    assert dialog.preview_table.horizontalHeaderItem(2).text() == "→ E (keV)"


def test_percent_and_polarization(qtbot: QtBot, tmp_path: Path) -> None:
    dialog = make(qtbot)
    dialog.set_path(write(tmp_path, "t,R,dR\n0.1,90,1\n0.2,50,2\n"))
    dialog.percent_check.setChecked(True)
    dialog.polarization_combo.setCurrentText("s")
    (dataset,) = dialog.datasets()
    assert dataset.curve.reflectivity == pytest.approx((0.9, 0.5))
    assert dataset.curve.uncertainty == pytest.approx((0.01, 0.02))
    assert dataset.curve.polarization == "s"
    assert float(cell(dialog, 0, 4)) == pytest.approx(0.9)
    layout = dialog.layout_options()
    assert layout is not None
    assert layout.reflectivity_scale == 0.01


COUNTS_FILE = "t,N\n0.1,100\n0.2,400\n0.3,25\n"


def counts_dialog(qtbot: QtBot, tmp_path: Path, text: str = COUNTS_FILE) -> ImportDialog:
    dialog = make(qtbot)
    dialog.set_path(write(tmp_path, text))
    dialog.values_combo.setCurrentIndex(dialog.values_combo.findData(True))
    return dialog


def test_reflectivity_mode_is_default(qtbot: QtBot, tmp_path: Path) -> None:
    dialog = make(qtbot)
    dialog.set_path(write(tmp_path, HEADER_FILE))
    assert dialog.values_combo.currentData() is False
    assert not dialog.percent_check.isHidden()
    assert dialog.normalize_widget.isHidden()
    assert dialog.r_label.text() == "R"
    layout = dialog.layout_options()
    assert layout is not None
    assert not layout.counts
    assert layout.full_scale is None


def test_counts_largest_normalization_and_sqrt_uncertainty(qtbot: QtBot, tmp_path: Path) -> None:
    dialog = counts_dialog(qtbot, tmp_path)
    assert dialog.error_label.text() == ""
    (dataset,) = dialog.datasets()
    assert dataset.curve.reflectivity == pytest.approx((0.25, 1.0, 0.0625))
    assert dataset.curve.uncertainty == pytest.approx((10 / 400, 20 / 400, 5 / 400))
    layout = dialog.layout_options()
    assert layout is not None
    assert layout.counts
    assert layout.full_scale is None
    assert float(cell(dialog, 1, 3)) == pytest.approx(1.0)


def test_counts_full_scale_normalization(qtbot: QtBot, tmp_path: Path) -> None:
    dialog = counts_dialog(qtbot, tmp_path)
    dialog.normalize_combo.setCurrentIndex(dialog.normalize_combo.findData(True))
    dialog.full_scale_spin.setValue(1000)
    (dataset,) = dialog.datasets()
    assert dataset.curve.reflectivity == pytest.approx((0.1, 0.4, 0.025))
    assert dataset.curve.uncertainty == pytest.approx((0.01, 0.02, 0.005))
    layout = dialog.layout_options()
    assert layout is not None
    assert layout.full_scale == 1000.0


def test_counts_factor(qtbot: QtBot, tmp_path: Path) -> None:
    dialog = counts_dialog(qtbot, tmp_path)
    dialog.scale_spin.setValue(0.5)
    (dataset,) = dialog.datasets()
    assert dataset.curve.reflectivity == pytest.approx((0.125, 0.5, 0.03125))
    layout = dialog.layout_options()
    assert layout is not None
    assert layout.reflectivity_scale == 0.5


def test_counts_with_uncertainty_column(qtbot: QtBot, tmp_path: Path) -> None:
    dialog = counts_dialog(qtbot, tmp_path, "t,N,dN\n0.1,100,5\n0.2,400,8\n")
    (dataset,) = dialog.datasets()
    assert dataset.curve.uncertainty == pytest.approx((5 / 400, 8 / 400))


def test_counts_zero_count_has_one_count_uncertainty(qtbot: QtBot, tmp_path: Path) -> None:
    dialog = counts_dialog(qtbot, tmp_path, "t,N\n0.1,0\n0.2,400\n")
    (dataset,) = dialog.datasets()
    assert dataset.curve.uncertainty == pytest.approx((1 / 400, 20 / 400))


def test_counts_visibility(qtbot: QtBot, tmp_path: Path) -> None:
    dialog = counts_dialog(qtbot, tmp_path)
    assert dialog.percent_check.isHidden()
    assert not dialog.normalize_widget.isHidden()
    assert dialog.full_scale_spin.isHidden()
    assert not dialog.scale_spin.isHidden()
    assert dialog.r_label.text() == "N"
    dialog.normalize_combo.setCurrentIndex(dialog.normalize_combo.findData(True))
    assert not dialog.full_scale_spin.isHidden()
    dialog.normalize_combo.setCurrentIndex(dialog.normalize_combo.findData(False))
    assert dialog.full_scale_spin.isHidden()
    dialog.values_combo.setCurrentIndex(dialog.values_combo.findData(False))
    assert not dialog.percent_check.isHidden()
    assert dialog.normalize_widget.isHidden()
    assert dialog.r_label.text() == "R"


def test_count_widgets_disabled_for_export(qtbot: QtBot, tmp_path: Path) -> None:
    dialog = make(qtbot)
    dialog.set_path(write_export(tmp_path))
    assert not dialog.values_combo.isEnabled()
    assert not dialog.normalize_combo.isEnabled()
    assert not dialog.scale_spin.isEnabled()


def test_broken_file_shows_error(qtbot: QtBot, tmp_path: Path) -> None:
    dialog = make(qtbot)
    dialog.set_path(write(tmp_path, "0.1,0.9\n0.2,abc\n"))
    assert "abc" in dialog.error_label.text()
    assert not dialog.ok_button.isEnabled()
    assert dialog.datasets() == []
    assert dialog.preview_table.rowCount() == 0


def test_missing_file_shows_error(qtbot: QtBot, tmp_path: Path) -> None:
    dialog = make(qtbot)
    dialog.set_path(tmp_path / "nothing.csv")
    assert dialog.error_label.text() != ""
    assert not dialog.ok_button.isEnabled()


def test_error_clears_with_next_file(qtbot: QtBot, tmp_path: Path) -> None:
    dialog = make(qtbot)
    dialog.set_path(write(tmp_path, "0.1,0.9\n0.2,abc\n", "bad.csv"))
    dialog.set_path(write(tmp_path, "0.1,0.9\n0.2,0.5\n", "good.csv"))
    assert dialog.error_label.text() == ""
    assert dialog.ok_button.isEnabled()


def test_layout_error_disables_ok(qtbot: QtBot, tmp_path: Path) -> None:
    dialog = make(qtbot)
    dialog.set_path(write(tmp_path, "0.1,0.9\n0.2,0.5\n"))
    dialog.x_combo.setCurrentIndex(1)  # x = reflectivity column: 0.9/0.5 deg, valid
    dialog.quantity_combo.setCurrentIndex(dialog.quantity_combo.findData("q"))
    dialog.unit_combo.setCurrentText("1/nm")
    dialog.x_combo.setCurrentIndex(0)
    # q = 0.1/nm at 8 keV is valid; q far above 4π/λ is not
    dialog.set_path(write(tmp_path, "1000,0.9\n", "big.csv"))
    dialog.quantity_combo.setCurrentIndex(dialog.quantity_combo.findData("q"))
    assert "4π/λ" in dialog.error_label.text()
    assert not dialog.ok_button.isEnabled()


def test_name_from_stem_is_identifier(qtbot: QtBot, tmp_path: Path) -> None:
    dialog = make(qtbot)
    dialog.set_path(write(tmp_path, "0.1,0.9\n", "17 W-Si.v2.csv"))
    assert dialog.name_edit.text() == "d_17_W_Si_v2"
    assert dialog.datasets()[0].name == "d_17_W_Si_v2"


def test_name_unique_against_used(qtbot: QtBot, tmp_path: Path) -> None:
    dialog = make(qtbot, ("scan", "scan_2"))
    dialog.set_path(write(tmp_path, "0.1,0.9\n"))
    assert dialog.name_edit.text() == "scan_3"
    assert dialog.ok_button.isEnabled()


def test_edited_name_validated(qtbot: QtBot, tmp_path: Path) -> None:
    dialog = make(qtbot, ("taken",))
    dialog.set_path(write(tmp_path, "0.1,0.9\n"))
    dialog.name_edit.setText("taken")
    dialog.name_edit.textEdited.emit("taken")
    assert "already exists" in dialog.error_label.text()
    assert not dialog.ok_button.isEnabled()
    dialog.name_edit.setText("1bad")
    dialog.name_edit.textEdited.emit("1bad")
    assert "not a valid" in dialog.error_label.text()
    assert dialog.datasets() == []
    dialog.name_edit.setText("fresh")
    dialog.name_edit.textEdited.emit("fresh")
    assert dialog.error_label.text() == ""
    assert [d.name for d in dialog.datasets()] == ["fresh"]


@pytest.mark.parametrize(
    ("text", "used", "expected"),
    [
        ("scan", (), "scan"),
        ("a b.c", (), "a_b_c"),
        ("3d", (), "d_3d"),
        ("", (), "data"),
        ("x", ("x",), "x_2"),
        ("x", ("x", "x_2"), "x_3"),
    ],
)
def test_unique_identifier(text: str, used: tuple[str, ...], expected: str) -> None:
    assert unique_identifier(text, used) == expected


def write_export(tmp_path: Path) -> Path:
    si = materials.get("Si")
    stack = Stack([Layer(materials.get("W"), 2.0, 0.3, "top")], si, 0.5, si)
    angles = AngleScan(
        energy=8000.0,
        angle_min=math.radians(0.2),
        angle_max=math.radians(1.0),
        name="angles",
        points=5,
    )
    energies = EnergyScan(
        angle=math.radians(0.5), energy_min=5000.0, energy_max=20000.0, name="scan", points=7
    )
    path = tmp_path / "export.csv"
    write_reflectivity_csv(path, stack, [angles, energies], ParrattModel())
    return path


def test_export_detected(qtbot: QtBot, tmp_path: Path) -> None:
    dialog = make(qtbot, ("angles",))
    dialog.set_path(write_export(tmp_path))
    assert "export" in dialog.format_label.text()
    assert not dialog.note_label.isHidden()
    assert not dialog.x_combo.isEnabled()
    assert not dialog.quantity_combo.isEnabled()
    assert not dialog.name_edit.isEnabled()
    assert dialog.layout_options() is None
    assert dialog.error_label.text() == ""
    assert dialog.ok_button.isEnabled()
    datasets = dialog.datasets()
    assert [d.name for d in datasets] == ["angles_2", "scan"]
    assert [d.curve.axis for d in datasets] == ["angle", "energy"]
    assert datasets[0].curve.energy == pytest.approx(8000.0)
    assert datasets[1].curve.angle == pytest.approx(math.radians(0.5))
    assert "2 curves" in dialog.count_label.text()
    assert dialog.preview_table.columnCount() == 7


def test_layout_controls_enabled_for_plain_file(qtbot: QtBot, tmp_path: Path) -> None:
    dialog = make(qtbot)
    dialog.set_path(write_export(tmp_path))
    dialog.set_path(write(tmp_path, "0.1,0.9\n"))
    assert dialog.x_combo.isEnabled()
    assert dialog.note_label.isHidden()
    assert isinstance(dialog.layout_options(), CsvLayout)


def test_browse_uses_file_dialog(
    qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = write(tmp_path, "0.1,0.9\n")
    seen: list[str] = []

    def fake(parent: object, title: str, start: str, file_filter: str) -> tuple[str, str]:
        seen.append(file_filter)
        return str(path), file_filter

    monkeypatch.setattr(QFileDialog, "getOpenFileName", fake)
    dialog = make(qtbot)
    dialog.browse_button.click()
    assert seen == ["Data files (*.csv *.tsv *.txt *.dat);;All files (*)"]
    assert dialog.path_edit.text() == str(path)
    assert dialog.ok_button.isEnabled()


EXAMPLE_DATA = Path(__file__).parents[3] / "examples" / "data"


@pytest.mark.parametrize(
    ("name", "energy_kev"),
    [("WSi_multilayer_cu_ka.csv", 8.048), ("WSi_multilayer_mo_ka.csv", 17.479)],
)
def test_example_data_imports_with_energy_from_header(
    qtbot: QtBot, name: str, energy_kev: float
) -> None:
    dialog = ImportDialog()
    qtbot.addWidget(dialog)
    dialog.set_path(EXAMPLE_DATA / name)
    (dataset,) = dialog.datasets()
    assert dataset.curve.energy == pytest.approx(energy_kev * 1000.0)
    assert dataset.curve.uncertainty is not None
    assert "from the file header" in dialog.format_label.text()


def test_energy_hint() -> None:
    assert energy_hint(["# Photon energy: 17.479 keV"]) == pytest.approx(17479.0)
    assert energy_hint(["# energy = 8048 eV"]) == pytest.approx(8048.0)
    assert energy_hint(["# wavelength 0.15406 nm"]) == pytest.approx(8048.0, abs=1.0)
    assert energy_hint(["# lambda: 1.5406 Å"]) == pytest.approx(8048.0, abs=1.0)
    assert energy_hint(["# sample: W/Si", "# 300 K"]) is None
