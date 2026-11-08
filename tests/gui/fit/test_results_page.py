"""Tests for `grazing.gui.fit.steps.results`."""

import csv
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication
from pytestqt.qtbot import QtBot

from grazing.core import materials
from grazing.core.measurement import MeasuredCurve
from grazing.core.parameters import free
from grazing.core.structure import Mirror, SingleLayer
from grazing.fit.dataset import Dataset
from grazing.fit.problem import FitProblem
from grazing.fit.project import (
    Command,
    FitProject,
    FitState,
    run_fit,
    store_result,
)
from grazing.fit.report import format_result
from grazing.fit.results import FitResult
from grazing.gui.fit.preview import compute_preview
from grazing.gui.fit.steps.results import EXPORT_COLUMNS, ResultsPage

ANGLES = np.linspace(0.002, 0.012, 30)


def make_mirror(t: float = 30.0) -> Mirror:
    coating = SingleLayer.create(
        materials.get("Pt"), t=free(t, 10.0, 60.0), sigma=free(0.5, 0.0, 3.0)
    )
    return Mirror(coating, materials.get("Si"))


def make_state(start: float = 28.0, weighting: str = "chi2") -> FitState:
    truth = make_mirror(30.0)
    blank = Dataset.create("a", MeasuredCurve("angle", ANGLES, np.ones(30), energy=8048.0))
    sim = FitProblem(truth, (blank,), weighting="linear").simulate(blank)
    rng = np.random.default_rng(1)
    noisy = sim * (1 + 0.02 * rng.standard_normal(30))
    curve = MeasuredCurve("angle", ANGLES, noisy, 0.02 * sim, energy=8048.0)
    dataset = Dataset.create("a", curve, offset=free(0.0, -0.002, 0.002))
    state = FitState(
        name="demo",
        mirror=make_mirror(start),
        datasets=(dataset,),
        weighting=weighting,  # type: ignore[arg-type]
    )
    return state


@pytest.fixture(scope="module")
def fitted() -> tuple[FitState, FitResult]:
    state = make_state()
    return state, run_fit(state)


def project_with_fits(fitted: tuple[FitState, FitResult], n: int = 1) -> FitProject:
    state, result = fitted
    project = FitProject(state)
    for _ in range(n):
        project.execute(store_result(result, apply=False))
    return project


@pytest.fixture
def page(qtbot: QtBot) -> ResultsPage:
    p = ResultsPage()
    qtbot.addWidget(p)
    return p


def capture(page: ResultsPage) -> list[Command]:
    commands: list[Command] = []
    page.commandRequested.connect(commands.append)
    return commands


def test_empty(page: ResultsPage) -> None:
    page.set_state(FitState())
    assert page.fit_list.count() == 0
    assert page.selected_fit() is None
    assert not page.hint.isHidden() or page.hint.isVisibleTo(page)
    assert not page.apply_button.isEnabled()
    assert not page.copy_button.isEnabled()
    assert not page.curves_button.isEnabled()


def test_list_contents(page: ResultsPage, fitted: tuple[FitState, FitResult]) -> None:
    project = project_with_fits(fitted, 2)
    page.set_state(project.state)
    assert page.fit_list.count() == 2
    text = page.fit_list.item(0).text()
    assert text.startswith("fit 1 — χ²_red ")
    assert fitted[1].method in text
    assert page.fit_list.item(0).toolTip()
    assert page.selected_fit() is project.state.results[1]


def test_selection_persists_and_newest_selected(
    page: ResultsPage, fitted: tuple[FitState, FitResult]
) -> None:
    project = project_with_fits(fitted, 2)
    page.set_state(project.state)
    page.fit_list.setCurrentRow(0)
    assert page.selected_fit().name == "fit 1"  # type: ignore[union-attr]
    page.set_state(project.state)
    assert page.selected_fit().name == "fit 1"  # type: ignore[union-attr]
    project.execute(store_result(fitted[1], apply=False))
    page.set_state(project.state)
    assert page.selected_fit().name == "fit 3"  # type: ignore[union-attr]


def test_fit_selected_signal(page: ResultsPage, fitted: tuple[FitState, FitResult]) -> None:
    project = project_with_fits(fitted, 2)
    page.set_state(project.state)
    received: list[object] = []
    page.fitSelected.connect(received.append)
    page.fit_list.setCurrentRow(0)
    assert received[-1].name == "fit 1"  # type: ignore[attr-defined]
    page.set_state(FitState())
    assert received[-1] is None


def test_apply_rename_delete(page: ResultsPage, fitted: tuple[FitState, FitResult]) -> None:
    project = project_with_fits(fitted, 2)
    page.set_state(project.state)
    commands = capture(page)
    page.fit_list.setCurrentRow(0)

    page.apply_button.click()
    project.execute(commands[-1])

    page._ask_name = lambda current: "best"  # type: ignore[method-assign]
    page.rename_button.click()
    project.execute(commands[-1])
    assert [f.name for f in project.state.results] == ["best", "fit 2"]
    page.set_state(project.state)
    assert page.selected_fit().name == "best"  # type: ignore[union-attr]

    page.delete_button.click()
    project.execute(commands[-1])
    assert [f.name for f in project.state.results] == ["fit 2"]


def test_apply_restores_values(page: ResultsPage, fitted: tuple[FitState, FitResult]) -> None:
    _state, result = fitted
    project = project_with_fits(fitted)
    page.set_state(project.state)
    commands = capture(page)
    page.apply_button.click()
    project.execute(commands[-1])
    t = project.state.problem().parameters["t"].value
    assert t == pytest.approx(result.parameters["t"].value)
    assert t != pytest.approx(28.0)


def test_rename_cancel_and_double_click(
    page: ResultsPage, fitted: tuple[FitState, FitResult]
) -> None:
    page.set_state(project_with_fits(fitted).state)
    commands = capture(page)
    page._ask_name = lambda current: None  # type: ignore[method-assign]
    page.rename_button.click()
    assert not commands
    page._ask_name = lambda current: "x"  # type: ignore[method-assign]
    page.fit_list.itemDoubleClicked.emit(page.fit_list.item(0))
    assert len(commands) == 1


def test_rename_duplicate_raises_in_command(
    page: ResultsPage, fitted: tuple[FitState, FitResult]
) -> None:
    project = project_with_fits(fitted, 2)
    page.set_state(project.state)
    commands = capture(page)
    page.fit_list.setCurrentRow(0)
    page._ask_name = lambda current: "fit 2"  # type: ignore[method-assign]
    page.rename_button.click()
    with pytest.raises(ValueError):
        project.execute(commands[-1])


def test_summary_chi2_and_log(page: ResultsPage, fitted: tuple[FitState, FitResult]) -> None:
    page.set_state(project_with_fits(fitted).state)
    text = page.summary.text()
    assert "χ²_red" in text
    assert "30 points" in text
    assert "free parameters" in text
    assert fitted[1].method in text

    state = make_state(weighting="log")
    result = run_fit(state)
    project = FitProject(state)
    project.execute(store_result(result, apply=False))
    page.set_state(project.state)
    assert "objective/dof" in page.summary.text()
    assert "χ²_red" not in page.summary.text()


def test_summary_without_statistics(page: ResultsPage, fitted: tuple[FitState, FitResult]) -> None:
    state, result = fitted
    bare = replace(result, statistics=None, note="singular matrix")
    project = FitProject(state)
    project.execute(store_result(bare, apply=False))
    page.set_state(project.state)
    assert "singular matrix" in page.summary.text()
    assert page.table.rowCount() == 0
    assert "no statistics" in page.fit_list.item(0).text()


def test_summary_undetermined(page: ResultsPage, fitted: tuple[FitState, FitResult]) -> None:
    state, result = fitted
    assert result.statistics is not None
    stats = replace(result.statistics, errors=(math.nan, *result.statistics.errors[1:]))
    project = FitProject(state)
    project.execute(store_result(replace(result, statistics=stats), apply=False))
    page.set_state(project.state)
    assert "do not determine all free parameters" in page.summary.text()
    assert "undetermined" in page.table.item(0, 6).text()
    assert page.table.item(0, 2).text() == ""


def cells(page: ResultsPage, row: int) -> list[str]:
    return [page.table.item(row, c).text() for c in range(page.table.columnCount())]


def test_parameter_table(page: ResultsPage, fitted: tuple[FitState, FitResult]) -> None:
    _state, result = fitted
    page.set_state(project_with_fits(fitted).state)
    stats = result.statistics
    assert stats is not None
    assert page.table.rowCount() == len(stats.names)
    rows = {cells(page, r)[0]: cells(page, r) for r in range(page.table.rowCount())}
    t = rows["t"]
    assert float(t[1]) == pytest.approx(result.parameters["t"].value, rel=1e-5)
    assert t[2].startswith("± ")
    assert t[3].endswith("%")
    assert t[4] == "nm"
    assert float(t[5]) == pytest.approx(28.0)
    offset = rows["a.offset"]
    assert offset[4] == "deg"
    assert float(offset[1]) == pytest.approx(
        math.degrees(result.parameters["a.offset"].value), abs=1e-5
    )
    assert float(offset[5]) == pytest.approx(0.0)


def test_at_bound_note(page: ResultsPage, fitted: tuple[FitState, FitResult]) -> None:
    state, result = fitted
    t = result.parameters["t"]
    parameters = result.parameters.replace("t", value=t.max)
    project = FitProject(state)
    project.execute(store_result(replace(result, parameters=parameters), apply=False))
    page.set_state(project.state)
    rows = {cells(page, r)[0]: cells(page, r) for r in range(page.table.rowCount())}
    assert "at bound" in rows["t"][6]
    assert "at bound" not in rows["sigma"][6]


def test_correlations(page: ResultsPage, fitted: tuple[FitState, FitResult]) -> None:
    state, result = fitted
    assert result.statistics is not None
    n = len(result.statistics.names)
    corr = tuple(tuple(1.0 if i == j else 0.95 for j in range(n)) for i in range(n))
    stats = replace(result.statistics, correlation=corr)
    project = FitProject(state)
    project.execute(store_result(replace(result, statistics=stats), apply=False))
    page.set_state(project.state)
    items = [page.correlation_list.item(i).text() for i in range(page.correlation_list.count())]
    assert len(items) == n * (n - 1) // 2
    assert "r = +0.950" in items[0]

    uncorrelated = replace(
        result.statistics,
        correlation=tuple(tuple(float(i == j) for j in range(n)) for i in range(n)),
    )
    project = FitProject(state)
    project.execute(store_result(replace(result, statistics=uncorrelated), apply=False))
    page.set_state(project.state)
    assert page.correlation_list.item(0).text() == "none"


def test_copy_report(page: ResultsPage, fitted: tuple[FitState, FitResult]) -> None:
    page.set_state(project_with_fits(fitted).state)
    page.copy_button.click()
    assert QApplication.clipboard().text() == format_result(fitted[1])


def test_export_report(
    page: ResultsPage, fitted: tuple[FitState, FitResult], tmp_path: Path
) -> None:
    assert not page.export_report(tmp_path / "none.txt")
    page.set_state(project_with_fits(fitted).state)
    path = tmp_path / "report.txt"
    assert page.export_report(path)
    assert path.read_text(encoding="utf-8").strip() == format_result(fitted[1])


def test_export_curves(
    page: ResultsPage, fitted: tuple[FitState, FitResult], tmp_path: Path
) -> None:
    errors: list[str] = []
    page.errorOccurred.connect(errors.append)
    project = project_with_fits(fitted)
    page.set_state(project.state)
    path = tmp_path / "curves.csv"
    assert not page.export_curves(path)
    assert errors
    assert not path.exists()

    page.set_preview(compute_preview(project.state))
    assert page.export_curves(path)
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert tuple(rows[0]) == EXPORT_COLUMNS
    assert len(rows) == 31
    first = rows[1]
    assert first[0] == "a"
    assert first[1] == "angle"
    assert first[3] == "deg"
    assert float(first[2]) == pytest.approx(math.degrees(ANGLES[0]))
    assert float(first[6]) > 0
    assert first[7] == "1"
