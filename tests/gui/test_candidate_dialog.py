"""Tests for the candidate parameters dialog."""

import pytest
from PySide6.QtGui import QGuiApplication
from pytestqt.qtbot import QtBot

from xstack.gui.widgets.candidate_dialog import CandidateParametersDialog
from xstack.project.project import Candidate
from xstack.project.templates import new_state
from xstack.structure.coating import Mirror

pytestmark = pytest.mark.gui


def _mirror() -> Mirror:
    mirror = new_state("periodic").mirror
    assert mirror is not None
    return mirror


def _candidate(
    mirror: Mirror | None = None,
    merit: float | None = 0.123456,
    objective: float | None = 0.654321,
    coating_time: float | None = 7200.0,
) -> Candidate:
    return Candidate(
        "refine-1",
        mirror or _mirror(),
        "refine",
        "2026-01-02T03:04:05+00:00",
        objective,
        merit,
        coating_time,
    )


def _rows(dialog: CandidateParametersDialog) -> dict[str, list[str]]:
    t = dialog.table()
    return {
        t.item(r, 0).text(): [t.item(r, c).text() for c in range(t.columnCount())]
        for r in range(t.rowCount())
    }


def test_title_and_rows(qtbot: QtBot) -> None:
    mirror = _mirror()
    dialog = CandidateParametersDialog(_candidate(mirror))
    qtbot.addWidget(dialog)
    assert dialog.windowTitle() == "Parameters – refine-1"
    rows = _rows(dialog)
    assert list(rows) == list(mirror.parameters)
    assert rows["N"][:4] == ["N", "40", "", "10 – 200"]
    assert rows["d"][1:4] == ["4", "nm", "2 – 8"]
    assert rows["gamma"][3] == "0.15 – 0.7"
    assert rows["sigma_H"][3] == "fixed"
    assert rows["d"][4] == "Period"


def test_expression_and_infinite_bounds(qtbot: QtBot) -> None:
    mirror = _mirror()
    coating = mirror.coating.with_parameter("sigma_L", expr="sigma_H")
    coating = coating.with_parameter("d", min=0.0, max=float("inf"))
    dialog = CandidateParametersDialog(_candidate(mirror.with_coating(coating)))
    qtbot.addWidget(dialog)
    rows = _rows(dialog)
    assert rows["sigma_L"][3] == "= sigma_H"
    assert rows["d"][3] == "0 – ∞"


def test_header(qtbot: QtBot) -> None:
    dialog = CandidateParametersDialog(_candidate())
    qtbot.addWidget(dialog)
    text = dialog.header_text()
    assert "refine-1" in text
    assert "refine" in text
    assert "2026-01-02" in text
    assert "H: W · L: Si" in text
    assert "substrate Si" in text


def test_summary(qtbot: QtBot) -> None:
    mirror = _mirror()
    stack = mirror.stack()
    dialog = CandidateParametersDialog(_candidate(mirror))
    qtbot.addWidget(dialog)
    text = dialog.summary_text()
    assert f"{stack.num_layers} layers" in text
    assert f"{stack.total_thickness:.5g} nm" in text
    assert "0.12346" in text
    assert "0.65432" in text
    assert "2 h" in text


def test_missing_values(qtbot: QtBot) -> None:
    dialog = CandidateParametersDialog(_candidate(merit=None, objective=None, coating_time=None))
    qtbot.addWidget(dialog)
    lines = dialog.summary_text().splitlines()
    assert lines[0].endswith("—")
    assert lines[1].endswith("—")
    assert lines[2].endswith("—")


def test_unbuildable_mirror(qtbot: QtBot) -> None:
    mirror = _mirror()
    bad = mirror.with_coating(mirror.coating.with_parameter("gamma", expr="1.5"))
    dialog = CandidateParametersDialog(_candidate(bad))
    qtbot.addWidget(dialog)
    assert "not available" in dialog.summary_text()
    assert _rows(dialog)["gamma"][3] == "= 1.5"


def test_copy_button(qtbot: QtBot) -> None:
    mirror = _mirror()
    dialog = CandidateParametersDialog(_candidate(mirror))
    qtbot.addWidget(dialog)
    dialog.copy_button().click()
    text = QGuiApplication.clipboard().text()
    lines = text.splitlines()
    assert lines[0] == "Parameter\tValue\tUnit\tRange\tDescription"
    assert len(lines) == 1 + len(mirror.parameters)
    for name in mirror.parameters:
        assert name in text
    assert all(line.count("\t") == 4 for line in lines)


def test_close_button(qtbot: QtBot) -> None:
    dialog = CandidateParametersDialog(_candidate())
    qtbot.addWidget(dialog)
    dialog.show()
    assert dialog.isVisible()
    dialog.close_button().click()
    assert not dialog.isVisible()
