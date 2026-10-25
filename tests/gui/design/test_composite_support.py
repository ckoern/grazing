"""Design GUI pieces with combined (composite) coatings: summary, process, estimate, dialog."""

import pytest
from PySide6.QtWidgets import QTableWidget
from pytestqt.qtbot import QtBot

from xstack.core.materials import get
from xstack.core.structure import DepthGraded, Mirror, NLayer, Periodic
from xstack.core.structure.coatings.composite import Composite
from xstack.design.manufacturing.process import DepositionProcess
from xstack.design.project import ProjectState
from xstack.design.project.project import Candidate
from xstack.design.project.templates import new_state
from xstack.gui.design.main_window import MainWindow
from xstack.gui.design.materials_summary import coating_title, materials_text
from xstack.gui.design.steps import ProcessPage
from xstack.gui.design.steps.estimate import EstimatePage
from xstack.gui.design.widgets.candidate_dialog import CandidateParametersDialog

pytestmark = pytest.mark.gui


def composite() -> Composite:
    return Composite.create(
        [
            DepthGraded.create(get("Pt"), get("C"), N=10, d_min=3.0, d_max=6.0, name="ptc"),
            NLayer.create([get("B4C"), get("Mo")], [2.0, 3.0], name="mid"),
            Periodic.create(get("W"), get("Si"), N=20, d=4.0, name="wsi"),
        ]
    )


def composite_state() -> ProjectState:
    return new_state("periodic").evolve(mirror=Mirror(composite(), get("Si")))


def test_materials_text() -> None:
    assert coating_title(composite()) == "Combined coating"
    assert materials_text(composite()) == "ptc: Pt/C · mid: B4C, Mo · wsi: W/Si"


def test_materials_text_single_coating_unchanged() -> None:
    coating = Periodic.create(get("W"), get("Si"), N=20, d=4.0)
    assert coating_title(coating) == "Periodic"
    assert materials_text(coating) == "H: W · L: Si"


def test_process_page_lists_each_material_once(qtbot: QtBot) -> None:
    state = composite_state().evolve(process=DepositionProcess({"Si": 0.1}))
    page = ProcessPage()
    qtbot.addWidget(page)
    page.set_state(state)
    names = [page.table.item(r, 0).text() for r in range(page.table.rowCount())]
    assert names == sorted({"Pt", "C", "B4C", "Mo", "W", "Si"})


def test_estimate_unavailable_for_composite(qtbot: QtBot) -> None:
    page = EstimatePage()
    qtbot.addWidget(page)
    page.set_state(composite_state())
    assert not page.estimate_button.isEnabled()
    assert not page.unsupported_label.isHidden()
    assert "combined" in page.unsupported_label.text()


def test_candidate_dialog_header_for_composite(qtbot: QtBot) -> None:
    candidate = Candidate(
        "c-1",
        Mirror(composite(), get("Si")),
        "refine",
        "2026-01-02T03:04:05+00:00",
        None,
        None,
        None,
    )
    dialog = CandidateParametersDialog(candidate)
    qtbot.addWidget(dialog)
    text = dialog.header_text()
    assert "ptc: Pt/C · mid: B4C, Mo · wsi: W/Si" in text
    assert isinstance(dialog.table(), QTableWidget)


def test_main_window_summary_for_composite(window: MainWindow) -> None:
    window._on_state_changed(composite_state())
    text = window.design_label.text()
    assert "Combined coating" in text
    assert "ptc: Pt/C · mid: B4C, Mo · wsi: W/Si" in text
    assert "substrate Si" in text
