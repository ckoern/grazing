"""Tests of the Analyze workflow page."""

from pytestqt.qtbot import QtBot

from xstack.core.optim.events import CancellationToken
from xstack.design.manufacturing.tolerance import ErrorModel, ToleranceResult
from xstack.design.project.templates import new_state
from xstack.gui.common.workers import Job
from xstack.gui.design.steps import AnalyzePage


def make_page(qtbot: QtBot) -> AnalyzePage:
    page = AnalyzePage()
    qtbot.addWidget(page)
    page.set_state(new_state("periodic"))
    return page


def test_default_model_is_thickness_only(qtbot: QtBot) -> None:
    page = make_page(qtbot)
    assert page.error_model() == ErrorModel(thickness_random=0.02)


def test_presets(qtbot: QtBot) -> None:
    page = make_page(qtbot)
    page.preset_combo.setCurrentText("Roughness only")
    assert page.error_model() == ErrorModel(roughness_random=0.1, roughness_systematic=0.05)
    page.preset_combo.setCurrentText("Interlayer mixing only")
    assert page.error_model() == ErrorModel(interlayer_thickness=0.5, interlayer_random=0.2)
    page.preset_combo.setCurrentText("All errors")
    model = page.error_model()
    assert model.thickness_random == 0.02
    assert model.roughness_random == 0.1
    assert model.interlayer_thickness == 0.5
    page.preset_combo.setCurrentText("Thickness errors only")
    assert page.error_model() == ErrorModel(thickness_random=0.02)
    assert page.preset_combo.currentText() == "Thickness errors only"


def test_custom_edit_converts_percent_and_switches_preset(qtbot: QtBot) -> None:
    page = make_page(qtbot)
    page.preset_combo.setCurrentText("All errors")
    page.systematic_error.setValue(3.0)
    assert page.preset_combo.currentText() == "Custom"
    page.drift_error.setValue(1.5)
    page.interlayer_random.setValue(30.0)
    model = page.error_model()
    assert model.thickness_systematic == 0.03
    assert model.thickness_drift == 0.015
    assert model.interlayer_random == 0.3


def test_disabled_sections_give_zeros(qtbot: QtBot) -> None:
    page = make_page(qtbot)
    page.preset_combo.setCurrentText("All errors")
    page.thickness_box.setChecked(False)
    assert page.preset_combo.currentText() == "Custom"
    model = page.error_model()
    assert model.thickness_random == model.thickness_systematic == model.thickness_drift == 0.0
    assert model.roughness_random > 0


def test_run_emits_job_with_error_model(qtbot: QtBot) -> None:
    page = make_page(qtbot)
    page.preset_combo.setCurrentText("All errors")
    page.samples.setValue(3)
    page.seed.setText("1")
    jobs: list[Job] = []
    finished: list[tuple[object, ...]] = []
    page.jobRequested.connect(jobs.append)
    page.toleranceFinished.connect(lambda *args: finished.append(args))
    page.run()
    outcome = jobs[0].function(lambda _: None, CancellationToken())
    result, curves, ranges = outcome
    assert isinstance(result, ToleranceResult)
    assert curves is None
    assert ranges == {}
    assert len(result.merits) == 3
    assert result.errors == page.error_model()
    jobs[0].on_finished(outcome)
    text = page.summary.text()
    for word in ("Nominal", "median", "10", "worst", "feasible", "roughness", "mixed layer"):
        assert word in text
    assert finished == [(result, None, {})]


def test_run_without_errors_reports(qtbot: QtBot) -> None:
    page = make_page(qtbot)
    errors: list[str] = []
    jobs: list[Job] = []
    page.errorOccurred.connect(errors.append)
    page.jobRequested.connect(jobs.append)
    page.thickness_box.setChecked(False)
    page.run()
    assert errors == ["No errors selected"]
    page.thickness_box.setChecked(True)
    page.random_error.setValue(0.0)
    page.run()
    assert errors == ["No errors selected"] * 2
    assert not jobs


def test_help_text_describes_errors() -> None:
    text = AnalyzePage.help_text.lower()
    for word in ("roughness", "mixing", "thickness"):
        assert word in text
