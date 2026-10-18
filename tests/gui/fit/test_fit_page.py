"""Tests for the fit page of the fit application."""

import pytest
from PySide6.QtWidgets import QCheckBox
from pytestqt.qtbot import QtBot

from tests.gui.fit.helpers import make_state
from xstack.core.optim.events import CancellationToken
from xstack.core.optim.optimizers import LmfitOptimizer
from xstack.fit.project import FitProject, FitState, set_parameter
from xstack.gui.common.workers import Job
from xstack.gui.fit.steps.fit import FitPage


@pytest.fixture
def page(qtbot: QtBot) -> FitPage:
    p = FitPage()
    qtbot.addWidget(p)
    return p


def capture(page: FitPage) -> dict[str, list]:  # type: ignore[type-arg]
    out: dict[str, list] = {"commands": [], "errors": [], "jobs": [], "fits": []}  # type: ignore[type-arg]
    page.commandRequested.connect(out["commands"].append)
    page.errorOccurred.connect(out["errors"].append)
    page.jobRequested.connect(out["jobs"].append)
    page.fitFinished.connect(out["fits"].append)
    return out


def execute(state: FitState, command: object) -> FitState:
    project = FitProject(state)
    project.execute(command)  # type: ignore[arg-type]
    return project.state


def global_stages(seed: int = 0, pop: int = 15) -> tuple[LmfitOptimizer, ...]:
    return (
        LmfitOptimizer(
            "differential_evolution",
            4000,
            (("seed", seed), ("popsize", pop), ("polish", False)),
        ),
        LmfitOptimizer("least_squares"),
    )


def test_weighting_radios(page: FitPage) -> None:
    out = capture(page)
    state = make_state()
    page.set_state(state)
    assert page.weighting_buttons["log"].isChecked()
    page.weighting_buttons["linear"].setChecked(True)
    assert len(out["commands"]) == 1
    new = execute(state, out["commands"][0])
    assert new.weighting == "linear"
    page.set_state(new)
    assert page.weighting_buttons["linear"].isChecked()
    assert len(out["commands"]) == 1


def test_chi2_disabled_without_uncertainties(page: FitPage) -> None:
    page.set_state(make_state(uncertainty=False))
    assert not page.weighting_buttons["chi2"].isEnabled()
    assert "uncertainties" in page.weighting_buttons["chi2"].toolTip()
    page.set_state(make_state(uncertainty=True))
    assert page.weighting_buttons["chi2"].isEnabled()


def test_chi2_selectable(page: FitPage) -> None:
    out = capture(page)
    page.set_state(make_state())
    page.weighting_buttons["chi2"].setChecked(True)
    assert execute(make_state(), out["commands"][0]).weighting == "chi2"


def test_default_state_is_local(page: FitPage) -> None:
    page.set_state(make_state())
    assert page.current_strategy() == "local"
    assert page.optimizers() == (LmfitOptimizer("least_squares"),)


def test_strategy_global(page: FitPage) -> None:
    out = capture(page)
    state = make_state()
    page.set_state(state)
    page.strategy_combo.setCurrentIndex(1)
    assert len(out["commands"]) == 1
    new = execute(state, out["commands"][0])
    assert new.optimizers == global_stages()
    page.set_state(new)
    assert page.current_strategy() == "global"
    page.strategy_combo.setCurrentIndex(0)
    assert execute(new, out["commands"][1]).optimizers == (LmfitOptimizer("least_squares"),)


def test_seed_and_population(page: FitPage) -> None:
    out = capture(page)
    state = make_state().evolve(optimizers=global_stages())
    page.set_state(state)
    page.seed.setValue(7)
    page.population.setValue(20)
    assert len(out["commands"]) == 2
    new = execute(state, out["commands"][-1])
    assert new.optimizers == global_stages(7, 20)
    page.set_state(new)
    assert (page.seed.value(), page.population.value()) == (7, 20)
    assert page.current_strategy() == "global"


def test_custom_edit(page: FitPage) -> None:
    out = capture(page)
    state = make_state()
    page.set_state(state)
    page.stage_table.cellWidget(1, 2).setCurrentText("nelder")
    assert page.current_strategy() == "custom"
    assert len(out["commands"]) == 1
    new = execute(state, out["commands"][0])
    assert new.optimizers == (LmfitOptimizer("nelder"),)
    page.stage_table.cellWidget(1, 3).setText("500")
    page.stage_table.cellWidget(1, 3).editingFinished.emit()
    assert execute(new, out["commands"][1]).optimizers == (LmfitOptimizer("nelder", 500),)


def test_custom_round_trip(page: FitPage) -> None:
    state = make_state().evolve(
        optimizers=(LmfitOptimizer("powell", 100), LmfitOptimizer("leastsq"))
    )
    page.set_state(state)
    assert page.current_strategy() == "custom"
    assert page.optimizers() == state.optimizers
    out = capture(page)
    page.set_state(state)
    assert out["commands"] == []


def test_all_stages_disabled_is_an_error(page: FitPage) -> None:
    out = capture(page)
    page.set_state(make_state())
    page.stage_table.cellWidget(1, 0).findChild(QCheckBox).setChecked(False)
    assert out["commands"] == []
    assert len(out["errors"]) == 1


def test_invalid_evaluation_limit(page: FitPage) -> None:
    out = capture(page)
    page.set_state(make_state())
    page.stage_table.cellWidget(1, 3).setText("abc")
    page.stage_table.cellWidget(1, 3).editingFinished.emit()
    assert out["commands"] == []
    assert len(out["errors"]) == 1


def test_run_error_paths(page: FitPage) -> None:
    out = capture(page)
    page.run()
    assert len(out["errors"]) == 1
    page.set_state(FitState())
    page.run()
    assert len(out["errors"]) == 2
    page.set_state(make_state().evolve(datasets=()))
    page.run()
    assert len(out["errors"]) == 3
    state = make_state()
    fixed = set_parameter("t", vary=False).transform(state)
    fixed = set_parameter("sigma", vary=False).transform(fixed)
    page.set_state(fixed)
    page.run()
    assert "free" in out["errors"][-1]
    assert out["jobs"] == []


def run_job(job: Job) -> object:
    result = job.function(lambda e: None, CancellationToken())
    job.on_finished(result)
    return result


def prepared() -> FitState:
    state = make_state()
    state = set_parameter("t", value=33.0).transform(state)
    return state.evolve(optimizers=(LmfitOptimizer("least_squares", 200),))


def test_run_stores_result(page: FitPage) -> None:
    out = capture(page)
    state = prepared()
    page.set_state(state)
    page.run()
    assert len(out["jobs"]) == 1
    job = out["jobs"][0]
    assert job.name == "Fit"
    result = run_job(job)
    assert len(out["commands"]) == 1
    assert out["fits"] == [result]
    new = execute(state, out["commands"][0])
    assert len(new.results) == 1
    assert new.results[0].result is result
    assert new.mirror is not None
    assert new.mirror.parameters["t"].value == result.parameters["t"].value
    page.set_state(new)
    text = page.last_label.text()
    assert "fit 1" in text
    assert "least_squares" in text


def test_run_local_uses_least_squares_without_changing_state(page: FitPage) -> None:
    out = capture(page)
    state = prepared().evolve(optimizers=(LmfitOptimizer("nelder", 50),))
    page.set_state(state)
    page.run_local()
    result = run_job(out["jobs"][0])
    assert result.method == "least_squares"  # type: ignore[attr-defined]
    assert page._state is state
    assert state.optimizers == (LmfitOptimizer("nelder", 50),)


def test_last_fit_empty(page: FitPage) -> None:
    page.set_state(make_state())
    assert page.last_label.text() == "No fit yet."
