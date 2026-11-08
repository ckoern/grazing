"""Tests of the run name of the Optimize page and its use for the stored candidates."""

import re
from typing import Any

import pytest
from pytestqt.qtbot import QtBot

from grazing.core.optim.events import CancellationToken
from grazing.design.optim import Pipeline
from grazing.design.project.templates import new_state
from grazing.gui.common.workers import Job
from grazing.gui.design.main_window import MainWindow
from grazing.gui.design.steps import OptimizePage

pytestmark = pytest.mark.gui


@pytest.fixture
def page(qtbot: QtBot) -> OptimizePage:
    widget = OptimizePage()
    qtbot.addWidget(widget)
    widget.set_state(new_state("periodic"))
    return widget


def run_outline_job(page: OptimizePage) -> tuple[Any, ...]:
    """Run the outline stage on the page, call its job and finish it; return the emitted args."""
    jobs: list[Job] = []
    emitted: list[tuple[Any, ...]] = []
    page.jobRequested.connect(jobs.append)
    page.pipelineFinished.connect(lambda *args: emitted.append(args))
    page.run("outline")
    outcome = jobs[0].function(lambda _: None, CancellationToken())
    jobs[0].on_finished(outcome)
    assert len(emitted) == 1
    return emitted[0]


def test_run_name_is_emitted(page: OptimizePage) -> None:
    page.run_name.setText("my run")
    _problem, _result, candidate_name = run_outline_job(page)
    assert candidate_name == "my run"


def test_empty_run_name_is_none(page: OptimizePage) -> None:
    page.run_name.setText("   ")
    _problem, _result, candidate_name = run_outline_job(page)
    assert candidate_name is None


def stage_outcome(window: MainWindow) -> tuple[Any, Any]:
    """Run the outline stage directly and return the problem and result."""
    problem = window.controller.state.problem()
    outline = window.optimize_page.pipeline(problem).stages[0]
    result = Pipeline((outline,)).run(problem)
    return problem, result


def test_named_run_stores_candidates_and_selects_final(window: MainWindow) -> None:
    problem, result = stage_outcome(window)
    window._apply_pipeline_result(problem, result, "my run")
    state = window.controller.state
    names = [c.name for c in state.candidates]
    assert names == ["my run"]
    assert state.selected == "my run"
    window.controller.undo_stack.setClean()


def test_unnamed_run_uses_run_number(window: MainWindow) -> None:
    problem, result = stage_outcome(window)
    window._apply_pipeline_result(problem, result, None)
    state = window.controller.state
    names = [c.name for c in state.candidates]
    assert re.fullmatch(r"run \d+", names[-1])
    assert names[-1] == f"run {len(window.controller.project.journal.entries('run'))}"
    assert state.selected == names[-1]
    window.controller.undo_stack.setClean()
