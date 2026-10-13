"""Tests of the Trade-off workflow page, its plot and the integration in the main window."""

import math

import pytest
from pytestqt.qtbot import QtBot

from tests.gui.test_optimization_gui import small_state, wait_job
from xstack.gui.main_window import MainWindow
from xstack.gui.plots import MERIT_QUANTITY, ReferenceDesign, TradeoffPlot
from xstack.gui.steps import TradeoffPage
from xstack.manufacturing.process import DepositionProcess
from xstack.optim.tradeoff import TradeoffPoint, TradeoffResult
from xstack.parameters import Parameter, ParameterSet
from xstack.project.commands import set_mirror, set_process
from xstack.project.templates import new_state

pytestmark = pytest.mark.gui

HOUR = 3600.0
PROCESS = DepositionProcess({"W": 0.05, "Si": 0.1}, overhead_per_layer=5.0)


def point(
    limit: float,
    hours: float,
    merit: float,
    feasible: bool = True,
    replaced: bool = False,
    n: float = 20.0,
) -> TradeoffPoint:
    params = ParameterSet([Parameter("N", n, vary=True, min=10, max=40, integer=True)])
    return TradeoffPoint(
        limit,
        params,
        merit,
        merit,
        hours * HOUR,
        feasible,
        (("mean reflectivity", 1.0 - merit),),
        replaced,
    )


def sample_result() -> TradeoffResult:
    return TradeoffResult(
        (
            point(0.5 * HOUR, 1.0, 0.9, feasible=False),
            point(1 * HOUR, 1.0, 0.6),
            point(2 * HOUR, 2.0, 0.3),
            point(3 * HOUR, 2.0, 0.3, replaced=True),
            point(4 * HOUR, 4.0, 0.25),
            point(math.inf, 8.0, 0.24),
        )
    )


def make_window(window: MainWindow) -> MainWindow:
    window.controller.execute(set_mirror(small_state().mirror))
    window.optimize_page.seed.setText("1")
    window.optimize_page.iterations.setValue(1)
    window.optimize_page.population.setValue(3)
    window.optimize_page.coarse.setValue(10)
    window.optimize_page.scan_points.setValue(2)
    return window


def run_small_tradeoff(qtbot: QtBot, window: MainWindow) -> None:
    window.controller.execute(set_process(PROCESS))
    page = window.tradeoff_page
    page.explicit_radio.setChecked(True)
    page.min_hours.setValue(0.22)
    page.max_hours.setValue(0.4)
    page.count.setValue(2)
    page.run()
    wait_job(qtbot, window.job_runner)


def test_plot_marks_points_knee_and_references(qtbot: QtBot) -> None:
    plot = TradeoffPlot()
    qtbot.addWidget(plot)
    result = sample_result()
    plot.set_result(result, MERIT_QUANTITY, [ReferenceDesign("input design", 5 * HOUR, 0.5)])
    assert plot.feasible_xy == ([1.0, 2.0, 4.0, 8.0], [0.6, 0.3, 0.25, 0.24])
    assert plot.infeasible_xy == ([1.0], [0.9])
    assert plot.replaced_xy == ([2.0], [0.3])
    assert plot.reference_xy == ([5.0], [0.5])
    assert plot.knee_xy is not None
    assert "no limit" in plot.labels
    assert "knee" in plot.labels
    assert "input design" in plot.labels
    assert len(plot.gains) == len(result.feasible_points()) - 1


def test_plot_other_quantity_hides_references(qtbot: QtBot) -> None:
    plot = TradeoffPlot()
    qtbot.addWidget(plot)
    plot.set_result(
        sample_result(), "mean reflectivity", [ReferenceDesign("input design", 5 * HOUR, 0.5)]
    )
    assert plot.feasible_xy[1] == pytest.approx([0.4, 0.7, 0.75, 0.76])
    assert plot.reference_xy == ([], [])
    plot.set_result(None)
    assert plot.feasible_xy == ([], [])


def test_plot_click_selects_point(qtbot: QtBot) -> None:
    plot = TradeoffPlot()
    qtbot.addWidget(plot)
    plot.set_result(sample_result())
    with qtbot.waitSignal(plot.pointSelected) as blocker:
        plot._clicked(None, [type("P", (), {"data": lambda self: 2})()])
    assert blocker.args == [2]
    assert plot.selected == 2


def test_page_shows_stored_result_in_table(qtbot: QtBot) -> None:
    page = TradeoffPage()
    qtbot.addWidget(page)
    state = new_state("periodic").evolve(process=PROCESS, tradeoff=sample_result())
    page.set_state(state)
    table = page.table
    assert table.rowCount() == 6
    assert table.item(5, 0).text() == "no limit"
    assert "infeasible" in table.item(0, 5).text()
    assert "replaced" in table.item(3, 5).text()
    assert "knee" in table.item(2, 5).text()
    assert table.item(0, 4).text() == ""
    assert float(table.item(2, 4).text()) == pytest.approx(0.3)
    assert table.item(1, 6).text() == "N=20"
    assert page.quantity_combo.findData("mean reflectivity") >= 0
    assert table.isColumnHidden(3)
    page.quantity_combo.setCurrentIndex(page.quantity_combo.findData("mean reflectivity"))
    page._quantity_activated()
    assert not table.isColumnHidden(3)
    assert page.quantity() == "mean reflectivity"


def test_run_disabled_without_process(qtbot: QtBot) -> None:
    page = TradeoffPage()
    qtbot.addWidget(page)
    page.set_state(new_state("periodic"))
    assert not page.run_button.isEnabled()
    assert "deposition process" in page.message.text()
    page.set_state(new_state("periodic").evolve(process=PROCESS))
    assert page.run_button.isEnabled()
    assert page.message.text() == ""


def test_limits(qtbot: QtBot) -> None:
    page = TradeoffPage()
    qtbot.addWidget(page)
    assert page.limits() is None
    page.explicit_radio.setChecked(True)
    page.min_hours.setValue(1.0)
    page.max_hours.setValue(4.0)
    page.count.setValue(3)
    assert page.limits() == pytest.approx((1 * HOUR, 2 * HOUR, 4 * HOUR))
    page.min_hours.setValue(5.0)
    with pytest.raises(ValueError):
        page.limits()
    errors: list[str] = []
    page.errorOccurred.connect(errors.append)
    page.set_state(new_state("periodic").evolve(process=PROCESS))
    page.run()
    assert errors
    assert "Cannot run" in errors[0]


def test_window_has_tradeoff_step_and_tab(window: MainWindow) -> None:
    labels = [window.step_tabs.tabText(i) for i in range(window.step_tabs.count())]
    assert any(label.endswith("Trade-off") for label in labels)
    assert labels.index(next(t for t in labels if t.endswith("Trade-off"))) == (
        labels.index(next(t for t in labels if t.endswith("Optimize"))) + 1
    )
    tabs = [window.plot_tabs.tabText(i) for i in range(window.plot_tabs.count())]
    assert "Trade-off" in tabs
    assert not window.tradeoff_page.run_button.isEnabled()


def test_window_runs_tradeoff_and_stores_result(qtbot: QtBot, window: MainWindow) -> None:
    make_window(window)
    run_small_tradeoff(qtbot, window)
    result = window.controller.state.tradeoff
    assert result is not None
    assert len(result.points) == 2
    assert window.convergence_plot.tasks()
    page = window.tradeoff_page
    assert page.table.rowCount() == 2
    plot = window.tradeoff_plot
    shown = len(plot.feasible_xy[0]) + len(plot.infeasible_xy[0]) + len(plot.replaced_xy[0])
    assert shown == len([p for p in result.points if math.isfinite(p.coating_time)]) > 0
    assert plot.feasible_xy[0] or plot.replaced_xy[0]
    assert window.plot_tabs.currentWidget() is window.tradeoff_plot
    # the input design is shown for comparison
    qtbot.waitUntil(lambda: window.tradeoff_plot.reference_xy[0] != [], timeout=10000)
    assert "input design" in window.tradeoff_plot.labels
    window.controller.undo()
    assert window.controller.state.tradeoff is None
    assert window.tradeoff_page.table.rowCount() == 0
    assert window.tradeoff_plot.feasible_xy == ([], [])
    window.controller.undo_stack.setClean()


def test_quantity_combo_changes_plotted_data(qtbot: QtBot, window: MainWindow) -> None:
    make_window(window)
    run_small_tradeoff(qtbot, window)
    page = window.tradeoff_page
    merit_data = [
        list(v) for v in (window.tradeoff_plot.feasible_xy, window.tradeoff_plot.replaced_xy)
    ]
    assert page.quantity_combo.count() >= 2
    page.quantity_combo.setCurrentIndex(1)
    page._quantity_activated()
    other = [list(v) for v in (window.tradeoff_plot.feasible_xy, window.tradeoff_plot.replaced_xy)]
    assert merit_data[0] != []
    assert merit_data != other
    assert window.tradeoff_plot.reference_xy == ([], [])
    window.controller.undo_stack.setClean()


def test_selected_row_shows_overlay_and_stores_candidate(qtbot: QtBot, window: MainWindow) -> None:
    make_window(window)
    run_small_tradeoff(qtbot, window)
    page = window.tradeoff_page
    page.select_point(0)
    assert window.tradeoff_plot.selected == 0
    assert window.tradeoff_point is not None
    qtbot.waitUntil(lambda: bool(window.tradeoff_curves), timeout=10000)
    assert page.store_button.isEnabled()
    assert page.store_selected()
    candidates = window.controller.state.candidates
    assert len(candidates) == 1
    assert candidates[0].source == "tradeoff"
    assert candidates[0].name.startswith("trade-off ")
    assert window.controller.state.selected == candidates[0].name
    assert candidates[0].coating_time is not None
    window.controller.undo_stack.setClean()


def test_automatic_range_includes_unconstrained_design(qtbot: QtBot, window: MainWindow) -> None:
    make_window(window)
    window.controller.execute(set_process(PROCESS))
    page = window.tradeoff_page
    page.count.setValue(1)
    page.lowest.setValue(60.0)
    page.run()
    wait_job(qtbot, window.job_runner)
    result = window.controller.state.tradeoff
    assert result is not None
    assert [math.isinf(p.limit) for p in result.points] == [False, True]
    assert page.table.item(1, 0).text() == "no limit"
    window.controller.undo_stack.setClean()
