"""Automatic plot views and detached plots in both main windows."""

from typing import Any

import pytest
from PySide6.QtCore import QSettings
from pytestqt.qtbot import QtBot

from grazing.gui.common.workers import Job
from grazing.gui.design.main_window import WORKFLOW_STEPS as DESIGN_STEPS
from grazing.gui.design.main_window import MainWindow as DesignWindow
from grazing.gui.fit.main_window import WORKFLOW_STEPS as FIT_STEPS
from grazing.gui.fit.main_window import MainWindow as FitWindow

pytestmark = pytest.mark.gui


@pytest.fixture(params=["design", "fit"])
def window(request: pytest.FixtureRequest, qtbot: QtBot, settings: QSettings) -> Any:
    cls = DesignWindow if request.param == "design" else FitWindow
    w = cls(settings=settings, confirm_close=False)
    qtbot.addWidget(w)
    w.show()
    return w


def steps(window: Any) -> tuple[str, ...]:
    return DESIGN_STEPS if isinstance(window, DesignWindow) else FIT_STEPS


def go(window: Any, step: str) -> None:
    window.step_tabs.setCurrentIndex(steps(window).index(step))


def test_step_changes_switch_plot(window: Any) -> None:
    if isinstance(window, DesignWindow):
        go(window, "Goal")
        go(window, "Coating")
        assert window.plot_tabs.currentWidget() is window.layer_view
        go(window, "Scan")
        assert window.plot_tabs.currentWidget() is window.design_map
        go(window, "Analyze")
        assert window.plot_tabs.currentWidget() is window.tolerance_view
        go(window, "Goal")
        assert window.plot_tabs.currentWidget() is window.reflectivity_view
    else:
        go(window, "Structure")
        assert window.plot_tabs.currentWidget() is window.layer_plot
        go(window, "Results")
        assert window.plot_tabs.currentWidget() is window.correlation_plot
        go(window, "Data")
        assert window.plot_tabs.currentWidget() is window.reflectivity_view


def test_every_step_has_a_view(window: Any) -> None:
    assert set(window.step_views) == set(steps(window))


def test_switching_can_be_turned_off(window: Any, settings: QSettings) -> None:
    window.auto_views_action.setChecked(False)
    assert str(settings.value("plots/auto_switch")).lower() == "false"
    window.plot_tabs.setCurrentWidget(window.convergence_plot)
    go(window, steps(window)[1])
    go(window, steps(window)[2])
    assert window.plot_tabs.currentWidget() is window.convergence_plot
    window.auto_views_action.setChecked(True)
    assert window.plot_tabs.currentWidget() is window.step_views[steps(window)[2]]


def test_setting_is_restored(qtbot: QtBot, settings: QSettings) -> None:
    settings.setValue("plots/auto_switch", False)
    w = FitWindow(settings=settings, confirm_close=False)
    qtbot.addWidget(w)
    assert not w.auto_views_action.isChecked()


def test_job_shows_convergence_and_returns(window: Any, qtbot: QtBot) -> None:
    go(window, steps(window)[0])
    done: list[object] = []
    job = Job("Test", lambda progress, cancel: 1, done.append)
    assert window.start_job(job)
    assert window.plot_tabs.currentWidget() is window.convergence_plot
    qtbot.waitUntil(lambda: not window.job_runner.running and bool(done), timeout=5000)
    assert window.plot_tabs.currentWidget() is window.step_views[steps(window)[0]]


def test_detached_view_stays_detached_on_step_change(window: Any) -> None:
    reflectivity = window.reflectivity_view
    detached = window.plot_tabs.detach(reflectivity)
    go(window, steps(window)[1])
    go(window, steps(window)[0])
    assert window.plot_tabs.is_detached(reflectivity)
    assert detached.isVisible()
    window.close()
    assert not window.plot_tabs.is_detached(reflectivity)


def test_detach_action(window: Any) -> None:
    window.plot_tabs.setCurrentWidget(window.convergence_plot)
    window.detach_action.trigger()
    assert window.plot_tabs.is_detached(window.convergence_plot)
    window.plot_tabs.attach_all()
