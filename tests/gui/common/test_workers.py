"""Tests for `grazing.gui.common.workers`."""

import functools
import threading
import time
from typing import Any

import pytest
from PySide6.QtCore import QThreadPool
from pytestqt.qtbot import QtBot

from grazing.gui.common.workers import LatestRunner, Task

pytestmark = pytest.mark.gui


@pytest.fixture
def pool() -> QThreadPool:
    p = QThreadPool()
    p.setMaxThreadCount(2)
    return p


def test_task_finished_with_result(qtbot: QtBot, pool: QThreadPool) -> None:
    task = Task(lambda: 6 * 7)
    with qtbot.waitSignal(task.signals.finished, timeout=5000) as blocker:
        pool.start(task)
    assert blocker.args == [42]


def test_task_failed_with_message(qtbot: QtBot, pool: QThreadPool) -> None:
    def fail() -> None:
        raise ValueError("boom")

    task = Task(fail)
    with qtbot.waitSignal(task.signals.failed, timeout=5000) as blocker:
        pool.start(task)
    assert blocker.args == ["ValueError: boom"]


def test_task_progress_keyword(qtbot: QtBot, pool: QThreadPool) -> None:
    def work(progress: Any) -> str:
        progress(1)
        progress("half")
        return "done"

    task = Task(work, with_progress=True)
    seen: list[Any] = []
    task.signals.progress.connect(seen.append)
    with qtbot.waitSignal(task.signals.finished, timeout=5000) as blocker:
        pool.start(task)
    assert blocker.args == ["done"]
    qtbot.waitUntil(lambda: len(seen) == 2, timeout=5000)
    assert seen == [1, "half"]


def test_runner_only_last_quick_request(qtbot: QtBot, pool: QThreadPool) -> None:
    runner = LatestRunner(delay_ms=50, pool=pool)
    results: list[Any] = []
    runner.finished.connect(results.append)
    for i in range(5):
        runner.request(functools.partial(int, i))
    assert runner.is_busy()
    qtbot.waitUntil(lambda: bool(results), timeout=5000)
    qtbot.wait(150)
    assert results == [4]
    assert not runner.is_busy()


def test_runner_discards_outdated_result(qtbot: QtBot, pool: QThreadPool) -> None:
    runner = LatestRunner(delay_ms=1, pool=pool)
    results: list[Any] = []
    runner.finished.connect(results.append)
    started = threading.Event()
    release = threading.Event()

    def slow() -> str:
        started.set()
        release.wait(5)
        return "old"

    runner.request(slow)
    qtbot.waitUntil(started.is_set, timeout=5000)
    runner.request(lambda: "new")
    qtbot.waitUntil(lambda: bool(results), timeout=5000)
    release.set()
    qtbot.waitUntil(lambda: not runner.is_busy(), timeout=5000)
    qtbot.wait(50)
    assert results == ["new"]


def test_runner_failure_emits_failed(qtbot: QtBot, pool: QThreadPool) -> None:
    runner = LatestRunner(delay_ms=1, pool=pool)

    def fail() -> None:
        raise RuntimeError("bad")

    with qtbot.waitSignal(runner.failed, timeout=5000) as blocker:
        runner.request(fail)
    assert blocker.args == ["RuntimeError: bad"]


def test_runner_is_busy_while_running(qtbot: QtBot, pool: QThreadPool) -> None:
    runner = LatestRunner(delay_ms=1, pool=pool)
    assert not runner.is_busy()
    runner.request(lambda: time.sleep(0.05))
    assert runner.is_busy()
    with qtbot.waitSignal(runner.finished, timeout=5000):
        pass
    qtbot.waitUntil(lambda: not runner.is_busy(), timeout=5000)
