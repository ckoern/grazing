"""Tests for `grazing.gui.common.garbage`."""

import gc
import os
import subprocess
import sys
import textwrap
from collections.abc import Iterator

import pytest
from pytestqt.qtbot import QtBot

from grazing.gui.common.garbage import GarbageCollector

pytestmark = pytest.mark.gui


@pytest.fixture
def thresholds() -> Iterator[None]:
    saved = gc.get_threshold()
    yield
    gc.set_threshold(*saved)
    gc.enable()


@pytest.mark.usefixtures("thresholds")
def test_disables_automatic_collection_until_stopped(qtbot: QtBot) -> None:
    collector = GarbageCollector()
    assert not gc.isenabled()
    collector.stop()
    assert gc.isenabled()


@pytest.mark.usefixtures("thresholds")
def test_collects_unreachable_cycles_in_gui_thread(qtbot: QtBot) -> None:
    collector = GarbageCollector(interval_ms=10)
    gc.set_threshold(10, 10, 10)
    for _ in range(50):
        a: list[object] = []
        a.append(a)
    qtbot.waitUntil(lambda: gc.get_count()[0] < 10, timeout=2000)
    collector.stop()


@pytest.mark.usefixtures("thresholds")
def test_check_without_full_generation_collects_nothing(qtbot: QtBot) -> None:
    collector = GarbageCollector(interval_ms=60_000)
    gc.collect()
    gc.set_threshold(1_000_000, 10, 10)
    assert collector.check() == 0
    collector.stop()


SCRIPT = textwrap.dedent(
    """
    import threading, time
    from PySide6.QtWidgets import QApplication
    from grazing.core.structure.presets import StructureChoice, build_mirror
    from grazing.core.physics.models import ParrattModel
    from grazing.core.targets import EnergyScan
    from grazing.gui.common.wizard import ProjectWizard

    app = QApplication([])
    from grazing.gui.common.garbage import GarbageCollector

    collector = GarbageCollector(interval_ms=20)

    def drop_wizard():
        wizard = ProjectWizard("design project", ".grazing", "x (*.grazing)")
        wizard.show()
        app.processEvents()
        wizard.close()

    for _ in range(3):
        drop_wizard()
    app.processEvents()

    mirror = build_mirror(StructureChoice("periodic", ("W", "B4C")))
    target = EnergyScan(angle=0.0175, energy_min=7900.0, energy_max=8200.0, points=61)
    model = ParrattModel()
    done = []

    def work():
        for n in range(400):
            stack = mirror.stack({"N": 20 + n % 100})
            model.evaluate(stack, target)
        done.append(True)

    thread = threading.Thread(target=work)
    thread.start()
    while not done:
        app.processEvents()
        time.sleep(0.005)
    thread.join()
    print("ok")
    """
)


def test_worker_thread_never_collects_gui_objects() -> None:
    """Regression: a dropped wizard was collected in a worker thread and crashed the process.

    Without `GarbageCollector`, this script ends with a segmentation fault.
    """
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT],
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip().endswith("ok")
