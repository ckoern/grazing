"""Garbage collection in the GUI thread.

Python's cyclic garbage collector runs in whichever thread happens to allocate when a generation
is full. If it runs in a worker thread (e.g. an optimization) and frees a Qt object that lives in
the GUI thread, the application crashes. `GarbageCollector` prevents this: it switches automatic
collection off and collects from a timer in the GUI thread instead, with the collector's own
generation thresholds.
"""

import gc

from PySide6.QtCore import QObject, QTimer

__all__ = ["GarbageCollector"]


class GarbageCollector(QObject):
    """Runs Python's cyclic garbage collector only in the GUI thread.

    Automatic collection is disabled while the collector exists and enabled again when it is
    stopped.

    Args:
        interval_ms: Interval in ms at which the generation counts are checked.
        parent: Parent object; the collector lives as long as its parent.
    """

    def __init__(self, interval_ms: int = 500, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._timer = QTimer(self)
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self.check)
        gc.disable()
        self._timer.start()

    def check(self) -> int:
        """Collect the oldest generation whose count exceeds its threshold, if any.

        Returns:
            The number of unreachable objects found.
        """
        counts = gc.get_count()
        thresholds = gc.get_threshold()
        generation = -1
        for index, (count, threshold) in enumerate(zip(counts, thresholds, strict=False)):
            if threshold and count > threshold:
                generation = index
        return gc.collect(generation) if generation >= 0 else 0

    def stop(self) -> None:
        """Stop collecting from the timer and enable automatic collection again."""
        self._timer.stop()
        gc.enable()
