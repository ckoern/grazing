"""Background execution of calculations."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Signal

from xstack.design.optim.events import CancellationToken

__all__ = ["Job", "JobRunner", "LatestRunner", "Task", "TaskSignals"]


class TaskSignals(QObject):
    """Signals of a background task.

    Signals:
        finished(object): The task returned a result.
        failed(str): The task raised an exception; the message describes it.
        progress(object): Intermediate progress information sent by the task.
    """

    finished = Signal(object)
    failed = Signal(str)
    progress = Signal(object)


class Task(QRunnable):
    """Runs a callable in a thread pool and reports the outcome through signals.

    Args:
        function: Callable to run. If it accepts a keyword argument ``progress``, a function
            emitting the `TaskSignals.progress` signal is passed.
        with_progress: Pass the ``progress`` keyword argument.
    """

    def __init__(self, function: Callable[..., Any], with_progress: bool = False) -> None:
        super().__init__()
        self.signals = TaskSignals()
        self._function = function
        self._with_progress = with_progress

    def run(self) -> None:
        """Run the callable (called by the thread pool)."""
        try:
            if self._with_progress:
                result = self._function(progress=self.signals.progress.emit)
            else:
                result = self._function()
        except Exception as exc:  # any failure is reported to the user interface
            self.signals.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        self.signals.finished.emit(result)


class LatestRunner(QObject):
    """Runs the most recent request of a recurring calculation after a short delay.

    Requests arriving while a calculation is pending replace the pending one; results of
    outdated calculations are discarded. Used for live previews while the user edits.

    Signals:
        finished(object): Result of the most recent calculation.
        failed(str): The most recent calculation failed.

    Args:
        delay_ms: Delay before a request is started, in ms.
        pool: Thread pool. Defaults to the global pool.
        parent: Parent object.
    """

    finished = Signal(object)
    failed = Signal(str)

    def __init__(
        self, delay_ms: int = 150, pool: QThreadPool | None = None, parent: QObject | None = None
    ) -> None:
        super().__init__(parent)
        self._pool = pool or QThreadPool.globalInstance()
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(delay_ms)
        self._timer.timeout.connect(self._start)
        self._function: Callable[[], Any] | None = None
        self._generation = 0
        self._tasks: dict[TaskSignals, tuple[Task, int]] = {}

    def request(self, function: Callable[[], Any]) -> None:
        """Schedule a calculation, replacing any pending request.

        Args:
            function: Callable without arguments returning the result.
        """
        self._function = function
        self._timer.start()

    def _start(self) -> None:
        if self._function is None:
            return
        self._generation += 1
        task = Task(self._function)
        self._function = None
        task.setAutoDelete(False)
        self._tasks[task.signals] = (task, self._generation)
        # Bound-method slots are disconnected automatically if this runner is destroyed first.
        task.signals.finished.connect(self._task_finished)
        task.signals.failed.connect(self._task_failed)
        self._pool.start(task)

    def _take(self) -> bool:
        """Forget the task that sent the current signal; return whether it is the latest."""
        entry = self._tasks.pop(self.sender(), None)
        return entry is not None and entry[1] == self._generation

    def _task_finished(self, result: Any) -> None:
        if self._take():
            self.finished.emit(result)

    def _task_failed(self, message: str) -> None:
        if self._take():
            self.failed.emit(message)

    def is_busy(self) -> bool:
        """Whether a request is pending or running."""
        return self._timer.isActive() or bool(self._tasks)


@dataclass(frozen=True)
class Job:
    """A long-running, cancellable calculation requested by a page.

    Attributes:
        name: Name shown in the user interface.
        function: Callable ``(progress, cancel) -> result`` run in a worker thread; `progress`
            forwards `xstack.design.optim.ProgressEvent` objects, `cancel` is a
            `xstack.design.optim.CancellationToken`.
        on_finished: Called in the GUI thread with the result.
    """

    name: str
    function: Callable[[Callable[[Any], None], CancellationToken], Any]
    on_finished: Callable[[Any], None]


class JobRunner(QObject):
    """Runs one job at a time in the thread pool.

    Signals:
        started(str): A job started; its name.
        progress(object): Progress information of the running job.
        finished(str): The running job finished; its name. The job's ``on_finished`` has been
            called before.
        failed(str): The running job failed; the message describes why.

    Args:
        pool: Thread pool. Defaults to the global pool.
        parent: Parent object.
    """

    started = Signal(str)
    progress = Signal(object)
    finished = Signal(str)
    failed = Signal(str)

    def __init__(self, pool: QThreadPool | None = None, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._pool = pool or QThreadPool.globalInstance()
        self._job: Job | None = None
        self._task: Task | None = None
        self._cancel: CancellationToken | None = None

    @property
    def running(self) -> bool:
        """Whether a job is running."""
        return self._job is not None

    def start(self, job: Job) -> bool:
        """Start a job unless another one is running.

        Args:
            job: The job.

        Returns:
            Whether the job was started.
        """
        if self._job is not None:
            return False
        cancel = CancellationToken()
        task = Task(lambda progress: job.function(progress, cancel), with_progress=True)
        task.setAutoDelete(False)
        task.signals.progress.connect(self.progress)
        task.signals.finished.connect(self._finished)
        task.signals.failed.connect(self._failed)
        self._job, self._task, self._cancel = job, task, cancel
        self.started.emit(job.name)
        self._pool.start(task)
        return True

    def cancel(self) -> None:
        """Request cancellation of the running job; it finishes with its best result so far."""
        if self._cancel is not None:
            self._cancel.cancel()

    def _finished(self, result: Any) -> None:
        job = self._job
        self._job = self._task = self._cancel = None
        if job is None:
            return
        try:
            job.on_finished(result)
        except Exception as exc:  # errors while applying a result are reported, not raised
            self.failed.emit(f"{job.name}: {exc}")
            return
        self.finished.emit(job.name)

    def _failed(self, message: str) -> None:
        name = self._job.name if self._job is not None else "job"
        self._job = self._task = self._cancel = None
        self.failed.emit(f"{name}: {message}")
