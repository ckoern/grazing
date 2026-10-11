"""Progress reporting and cancellation of long-running calculations."""

import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass

__all__ = ["CancellationToken", "EventSink", "ProgressEvent"]


@dataclass(frozen=True)
class ProgressEvent:
    """Progress of an optimization or scan.

    Attributes:
        task: Name of the running task, e.g. the stage name or the optimizer method.
        evaluations: Number of objective evaluations so far.
        total: Expected total number of evaluations, if known.
        best_objective: Best objective so far: the figure of merit plus constraint penalties.
        current_objective: Objective of the latest evaluation.
        best_values: Parameter values of the best design so far.
        elapsed: Elapsed time in s.
        best_merit: Figure of merit (without penalties) of the best design so far; ``nan`` if
            that design could not be built.
    """

    task: str
    evaluations: int
    total: int | None
    best_objective: float
    current_objective: float
    best_values: Mapping[str, float]
    elapsed: float
    best_merit: float = float("nan")


type EventSink = Callable[[ProgressEvent], None]
"""Callable that receives progress events."""


class CancellationToken:
    """Thread-safe flag to request cancellation of a running calculation.

    Examples:
        >>> token = CancellationToken()
        >>> token.cancelled
        False
        >>> token.cancel()
        >>> token.cancelled
        True
    """

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        """Request cancellation."""
        self._event.set()

    @property
    def cancelled(self) -> bool:
        """Whether cancellation has been requested."""
        return self._event.is_set()
