"""State builders shared by the fit page tests."""

import numpy as np

from xstack.core.measurement import MeasuredCurve
from xstack.fit.dataset import Dataset
from xstack.fit.problem import FitProblem
from xstack.fit.project import FitProject, FitState
from xstack.fit.templates import new_state

ANGLES = np.linspace(0.002, 0.02, 40)


def make_curve(uncertainty: bool = True, energy: float = 8048.0) -> MeasuredCurve:
    state = new_state("single_layer")
    blank = Dataset.create("x", MeasuredCurve("angle", ANGLES, np.ones(ANGLES.size), energy=energy))
    assert state.mirror is not None
    sim = FitProblem(state.mirror, (blank,), weighting="linear").simulate(blank)
    return MeasuredCurve(
        "angle", ANGLES, sim, 0.05 * sim + 1e-7 if uncertainty else None, energy=energy
    )


def make_state(names: tuple[str, ...] = ("a",), uncertainty: bool = True) -> FitState:
    state = new_state("single_layer")
    datasets = tuple(Dataset.create(n, make_curve(uncertainty)) for n in names)
    return state.evolve(datasets=datasets)


def apply(state: FitState, command: object) -> FitState:
    """Apply a command to a state through a project (checks undo support)."""
    project = FitProject(state)
    project.execute(command)  # type: ignore[arg-type]
    return project.state
