"""Round-trip tests: simulate a measurement, add noise, fit, recover the parameters."""

import math
from collections.abc import Mapping

import numpy as np
import pytest

from xstack.core.materials import get
from xstack.core.measurement import MeasuredCurve
from xstack.core.optim import LmfitOptimizer
from xstack.core.parameters import ParameterValue, free
from xstack.core.structure import Mirror, Periodic, SingleLayer
from xstack.core.units import deg
from xstack.fit.dataset import Dataset
from xstack.fit.problem import FitProblem, Weighting
from xstack.fit.results import FitResult

CU_KA = 8048.0
MO_KA = 17479.0
TOLERANCE = 4.0  # allowed deviation in standard uncertainties


def multilayer(
    d: ParameterValue = 4.0,
    gamma: ParameterValue = 0.4,
    sigma_h: ParameterValue = 0.3,
    sigma_l: ParameterValue = 0.3,
) -> Mirror:
    coating = Periodic.create(
        get("W"), get("Si"), N=15, d=d, gamma=gamma, sigma_H=sigma_h, sigma_L=sigma_l
    )
    return Mirror(coating, get("Si"))


def angle_curve(lo: float, hi: float, points: int, energy: float) -> MeasuredCurve:
    x = np.linspace(deg(lo), deg(hi), points)
    return MeasuredCurve("angle", x, np.ones_like(x), energy=energy)


def measure(
    truth: Mirror, datasets: tuple[Dataset, ...], rel_noise: float, seed: int
) -> tuple[Dataset, ...]:
    """Return the datasets with simulated noisy reflectivity and uncertainties."""
    problem = FitProblem(truth, datasets)
    rng = np.random.default_rng(seed)
    measured = []
    for dataset in datasets:
        clean = problem.simulate(dataset)
        sigma = rel_noise * clean
        noisy = clean + rng.normal(0.0, sigma)
        curve = MeasuredCurve(
            dataset.curve.axis,
            dataset.curve.x,
            noisy,
            sigma,
            energy=dataset.curve.energy,
            angle=dataset.curve.angle,
            name=dataset.name,
        )
        measured.append(Dataset(dataset.name, curve, dataset.instrument))
    return tuple(measured)


def fit(problem: FitProblem, method: str = "least_squares") -> FitResult:
    result = FitResult.from_optimization(problem, LmfitOptimizer(method).run(problem))
    assert result.statistics is not None, result.note
    return result


def assert_recovered(result: FitResult, truth: Mapping[str, float]) -> None:
    stats = result.statistics
    assert stats is not None
    assert stats.determined
    for name, true in truth.items():
        value, error = result.parameters[name].value, stats.error(name)
        assert math.isfinite(error)
        assert error > 0.0
        assert abs(value - true) < TOLERANCE * error, (name, value, true, error)


@pytest.mark.parametrize("weighting", ["chi2", "log", "linear"])
def test_single_curve(weighting: Weighting) -> None:
    truth = multilayer(d=4.05, gamma=0.38, sigma_h=0.35)
    instrument = {"scale": 0.9, "background": 1e-6, "offset": deg(0.004), "resolution": deg(0.004)}
    (data,) = measure(
        truth, (Dataset.create("a", angle_curve(0.1, 2.5, 400, CU_KA), **instrument),), 0.03, 1
    )
    start = multilayer(
        d=free(4.0, 3.8, 4.2), gamma=free(0.4, 0.3, 0.5), sigma_h=free(0.3, 0.0, 0.8)
    )
    data = Dataset.create(
        "a",
        data.curve,
        scale=free(1.0, 0.5, 1.5),
        background=free(1e-6, 0.0, 1e-4),
        offset=free(0.0, -deg(0.01), deg(0.01)),
        resolution=deg(0.004),
    )
    result = fit(FitProblem(start, (data,), weighting=weighting))
    assert_recovered(
        result,
        {"d": 4.05, "gamma": 0.38, "sigma_H": 0.35, "a.scale": 0.9, "a.offset": deg(0.004)},
    )
    if weighting == "chi2":
        assert result.statistics is not None
        assert 0.8 < result.statistics.reduced_objective < 1.2


def test_fit_ranges_select_points() -> None:
    """Points outside the fit ranges are ignored, even if they disagree with the model."""
    truth = Mirror(SingleLayer.create(get("Pt"), t=32.0, sigma=0.4), get("Si"))
    (data,) = measure(truth, (Dataset.create("a", angle_curve(0.1, 3.0, 300, CU_KA)),), 0.02, 2)
    x, r, u = data.curve.arrays()
    spoiled = np.where((x > deg(1.0)) & (x < deg(1.3)), 50.0 * r, r)
    curve = MeasuredCurve("angle", x, spoiled, u, energy=CU_KA)
    ranges = [(deg(0.15), deg(1.0)), (deg(1.3), deg(2.0)), (deg(2.2), deg(3.0))]
    data = Dataset.create("a", curve, scale=free(1.0, 0.5, 1.5)).with_ranges(ranges)
    start = Mirror(
        SingleLayer.create(get("Pt"), t=free(31.0, 25.0, 40.0), sigma=free(0.3, 0.0, 1.0)),
        get("Si"),
    )
    result = fit(FitProblem(start, (data,), weighting="chi2"))
    assert_recovered(result, {"t": 32.0, "sigma": 0.4, "a.scale": 1.0})
    stats = result.statistics
    assert stats is not None
    assert stats.points == int(data.mask().sum()) < data.curve.size


def test_simultaneous_fit_with_shared_offset() -> None:
    """Two energies, one structure; the angle offset is shared, the scales are separate."""
    truth = multilayer(d=3.95, gamma=0.42)
    offset = deg(0.003)
    datasets = (
        Dataset.create("cu", angle_curve(0.1, 2.5, 300, CU_KA), scale=0.95, offset=offset),
        Dataset.create("mo", angle_curve(0.05, 1.2, 300, MO_KA), scale=0.85, offset=offset),
    )
    measured = measure(truth, datasets, 0.03, 3)
    cu = Dataset.create(
        "cu", measured[0].curve, scale=free(1.0, 0.5, 1.5), offset=free(0.0, -0.001, 0.001)
    )
    mo = Dataset.create("mo", measured[1].curve, scale=free(1.0, 0.5, 1.5))
    mo = mo.with_parameters(mo.parameters.replace("mo.offset", expr="cu.offset"))
    start = multilayer(d=free(4.0, 3.8, 4.2), gamma=free(0.4, 0.3, 0.5))
    problem = FitProblem(start, (cu, mo), weighting="chi2")
    assert "mo.offset" not in problem.parameters.free_names()
    result = fit(problem)
    assert_recovered(
        result, {"d": 3.95, "gamma": 0.42, "cu.scale": 0.95, "mo.scale": 0.85, "cu.offset": offset}
    )
    resolved = problem.with_parameters(result.parameters).resolve()
    assert resolved["mo.offset"] == pytest.approx(resolved["cu.offset"])


def test_energy_scan() -> None:
    truth = multilayer(d=4.1)
    energies = np.linspace(6000.0, 12000.0, 300)
    curve = MeasuredCurve("energy", energies, np.ones_like(energies), angle=deg(1.0))
    (data,) = measure(truth, (Dataset.create("e", curve, resolution=5.0, scale=0.8),), 0.02, 4)
    data = Dataset.create("e", data.curve, scale=free(1.0, 0.5, 1.5), resolution=5.0)
    start = multilayer(d=free(4.0, 3.8, 4.3))
    result = fit(FitProblem(start, (data,), weighting="chi2"))
    assert_recovered(result, {"d": 4.1, "e.scale": 0.8})


def test_uncertainties_scale_with_noise() -> None:
    """Uncertainties from the Jacobian scale linearly with the measurement noise."""
    truth = Mirror(SingleLayer.create(get("Pt"), t=30.0, sigma=0.5), get("Si"))
    errors = []
    for noise in (0.01, 0.04):
        (data,) = measure(
            truth, (Dataset.create("a", angle_curve(0.1, 2.0, 200, CU_KA)),), noise, 5
        )
        start = Mirror(
            SingleLayer.create(get("Pt"), t=free(29.0, 20.0, 40.0), sigma=0.5), get("Si")
        )
        result = fit(FitProblem(start, (data,), weighting="chi2"))
        assert result.statistics is not None
        errors.append(result.statistics.error("t"))
    assert errors[1] / errors[0] == pytest.approx(4.0, rel=0.1)
