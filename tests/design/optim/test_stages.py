"""Tests for design stages and the analytic Bragg outline."""

import math

import pytest

from tests.design.optim.test_problem import ENERGY, make_problem
from xstack.core.materials import default_library, get
from xstack.core.parameters import free
from xstack.core.physics import bragg
from xstack.core.physics.models import KinematicModel, ParrattModel
from xstack.core.structure import DepthGraded, Mirror, Periodic, SingleLayer
from xstack.core.targets import AngleScan, EnergyScan, TargetSet
from xstack.core.units import deg, energy_to_wavelength, kev
from xstack.design.merit import MeanReflectivity, PeakReflectivity
from xstack.design.optim import CancellationToken, DesignProblem, ProgressEvent
from xstack.design.optim.optimizers import LmfitOptimizer
from xstack.design.optim.stages import BraggOutline, Stage

W = get("W")
SI = get("Si")


def periodic_problem(angle: float = deg(1.0), n: object = None, **kwargs: object) -> DesignProblem:
    """W/Si problem with free d, gamma and N for a narrow band around 8048 eV."""
    coating = Periodic.create(
        W,
        SI,
        N=free(30, 10, 150) if n is None else n,  # type: ignore[arg-type]
        d=free(4.0, 2.0, 12.0),
        gamma=free(0.4, 0.1, 0.8),
    )
    targets = TargetSet(
        EnergyScan(angle=angle, energy_min=ENERGY - 80, energy_max=ENERGY + 80, points=5)
    )
    return DesignProblem(Mirror(coating, SI), targets, MeanReflectivity(), **kwargs)  # type: ignore[arg-type]


def graded_problem(targets: TargetSet) -> DesignProblem:
    coating = DepthGraded.create(
        W,
        SI,
        N=20,
        d_min=free(2.0, 0.5, 20.0),
        d_max=free(6.0, 0.5, 60.0),
        gamma_top=free(0.4, 0.1, 0.8),
        gamma_bottom=free(0.4, 0.1, 0.8),
    )
    return DesignProblem(Mirror(coating, SI), targets, MeanReflectivity())


def wavelength(energy: float) -> float:
    return float(energy_to_wavelength(energy))


class TestStageProblem:
    def test_unchanged_when_nothing_overridden(self) -> None:
        problem = make_problem()
        stage = Stage("s", LmfitOptimizer("leastsq"))
        assert stage.problem_for(problem) is problem

    def test_overrides(self) -> None:
        problem = make_problem()
        model = KinematicModel()
        merit = PeakReflectivity()
        stage = Stage("s", LmfitOptimizer(), model=model, points=7, merit=merit)
        variant = stage.problem_for(problem)
        assert variant.model is model
        assert variant.points == 7
        assert variant.merit is merit
        assert isinstance(problem.model, ParrattModel)
        assert problem.points is None
        assert variant.targets == problem.targets

    def test_partial_override(self) -> None:
        problem = make_problem()
        variant = Stage("s", LmfitOptimizer(), points=3).problem_for(problem)
        assert variant.points == 3
        assert variant.model is problem.model
        assert variant.merit is problem.merit


class TestStageRun:
    def test_free_restricts_and_restores_flags(self) -> None:
        problem = make_problem()
        stage = Stage("only_d", LmfitOptimizer("leastsq", 40), free=("d",))
        result = stage.run(problem)
        start = problem.parameters
        assert result.parameters["gamma"].value == start["gamma"].value
        assert result.parameters["d"].value != start["d"].value
        for name in start:
            assert result.parameters[name].vary == start[name].vary
        assert result.parameters.free_names() == start.free_names()

    def test_free_none_varies_all_free(self) -> None:
        problem = make_problem()
        result = Stage("all", LmfitOptimizer("leastsq", 60)).run(problem)
        assert result.parameters["gamma"].value != problem.parameters["gamma"].value
        assert result.parameters["d"].value != problem.parameters["d"].value

    def test_events_and_method_prefixed(self) -> None:
        problem = make_problem()
        events: list[ProgressEvent] = []
        result = Stage("fit", LmfitOptimizer("leastsq", 15)).run(problem, events=events.append)
        assert events
        assert all(e.task.startswith("fit: ") for e in events)
        assert result.method == "fit: leastsq"

    def test_start_parameters_used(self) -> None:
        problem = make_problem()
        start = problem.parameters.replace("gamma", value=0.3)
        stage = Stage("fit", LmfitOptimizer("leastsq", 30), free=("d",))
        result = stage.run(problem, start)
        assert result.parameters["gamma"].value == 0.3

    def test_cancelled_token_stops(self) -> None:
        problem = make_problem()
        token = CancellationToken()
        token.cancel()
        result = Stage("fit", LmfitOptimizer("leastsq")).run(problem, cancel=token)
        assert result.cancelled


class TestBraggOutlinePeriodic:
    def test_result_basics(self) -> None:
        problem = periodic_problem()
        result = BraggOutline().run(problem)
        assert result.evaluations == 1
        assert result.method == "bragg_outline"

    def test_period_and_bounds(self) -> None:
        problem = periodic_problem()
        result = BraggOutline(margin=0.2).run(problem)
        d_expected = float(bragg.bragg_period(deg(1.0), wavelength(ENERGY)))
        p = result.parameters["d"]
        assert p.value == pytest.approx(d_expected, rel=1e-3)
        assert p.min == pytest.approx(0.8 * d_expected, rel=1e-3)
        assert p.max == pytest.approx(1.2 * d_expected, rel=1e-3)

    def test_bounds_intersect_original(self) -> None:
        problem = periodic_problem()
        d_expected = float(bragg.bragg_period(deg(1.0), wavelength(ENERGY)))
        tight = problem.parameters.replace("d", min=d_expected * 0.95, max=d_expected * 1.5)
        result = BraggOutline(margin=0.2).run(problem, tight)
        p = result.parameters["d"]
        assert p.min == pytest.approx(0.95 * d_expected)
        assert p.max == pytest.approx(1.2 * d_expected, rel=1e-3)

    def test_gamma_is_optimal(self) -> None:
        problem = periodic_problem()
        result = BraggOutline().run(problem)
        db_h = default_library().delta_beta(W, ENERGY)
        db_l = default_library().delta_beta(SI, ENERGY)
        expected = bragg.optimal_gamma(float(db_h.imag), float(db_l.imag))
        assert result.parameters["gamma"].value == pytest.approx(expected, rel=1e-3)

    def test_n_narrowed_and_integer(self) -> None:
        problem = periodic_problem()
        result = BraggOutline().run(problem)
        n = result.parameters["N"]
        original = problem.parameters["N"]
        assert n.value == round(n.value)
        assert n.min == round(n.min)
        assert n.max == round(n.max)
        assert original.min <= n.min <= n.value <= n.max <= original.max
        assert (n.min, n.max) != (original.min, original.max)
        assert n.vary

    def test_fixed_parameters_untouched(self) -> None:
        problem = periodic_problem(n=25)
        problem = problem.with_parameters(problem.parameters.replace("gamma", vary=False))
        result = BraggOutline().run(problem)
        assert result.parameters["N"] == problem.parameters["N"]
        assert result.parameters["gamma"] == problem.parameters["gamma"]
        assert result.parameters["d"].value != problem.parameters["d"].value

    def test_refraction_gives_larger_period(self) -> None:
        # Refraction shifts the Bragg peak to a larger angle at fixed period, so a fixed angle
        # needs a larger period: d = mλ / (2·sqrt(sin²θ − 2δ̄)).
        problem = periodic_problem()
        plain = BraggOutline(refraction=False).run(problem).parameters["d"].value
        result = BraggOutline(refraction=True).run(problem)
        refracted = result.parameters["d"].value
        assert refracted > plain
        # The period is computed with the estimated Γ of the outline.
        gamma = result.parameters["gamma"].value
        delta = bragg.mean_delta(
            default_library().delta_beta(W, ENERGY).real,
            default_library().delta_beta(SI, ENERGY).real,
            gamma,
        )
        expected = float(bragg.bragg_period(deg(1.0), wavelength(ENERGY), 1, float(delta)))
        assert refracted == pytest.approx(expected, rel=0.02)

    def test_outline_improves_start(self) -> None:
        problem = periodic_problem()
        result = BraggOutline().run(problem)
        assert result.best.merit < problem.evaluate().merit

    def test_other_coatings_unchanged(self) -> None:
        coating = SingleLayer.create(SI, t=free(20.0, 1.0, 100.0))
        targets = TargetSet(
            EnergyScan(angle=deg(0.5), energy_min=ENERGY - 50, energy_max=ENERGY + 50, points=5)
        )
        problem = DesignProblem(Mirror(coating, SI), targets, MeanReflectivity())
        result = BraggOutline().run(problem)
        assert result.parameters == problem.parameters
        assert result.evaluations == 1

    def test_angle_dependent_band_centre(self) -> None:
        coating = Periodic.create(W, SI, N=30, d=free(4.0, 1.0, 20.0), gamma=0.4)
        targets = TargetSet(
            AngleScan(energy=ENERGY, angle_min=deg(0.8), angle_max=deg(1.2), points=5)
        )
        problem = DesignProblem(Mirror(coating, SI), targets, MeanReflectivity())
        d = BraggOutline().run(problem).parameters["d"].value
        assert d == pytest.approx(float(bragg.bragg_period(deg(1.0), wavelength(ENERGY))), rel=1e-3)


class TestBraggOutlineTargetSelection:
    low, high = ENERGY, 2 * ENERGY

    def two_targets(self) -> DesignProblem:
        problem = periodic_problem()
        targets = TargetSet(
            (
                EnergyScan(angle=deg(1.0), energy_min=self.low - 40, energy_max=self.low + 40),
                EnergyScan(angle=deg(1.0), energy_min=self.high - 40, energy_max=self.high + 40),
            ),
            "average_merit",
        )
        return DesignProblem(problem.mirror, targets, MeanReflectivity())

    def test_selected_target_alone(self) -> None:
        problem = self.two_targets()
        for index, energy in enumerate((self.low, self.high)):
            d = BraggOutline(target=index).run(problem).parameters["d"].value
            expected = float(bragg.bragg_period(deg(1.0), wavelength(energy)))
            assert d == pytest.approx(expected, rel=1e-3)

    def test_none_pools_targets(self) -> None:
        problem = self.two_targets()
        pooled = BraggOutline().run(problem).parameters["d"].value
        first = BraggOutline(target=0).run(problem).parameters["d"].value
        second = BraggOutline(target=1).run(problem).parameters["d"].value
        assert pooled != pytest.approx(first, rel=1e-3)
        assert pooled != pytest.approx(second, rel=1e-3)
        assert second < pooled <= first

    @pytest.mark.parametrize("index", [2, -1])
    def test_out_of_range_raises(self, index: int) -> None:
        with pytest.raises(ValueError, match="out of range"):
            BraggOutline(target=index).run(self.two_targets())


class TestBraggOutlineGraded:
    def test_energy_scan(self) -> None:
        theta = deg(0.2)
        targets = TargetSet(
            EnergyScan(angle=theta, energy_min=kev(15), energy_max=kev(70), points=5)
        )
        result = BraggOutline().run(graded_problem(targets))
        d_max = float(bragg.bragg_period(theta, wavelength(kev(15))))
        d_min = float(bragg.bragg_period(theta, wavelength(kev(70))))
        assert result.parameters["d_max"].value == pytest.approx(d_max, rel=1e-6)
        assert result.parameters["d_min"].value == pytest.approx(d_min, rel=1e-6)
        assert result.parameters["d_max"].min == pytest.approx(0.8 * d_max, rel=1e-6)
        assert result.parameters["d_min"].max == pytest.approx(1.2 * d_min, rel=1e-6)
        assert result.evaluations == 1

    def test_angle_scan(self) -> None:
        e = kev(30)
        targets = TargetSet(AngleScan(energy=e, angle_min=deg(0.2), angle_max=deg(0.6), points=5))
        result = BraggOutline().run(graded_problem(targets))
        d_max = float(bragg.bragg_period(deg(0.2), wavelength(e)))
        d_min = float(bragg.bragg_period(deg(0.6), wavelength(e)))
        assert result.parameters["d_max"].value == pytest.approx(d_max, rel=1e-6)
        assert result.parameters["d_min"].value == pytest.approx(d_min, rel=1e-6)

    def test_coupled_targets_use_extremes(self) -> None:
        angles = (deg(0.15), deg(0.2), deg(0.27))
        targets = TargetSet(
            [EnergyScan(angle=a, energy_min=kev(20), energy_max=kev(60), points=5) for a in angles]
        )
        result = BraggOutline().run(graded_problem(targets))
        d_max = float(bragg.bragg_period(angles[0], wavelength(kev(20))))
        d_min = float(bragg.bragg_period(angles[-1], wavelength(kev(60))))
        assert result.parameters["d_max"].value == pytest.approx(d_max, rel=1e-6)
        assert result.parameters["d_min"].value == pytest.approx(d_min, rel=1e-6)

    def test_gamma_set(self) -> None:
        targets = TargetSet(
            EnergyScan(angle=deg(0.2), energy_min=kev(15), energy_max=kev(70), points=5)
        )
        result = BraggOutline().run(graded_problem(targets))
        gammas = {result.parameters["gamma_top"].value, result.parameters["gamma_bottom"].value}
        assert len(gammas) == 1
        assert math.isfinite(gammas.pop())
        assert result.parameters["gamma_top"].value != 0.4
