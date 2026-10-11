"""Tests for the conversion of model objects to and from JSON data."""

import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import pytest

from xstack import materials
from xstack.manufacturing import (
    DepositionProcess,
    MaxCoatingTime,
    MaxLayers,
    MaxTotalThickness,
    MinLayerThickness,
)
from xstack.materials import Material
from xstack.merit import (
    CallableMerit,
    EvaluationContext,
    Flatness,
    MeanReflectivity,
    PeakReflectivity,
    ProfileDeviation,
    SoftMinReflectivity,
    TimePenalized,
    WeightedSum,
)
from xstack.optim import BraggOutline, DesignProblem, GridScan, LmfitOptimizer, Pipeline, Stage
from xstack.parameters import Parameter, ParameterSet, free
from xstack.physics.models import KinematicModel, ParrattModel
from xstack.project import (
    Candidate,
    ProjectState,
    SerializationError,
    from_data,
    register_serializable,
    to_data,
)
from xstack.structure import (
    Composite,
    DepthGraded,
    Layer,
    Mirror,
    NLayer,
    Periodic,
    SingleLayer,
    Stack,
    register_coating,
)
from xstack.structure.coating import Coating, coatings
from xstack.targets import AngleScan, EnergyScan, Profile, TargetSet
from xstack.units import deg, kev

W = materials.get("W")
SI = materials.get("Si")
B4C = materials.get("B4C")
C = materials.get("C")


def roundtrip(obj: Any) -> Any:
    """Convert to data, through strict JSON text and back."""
    text = json.dumps(to_data(obj), allow_nan=False)
    return from_data(json.loads(text))


def module_merit(ctx: EvaluationContext) -> float:
    return 0.5


@dataclass(frozen=True)
class Point:
    x: float
    y: tuple[int, ...] = ()


@dataclass(frozen=True)
class OtherPoint:
    x: float


class NotADataclass:
    pass


@dataclass(frozen=True)
class DummyCoating(Coating):
    type_name: ClassVar[str] = "m8_dummy_coating"
    roles: ClassVar[tuple[str, ...]] = ("M",)

    def _build(self, values: dict[str, float]) -> tuple[Layer, ...]:
        return (Layer(self.material("M"), values["t"]),)


@pytest.fixture
def dummy_coating() -> Iterator[type[DummyCoating]]:
    register_coating(DummyCoating)
    try:
        yield DummyCoating
    finally:
        coatings.unregister(DummyCoating.type_name)


class TestPrimitives:
    @pytest.mark.parametrize("value", [None, True, 3, 2.5, "text", [1, 2], (1, "a", None)])
    def test_plain_values(self, value: Any) -> None:
        expected = tuple(value) if isinstance(value, list) else value
        assert roundtrip(value) == expected

    def test_non_finite_floats(self) -> None:
        assert roundtrip(float("inf")) == float("inf")
        assert roundtrip(-float("inf")) == -float("inf")
        assert np.isnan(roundtrip(float("nan")))
        json.dumps(to_data(float("inf")), allow_nan=False)

    def test_numpy_scalars(self) -> None:
        assert to_data(np.float64(1.5)) == 1.5
        assert to_data(np.int64(3)) == 3
        assert to_data(np.bool_(True)) is True
        assert isinstance(to_data(np.int64(3)), int)

    def test_ndarray(self) -> None:
        array = np.array([[1.0, np.inf], [-np.inf, 2.5]])
        result = roundtrip(array)
        assert isinstance(result, np.ndarray)
        assert result.dtype == array.dtype
        np.testing.assert_array_equal(result, array)

    def test_integer_ndarray(self) -> None:
        array = np.arange(6).reshape(2, 3)
        result = roundtrip(array)
        assert result.dtype == array.dtype
        np.testing.assert_array_equal(result, array)

    def test_dict(self) -> None:
        value = {"a": 1, "b": {"c": 2.5}, 3: "x"}
        assert roundtrip(value) == value

    def test_dict_with_tuple_value(self) -> None:
        assert roundtrip({"a": [1, 2]}) == {"a": (1, 2)}


class TestMaterialAndParameters:
    def test_material_with_provider(self) -> None:
        material = Material("B4C (cxro)", "B4C", 2.52, provider="cxro")
        data = to_data(material)
        assert data["fields"]["provider"] == "cxro"
        assert roundtrip(material) == material

    def test_library_material(self) -> None:
        assert roundtrip(W) == W

    def test_parameter_with_infinite_bounds(self) -> None:
        p = Parameter("x", 1.0, vary=True)
        assert p.max == float("inf")
        json.dumps(to_data(p), allow_nan=False)
        result = roundtrip(p)
        assert result == p
        assert result.min == -float("inf")

    def test_parameter_full(self) -> None:
        p = Parameter("N", 20, vary=True, min=1, max=50, integer=True, unit="", description="n")
        assert roundtrip(p) == p

    def test_parameter_set(self) -> None:
        ps = ParameterSet(
            [Parameter("a", 1.0, vary=True, min=0.0), Parameter("b", 2.0, expr="a * 2")]
        )
        result = roundtrip(ps)
        assert isinstance(result, ParameterSet)
        assert result == ps

    def test_layer_and_stack(self) -> None:
        layers = [Layer(W, 2.0, 0.3, "a"), Layer(SI, 3.0)]
        stack = Stack(layers, SI, 0.4)
        assert roundtrip(layers[0]) == layers[0]
        assert roundtrip(stack) == stack


class TestCoatings:
    def test_single_layer(self) -> None:
        c = SingleLayer.create(W, t=free(5.0, 1.0, 10.0), sigma=0.3, name="top")
        assert roundtrip(c) == c

    def test_n_layer(self) -> None:
        c = NLayer.create([W, SI, C], [free(1.0, 0.5, 2.0), 2.0, 3.0])
        assert roundtrip(c) == c

    def test_periodic_with_interlayers(self) -> None:
        c = Periodic.create(
            W,
            SI,
            N=free(20, 10, 40),
            d=free(4.0, 3.0, 6.0),
            gamma=0.4,
            sigma_H=0.3,
            interlayers={"HonL": B4C, "LonH": C},
            t_HonL=0.5,
            t_LonH=0.4,
            top="L",
        )
        result = roundtrip(c)
        assert result == c
        assert type(result) is Periodic

    def test_depth_graded_cap_linear(self) -> None:
        c = DepthGraded.create(
            W, SI, N=40, law="linear", cap=True, d_min=free(2.0, 1.0, 3.0), d_max=6.0
        )
        result = roundtrip(c)
        assert result == c
        assert result.law == "linear"
        assert result.cap is True

    def test_depth_graded_power_law(self) -> None:
        c = DepthGraded.create(W, SI, N=40, d_min=2.0, d_max=6.0, c=0.3)
        assert roundtrip(c) == c

    def test_composite_nested(self) -> None:
        inner = Composite.create(
            [
                SingleLayer.create(B4C, t=2.0, name="cap"),
                Periodic.create(W, SI, N=free(5, 1, 10), name="ml"),
            ],
            name="inner",
        )
        outer = Composite.create([inner, SingleLayer.create(C, t=1.0, name="base")], name="outer")
        result = roundtrip(outer)
        assert result == outer
        assert result.layers() == outer.layers()

    def test_mirror(self) -> None:
        mirror = Mirror(Periodic.create(W, SI), SI, substrate_roughness=0.4, ambient=B4C)
        assert roundtrip(mirror) == mirror

    def test_coating_tag(self) -> None:
        assert to_data(Periodic.create(W, SI))["type"] == "coating.periodic"

    def test_registered_custom_coating(self, dummy_coating: type[DummyCoating]) -> None:
        params = ParameterSet([Parameter("t", 2.0, vary=True, min=0.0)])
        coating = dummy_coating("dummy", (("M", W),), params)
        data = to_data(coating)
        assert data["type"] == "coating.m8_dummy_coating"
        assert roundtrip(coating) == coating


class TestTargets:
    def test_profile(self) -> None:
        profile = Profile([1.0, 2.0, 3.0], [0.1, 0.5, 0.2])
        assert roundtrip(profile) == profile

    def test_energy_scan_full(self) -> None:
        t = EnergyScan(
            angle=deg(0.3),
            energy_min=kev(10),
            energy_max=kev(30),
            weight=2.0,
            points=50,
            polarization="s",
            profile=Profile.constant(0.5),
            spectrum=Profile([kev(10), kev(30)], [1.0, 2.0]),
            resolution=12.0,
            name="band",
        )
        assert roundtrip(t) == t

    def test_angle_scan_full(self) -> None:
        t = AngleScan(
            energy=kev(8),
            angle_min=deg(0.1),
            angle_max=deg(0.5),
            profile=Profile([deg(0.1), deg(0.5)], [0.2, 0.8]),
            spectrum=Profile.constant(1.0),
            resolution=1e-4,
        )
        assert roundtrip(t) == t

    @pytest.mark.parametrize("combine", ["average_reflectivity", "average_merit"])
    def test_target_set(self, combine: str) -> None:
        a = EnergyScan(angle=deg(0.3), energy_min=kev(10), energy_max=kev(30))
        b = EnergyScan(angle=deg(0.5), energy_min=kev(10), energy_max=kev(30), weight=2.0)
        ts = TargetSet([a, b], combine=combine)  # type: ignore[arg-type]
        result = roundtrip(ts)
        assert result == ts
        assert result.combine == combine


class TestManufacturing:
    def test_process_with_material_overheads(self) -> None:
        process = DepositionProcess(
            {"W": 0.1, "Si": 0.2},
            overhead_per_layer=5.0,
            overhead_per_material={"W": 2.0, "Si": 1.0},
            fixed_overhead=60.0,
        )
        result = roundtrip(process)
        assert result == process
        assert result.rate("W") == 0.1

    @pytest.mark.parametrize(
        "constraint",
        [
            MinLayerThickness(0.5, ("W",)),
            MaxTotalThickness(500.0),
            MaxLayers(100),
            MaxCoatingTime(3600.0),
            MinLayerThickness(0.5, severity="soft"),
            MaxTotalThickness(500.0, severity="soft"),
            MaxLayers(100, severity="soft"),
            MaxCoatingTime(3600.0, severity="soft"),
        ],
    )
    def test_constraints(self, constraint: Any) -> None:
        result = roundtrip(constraint)
        assert result == constraint
        assert result.severity == constraint.severity


class TestMerits:
    @pytest.mark.parametrize(
        "merit",
        [
            MeanReflectivity(),
            SoftMinReflectivity(softness=0.05),
            ProfileDeviation(mode="below"),
            Flatness(),
            PeakReflectivity(),
            WeightedSum(((MeanReflectivity(), 1.0), (Flatness(), 0.5))),
            TimePenalized(MeanReflectivity(), weight=0.2, reference_time=100.0),
            TimePenalized(
                WeightedSum(((PeakReflectivity(), 2.0), (SoftMinReflectivity(), 1.0))),
                weight=0.3,
            ),
            WeightedSum(((TimePenalized(Flatness(), 0.1, 50.0), 1.0),)),
        ],
    )
    def test_merits(self, merit: Any) -> None:
        assert roundtrip(merit) == merit

    def test_callable_merit_by_reference(self) -> None:
        merit = CallableMerit(module_merit, "mine")
        data = to_data(merit)
        assert data["fields"]["function"] == {
            "type": "callable",
            "ref": f"{__name__}:module_merit",
        }
        result = roundtrip(merit)
        assert result.function is module_merit
        assert result == merit

    def test_lambda_fails(self) -> None:
        with pytest.raises(SerializationError):
            to_data(CallableMerit(lambda ctx: 0.0))

    def test_local_function_fails(self) -> None:
        def local(ctx: EvaluationContext) -> float:
            return 0.0

        with pytest.raises(SerializationError):
            to_data(CallableMerit(local))

    def test_unresolvable_callable_reference(self) -> None:
        with pytest.raises(SerializationError):
            from_data({"type": "callable", "ref": "no_such_module_xyz:f"})
        with pytest.raises(SerializationError):
            from_data({"type": "callable", "ref": f"{__name__}:missing"})
        with pytest.raises(SerializationError):
            from_data({"type": "callable", "ref": f"{__name__}:W"})


class TestModelsAndOptimizers:
    @pytest.mark.parametrize(
        "model",
        [ParrattModel(), KinematicModel(refraction=False), KinematicModel(interface_model="x")],
    )
    def test_models(self, model: Any) -> None:
        assert roundtrip(model) == model

    def test_lmfit_with_options(self) -> None:
        opt = LmfitOptimizer("differential_evolution", 500, (("popsize", 20), ("seed", 1)))
        result = roundtrip(opt)
        assert result == opt
        assert result.options == (("popsize", 20), ("seed", 1))

    def test_grid_scan_with_values(self) -> None:
        opt = GridScan(points=5, values=(("gamma", (0.3, 0.4, 0.5)), ("N", (10.0, 20.0))))
        assert roundtrip(opt) == opt

    def test_bragg_outline(self) -> None:
        opt = BraggOutline(refraction=True, margin=0.1)
        assert roundtrip(opt) == opt

    def test_stage(self) -> None:
        stage = Stage(
            "refine",
            LmfitOptimizer("nelder", 30),
            model=KinematicModel(),
            points=20,
            merit=Flatness(),
            free=("d", "gamma"),
        )
        assert roundtrip(stage) == stage

    def test_stage_defaults(self) -> None:
        stage = Stage("outline", BraggOutline())
        assert roundtrip(stage) == stage


def make_problem() -> DesignProblem:
    mirror = Mirror(
        Periodic.create(W, SI, N=20, d=free(4.0, 3.0, 5.0), gamma=free(0.4, 0.1, 0.9)), SI
    )
    targets = TargetSet(
        EnergyScan(angle=deg(0.5), energy_min=kev(10), energy_max=kev(12), points=5)
    )
    return DesignProblem(
        mirror,
        targets,
        MeanReflectivity(),
        (MaxTotalThickness(1000.0), MinLayerThickness(0.5, severity="soft")),
        DepositionProcess({"W": 0.1, "Si": 0.2}, overhead_per_material={"W": 1.0}),
        KinematicModel(),
        points=10,
    )


class TestPipelineAndProblem:
    def test_pipeline_default(self) -> None:
        problem = make_problem()
        pipeline = Pipeline.default(problem, seed=3)
        result = roundtrip(pipeline)
        assert result == pipeline
        assert len(result.stages) == len(pipeline.stages)

    def test_design_problem(self) -> None:
        problem = make_problem()
        data = to_data(problem)
        assert "library" not in data["fields"]
        assert roundtrip(problem) == problem

    def test_candidate_and_state(self) -> None:
        problem = make_problem()
        candidate = Candidate("a", problem.mirror, "manual", "2026-01-01T00:00:00+00:00", 0.1, 0.2)
        state = ProjectState(
            "demo",
            problem.mirror,
            problem.targets,
            problem.merit,
            problem.constraints,
            problem.process,
            problem.model,
            Pipeline.default(problem),
            (candidate,),
            "some notes",
        )
        assert roundtrip(candidate) == candidate
        assert roundtrip(state) == state

    def test_default_state(self) -> None:
        assert roundtrip(ProjectState()) == ProjectState()


class TestErrors:
    def test_unknown_tag(self) -> None:
        with pytest.raises(SerializationError):
            from_data({"type": "NoSuchType", "fields": {}})

    def test_missing_tag(self) -> None:
        with pytest.raises(SerializationError):
            from_data({"fields": {}})

    def test_unexpected_data(self) -> None:
        with pytest.raises(SerializationError):
            from_data(object())

    def test_bad_fields(self) -> None:
        with pytest.raises(SerializationError):
            from_data({"type": "Layer", "fields": {"nonsense": 1}})

    def test_invalid_values(self) -> None:
        with pytest.raises(SerializationError):
            from_data({"type": "Layer", "fields": {"material": to_data(W), "thickness": -1.0}})

    def test_non_serializable_object(self) -> None:
        with pytest.raises(SerializationError):
            to_data(object())

    def test_non_serializable_nested(self) -> None:
        with pytest.raises(SerializationError):
            to_data({"a": [1, object()]})

    def test_unregistered_dataclass(self) -> None:
        with pytest.raises(SerializationError):
            to_data(Point(1.0))

    def test_is_value_error(self) -> None:
        assert issubclass(SerializationError, ValueError)


class TestRegisterSerializable:
    def test_custom_dataclass(self) -> None:
        register_serializable("test_m8.Point", Point)
        point = Point(1.5, (1, 2, 3))
        assert to_data(point)["type"] == "test_m8.Point"
        assert roundtrip(point) == point
        assert roundtrip({"p": point}) == {"p": point}

    def test_idempotent(self) -> None:
        register_serializable("test_m8.Point", Point)
        register_serializable("test_m8.Point", Point)

    def test_conflicting_tag(self) -> None:
        register_serializable("test_m8.Point", Point)
        with pytest.raises(ValueError):
            register_serializable("test_m8.Point", OtherPoint)

    def test_conflicting_class(self) -> None:
        register_serializable("test_m8.Point", Point)
        with pytest.raises(ValueError):
            register_serializable("test_m8.Other", Point)

    def test_builtin_tag_conflict(self) -> None:
        with pytest.raises(ValueError):
            register_serializable("Layer", Point)

    def test_non_dataclass(self) -> None:
        with pytest.raises(ValueError):
            register_serializable("test_m8.NotADataclass", NotADataclass)


def test_unregistered_coating_is_not_serializable() -> None:
    from xstack.structure.coating import register_coating

    @dataclass(frozen=True)
    class Temporary(SingleLayer):
        type_name: ClassVar[str] = "temporary_test_coating"

    register_coating(Temporary)
    si = materials.get("Si")
    coating = Temporary("layer", (("M", si),), SingleLayer.create(si).parameters)
    try:
        assert to_data(coating)["type"] == "coating.temporary_test_coating"
    finally:
        coatings.unregister("temporary_test_coating")
    with pytest.raises(SerializationError):
        to_data(coating)
