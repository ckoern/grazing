"""Density parameters of `grazing.core.structure.coating.Mirror`."""

import math

import numpy as np
import pytest

from grazing.core.materials import get
from grazing.core.parameters import ParameterSet
from grazing.core.physics.models import ParrattModel
from grazing.core.serialization import from_data, to_data
from grazing.core.structure import Mirror, Periodic, SingleLayer
from grazing.core.structure.coating import (
    DENSITY_RANGE,
    SUBSTRATE,
    CoatingBuildError,
    density_parameter_name,
)
from grazing.core.structure.coatings.composite import Composite
from grazing.core.targets import AngleScan
from grazing.core.units import deg

W, SI, PT, C = get("W"), get("Si"), get("Pt"), get("C")


def multilayer() -> Mirror:
    return Mirror(Periodic.create(W, SI, N=10, d=4.0, gamma=0.4), SI)


def test_no_density_parameters_by_default() -> None:
    mirror = multilayer()
    assert len(mirror.densities) == 0
    assert mirror.parameters is mirror.coating.parameters


def test_density_roles_include_substrate() -> None:
    roles = multilayer().density_roles()
    assert list(roles) == ["H", "L", SUBSTRATE]
    assert roles["H"] == W
    assert roles[SUBSTRATE] == SI


def test_with_density_parameters() -> None:
    mirror = multilayer().with_density_parameters(["L", "H"])
    assert list(mirror.densities) == ["rho_H", "rho_L"]
    rho = mirror.densities["rho_H"]
    assert rho.value == W.density
    assert not rho.vary
    assert (rho.min, rho.max) == pytest.approx(
        (DENSITY_RANGE[0] * W.density, DENSITY_RANGE[1] * W.density)
    )
    assert rho.unit == "g/cm³"
    assert list(mirror.parameters)[-2:] == ["rho_H", "rho_L"]


def test_all_roles_and_existing_parameters_kept() -> None:
    mirror = multilayer().with_density_parameters(["H"])
    mirror = mirror.with_parameter("rho_H", value=17.0)
    mirror = mirror.with_density_parameters()
    assert list(mirror.densities) == ["rho_H", "rho_L", "rho_substrate"]
    assert mirror.densities["rho_H"].value == 17.0


def test_unknown_role() -> None:
    with pytest.raises(KeyError, match="unknown material roles"):
        multilayer().with_density_parameters(["X"])


def test_unknown_density_parameter_rejected() -> None:
    mirror = multilayer().with_density_parameters(["H"])
    with pytest.raises(ValueError, match="belong to no material role"):
        Mirror(SingleLayer.create(PT), SI, densities=mirror.densities)


def test_stack_uses_density_parameters() -> None:
    mirror = multilayer().with_density_parameters(["H", SUBSTRATE])
    mirror = mirror.with_parameter("rho_H", value=17.0).with_parameter("rho_substrate", value=2.2)
    stack = mirror.stack()
    densities = {layer.material.name: layer.material.density for layer in stack.layers}
    assert densities == {"W": 17.0, "Si": SI.density}
    assert stack.substrate.density == 2.2
    assert mirror.coating.materials["H"].density == W.density


def test_values_override_density() -> None:
    mirror = multilayer().with_density_parameters(["H"])
    stack = mirror.stack({"rho_H": 15.0, "d": 4.2})
    assert stack.layers[0].material.density == 15.0
    assert sum(layer.thickness for layer in stack.layers[:2]) == pytest.approx(4.2)


def test_density_expression() -> None:
    mirror = multilayer().with_density_parameters(["H", "L"])
    mirror = mirror.with_parameter("rho_L", expr="rho_H / 10")
    stack = mirror.stack({"rho_H": 18.0})
    assert stack.layers[1].material.density == pytest.approx(1.8)


def test_unknown_value_names_rejected() -> None:
    with pytest.raises(KeyError, match="unknown parameters"):
        multilayer().stack({"rho_H": 15.0})


def test_non_positive_density() -> None:
    mirror = multilayer().with_density_parameters(["H"])
    with pytest.raises(CoatingBuildError, match="positive"):
        mirror.stack({"rho_H": 0.0})


def test_composite_roles_are_prefixed() -> None:
    cap = SingleLayer.create(PT, t=2.0, name="cap")
    ml = Periodic.create(W, C, N=5, d=3.0, name="ml")
    mirror = Mirror(Composite.create([cap, ml]), SI)
    assert list(mirror.density_roles()) == ["cap.M", "ml.H", "ml.L", SUBSTRATE]
    mirror = mirror.with_density_parameters(["ml.L"])
    assert list(mirror.densities) == ["ml.rho_L"]
    stack = mirror.stack({"ml.rho_L": 1.8})
    carbon = [layer.material.density for layer in stack.layers if layer.material.name == "C"]
    assert carbon == [1.8] * 5
    assert stack.layers[0].material.density == PT.density


def test_parameter_name() -> None:
    assert density_parameter_name("H") == "rho_H"
    assert density_parameter_name(SUBSTRATE) == "rho_substrate"
    assert density_parameter_name("cap.M") == "cap.rho_M"


def test_with_parameters_routes_densities() -> None:
    mirror = multilayer().with_density_parameters(["H"])
    params = mirror.parameters.with_values({"rho_H": 16.0, "d": 4.4})
    changed = mirror.with_parameters(params)
    assert changed.densities["rho_H"].value == 16.0
    assert changed.coating.parameters["d"].value == 4.4
    with pytest.raises(ValueError, match="must not change"):
        mirror.with_parameters(mirror.coating.parameters)


def test_with_coating_drops_densities_of_removed_roles() -> None:
    mirror = multilayer().with_density_parameters()
    single = mirror.with_coating(SingleLayer.create(PT))
    assert list(single.densities) == ["rho_substrate"]


def test_without_density_parameters() -> None:
    mirror = multilayer().with_density_parameters()
    assert list(mirror.without_density_parameters(["L"]).densities) == ["rho_H", "rho_substrate"]
    assert len(mirror.without_density_parameters().densities) == 0


def test_serialization_round_trip_and_old_data() -> None:
    mirror = multilayer().with_density_parameters(["H"]).with_parameter("rho_H", vary=True)
    assert from_data(to_data(mirror)) == mirror
    old = to_data(multilayer())
    del old["fields"]["densities"]
    assert from_data(old) == multilayer()
    assert from_data(old).densities == ParameterSet()


def test_critical_angle_follows_density() -> None:
    """Below the critical angle of the top layer the reflectivity is high; θc ∝ √ρ."""
    single = Mirror(SingleLayer.create(PT, t=50.0), SI).with_density_parameters(["M"])
    target = AngleScan(energy=8048.0, angle_min=deg(0.2), angle_max=deg(0.8), points=601)
    model = ParrattModel()

    def critical_angle(density: float) -> float:
        result = model.evaluate(single.stack({"rho_M": density}), target)
        return float(result.x[np.argmax(result.reflectivity < 0.5)])

    ratio = critical_angle(0.8 * PT.density) / critical_angle(PT.density)
    assert ratio == pytest.approx(math.sqrt(0.8), rel=0.02)
