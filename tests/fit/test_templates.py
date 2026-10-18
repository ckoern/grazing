"""Tests for `xstack.fit.templates`."""

import pytest

from xstack.core.materials import get
from xstack.core.structure import Mirror, SingleLayer
from xstack.core.structure.presets import StructureChoice, build_mirror
from xstack.fit.templates import ROUGHNESS_RANGE, TEMPLATES, for_fit, new_state


@pytest.mark.parametrize("name", list(TEMPLATES))
def test_templates_create_states(name: str) -> None:
    state = new_state(name)
    assert state.datasets == ()


def test_unknown_template() -> None:
    with pytest.raises(KeyError, match="unknown template"):
        new_state("nope")


def test_for_fit_fixes_integers_and_frees_roughness() -> None:

    mirror = for_fit(build_mirror(StructureChoice("periodic", ("W", "Si"))))
    params = mirror.parameters
    assert not params["N"].vary
    for name in ("sigma_H", "sigma_L"):
        assert params[name].vary
        assert (params[name].min, params[name].max) == ROUGHNESS_RANGE
        assert params[name].value == 0.3
    assert params["d"].vary


def test_for_fit_keeps_expressions_and_nonzero_roughness() -> None:

    coating = SingleLayer.create(get("Pt"), t=30.0, sigma=0.8)
    mirror = for_fit(Mirror(coating, get("Si")))
    assert mirror.parameters["sigma"].value == 0.8
    assert mirror.parameters["sigma"].vary
