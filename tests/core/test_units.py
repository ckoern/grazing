import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from grazing.core import units

positive = st.floats(min_value=1e-6, max_value=1e9, allow_nan=False, allow_infinity=False)

ROUND_TRIPS = [
    (units.kev, units.to_kev),
    (units.deg, units.to_deg),
    (units.mrad, units.to_mrad),
    (units.angstrom, units.to_angstrom),
    (units.um, units.to_um),
    (units.energy_to_wavelength, units.wavelength_to_energy),
]


def test_cu_kalpha_wavelength() -> None:
    assert units.energy_to_wavelength(8047.8) == pytest.approx(0.154060, rel=1e-5)


def test_hc_from_si_constants() -> None:
    h, c, e = 6.62607015e-34, 299792458.0, 1.602176634e-19
    assert pytest.approx(h * c / e * 1e9, rel=1e-15) == units.HC


@pytest.mark.parametrize(
    ("func", "value", "expected"),
    [
        (units.kev, 1.0, 1000.0),
        (units.deg, 180.0, np.pi),
        (units.mrad, 1.0, 1e-3),
        (units.angstrom, 10.0, 1.0),
        (units.um, 1.0, 1000.0),
    ],
)
def test_known_values(func, value, expected) -> None:
    assert func(value) == pytest.approx(expected)


@pytest.mark.parametrize(("forward", "backward"), ROUND_TRIPS)
@given(value=positive)
def test_round_trip(forward, backward, value: float) -> None:
    assert backward(forward(value)) == pytest.approx(value, rel=1e-12)


@pytest.mark.parametrize(("forward", "backward"), ROUND_TRIPS)
def test_arrays_preserve_shape(forward, backward) -> None:
    values = np.linspace(1.0, 10.0, 12).reshape(3, 4)
    result = forward(values)
    assert isinstance(result, np.ndarray)
    assert result.shape == values.shape
    np.testing.assert_allclose(backward(result), values, rtol=1e-12)


def test_angle_to_q_known_value() -> None:
    theta = units.deg(0.5)
    expected = 4.0 * np.pi * np.sin(theta) / units.energy_to_wavelength(8048.0)
    assert units.angle_to_q(theta, 8048.0) == pytest.approx(expected, rel=1e-12)
    assert units.q_to_angle(expected, 8048.0) == pytest.approx(theta, rel=1e-12)


@given(
    st.lists(st.floats(min_value=1e-4, max_value=np.pi / 2 - 1e-3), min_size=1, max_size=10),
    st.floats(min_value=100.0, max_value=1e5),
)
def test_q_angle_round_trip_on_arrays(thetas: list[float], energy: float) -> None:
    theta = np.array(thetas)
    q = units.angle_to_q(theta, energy)
    np.testing.assert_allclose(units.q_to_angle(q, energy), theta, rtol=1e-9)


def test_q_above_maximum_is_nan() -> None:
    q_max = 4.0 * np.pi / units.energy_to_wavelength(8048.0)
    result = units.q_to_angle(np.array([0.5 * q_max, 1.01 * q_max]), 8048.0)
    assert np.isfinite(result[0])
    assert np.isnan(result[1])
    assert units.q_to_angle(q_max, 8048.0) == pytest.approx(np.pi / 2)
