import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from xstack import units

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
