"""Tests for `xstack.fit.dataset`."""

import numpy as np
import pytest

from xstack.core.measurement import MeasuredCurve
from xstack.core.parameters import free
from xstack.core.serialization import from_data, to_data
from xstack.fit.dataset import Dataset
from xstack.fit.instrument import Instrument

X = [0.01, 0.02, 0.03, 0.04, 0.05]


def make_curve(axis: str = "angle") -> MeasuredCurve:
    if axis == "angle":
        return MeasuredCurve("angle", X, [1.0] * 5, [0.1] * 5, energy=8048.0)
    return MeasuredCurve("energy", [8000.0, 8010.0], [0.5, 0.4], angle=0.01)


class TestValidation:
    def test_name_must_be_identifier(self) -> None:
        for name in ["", "a b", "1a", "a.b", "a-b"]:
            with pytest.raises(ValueError, match="not an identifier"):
                Dataset.create(name, make_curve())
        assert Dataset.create("_a1", make_curve()).name == "_a1"

    def test_axis_mismatch(self) -> None:
        with pytest.raises(ValueError, match="same axis"):
            Dataset("a", make_curve("angle"), Instrument.create("energy"))

    @pytest.mark.parametrize("weight", [0.0, -1.0, float("nan")])
    def test_weight_must_be_positive(self, weight: float) -> None:
        with pytest.raises(ValueError, match="weight"):
            Dataset("a", make_curve(), Instrument.create("angle"), weight=weight)

    def test_invalid_range(self) -> None:
        with pytest.raises(ValueError, match="invalid range"):
            Dataset("a", make_curve(), Instrument.create("angle"), ranges=((0.03, 0.01),))
        with pytest.raises(ValueError, match="invalid range"):
            Dataset("a", make_curve(), Instrument.create("angle"), excluded=((0.03, 0.01),))
        with pytest.raises(ValueError, match="invalid range"):
            Dataset.create("a", make_curve()).with_ranges([(1.0, 0.0)])

    def test_degenerate_range_allowed(self) -> None:
        data = Dataset.create("a", make_curve()).with_ranges([(0.02, 0.02)])
        assert data.mask().tolist() == [False, True, False, False, False]

    def test_ranges_normalized(self) -> None:
        data = Dataset(
            "a",
            make_curve(),
            Instrument.create("angle"),
            ranges=[(1, 2), (0, 1)],  # type: ignore[arg-type]
            excluded=[(np.float64(3.0), 4)],  # type: ignore[arg-type]
        )
        assert data.ranges == ((0.0, 1.0), (1.0, 2.0))
        assert all(isinstance(v, float) for r in data.ranges + data.excluded for v in r)
        assert isinstance(data.ranges, tuple)

    def test_create_passes_instrument_specs(self) -> None:
        data = Dataset.create("a", make_curve(), scale=free(1.0, 0.5, 2.0), background=1e-5)
        assert data.instrument.parameters.free_names() == ("scale",)
        assert data.instrument.parameters["background"].value == 1e-5
        assert data.instrument.axis == "angle"


class TestMask:
    def test_no_ranges_selects_all(self) -> None:
        assert Dataset.create("a", make_curve()).mask().tolist() == [True] * 5

    def test_single_range_inclusive(self) -> None:
        data = Dataset.create("a", make_curve()).with_ranges([(0.02, 0.04)])
        assert data.mask().tolist() == [False, True, True, True, False]

    def test_disjoint_ranges(self) -> None:
        data = Dataset.create("a", make_curve()).with_ranges([(0.04, 0.05), (0.0, 0.015)])
        assert data.mask().tolist() == [True, False, False, True, True]

    def test_overlapping_ranges(self) -> None:
        data = Dataset.create("a", make_curve()).with_ranges([(0.01, 0.03), (0.02, 0.04)])
        assert data.mask().tolist() == [True, True, True, True, False]

    def test_excluded(self) -> None:
        data = Dataset.create("a", make_curve()).with_ranges([], excluded=[(0.02, 0.03)])
        assert data.mask().tolist() == [True, False, False, True, True]

    def test_excluded_overrides_ranges(self) -> None:
        data = Dataset.create("a", make_curve()).with_ranges(
            [(0.0, 0.05)], excluded=[(0.015, 0.025), (0.05, 0.06)]
        )
        assert data.mask().tolist() == [True, False, True, True, False]

    def test_everything_excluded(self) -> None:
        data = Dataset.create("a", make_curve()).with_ranges([], excluded=[(0.0, 1.0)])
        assert not data.mask().any()

    def test_mask_shape(self) -> None:
        mask = Dataset.create("a", make_curve("energy")).mask()
        assert mask.shape == (2,)
        assert mask.dtype == bool


class TestWithRanges:
    def test_keeps_excluded_by_default(self) -> None:
        data = Dataset.create("a", make_curve()).with_ranges([], excluded=[(0.01, 0.02)])
        changed = data.with_ranges([(0.0, 0.03)])
        assert changed.excluded == ((0.01, 0.02),)
        assert changed.ranges == ((0.0, 0.03),)

    def test_replaces_excluded(self) -> None:
        data = Dataset.create("a", make_curve()).with_ranges([], excluded=[(0.01, 0.02)])
        assert data.with_ranges([], excluded=[]).excluded == ()
        assert data.with_ranges([], excluded=[(0.03, 0.04)]).excluded == ((0.03, 0.04),)

    def test_original_unchanged(self) -> None:
        data = Dataset.create("a", make_curve())
        data.with_ranges([(0.0, 0.02)])
        assert data.ranges == ()


class TestParameters:
    def test_prefixed(self) -> None:
        data = Dataset.create("scan", make_curve(), scale=free(1.0, 0.5, 2.0))
        assert list(data.parameters) == [
            "scan.scale",
            "scan.background",
            "scan.offset",
            "scan.resolution",
            "scan.beam_width",
            "scan.sample_length",
        ]
        assert data.parameters.free_names() == ("scan.scale",)

    def test_prefixed_expression(self) -> None:
        data = Dataset.create("s", make_curve(), background="scale * 1e-3")
        assert data.parameters["s.background"].expr == "s.scale * 1e-3"

    def test_with_parameters_round_trip(self) -> None:
        data = Dataset.create("scan", make_curve(), offset=free(0.0, -0.01, 0.01))
        assert data.with_parameters(data.parameters) == data

    def test_with_parameters_changes_values(self) -> None:
        data = Dataset.create("scan", make_curve(), offset=free(0.0, -0.01, 0.01))
        params = data.parameters.with_values({"scan.offset": 0.003})
        changed = data.with_parameters(params)
        assert changed.instrument.parameters["offset"].value == 0.003
        assert changed.instrument.parameters["offset"].is_free
        assert changed.curve is data.curve
        assert data.instrument.parameters["offset"].value == 0.0


class TestSerialization:
    def test_round_trip(self) -> None:
        data = Dataset.create(
            "scan", make_curve(), scale=free(1.0, 0.5, 2.0), resolution=1e-4, background="scale"
        )
        data = Dataset(
            data.name,
            data.curve,
            data.instrument,
            ranges=((0.01, 0.03),),
            excluded=((0.02, 0.025),),
            weight=2.5,
            enabled=False,
        )
        restored = from_data(to_data(data))
        assert isinstance(restored, Dataset)
        assert restored == data
        assert restored.weight == 2.5
        assert not restored.enabled

    def test_round_trip_energy_axis(self) -> None:
        data = Dataset.create("e", make_curve("energy"), offset=free(0.0, -5.0, 5.0))
        assert from_data(to_data(data)) == data
