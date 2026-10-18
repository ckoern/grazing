"""Tests for `xstack.fit.report`."""

import math

import pytest

from xstack.core.parameters import Parameter, ParameterSet, free
from xstack.fit.instrument import Instrument
from xstack.fit.report import display_value, format_parameters, format_result
from xstack.fit.results import FitResult, FitStatistics


class TestDisplayValue:
    def test_rad_to_deg(self) -> None:
        value, unit = display_value(Parameter("offset", math.pi / 180, unit="rad"))
        assert value == pytest.approx(1.0)
        assert unit == "deg"

    def test_explicit_value(self) -> None:
        value, unit = display_value(Parameter("offset", 0.0, unit="rad"), math.pi)
        assert value == pytest.approx(180.0)
        assert unit == "deg"

    @pytest.mark.parametrize(
        "name", ["beam_width", "sample_length", "a.beam_width", "x.sample_length"]
    )
    def test_nm_to_mm(self, name: str) -> None:
        value, unit = display_value(Parameter(name, 2e6, unit="nm"))
        assert value == pytest.approx(2.0)
        assert unit == "mm"

    def test_other_thickness_unchanged(self) -> None:
        assert display_value(Parameter("t", 30.0, unit="nm")) == (30.0, "nm")
        assert display_value(Parameter("a.t_beam_width_x", 5.0, unit="nm")) == (5.0, "nm")

    def test_energy_and_dimensionless_unchanged(self) -> None:
        assert display_value(Parameter("resolution", 3.0, unit="eV")) == (3.0, "eV")
        assert display_value(Parameter("scale", 1.5)) == (1.5, "")


def parameters() -> ParameterSet:
    instrument = Instrument.create(
        "angle",
        scale=free(1.25, 0.5, 2.0),
        background=1e-6,
        offset=free(0.001, -0.01, 0.01),
        resolution="offset * 2",
        beam_width=2e5,
        sample_length=free(5e7, 1e7, 1e8),
    )
    return ParameterSet([Parameter("t", 30.0, unit="nm", vary=True, min=0, max=100)]).merge(
        instrument.parameters.prefixed("a")
    )


def lines_by_name(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        if line.strip():
            out.setdefault(line.split()[0], line)
    return out


class TestFormatParameters:
    def test_without_result(self) -> None:
        text = format_parameters(parameters())
        lines = lines_by_name(text)
        assert len(text.splitlines()) == 7
        assert "±" not in text
        assert lines["t"].split() == ["t", "30", "nm", "free"]
        assert lines["a.scale"].split() == ["a.scale", "1.25", "free"]
        assert lines["a.background"].split() == ["a.background", "1e-06", "fixed"]
        assert lines["a.offset"].split()[:3] == ["a.offset", "0.0572958", "deg"]
        assert lines["a.resolution"].endswith("= a.offset * 2")
        assert lines["a.beam_width"].split() == ["a.beam_width", "0.2", "mm", "fixed"]
        assert lines["a.sample_length"].split() == ["a.sample_length", "50", "mm", "free"]

    def test_names_aligned(self) -> None:
        lines = format_parameters(parameters()).splitlines()
        width = len("a.sample_length")
        assert all(line[width : width + 2] == "  " for line in lines)
        assert lines[0].startswith("t" + " " * (width - 1))

    def test_empty(self) -> None:
        assert format_parameters(ParameterSet()) == ""

    def stats(self, errors: tuple[float, ...]) -> FitStatistics:
        names = ("t", "a.offset", "a.scale")[: len(errors)]
        n = len(names)
        corr = tuple(tuple(1.0 if i == j else 0.0 for j in range(n)) for i in range(n))
        return FitStatistics(names, (1.0,) * n, errors, corr, 3.0, 20, "chi2")

    def test_uncertainties_in_display_units(self) -> None:
        stats = self.stats((0.123, math.radians(0.01), 0.04))
        result = FitResult(parameters(), stats)
        lines = lines_by_name(format_parameters(parameters(), result))
        assert "30 ± 0.12" in lines["t"]
        assert "0.0572958 ± 0.01" in lines["a.offset"]
        assert "1.25 ± 0.04" in lines["a.scale"]
        # Fixed parameters and parameters outside the statistics show no uncertainty.
        assert "±" not in lines["a.background"]
        assert "±" not in lines["a.sample_length"]

    def test_undetermined(self) -> None:
        stats = self.stats((0.1, math.nan))
        lines = lines_by_name(format_parameters(parameters(), FitResult(parameters(), stats)))
        assert "± (undetermined)" in lines["a.offset"]
        assert "± 0.1" in lines["t"]

    def test_result_without_statistics(self) -> None:
        text = format_parameters(parameters(), FitResult(parameters(), None))
        assert "±" not in text

    def test_negative_error_shown_positive(self) -> None:
        stats = self.stats((-0.5,))
        text = format_parameters(parameters(), FitResult(parameters(), stats))
        assert "± 0.5" in text
        assert "± -" not in text


def make_result(
    weighting: str = "chi2", with_stats: bool = True, errors=(0.2, 0.01), corr=0.95
) -> FitResult:
    params = parameters()
    stats = FitStatistics(
        ("t", "a.scale"),
        (30.0, 1.25),
        errors,
        ((1.0, corr), (corr, 1.0)),
        objective=36.0,
        points=20,
        weighting=weighting,  # type: ignore[arg-type]
    )
    return FitResult(
        params,
        stats if with_stats else None,
        method="least_squares",
        evaluations=42,
        elapsed=1.25,
        note="" if with_stats else "no free parameters",
    )


class TestFormatResult:
    def test_chi2_with_statistics(self) -> None:
        text = format_result(make_result())
        lines = text.splitlines()
        assert lines[0].startswith("method: least_squares, 42 evaluations, 1.")
        assert "weighting: chi2, points: 20, free: 2, χ²_red: 2" in text
        assert "warning" not in text
        assert "± 0.2" in lines_by_name(text)["t"]

    def test_other_weighting_label(self) -> None:
        text = format_result(make_result("log"))
        assert "objective per degree of freedom: 2" in text
        assert "χ²_red" not in text

    def test_without_statistics_shows_note(self) -> None:
        text = format_result(make_result(with_stats=False))
        assert "no uncertainties: no free parameters" in text
        assert "weighting:" not in text
        assert "correlations" not in text
        assert "a.scale" in text  # the parameter table is still shown

    def test_cancelled(self) -> None:
        result = FitResult(parameters(), None, method="nelder", cancelled=True)
        assert "(cancelled)" in format_result(result).splitlines()[0]
        assert "(cancelled)" not in format_result(make_result())

    def test_correlation_section(self) -> None:
        text = format_result(make_result(corr=0.95))
        assert "correlations above 0.8:" in text
        assert "  t / a.scale: +0.950" in text
        assert "correlations above 0.99:" in format_result(make_result(corr=0.995), 0.99)

    def test_negative_correlation(self) -> None:
        assert "t / a.scale: -0.950" in format_result(make_result(corr=-0.95))

    def test_no_correlation_section_below_threshold(self) -> None:
        assert "correlations" not in format_result(make_result(corr=0.5))
        assert "correlations" not in format_result(
            make_result(corr=0.95), correlation_threshold=0.99
        )

    def test_undetermined_warning(self) -> None:
        text = format_result(make_result(errors=(0.2, math.nan), corr=math.nan))
        assert "warning: the data do not determine all free parameters" in text
        assert "(undetermined)" in text
        assert "correlations" not in text

    def test_table_after_blank_line(self) -> None:
        lines = format_result(make_result()).splitlines()
        assert lines[2] == ""
        assert lines[3].startswith("t")


def test_expression_parameters_show_evaluated_value() -> None:
    params = ParameterSet(
        [
            Parameter("a.offset", 0.001, vary=True, min=-1.0, max=1.0, unit="rad"),
            Parameter("b.offset", 0.0, min=-1.0, max=1.0, expr="a.offset", unit="rad"),
        ]
    )
    lines = lines_by_name(format_parameters(params))
    assert lines["b.offset"].split()[1] == lines["a.offset"].split()[1]
    assert "= a.offset" in lines["b.offset"]
