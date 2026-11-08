"""Tests for `grazing.core.io.data`."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from grazing.core.io import files
from grazing.core.io.data import data_parsers, parser_for, read_curves
from grazing.core.io.delimited import CsvParser
from grazing.core.measurement import MeasuredCurve


class FakeParser:
    description = "fake data"
    suffixes = (".fake",)
    options_type = dict

    def __init__(self) -> None:
        self.calls: list[tuple[Path, Any]] = []

    def read(self, path: Path, options: Any = None) -> tuple[MeasuredCurve, ...]:
        self.calls.append((path, options))
        return (MeasuredCurve("energy", [1000.0], [0.5], name=path.stem),)


@pytest.fixture
def fake_parser(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeParser]:
    monkeypatch.setattr(files, "load_plugins", lambda: ())
    parser = FakeParser()
    data_parsers.register("fake", parser)
    yield parser
    data_parsers.unregister("fake")


@pytest.mark.parametrize("name", ["a.csv", "a.TSV", "a.txt", "a.dat", "dir/b.CSV"])
def test_csv_parser_chosen(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(files, "load_plugins", lambda: ())
    assert isinstance(parser_for(name), CsvParser)


def test_unknown_suffix_lists_supported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(files, "load_plugins", lambda: ())
    with pytest.raises(ValueError, match=r"'\.xyz'.*supported: \.csv, \.dat, \.tsv, \.txt"):
        parser_for("a.xyz")


def test_read_curves_passes_options(fake_parser: FakeParser) -> None:
    options = {"a": 1}
    (curve,) = read_curves("data/scan.fake", options)
    assert curve.name == "scan"
    assert fake_parser.calls == [(Path("data/scan.fake"), options)]


def test_read_curves_default_options_none(fake_parser: FakeParser) -> None:
    read_curves(Path("scan.fake"))
    assert fake_parser.calls == [(Path("scan.fake"), None)]
