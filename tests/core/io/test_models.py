"""Tests for `grazing.core.io.models`."""

from collections.abc import Iterator
from pathlib import Path

import pytest

from grazing.core.io import files
from grazing.core.io.models import ImportedModel, model_readers, read_models, reader_for
from grazing.core.materials import get
from grazing.core.structure.coating import Mirror
from grazing.core.structure.coatings import SingleLayer

MIRROR = Mirror(SingleLayer.create(get("Pt"), t=30.0), get("Si"))


class FakeReader:
    description = "fake model file"
    suffixes = (".fake", ".fk")

    def read(self, path: Path) -> tuple[ImportedModel, ...]:
        return (ImportedModel(path.stem, MIRROR),)


@pytest.fixture
def fake_reader(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeReader]:
    monkeypatch.setattr(files, "load_plugins", lambda: ())
    reader = FakeReader()
    model_readers.register("fake", reader)
    yield reader
    model_readers.unregister("fake")


def test_reader_chosen_by_suffix_ignoring_case(fake_reader: FakeReader) -> None:
    assert reader_for("data/sample.FK") is fake_reader
    assert reader_for(Path("sample.fake")) is fake_reader


def test_read_models(fake_reader: FakeReader) -> None:
    (model,) = read_models("sample.fake")
    assert model.name == "sample"
    assert model.mirror == MIRROR
    assert model.model.name == "parratt"


@pytest.mark.usefixtures("fake_reader")
def test_unknown_suffix_lists_supported() -> None:
    with pytest.raises(ValueError, match=r"'\.txt'.*\.fake, \.fk"):
        reader_for("sample.txt")


@pytest.mark.usefixtures("fake_reader")
def test_missing_suffix() -> None:
    with pytest.raises(ValueError, match="files without suffix"):
        reader_for("sample")


def test_plugins_loaded_before_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[bool] = []

    def load_plugins() -> tuple[str, ...]:
        calls.append(True)
        return ()

    monkeypatch.setattr(files, "load_plugins", load_plugins)
    with pytest.raises(ValueError):
        reader_for("sample.unknown")
    assert calls == [True]
