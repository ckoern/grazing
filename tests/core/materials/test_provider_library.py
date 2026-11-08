"""Tests of the shared libraries per optical-constants provider."""

import pytest

from grazing.core.materials import (
    DEFAULT_PROVIDER,
    default_library,
    delta_beta,
    get,
    provider_library,
)


def test_default_provider_gives_default_library() -> None:
    assert provider_library(DEFAULT_PROVIDER) is default_library()


def test_libraries_are_shared_per_provider() -> None:
    library = provider_library("chantler")
    assert library is provider_library("chantler")
    assert library.default_provider == "chantler"
    assert library is not provider_library("henke")


def test_library_uses_its_provider() -> None:
    w = get("W")
    assert provider_library("chantler").delta_beta(w, 8048.0) == pytest.approx(
        delta_beta(w, 8048.0, provider="chantler")
    )


def test_unknown_provider() -> None:
    with pytest.raises(KeyError):
        provider_library("nonexistent")
