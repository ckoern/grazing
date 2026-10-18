"""Tests for `xstack.gui.design.steps.estimate.EstimatePage`."""

import dataclasses
import math

import pytest
from pytestqt.qtbot import QtBot

from xstack.core.targets import EnergyScan, TargetSet
from xstack.core.units import energy_to_wavelength
from xstack.design.project import Command, ProjectState
from xstack.design.project.templates import new_state
from xstack.gui.design.steps.estimate import EstimatePage

pytestmark = pytest.mark.gui


class Recorder:
    """Collects the commands and errors emitted by the page."""

    def __init__(self, page: EstimatePage) -> None:
        self.commands: list[Command] = []
        self.errors: list[str] = []
        page.commandRequested.connect(self.commands.append)
        page.errorOccurred.connect(self.errors.append)


def make_page(qtbot: QtBot, state: ProjectState) -> tuple[EstimatePage, Recorder]:
    page = EstimatePage()
    qtbot.addWidget(page)
    page.set_state(state)
    return page, Recorder(page)


def test_help_text() -> None:
    assert EstimatePage.title == "Estimate"
    for word in ("Bragg", "refraction", "start values", "free"):
        assert word in EstimatePage.help_text


def test_periodic_estimates_fill_table(qtbot: QtBot) -> None:
    page, _ = make_page(qtbot, new_state("periodic"))
    assert page.estimates == {}
    page.estimate_button.click()
    assert set(page.estimates) == {"plain", "refraction"}
    assert page.table.rowCount() == 3
    names = [page.table.item(r, 0).text() for r in range(3)]
    assert names == ["N", "d", "gamma"]
    assert "bounds:" in page.table.item(1, 3).toolTip()
    for label in page.merit_labels.values():
        assert "figure of merit" in label.text()
    assert page.refraction_radio.isChecked()
    assert not page.narrow_check.isChecked()


def test_refraction_increases_period(qtbot: QtBot) -> None:
    state = new_state("periodic")
    page, _ = make_page(qtbot, state)
    page.estimate()
    plain = page.estimates["plain"].parameters["d"].value
    corrected = page.estimates["refraction"].parameters["d"].value
    assert state.targets is not None
    target = state.targets.targets[0]
    assert isinstance(target, EnergyScan)
    energy = 0.5 * (target.energy_min + target.energy_max)
    wavelength = float(energy_to_wavelength(energy))
    assert plain == pytest.approx(wavelength / (2 * math.sin(target.angle)), rel=1e-3)
    assert corrected > plain


def test_depth_graded_estimates(qtbot: QtBot) -> None:
    page, _ = make_page(qtbot, new_state("depth_graded"))
    page.estimate()
    for result in page.estimates.values():
        p = result.parameters
        assert p["d_min"].value < p["d_max"].value
    assert page.table.rowCount() == 5


@pytest.mark.parametrize("kind", ["plain", "refraction"])
@pytest.mark.parametrize("narrow", [False, True])
def test_apply_sets_start_values(qtbot: QtBot, kind: str, narrow: bool) -> None:
    state = new_state("periodic")
    page, rec = make_page(qtbot, state)
    page.estimate()
    (page.plain_radio if kind == "plain" else page.refraction_radio).setChecked(True)
    page.narrow_check.setChecked(narrow)
    page.apply_button.click()
    assert len(rec.commands) == 1
    assert not page.confirmation.isHidden()
    assert f"{'plain Bragg' if kind == 'plain' else 'refraction-corrected'} estimate" in (
        page.confirmation.text()
    )
    new = rec.commands[0].transform(state).mirror.parameters  # type: ignore[union-attr]
    old = state.mirror.parameters  # type: ignore[union-attr]
    est = page.estimates[kind].parameters
    for name in ("d", "gamma", "N"):
        assert new[name].value == pytest.approx(est[name].value)
        if narrow:
            assert new[name].min == pytest.approx(est[name].min)
            assert new[name].max == pytest.approx(est[name].max)
        else:
            assert (new[name].min, new[name].max) == (old[name].min, old[name].max)
    assert new["sigma_H"] == old["sigma_H"]


def test_store_as_candidate(qtbot: QtBot) -> None:
    state = new_state("periodic")
    page, rec = make_page(qtbot, state)
    page.estimate()
    page.store_button.click()
    assert len(rec.commands) == 1
    new = rec.commands[0].transform(state)
    assert [c.name for c in new.candidates] == ["estimate (refraction)"]
    assert new.mirror == state.mirror


def test_unsupported_coating(qtbot: QtBot) -> None:
    page, rec = make_page(qtbot, new_state("single_layer"))
    assert not page.unsupported_label.isHidden()
    assert not page.estimate_button.isEnabled()
    page.estimate()
    assert page.estimates == {}
    assert not page.table.isVisible()
    assert rec.errors == []


def test_invalid_state_reports_error(qtbot: QtBot) -> None:
    state = new_state("periodic").evolve(targets=None)
    page, rec = make_page(qtbot, state)
    page.estimate_button.click()
    assert len(rec.errors) == 1
    assert page.estimates == {}


def two_target_state() -> ProjectState:
    state = new_state("periodic")
    assert state.targets is not None
    first = state.targets.targets[0]
    assert isinstance(first, EnergyScan)
    second = dataclasses.replace(
        first, energy_min=2 * first.energy_min, energy_max=2 * first.energy_max
    )
    return state.evolve(targets=TargetSet((first, second), "average_merit"))


def test_target_selector_hidden_for_single_target(qtbot: QtBot) -> None:
    page, _ = make_page(qtbot, new_state("periodic"))
    page.show()
    assert not page.target_combo.isVisible()


def test_target_selector_lists_targets(qtbot: QtBot) -> None:
    page, _ = make_page(qtbot, two_target_state())
    page.show()
    assert page.target_combo.isVisible()
    assert page.target_combo.itemText(0) == "All targets"
    assert page.target_combo.itemData(0) is None
    assert [page.target_combo.itemData(i) for i in (1, 2)] == [0, 1]
    assert "Target" in EstimatePage.help_text


def test_target_selection_changes_estimate(qtbot: QtBot) -> None:
    page, _ = make_page(qtbot, two_target_state())
    page.estimate()
    pooled = page.estimates["plain"].parameters["d"].value
    page.target_combo.setCurrentIndex(1)
    page.estimate()
    first = page.estimates["plain"].parameters["d"].value
    page.target_combo.setCurrentIndex(2)
    page.estimate()
    second = page.estimates["plain"].parameters["d"].value
    assert second == pytest.approx(first / 2, rel=0.05)
    assert pooled not in (first, second)


def test_target_selection_kept_on_state_update(qtbot: QtBot) -> None:
    state = two_target_state()
    page, _ = make_page(qtbot, state)
    page.target_combo.setCurrentIndex(2)
    page.set_state(state)
    assert page.target_combo.currentData() == 1
