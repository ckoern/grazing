"""Screenshots of the fit application and of the fit with the library."""

import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from docs.guide.screenshots import IMAGES, fit_window, pump, save, screenshot
from xstack.core.optim import LmfitOptimizer
from xstack.core.units import deg
from xstack.fit.project import (
    add_dataset,
    set_optimizers,
    set_parameter,
    set_ranges,
    set_weighting,
)
from xstack.gui.common.wizard import ProjectWizard
from xstack.gui.fit.main_window import PROJECT_FILTER
from xstack.gui.fit.steps.parameters import share_command
from xstack.gui.fit.widgets.import_dialog import ImportDialog

if TYPE_CHECKING:
    from xstack.gui.fit.main_window import MainWindow

REPOSITORY = Path(__file__).parents[3]
DATA = REPOSITORY / "examples" / "data"
CU_KA = DATA / "WSi_multilayer_cu_ka.csv"
MO_KA = DATA / "WSi_multilayer_mo_ka.csv"

DATA_STEP, STRUCTURE_STEP, MATERIALS_STEP, PARAMETERS_STEP, FIT_STEP, RESULTS_STEP = range(6)
REFLECTIVITY_TAB, CORRELATIONS_TAB, PROFILE_TAB, CONVERGENCE_TAB = range(4)

FIT_RANGES = [(deg(0.35), deg(1.6)), (deg(1.8), deg(3.0))]
ARTIFACT = [(deg(2.05), deg(2.15))]


def _wizard() -> ProjectWizard:
    wizard = ProjectWizard("fit project", ".xfit", PROJECT_FILTER)
    wizard.set_name("W/Si multilayer, Cu Kα")
    wizard.set_kind("periodic")
    wizard.set_pair("W/Si")
    return wizard


def _import_dialog(path: Path, name: str) -> ImportDialog:
    dialog = ImportDialog(())
    dialog.set_path(path)
    dialog.name_edit.setText(name)
    dialog.name_edit.textEdited.emit(name)
    return dialog


def _new_window() -> "MainWindow":
    """A fit window with the wizard's W/Si multilayer and no data."""
    window = fit_window()
    window.name_edit.clearFocus()
    wizard = _wizard()
    window.controller.new_from_setup(wizard.setup())
    pump(0.5)
    return window


def _add_data(window: "MainWindow", path: Path, name: str) -> None:
    for dataset in _import_dialog(path, name).datasets():
        window.controller.execute(add_dataset(dataset))
    pump(0.5)


def _set_up_parameters(window: "MainWindow", dataset: str = "cu_ka") -> None:
    """Parameter settings of the quick start: 20 bilayers, free roughnesses and instrument."""
    execute = window.controller.execute
    execute(set_parameter("N", value=20))
    _set_up_instrument(window, dataset)


def _set_up_instrument(window: "MainWindow", dataset: str) -> None:
    execute = window.controller.execute
    execute(set_parameter(f"{dataset}.scale", value=1.0, min=0.5, max=1.5, vary=True))
    execute(set_parameter(f"{dataset}.background", value=1e-7, min=0.0, max=1e-5, vary=True))
    execute(set_parameter(f"{dataset}.offset", value=0.0, min=-deg(0.02), max=deg(0.02), vary=True))
    execute(set_parameter(f"{dataset}.resolution", value=deg(0.004)))
    execute(set_parameter(f"{dataset}.beam_width", value=0.05e6))
    execute(set_parameter(f"{dataset}.sample_length", value=20e6))


def _set_ranges(window: "MainWindow", dataset: str = "cu_ka") -> None:
    window.controller.execute(set_ranges(dataset, FIT_RANGES, ARTIFACT))
    pump(0.5)


def _fit(window: "MainWindow", local: bool = True) -> None:
    window.step_tabs.setCurrentIndex(FIT_STEP)
    if local:
        window.fit_page.run_local()
    else:
        window.fit_page.run()
    pump(0.3)
    pump(until=lambda: not window.job_runner.running)
    pump(1.0)


def _prepared_window(fitted: bool = False) -> "MainWindow":
    window = _new_window()
    _add_data(window, CU_KA, "cu_ka")
    _set_ranges(window)
    _set_up_parameters(window)
    window.controller.execute(set_weighting("chi2"))
    pump(0.5)
    if fitted:
        _fit(window)
    return window


def _close(window: "MainWindow") -> None:
    """Close a window once its background calculations are done."""
    pump(until=lambda: not (window.job_runner.running or window.preview_runner.is_busy()))
    window.close()
    pump(0.2)


def _grab_step(window: "MainWindow", step: int, name: str) -> None:
    window.step_tabs.setCurrentIndex(step)
    pump(0.5)
    save(window.step_tabs, name)


@screenshot
def fit_wizard() -> None:
    """The coating page of the wizard with a W/Si multilayer."""
    wizard = _wizard()
    wizard.resize(700, 700)
    wizard.show()
    wizard.next()
    pump(0.3)
    save(wizard, "fit_wizard")
    wizard.close()


@screenshot
def fit_import_dialog() -> None:
    """The import dialog with the Cu Kα example file."""
    dialog = _import_dialog(CU_KA, "cu_ka")
    dialog.resize(720, 640)
    dialog.show()
    pump(0.5)
    save(dialog, "fit_import_dialog")
    dialog.close()


@screenshot
def fit_data_added() -> None:
    """The window after the data were added: all points are fitted."""
    window = _new_window()
    _add_data(window, CU_KA, "cu_ka")
    window.step_tabs.setCurrentIndex(DATA_STEP)
    pump(1.0)
    save(window, "fit_data_added")
    _close(window)


@screenshot
def fit_data_ranges() -> None:
    """The Data page with a fit range, an excluded range and the plot."""
    window = _new_window()
    _add_data(window, CU_KA, "cu_ka")
    _set_ranges(window)
    window.step_tabs.setCurrentIndex(DATA_STEP)
    pump(1.0)
    save(window, "fit_data_ranges")
    _close(window)


@screenshot
def fit_parameters() -> None:
    """The Parameters page with the quick-start settings."""
    window = _prepared_window()
    window.resize(1500, 1120)
    pump(0.5)
    _grab_step(window, PARAMETERS_STEP, "fit_parameters")
    _close(window)


@screenshot
def fit_fit_page() -> None:
    """The Fit page before the first fit."""
    window = _prepared_window()
    _grab_step(window, FIT_STEP, "fit_fit_page")
    _close(window)


@screenshot
def fit_results() -> None:
    """The window after a local fit, showing the Results page."""
    window = _prepared_window(fitted=True)
    window.resize(1500, 1120)
    window.step_tabs.setCurrentIndex(RESULTS_STEP)
    pump(1.0)
    save(window, "fit_results")
    window.plot_tabs.setCurrentIndex(CORRELATIONS_TAB)
    pump(0.5)
    save(window.plot_tabs, "fit_correlations")
    _close(window)


def _global_stages() -> list[LmfitOptimizer]:
    options = (("seed", 0), ("popsize", 8), ("polish", False))
    return [
        LmfitOptimizer("differential_evolution", max_evaluations=800, options=options),
        LmfitOptimizer("least_squares"),
    ]


@screenshot
def fit_global() -> None:
    """A local fit trapped in a local minimum, then a global search, both saved."""
    window = _prepared_window()
    window.resize(1500, 1120)
    execute = window.controller.execute
    execute(set_parameter("d", value=3.6, min=3.0, max=6.0))
    _fit(window)
    execute(set_parameter("d", min=3.5, max=4.6))
    execute(set_optimizers(_global_stages()))
    pump(0.5)
    _fit(window, local=False)
    window.step_tabs.setCurrentIndex(RESULTS_STEP)
    pump(0.5)
    save(window, "fit_saved_fits")
    window.plot_tabs.setCurrentIndex(CONVERGENCE_TAB)
    pump(0.5)
    save(window.plot_tabs, "fit_convergence")
    _close(window)


@screenshot
def fit_simultaneous() -> None:
    """Two datasets fitted with one structure and a shared angle offset."""
    window = _prepared_window()
    window.resize(1500, 1120)
    execute = window.controller.execute
    _add_data(window, MO_KA, "mo_ka")
    execute(set_ranges("mo_ka", [(deg(0.2), deg(1.4))]))
    _set_up_instrument(window, "mo_ka")
    execute(share_command("offset", "cu_ka", ["mo_ka"]))
    pump(0.5)
    window.parameters_page.parameter_combo.setCurrentIndex(
        window.parameters_page.parameter_combo.findData("offset")
    )
    _fit(window)
    window.step_tabs.setCurrentIndex(PARAMETERS_STEP)
    pump(0.5)
    save(window, "fit_simultaneous")
    _close(window)


@screenshot
def fit_library_plot() -> None:
    """The figure drawn by the plotting snippet of the library quick start."""
    source = (REPOSITORY / "docs" / "guide" / "quickstart_fit_library.md").read_text()
    blocks = re.findall(r"^```python\n(.*?)^```", source, re.MULTILINE | re.DOTALL)
    previous = Path.cwd()
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copytree(DATA, Path(tmp) / "examples" / "data")
        os.chdir(tmp)
        try:
            namespace: dict[str, object] = {"__name__": "__guide__"}
            for block in blocks:
                exec(compile(block, "quickstart_fit_library", "exec"), namespace)
            IMAGES.mkdir(parents=True, exist_ok=True)
            shutil.copy("fit_curves.png", IMAGES / "fit_library_plot.png")
        finally:
            os.chdir(previous)
