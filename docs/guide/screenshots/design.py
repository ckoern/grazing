"""Screenshots of the design application for the design guides."""

from typing import TYPE_CHECKING

from PySide6.QtWidgets import QScrollArea

from docs.guide.screenshots import design_window, pump, save, screenshot
from grazing.core.targets import EnergyScan, Target, TargetSet
from grazing.core.units import deg, kev
from grazing.design.manufacturing import DepositionProcess, MaxCoatingTime, MinLayerThickness
from grazing.design.merit import MeanReflectivity, TimePenalized
from grazing.design.project.commands import (
    add_constraint,
    promote_candidate,
    select_candidate,
    set_merit,
    set_parameter,
    set_process,
    set_targets,
)
from grazing.gui.common.wizard import ProjectWizard
from grazing.gui.design.main_window import PROJECT_FILTER

if TYPE_CHECKING:
    from grazing.gui.design.main_window import MainWindow

ENERGY = 8048.0
WINDOW = (1500, 1100)
STEPS_ALL = (
    "Goal",
    "Materials",
    "Coating",
    "Estimate",
    "Constraints",
    "Process",
    "Scan",
    "Optimize",
    "Trade-off",
    "Analyze",
)


def _wizard(kind: str, pair: str, name: str) -> ProjectWizard:
    wizard = ProjectWizard("design project", ".grazing", PROJECT_FILTER)
    wizard.resize(700, 620)
    wizard.show()
    wizard.set_name(name)
    wizard.next()
    wizard.set_kind(kind)  # type: ignore[arg-type]
    wizard.set_pair(pair)
    wizard.set_substrate("Si")
    pump(0.3)
    return wizard


def _window(wizard: ProjectWizard) -> "MainWindow":
    window = design_window()
    window.resize(*WINDOW)
    window.controller.new_from_setup(wizard.setup())
    wizard.close()
    pump(0.8)
    return window


def _step(window: "MainWindow", name: str) -> None:
    window.step_tabs.setCurrentIndex(STEPS_ALL.index(name))
    pump(0.5)


def _plot(window: "MainWindow", widget_name: str) -> None:
    window.plot_tabs.setCurrentWidget(getattr(window, widget_name))
    pump(0.4)


def _scroll(window: "MainWindow", top: bool = False) -> None:
    """Scroll the page of the current step to its end (or its start)."""
    bar = window.step_tabs.currentWidget().findChild(QScrollArea).verticalScrollBar()
    bar.setValue(bar.minimum() if top else bar.maximum())
    pump(0.3)


def _show_layers(window: "MainWindow") -> None:
    """Show the layer profile of the selected candidate."""
    window.layer_source.setCurrentIndex(window.layer_source.findData("selected"))
    window._show_layers()
    pump(0.4)


def _wait(window: "MainWindow") -> None:
    pump(0.5)
    pump(until=lambda: not window.job_runner.running)
    pump(1.0)


def _budget(
    window: "MainWindow", iterations: int, population: int, coarse: int, refine: int
) -> None:
    page = window.optimize_page
    page.seed.setText("0")
    page.iterations.setValue(iterations)
    page.population.setValue(population)
    page.coarse.setValue(coarse)
    page._rows["refine"].evaluations.setValue(refine)
    pump(0.3)


def _monochromator() -> "MainWindow":
    """Window with the goal, parameters, constraint and process of the quick start."""
    window = _window(_wizard("periodic", "W/B4C", "W/B4C Cu K-alpha monochromator"))
    row = window.goal_page.rows[0]
    for edit, text in (
        (row.fixed, "1.0"),
        (row.lo, f"{0.99 * ENERGY / 1e3:.4f}"),
        (row.hi, f"{1.01 * ENERGY / 1e3:.4f}"),
        (row.points, "61"),
    ):
        edit.setText(text)
        edit.editingFinished.emit()
    row.polarization.setCurrentText("s")
    row.polarization.activated.emit(row.polarization.currentIndex())
    spin = window.coating_page.roughness_spin
    spin.setValue(0.3)
    spin.editingFinished.emit()
    execute = window.controller.execute
    execute(set_parameter("N", value=60, min=20, max=150))
    execute(set_parameter("d", value=4.4, min=2.0, max=8.0))
    execute(set_parameter("gamma", value=0.4, min=0.1, max=0.8))
    execute(set_parameter("sigma_H", value=0.3))
    execute(set_parameter("sigma_L", value=0.3))
    window.controller.execute(
        set_process(DepositionProcess({"W": 0.1, "B4C": 0.05}, 15.0, fixed_overhead=1800.0))
    )
    pump(1.0)
    return window


_CACHE: dict[str, "MainWindow"] = {}


def _optimized() -> "MainWindow":
    """Monochromator window after a scan and a pipeline run."""
    if "optimized" not in _CACHE:
        window = _monochromator()
        _budget(window, 6, 8, 21, 400)
        window.optimize_page.run()
        _wait(window)
        _CACHE["optimized"] = window
    return _CACHE["optimized"]


@screenshot
def design_wizard_project() -> None:
    """First page of the project wizard."""
    wizard = _wizard("periodic", "W/B4C", "W/B4C Cu K-alpha monochromator")
    wizard.back()
    pump(0.2)
    save(wizard, "design_wizard_project")
    wizard.close()


@screenshot
def design_wizard_coating() -> None:
    """Coating page of the wizard with a periodic W/B4C multilayer."""
    wizard = _wizard("periodic", "W/B4C", "W/B4C Cu K-alpha monochromator")
    save(wizard, "design_wizard_coating")
    wizard.close()


@screenshot
def design_window_start() -> None:
    """The main window after the wizard."""
    window = _window(_wizard("periodic", "W/B4C", "W/B4C Cu K-alpha monochromator"))
    save(window, "design_window_start")
    window.close()


@screenshot
def design_goal() -> None:
    """Goal step with the Cu K-alpha target."""
    window = _monochromator()
    _step(window, "Goal")
    save(window, "design_goal")


@screenshot
def design_materials() -> None:
    """Materials step."""
    window = _monochromator()
    _step(window, "Materials")
    save(window.step_tabs, "design_materials")


@screenshot
def design_coating() -> None:
    """Coating step with the parameter table."""
    window = _monochromator()
    _step(window, "Coating")
    save(window.step_tabs, "design_coating")


@screenshot
def design_estimate() -> None:
    """Estimate step after pressing Estimate."""
    window = _monochromator()
    _step(window, "Estimate")
    window.estimate_page.estimate()
    pump(0.8)
    save(window, "design_estimate")


@screenshot
def design_constraints() -> None:
    """Constraints step."""
    window = _monochromator()
    _step(window, "Constraints")
    save(window.step_tabs, "design_constraints")


@screenshot
def design_process() -> None:
    """Process step."""
    window = _monochromator()
    _step(window, "Process")
    save(window.step_tabs, "design_process")


@screenshot
def design_scan() -> None:
    """Scan step with the design map."""
    window = _monochromator()
    _step(window, "Scan")
    window.scan_page.x_combo.setCurrentText("d")
    window.scan_page.y_combo.setCurrentText("gamma")
    window.scan_page.run()
    _wait(window)
    _plot(window, "design_map")
    save(window, "design_scan")


@screenshot
def design_optimize() -> None:
    """Optimize step after a pipeline run."""
    window = _optimized()
    _step(window, "Optimize")
    _plot(window, "convergence_plot")
    _scroll(window)
    save(window, "design_optimize")


@screenshot
def design_result() -> None:
    """Reflectivity of the optimized candidate."""
    window = _optimized()
    window.plot_tabs.setCurrentIndex(0)
    pump(0.5)
    save(window.plot_tabs, "design_result_reflectivity")
    window.plot_tabs.setCurrentWidget(window.comparison_view)
    pump(0.5)
    save(window.plot_tabs, "design_result_parameters")
    window.plot_tabs.setCurrentIndex(1)
    _show_layers(window)
    save(window.plot_tabs, "design_result_layers")


@screenshot
def design_analyze() -> None:
    """Analyze step with the tolerance analysis."""
    window = _optimized()
    _step(window, "Analyze")
    page = window.analyze_page
    page.design_combo.setCurrentIndex(page.design_combo.findData("selected"))
    page.preset_combo.setCurrentText("Thickness errors only")
    page.samples.setValue(60)
    page.seed.setText("1")
    page.run()
    _wait(window)
    save(window, "design_analyze")
    save(window.tolerance_view, "design_tolerance_plot")


@screenshot
def design_tolerance_errors() -> None:
    """Analyze step with all error types enabled."""
    window = _optimized()
    _step(window, "Analyze")
    page = window.analyze_page
    page.design_combo.setCurrentIndex(page.design_combo.findData("selected"))
    page.preset_combo.setCurrentText("All errors")
    page.samples.setValue(60)
    page.seed.setText("1")
    page.run()
    _wait(window)
    save(window.step_tabs, "design_tolerance_errors")
    save(window.tolerance_view, "design_tolerance_all")


@screenshot
def design_multi_goal() -> None:
    """Goal step with five coupled angles and the combine mode."""
    wizard = _wizard("graded_power_law", "W/Si", "Five-angle supermirror")
    window = _window(wizard)
    angles = (0.15, 0.18, 0.21, 0.24, 0.27)
    weights = (1.0, 2.0, 3.0, 2.0, 1.0)
    scans: list[Target] = [
        EnergyScan(
            angle=deg(a),
            energy_min=kev(20),
            energy_max=kev(60),
            points=100,
            weight=w,
            name=f"{a} deg",
        )
        for a, w in zip(angles, weights, strict=True)
    ]
    window.controller.execute(set_targets(TargetSet(scans, combine="average_reflectivity")))
    window.controller.execute(add_constraint(MinLayerThickness(0.8)))
    _step(window, "Goal")
    window.resize(1500, 1300)
    pump(0.8)
    save(window.step_tabs, "design_multi_goal")
    _CACHE["multi"] = window


@screenshot
def design_multi_result() -> None:
    """Reflectivity of the coupled-angle design."""
    window = _CACHE.get("multi")
    if window is None:
        design_multi_goal()
        window = _CACHE["multi"]
    window.optimize_page.run_button.setFocus()
    _step(window, "Optimize")
    _budget(window, 8, 6, 40, 120)
    window.optimize_page.run()
    _wait(window)
    window.controller.execute(promote_candidate("run 1"))
    window.controller.execute(select_candidate(None))
    window.plot_tabs.setCurrentIndex(0)
    pump(1.0)
    save(window.plot_tabs, "design_multi_result")


def _graded() -> "MainWindow":
    if "graded" in _CACHE:
        return _CACHE["graded"]
    window = _window(_wizard("graded_power_law", "W/Si", "Depth-graded W/Si supermirror"))
    spin = window.coating_page.roughness_spin
    spin.setValue(0.3)
    spin.editingFinished.emit()
    execute = window.controller.execute
    execute(
        set_targets(
            TargetSet(
                EnergyScan(angle=deg(0.2), energy_min=kev(15), energy_max=kev(70), points=150)
            )
        )
    )
    execute(set_parameter("N", value=100, min=20, max=300))
    execute(set_parameter("sigma_H", value=0.35))
    execute(set_parameter("sigma_L", value=0.35))
    execute(add_constraint(MaxCoatingTime(24 * 3600.0)))
    execute(set_process(DepositionProcess({"W": 0.1, "Si": 0.2}, 20.0, fixed_overhead=1800.0)))
    execute(set_merit(TimePenalized(MeanReflectivity(), weight=0.2, reference_time=8 * 3600.0)))
    pump(1.0)
    _CACHE["graded"] = window
    return window


@screenshot
def design_graded_coating() -> None:
    """Coating step of a depth-graded multilayer."""
    window = _graded()
    _step(window, "Coating")
    save(window.step_tabs, "design_graded_coating")


@screenshot
def design_graded_optimize() -> None:
    """Optimize step with the coating-time penalty and the resulting design."""
    window = _graded()
    _step(window, "Optimize")
    _budget(window, 4, 6, 40, 150)
    window.optimize_page.run()
    _wait(window)
    window.plot_tabs.setCurrentIndex(1)
    _show_layers(window)
    save(window.step_tabs, "design_graded_optimize")
    save(window.plot_tabs, "design_graded_layers")


@screenshot
def design_graded_tradeoff() -> None:
    """Trade-off step and plot."""
    window = _graded()
    _step(window, "Trade-off")
    page = window.tradeoff_page
    page.count.setValue(4)
    page.lowest.setValue(30)
    page.run()
    _wait(window)
    save(window, "design_graded_tradeoff")
    save(window.tradeoff_plot, "design_graded_tradeoff_plot")


def _combined() -> "MainWindow":
    """Window with the combined coating of the guide topic."""
    from grazing.core.materials import get
    from grazing.core.structure import Composite, Mirror, Periodic
    from grazing.core.structure.coatings import DepthGraded, NLayer
    from grazing.design.project.commands import set_mirror

    window = _window(_wizard("periodic", "W/Si", "Combined Pt/C and W/Si mirror"))
    coating = Composite.create(
        [
            DepthGraded.create(get("Pt"), get("C"), N=10, d_min=3.0, d_max=6.0, name="ptc"),
            NLayer.create(
                [get(m) for m in ("B4C", "Mo", "Si", "Ru", "C")],
                [2.0, 3.0, 4.0, 2.0, 3.0],
                name="mid",
            ),
            Periodic.create(get("W"), get("Si"), N=20, d=4.0, name="wsi"),
        ]
    )
    window.controller.execute(set_mirror(Mirror(coating, get("Si"), substrate_roughness=0.3)))
    pump(0.8)
    return window


@screenshot
def design_combined_coating() -> None:
    """Coating step with a combined coating and the periodic block selected."""
    window = _combined()
    _step(window, "Coating")
    window.coating_page.select_block("wsi")
    pump(0.5)
    save(window.step_tabs, "design_combined_coating")


@screenshot
def design_combined_materials() -> None:
    """Materials step with the roles grouped by block."""
    window = _combined()
    _step(window, "Materials")
    save(window.step_tabs, "design_combined_materials")
