"""Screenshots of the project wizard."""

from docs.guide.screenshots import pump, save, screenshot
from xstack.gui.common.wizard import ProjectWizard
from xstack.gui.fit.main_window import PROJECT_FILTER


@screenshot
def wizard_coating() -> None:
    """The coating page of the wizard with a depth-graded Pt/C multilayer."""
    wizard = ProjectWizard("fit project", ".xfit", PROJECT_FILTER)
    wizard.resize(700, 700)
    wizard.show()
    wizard.set_name("Pt/C supermirror")
    wizard.next()
    wizard.set_kind("graded_power_law")
    wizard.set_pair("Pt/C")
    pump(0.3)
    save(wizard, "wizard_coating")
    wizard.close()
