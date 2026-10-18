# `xstack` philosophy

xstack shall be a software and library that can be used to design x-ray multi-layer mirrors, both with fixed d-spacing and depth graded designs

xstack consists of a shared core library and two applications: a design tool (`xstack-design`) and a fitting tool (`xstack-fit`) that fits measured reflectivity curves with the same physics and structure models.

Every feature shall be accessible both as a library and in the GUI.

The goal is to step-by-step optimize the design of a multi-layer, taking manufacturability into account.

The design shall happen in several stages:
 - First the user provides the main goal, either a target energy at a fixed angle or a target angular range at a fixed energy for which the ML shall be optimized
 - Initially, simple assumptions should be used to obtain a rough outline of the design space. This should already include basic manufacturing constraints such as minimum layer thickness and maximum overall thickness
 - Optionally enter more manufacturing constraints such as deposition speed per layer and maximum overall coating time. This should also include a overhead duration per individual layer independent of the thickness (think about something like pre-sputtering).
 - a rough scan of the design space shall be performed looking at various gamma values to exclude regions that are unvafourable
 - Perform detailed optimization with all constraints. The metric to optimize for shall be configurable to also include the maufacturing time as a penalty (with a configurable weighting fraction between 0 and 1).
 
apply appropriate effort at the various stageseE.g. initally, a simple approximation of the reflected energy based on Braggs law without any shift due to refractive index is used to find a rough solution, and then this is later improved by actually solving the proper fresnel equations. This should also be combined with approriate optimizer strategies.

Part of the software shall be the reflectivity calculation itself (e.g. mased in transfer matrix formalism) using publically available databases (e.g. CXRO).

Depth-graded multi-layer shall also have the option to add a cap layer pair with a thickness that does not match the remaining power-law for the d-spacing and gamma. It shall also be possible to define other types of x-ray coatings, e.g. simple coating with N different materials (no multilayer) or periodic multilayers.

As part of the initial inputs, various options of the design coating shall be able to either be fixed or to define ranges to these parameters. (Eg fix the power law coefficient to 1 or define the number of bi-layer to be between X and Y)

The software shall be written in python (targetting 3.12) and employ type hints.
The GUI shall be written in PySide6. Calculations shall be done using numpy. lmfit shall be used for optimization. Various optimization methods shall be possible to use, standard minimizers such as lmfit but also alternatives like differential evolution, depending on what makes more sense each stage. To some extent, it should also be modifiable by the user.

On top of this, it shall also be possible to perform a coupled optimization of several coatings. For this, instead of defining one main goal, several inputs are given. E.g. fixed target energy range but 5 different fixed angles. Each angle input is given a relative weight. The overall reflectivity is then the weighted average of each of the fixed angles. This should also work for a fixed energy and several angle ranges.

The GUI shall support saving and loading a project and keep a journal of actions and a history to support undo and redo.

The fitting tool shall fit a model of the coating (the same coatings, materials and parameters as in the design) to measured reflectivity curves, including instrument parameters such as intensity scale, background, angle offset and resolution. Several datasets (e.g. different energies) shall be fitted simultaneously with shared parameters. The result shall contain parameter uncertainties and correlations. The design tool and the fitting tool shall be installable independently of each other, and they shall be usable from the command line as well as from their GUIs.
