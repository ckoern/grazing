"""Monte Carlo analysis of the sensitivity of a design to manufacturing errors.

An `ErrorModel` describes the errors of a coating run: random, systematic and drifting
thickness errors, variations of the interface roughness, and mixed interlayers that form at
the interfaces between different materials. `tolerance_analysis` evaluates many randomly
perturbed copies of a design and returns the distribution of the figure of merit together
with the reflectivity of every sample.
"""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
import xraylib
from numpy.typing import NDArray

from grazing.core.materials.material import Material
from grazing.core.structure.stack import Layer, Stack

if TYPE_CHECKING:
    from grazing.core.optim.events import CancellationToken, EventSink
    from grazing.design.optim.problem import DesignProblem, Evaluation

__all__ = [
    "ErrorModel",
    "ToleranceResult",
    "mixed_material",
    "perturb_stack",
    "perturb_thicknesses",
    "thickness_tolerance",
    "tolerance_analysis",
]


@dataclass(frozen=True)
class ErrorModel:
    """Statistical model of the manufacturing errors of one coating run.

    All errors are drawn independently for every Monte Carlo sample. Standard deviations of
    zero disable the corresponding error.

    Attributes:
        thickness_random: Relative standard deviation of independent per-layer thickness
            errors.
        thickness_systematic: Relative standard deviation of a rate calibration error, one
            per material and run.
        thickness_drift: Relative standard deviation of a linear rate drift during the run;
            the drawn value is the relative thickness change of the last deposited layer, the
            first layer is unchanged.
        roughness_random: Standard deviation in nm of independent per-interface changes of
            the roughness.
        roughness_systematic: Standard deviation in nm of a change of the roughness common to
            all interfaces of a run.
        interlayer_thickness: Mean thickness in nm of the mixed layer that forms at every
            interface between two layers of different materials. Half of it is taken from
            each neighbouring layer; its optical constants are the volume average of both
            materials.
        interlayer_random: Relative standard deviation of the mixed-layer thickness per
            interface.
    """

    thickness_random: float = 0.0
    thickness_systematic: float = 0.0
    thickness_drift: float = 0.0
    roughness_random: float = 0.0
    roughness_systematic: float = 0.0
    interlayer_thickness: float = 0.0
    interlayer_random: float = 0.0

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if value < 0:
                raise ValueError(f"{name} must not be negative")

    @property
    def is_zero(self) -> bool:
        """Whether the model contains no errors at all."""
        return all(value == 0 for value in self.__dict__.values())


def mixed_material(first: Material, second: Material) -> Material:
    """Return the 50/50 volume mixture of two materials.

    The atom densities of the mixture are the averages of those of both materials, so its
    δ and β are the averages as well.

    Args:
        first: First material (not vacuum).
        second: Second material (not vacuum).

    Returns:
        The mixed material, named ``"<first>/<second> mix"``.

    Raises:
        ValueError: If one of the materials is vacuum.
    """
    if first.is_vacuum or second.is_vacuum:
        raise ValueError("vacuum cannot be mixed")
    densities: dict[int, float] = {}
    for material in (first, second):
        for z, n in material.atom_densities().items():
            densities[z] = densities.get(z, 0.0) + 0.5 * n
    formula = "".join(
        f"{xraylib.AtomicNumberToSymbol(z)}{n:.6g}" for z, n in sorted(densities.items())
    )
    density = 0.5 * (first.density + second.density)
    return Material(f"{first.name}/{second.name} mix", formula, density)


def _mix_interfaces(
    layers: list[Layer], thicknesses: NDArray[np.float64], cache: dict[tuple[str, str], Material]
) -> list[Layer]:
    """Insert mixed layers of the given thicknesses at interfaces between different layers."""
    shrink = np.zeros(len(layers))
    inserts: dict[int, Layer] = {}
    for i in range(len(layers) - 1):
        upper, lower = layers[i], layers[i + 1]
        t = float(thicknesses[i])
        if t <= 0 or upper.material == lower.material or upper.material.is_vacuum:
            continue
        t = min(t, upper.thickness, lower.thickness)
        key = (upper.material.name, lower.material.name)
        if key not in cache:
            cache[key] = mixed_material(upper.material, lower.material)
        shrink[i] += 0.5 * t
        shrink[i + 1] += 0.5 * t
        inserts[i] = Layer(cache[key], t, lower.roughness, f"{upper.tag}|{lower.tag}.mix")
    result: list[Layer] = []
    for i, layer in enumerate(layers):
        if shrink[i]:
            layer = replace(layer, thickness=max(0.0, layer.thickness - shrink[i]))
        result.append(layer)
        if i in inserts:
            result.append(inserts[i])
    return result


def perturb_stack(
    stack: Stack,
    rng: np.random.Generator,
    errors: ErrorModel,
    mixtures: dict[tuple[str, str], Material] | None = None,
) -> Stack:
    """Return a randomly perturbed copy of a stack according to an error model.

    Thickness errors are applied first, then roughness changes, then mixed interlayers.
    Negative thicknesses and roughnesses are set to zero.

    Args:
        stack: The stack.
        rng: Random number generator.
        errors: The error model.
        mixtures: Cache of mixed materials by (upper, lower) material name, shared between
            samples to avoid creating the same material repeatedly.

    Returns:
        The perturbed stack.
    """
    return _perturb(stack, rng, errors, mixtures)[1]


def _perturb(
    stack: Stack,
    rng: np.random.Generator,
    errors: ErrorModel,
    mixtures: dict[tuple[str, str], Material] | None = None,
) -> tuple[Stack, Stack]:
    """Return the perturbed stack as deposited and with mixed interlayers."""
    n = stack.num_layers
    factors = 1.0 + errors.thickness_random * rng.standard_normal(n)
    materials = sorted({layer.material.name for layer in stack.layers})
    rate = {m: 1.0 + errors.thickness_systematic * rng.standard_normal() for m in materials}
    drift = errors.thickness_drift * rng.standard_normal()
    # Deposition runs from the substrate (last layer) to the top (first layer).
    progress = np.linspace(1.0, 0.0, n) if n > 1 else np.zeros(n)
    sigma_common = errors.roughness_systematic * rng.standard_normal()
    sigma_random = errors.roughness_random * rng.standard_normal(n + 1)
    layers = [
        replace(
            layer,
            thickness=max(
                0.0,
                layer.thickness
                * factors[i]
                * rate[layer.material.name]
                * (1.0 + drift * progress[i]),
            ),
            roughness=max(0.0, layer.roughness + sigma_common + sigma_random[i]),
        )
        for i, layer in enumerate(stack.layers)
    ]
    substrate_roughness = max(0.0, stack.substrate_roughness + sigma_common + sigma_random[n])
    deposited = Stack(layers, stack.substrate, substrate_roughness, stack.ambient)
    if errors.interlayer_thickness > 0 and n > 1:
        mix = errors.interlayer_thickness * (
            1.0 + errors.interlayer_random * rng.standard_normal(n - 1)
        )
        layers = _mix_interfaces(layers, np.maximum(mix, 0.0), {} if mixtures is None else mixtures)
        return deposited, Stack(layers, stack.substrate, substrate_roughness, stack.ambient)
    return deposited, deposited


def perturb_thicknesses(
    stack: Stack,
    rng: np.random.Generator,
    random_error: float = 0.01,
    systematic_error: float = 0.0,
) -> Stack:
    """Return a copy of a stack with randomly perturbed layer thicknesses.

    Every thickness t is replaced by t·(1 + σ_r·ξ)·(1 + σ_s·η) with independent standard normal
    ξ per layer and one η per material and stack (a rate calibration error). Negative results
    are set to zero.

    Args:
        stack: The stack.
        rng: Random number generator.
        random_error: Relative standard deviation σ_r of independent per-layer errors.
        systematic_error: Relative standard deviation σ_s of the per-material rate error.

    Returns:
        The perturbed stack.
    """
    materials = {layer.material.name for layer in stack.layers}
    rate_error = {m: 1.0 + systematic_error * rng.standard_normal() for m in sorted(materials)}
    factors = 1.0 + random_error * rng.standard_normal(stack.num_layers)
    layers = [
        replace(layer, thickness=max(0.0, layer.thickness * f * rate_error[layer.material.name]))
        for layer, f in zip(stack.layers, factors, strict=True)
    ]
    return Stack(layers, stack.substrate, stack.substrate_roughness, stack.ambient)


@dataclass(frozen=True)
class ToleranceResult:
    """Distribution of the figure of merit under random manufacturing errors.

    Attributes:
        nominal: Evaluation of the unperturbed design.
        merits: Figure of merit of each perturbed sample.
        objectives: Objective (including penalties) of each perturbed sample.
        feasible: Whether each perturbed sample satisfies all hard constraints.
        samples: Evaluations of the perturbed samples; their ``results`` hold the
            reflectivity curves on the target grids.
        errors: The error model used.
    """

    nominal: "Evaluation"
    merits: NDArray[np.float64]
    objectives: NDArray[np.float64]
    feasible: NDArray[np.bool_]
    samples: tuple["Evaluation", ...]
    errors: ErrorModel = ErrorModel()

    def percentile(self, q: float) -> float:
        """Return a percentile of the figure of merit of the perturbed samples.

        Args:
            q: Percentile in [0, 100].

        Returns:
            The percentile.
        """
        return float(np.percentile(self.merits, q))


def tolerance_analysis(
    problem: "DesignProblem",
    values: Mapping[str, float] | None = None,
    errors: ErrorModel | None = None,
    samples: int = 100,
    seed: int | None = None,
    cancel: "CancellationToken | None" = None,
    events: "EventSink | None" = None,
) -> ToleranceResult:
    """Evaluate a design for randomly perturbed copies according to an error model.

    The figure of merit is evaluated on the physical stack (including mixed interlayers);
    coating time and constraints are evaluated on the stack as deposited.

    Args:
        problem: The design problem.
        values: Parameter values of the design. Defaults to the problem's current values.
        errors: The error model. Defaults to 1 % random thickness errors.
        samples: Number of perturbed samples.
        seed: Random seed.
        cancel: Cancellation token; stops after the current sample.
        events: Receiver of progress events (task ``"tolerance"``); their ``best_values`` are
            empty because the samples are not parameterized designs.

    Returns:
        The distribution of the figure of merit and the evaluated samples.

    Raises:
        ValueError: If the nominal design cannot be built.
    """
    from grazing.core.optim.events import ProgressEvent

    model = errors if errors is not None else ErrorModel(thickness_random=0.01)
    nominal = problem.evaluate(values)
    if nominal.stack is None:
        raise ValueError(f"design cannot be built: {nominal.error}")
    rng = np.random.default_rng(seed)
    mixtures: dict[tuple[str, str], Material] = {}
    evaluations: list[Evaluation] = []
    for i in range(samples):
        if cancel is not None and cancel.cancelled:
            break
        deposited, stack = _perturb(nominal.stack, rng, model, mixtures)
        evaluation = problem.evaluate_stack(stack, nominal.values, deposited)
        evaluations.append(evaluation)
        if events is not None:
            best = min(evaluations, key=lambda e: e.objective)
            events(
                ProgressEvent(
                    "tolerance",
                    i + 1,
                    samples,
                    best.objective,
                    evaluation.objective,
                    {},
                    0.0,
                    best.merit,
                )
            )
    return ToleranceResult(
        nominal,
        np.array([e.merit for e in evaluations]),
        np.array([e.objective for e in evaluations]),
        np.array([e.feasible for e in evaluations], dtype=bool),
        tuple(evaluations),
        model,
    )


def thickness_tolerance(
    problem: "DesignProblem",
    values: Mapping[str, float] | None = None,
    random_error: float = 0.01,
    systematic_error: float = 0.0,
    samples: int = 100,
    seed: int | None = None,
    cancel: "CancellationToken | None" = None,
) -> ToleranceResult:
    """Evaluate a design for randomly perturbed layer thicknesses.

    This is `tolerance_analysis` with an error model containing only thickness errors.

    Args:
        problem: The design problem.
        values: Parameter values of the design. Defaults to the problem's current values.
        random_error: Relative standard deviation of independent per-layer errors.
        systematic_error: Relative standard deviation of per-material rate errors.
        samples: Number of perturbed samples.
        seed: Random seed.
        cancel: Cancellation token; stops after the current sample.

    Returns:
        The distribution of the figure of merit.

    Raises:
        ValueError: If the nominal design cannot be built.
    """
    errors = ErrorModel(thickness_random=random_error, thickness_systematic=systematic_error)
    return tolerance_analysis(problem, values, errors, samples, seed, cancel)
