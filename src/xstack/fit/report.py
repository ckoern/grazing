"""Text reports of fit parameters and results.

Angles are shown in degrees and beam and sample dimensions in mm; all other values in their
internal units.
"""

import math

from xstack.core.parameters import Parameter, ParameterSet
from xstack.core.units import to_deg
from xstack.fit.results import FitResult

__all__ = ["display_value", "format_parameters", "format_result"]

_MM_PER_NM = 1e-6


def display_value(parameter: Parameter, value: float | None = None) -> tuple[float, str]:
    """Return a parameter value in the unit used for display.

    Args:
        parameter: The parameter.
        value: Value to convert, or ``None`` for the parameter's value.

    Returns:
        The converted value and its unit.
    """
    v = parameter.value if value is None else value
    if parameter.unit == "rad":
        return to_deg(v), "deg"
    if parameter.name.rsplit(".", 1)[-1] in ("beam_width", "sample_length"):
        return v * _MM_PER_NM, "mm"
    return v, parameter.unit


def _state(p: Parameter) -> str:
    if p.expr is not None:
        return f"= {p.expr}"
    return "free" if p.is_free else "fixed"


def format_parameters(parameters: ParameterSet, result: FitResult | None = None) -> str:
    """Format a parameter table with uncertainties from a fit result.

    Args:
        parameters: The parameters.
        result: Fit result providing uncertainties of free parameters, or ``None``.

    Returns:
        One line per parameter: name, value ± uncertainty, unit and state.
    """
    stats = result.statistics if result is not None else None
    width = max((len(n) for n in parameters), default=0)
    lines = []
    for name, p in parameters.items():
        value, unit = display_value(p)
        text = f"{value:.6g}"
        if stats is not None and name in stats.names:
            error = stats.error(name)
            if math.isfinite(error):
                text += f" ± {abs(display_value(p, error)[0]):.2g}"
            else:
                text += " ± (undetermined)"
        lines.append(f"{name:<{width}}  {text:<24} {unit:<4} {_state(p)}")
    return "\n".join(lines)


def format_result(result: FitResult, correlation_threshold: float = 0.8) -> str:
    """Format the summary of a fit result.

    Args:
        result: The fit result.
        correlation_threshold: Minimum absolute correlation of the pairs listed.

    Returns:
        Method, goodness of fit, the parameter table and strongly correlated pairs.
    """
    lines = [
        f"method: {result.method}, {result.evaluations} evaluations, {result.elapsed:.1f} s"
        + (" (cancelled)" if result.cancelled else "")
    ]
    stats = result.statistics
    if stats is None:
        lines.append(f"no uncertainties: {result.note}")
    else:
        label = "χ²_red" if stats.weighting == "chi2" else "objective per degree of freedom"
        lines.append(
            f"weighting: {stats.weighting}, points: {stats.points}, free: {len(stats.names)}, "
            f"{label}: {stats.reduced_objective:.4g}"
        )
        if not stats.determined:
            lines.append("warning: the data do not determine all free parameters")
    lines.append("")
    lines.append(format_parameters(result.parameters, result))
    if stats is not None:
        pairs = stats.strong_correlations(correlation_threshold)
        if pairs:
            lines.append("")
            lines.append(f"correlations above {correlation_threshold:g}:")
            lines.extend(f"  {a} / {b}: {c:+.3f}" for a, b, c in pairs)
    return "\n".join(lines)
