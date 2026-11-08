"""Command-line commands of the fit tool.

Usage::

    xstack fit new OUTPUT.xfit --model MODEL_FILE --data DATA_FILE [--data DATA_FILE ...]
        [--candidate NAME] [--energy KEV] [--angle DEG] [--quantity angle|two_theta|q|energy]
        [--unit UNIT] [--columns X,R[,SIGMA]] [--percent | --scale FACTOR]
        [--counts [--full-scale COUNTS]] [--range LO:HI ...]
        [--weighting log|linear|chi2] [--name NAME]
    xstack fit run PROJECT.xfit [--output OUT.xfit] [--method METHOD] [--max-evaluations N]
    xstack fit info PROJECT.xfit
    xstack fit export PROJECT.xfit OUTPUT.csv
"""

import argparse
import csv
import re
from pathlib import Path
from typing import cast, get_args

from xstack.core.io.data import read_curves
from xstack.core.io.delimited import UNITS, CsvLayout, Quantity
from xstack.core.measurement import MeasuredCurve
from xstack.core.optim.events import ProgressEvent
from xstack.core.optim.optimizers import LmfitOptimizer
from xstack.core.units import deg, kev, to_deg, to_kev
from xstack.fit.dataset import Dataset, Interval
from xstack.fit.problem import Weighting
from xstack.fit.project import (
    FitProject,
    FitState,
    add_dataset,
    import_model,
    rename_project,
    run_fit,
    set_optimizer,
    set_ranges,
    set_weighting,
    store_result,
)
from xstack.fit.report import format_parameters, format_result

__all__ = ["add_commands"]

_EXPORT_COLUMNS = (
    "dataset",
    "axis",
    "x",
    "x_unit",
    "measured",
    "uncertainty",
    "model",
    "fitted",
)


def _print_progress(event: ProgressEvent) -> None:
    total = f"/{event.total}" if event.total else ""
    print(
        f"  {event.task}: {event.evaluations}{total} evaluations, "
        f"best {event.best_objective:.5g} ({event.elapsed:.0f} s)",
        flush=True,
    )


def _column(text: str) -> int | str:
    text = text.strip()
    return int(text) if text.lstrip("-").isdigit() else text


def _layout(args: argparse.Namespace) -> CsvLayout | None:
    """Build the file layout from the command line, or ``None`` to detect it."""
    options = (args.quantity, args.unit, args.columns, args.percent, args.scale, args.counts)
    if not any(options) and args.full_scale is None:
        return None
    columns = [_column(c) for c in args.columns.split(",")] if args.columns else [0, 1]
    if not 2 <= len(columns) <= 3:
        raise ValueError("--columns takes two or three columns: X,R[,SIGMA]")
    if args.percent and args.scale is not None:
        raise ValueError("--percent and --scale cannot be combined")
    if args.full_scale is not None and not args.counts:
        raise ValueError("--full-scale requires --counts")
    scale = 0.01 if args.percent else 1.0 if args.scale is None else args.scale
    return CsvLayout(
        x_column=columns[0],
        reflectivity_column=columns[1],
        uncertainty_column=columns[2] if len(columns) == 3 else None,
        quantity=cast(Quantity, args.quantity or "angle"),
        unit=args.unit,
        reflectivity_scale=scale,
        energy=None if args.energy is None else kev(args.energy),
        angle=None if args.angle is None else deg(args.angle),
        counts=args.counts,
        full_scale=args.full_scale,
    )


def _with_fixed(curve: MeasuredCurve, args: argparse.Namespace) -> MeasuredCurve:
    if curve.fixed is not None:
        return curve
    if curve.axis == "angle" and args.energy is not None:
        return curve.with_fixed(kev(args.energy))
    if curve.axis == "energy" and args.angle is not None:
        return curve.with_fixed(deg(args.angle))
    return curve


def _identifier(name: str, used: set[str]) -> str:
    base = re.sub(r"\W", "_", name) or "data"
    if base[0].isdigit():
        base = f"d_{base}"
    unique, n = base, 1
    while unique in used:
        n += 1
        unique = f"{base}_{n}"
    used.add(unique)
    return unique


def _range(text: str) -> tuple[float, float]:
    lo, sep, hi = text.partition(":")
    try:
        if not sep:
            raise ValueError
        return float(lo), float(hi)
    except ValueError:
        raise ValueError(f"invalid range {text!r}: expected LO:HI") from None


def _to_internal(axis: str, bounds: tuple[float, float]) -> Interval:
    convert = deg if axis == "angle" else kev
    return convert(bounds[0]), convert(bounds[1])


def _display(axis: str, interval: Interval) -> str:
    convert, unit = (to_deg, "deg") if axis == "angle" else (to_kev, "keV")
    return f"{convert(interval[0]):.6g}:{convert(interval[1]):.6g} {unit}"


def _fixed_text(curve: MeasuredCurve) -> str:
    if curve.fixed is None:
        return "not set"
    if curve.axis == "angle":
        return f"energy {to_kev(curve.fixed):.6g} keV"
    return f"angle {to_deg(curve.fixed):.6g} deg"


def _new(args: argparse.Namespace) -> int:
    ranges = [_range(r) for r in args.range or ()]
    layout = _layout(args)
    project = FitProject()
    project.execute(import_model(args.model, args.candidate))
    project.execute(set_weighting(args.weighting))
    used: set[str] = set()
    for path in args.data:
        for curve in read_curves(path, layout):
            curve = _with_fixed(curve, args)
            dataset = Dataset.create(_identifier(curve.name or Path(path).stem, used), curve)
            project.execute(add_dataset(dataset))
            if ranges:
                intervals = [_to_internal(curve.axis, r) for r in ranges]
                project.execute(set_ranges(dataset.name, intervals))
    if args.name:
        project.execute(rename_project(args.name))
    project.save(args.output)
    state = project.state
    points = sum(d.curve.size for d in state.datasets)
    print(f"created {args.output}: {len(state.datasets)} datasets, {points} points")
    print(f"model: {state.model_source}")
    print(format_parameters(state.problem().parameters))
    return 0


def _run(args: argparse.Namespace) -> int:
    project = FitProject.load(args.project)
    if args.method or args.max_evaluations is not None:
        current = project.state.optimizers[-1]
        method = args.method or (current.method if isinstance(current, LmfitOptimizer) else None)
        project.execute(
            set_optimizer(LmfitOptimizer(method or "least_squares", args.max_evaluations))
        )
    result = run_fit(project.state, events=_print_progress if args.verbose else None)
    project.execute(store_result(result))
    print(format_result(result))
    project.save(args.output or args.project)
    return 0


def _info(args: argparse.Namespace) -> int:
    state = FitProject.load(args.project).state
    print(f"project {state.name!r}")
    print(f"model source: {state.model_source or 'none'}")
    print(f"model: {state.model.name}")
    print(f"weighting: {state.weighting}")
    print(f"datasets: {len(state.datasets)}")
    for d in state.datasets:
        c = d.curve
        ranges = ", ".join(_display(c.axis, r) for r in d.ranges) or "all"
        flags = "" if d.enabled else " (disabled)"
        print(
            f"  {d.name}{flags}: {c.axis} scan, {c.size} points, "
            f"{int(d.mask().sum())} fitted, ranges {ranges}, {_fixed_text(c)}"
        )
    if state.mirror is not None and state.datasets:
        print(format_parameters(state.problem().parameters, state.result))
    elif state.mirror is not None:
        print(format_parameters(state.mirror.parameters))
    if state.result is not None:
        print()
        print(format_result(state.result))
    return 0


def _export(args: argparse.Namespace) -> int:
    state: FitState = FitProject.load(args.project).state
    problem = state.problem()
    rows = 0
    with Path(args.output).open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(_EXPORT_COLUMNS)
        for d in state.datasets:
            c = d.curve
            model = problem.simulate(d)
            fitted = problem.fit_mask(d)
            convert, unit = (to_deg, "deg") if c.axis == "angle" else (to_kev, "keV")
            for i in range(c.size):
                sigma = "" if c.uncertainty is None else repr(c.uncertainty[i])
                writer.writerow(
                    [
                        d.name,
                        c.axis,
                        repr(float(convert(c.x[i]))),
                        unit,
                        repr(c.reflectivity[i]),
                        sigma,
                        repr(float(model[i])),
                        int(fitted[i]),
                    ]
                )
                rows += 1
    print(f"wrote {rows} rows to {args.output}")
    return 0


def add_commands(parser: argparse.ArgumentParser) -> None:
    """Add the fit commands as subcommands of a parser.

    Every command sets ``func``, a callable that takes the parsed arguments and returns the
    process exit code.

    Args:
        parser: The parser of the ``fit`` command.
    """
    sub = parser.add_subparsers(dest="fit_command", required=True)

    new = sub.add_parser("new", help="create a fit project from a model and measured data")
    new.add_argument("output", help="fit project file to write (.xfit)")
    new.add_argument("--model", required=True, help="file with the start model (e.g. .xstack)")
    new.add_argument("--candidate", help="name of the model in the file (default: first)")
    new.add_argument(
        "--data", action="append", required=True, help="measured data file (repeatable)"
    )
    new.add_argument("--energy", type=float, help="photon energy of angle scans in keV")
    new.add_argument("--angle", type=float, help="grazing angle of energy scans in deg")
    new.add_argument(
        "--quantity", choices=get_args(Quantity.__value__), help="quantity in the x column"
    )
    units = sorted({u for per_quantity in UNITS.values() for u in per_quantity})
    new.add_argument("--unit", help=f"unit of the x column ({', '.join(units)})")
    new.add_argument("--columns", help="columns X,R[,SIGMA] as 0-based indices or header names")
    new.add_argument("--percent", action="store_true", help="reflectivity is given in percent")
    new.add_argument(
        "--scale",
        type=float,
        metavar="FACTOR",
        help="factor applied to the reflectivity (with --counts: after the normalization)",
    )
    new.add_argument(
        "--counts",
        action="store_true",
        help="the R column holds detector counts; σ defaults to sqrt(max(N, 1))",
    )
    new.add_argument(
        "--full-scale",
        type=float,
        metavar="COUNTS",
        help="counts for reflectivity 1 with --counts (default: the largest count)",
    )
    new.add_argument(
        "--range",
        action="append",
        metavar="LO:HI",
        help="fit range in deg (angle scans) or keV (energy scans); repeatable",
    )
    new.add_argument(
        "--weighting",
        choices=get_args(Weighting.__value__),
        default="log",
        help="residual weighting",
    )
    new.add_argument("--name", help="project name")
    new.set_defaults(func=_new)

    run = sub.add_parser("run", help="fit the free parameters of a project")
    run.add_argument("project", help="project file")
    run.add_argument("--output", help="output project file (default: overwrite input)")
    run.add_argument("--method", help="lmfit method (default: the project's optimizer)")
    run.add_argument("--max-evaluations", type=int, help="maximum number of evaluations")
    run.add_argument("--verbose", "-v", action="store_true", help="print progress")
    run.set_defaults(func=_run)

    info = sub.add_parser("info", help="summarize a project")
    info.add_argument("project", help="project file")
    info.set_defaults(func=_info)

    export = sub.add_parser("export", help="write measured and modeled curves as CSV")
    export.add_argument("project", help="project file")
    export.add_argument("output", help="CSV file to write")
    export.set_defaults(func=_export)
