"""Command-line commands of the design tool.

Usage::

    grazing design run PROJECT.grazing [--output OUT.grazing] [--seed N]
    grazing design info PROJECT.grazing
"""

import argparse

from grazing.core.optim.events import ProgressEvent
from grazing.design.optim.pipeline import Pipeline
from grazing.design.project import Project
from grazing.design.project.commands import run_pipeline

__all__ = ["add_commands"]


def _print_progress(event: ProgressEvent) -> None:
    total = f"/{event.total}" if event.total else ""
    print(
        f"  {event.task}: {event.evaluations}{total} evaluations, "
        f"best {event.best_objective:.5g} ({event.elapsed:.0f} s)",
        flush=True,
    )


def _run(args: argparse.Namespace) -> int:
    project = Project.load(args.project)
    problem = project.state.problem()
    pipeline = project.state.pipeline
    if pipeline is None:
        pipeline = Pipeline.default(problem, seed=args.seed)
    result = run_pipeline(project, pipeline, events=_print_progress if args.verbose else None)
    for stage in result.stages:
        print(
            f"{stage.stage.name:>10s}: objective {stage.result.best.objective:.5g} "
            f"({stage.result.evaluations} evaluations, {stage.result.elapsed:.1f} s)"
        )
    final = result.final
    print(f"final objective {final.objective:.5g}, figure of merit {final.merit:.5g}")
    if final.coating_time is not None:
        print(f"coating time {final.coating_time / 3600:.3g} h")
    if final.stack is not None:
        print(f"{final.stack.num_layers} layers, {final.stack.total_thickness:.4g} nm")
    project.save(args.output or args.project)
    return 0


def _info(args: argparse.Namespace) -> int:
    project = Project.load(args.project)
    state = project.state
    print(f"project {state.name!r}")
    if state.mirror is not None:
        print(f"coating: {state.mirror.coating.type_name}, substrate {state.mirror.substrate.name}")
        for name, p in state.mirror.parameters.items():
            rng = f" in [{p.min:g}, {p.max:g}]" if p.vary else " (fixed)"
            expr = f" = {p.expr}" if p.expr else ""
            print(f"  {name} = {p.value:g} {p.unit}{expr}{rng if not p.expr else ''}")
    print(f"targets: {len(state.targets) if state.targets else 0}")
    print(f"constraints: {len(state.constraints)}")
    print(f"candidates: {', '.join(c.name for c in state.candidates) or 'none'}")
    print(f"journal entries: {len(project.journal)}")
    return 0


def add_commands(parser: argparse.ArgumentParser) -> None:
    """Add the design commands as subcommands of a parser.

    Every command sets ``func``, a callable that takes the parsed arguments and returns the
    process exit code.

    Args:
        parser: The parser of the ``design`` command.
    """
    sub = parser.add_subparsers(dest="design_command", required=True)
    run = sub.add_parser("run", help="run the design pipeline of a project")
    run.add_argument("project", help="project file")
    run.add_argument("--output", help="output project file (default: overwrite input)")
    run.add_argument("--seed", type=int, default=None, help="random seed for the default pipeline")
    run.add_argument("--verbose", "-v", action="store_true", help="print progress")
    run.set_defaults(func=_run)
    info = sub.add_parser("info", help="summarize a project")
    info.add_argument("project", help="project file")
    info.set_defaults(func=_info)
