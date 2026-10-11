"""Command-line interface.

Usage::

    python -m xstack run PROJECT.xstack [--output OUT.xstack] [--seed N]
    python -m xstack info PROJECT.xstack
"""

import argparse
import sys
from collections.abc import Sequence

from xstack.optim.events import ProgressEvent
from xstack.optim.pipeline import Pipeline
from xstack.project import Project
from xstack.project.commands import run_pipeline

__all__ = ["main"]


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


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface.

    Args:
        argv: Command-line arguments without the program name. Defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(prog="xstack", description="X-ray multilayer design")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run the design pipeline of a project")
    run.add_argument("project", help="project file")
    run.add_argument("--output", help="output project file (default: overwrite input)")
    run.add_argument("--seed", type=int, default=None, help="random seed for the default pipeline")
    run.add_argument("--verbose", "-v", action="store_true", help="print progress")
    run.set_defaults(func=_run)
    info = sub.add_parser("info", help="summarize a project")
    info.add_argument("project", help="project file")
    info.set_defaults(func=_info)
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
