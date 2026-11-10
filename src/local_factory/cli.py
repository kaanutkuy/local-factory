"""Command-line entry point. Subcommands are added phase by phase (see docs/PLAN.md §14)."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from local_factory import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="factory",
        description="local-factory: intake -> verified build -> evidence-bound release.",
    )
    parser.add_argument("--version", action="version", version=f"local-factory {__version__}")
    parser.add_subparsers(dest="command", metavar="<command>")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    return 0  # pragma: no cover - no subcommands registered yet


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
