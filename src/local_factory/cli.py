"""Command-line entry point: ``factory`` (docs/specs/phase-2b.md § cli.py; docs/PLAN.md §14).

``factory project add`` adopts a repository, ``factory intake`` turns a request into a proposed
contract, ``questions`` / ``answer`` carry the conversation with the ambiguity critic, ``show``
prints a ticket and ``approve`` freezes its contract. Exit codes: 0 success (a ticket waiting for
answers is a success), 1 an operational failure or refusal (``error: ...`` on stderr), 2 usage
(argparse).

Only the commands that run models enter a site (the box); the rest read and write the ledger.
The API key is never read here: the SDK finds it in the environment of the process.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shlex
import sys
from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from local_factory import __version__
from local_factory.agents import AgentBackend
from local_factory.config import ConfigError, FactoryConfig, load_factory_config
from local_factory.gitops import BareRepo, GitError
from local_factory.hashing import utc_now
from local_factory.ledger import Ledger, LedgerError
from local_factory.models import HoldoutProbe, Ticket, TicketStatus
from local_factory.oracle import HoldoutError, HoldoutStore
from local_factory.pipeline.flow import Site
from local_factory.pipeline.intake import (
    HUMAN,
    Intake,
    IntakeError,
    IntakeOutcome,
    answer,
    approve,
    open_questions,
    questions,
    receive,
    run_intake,
)
from local_factory.pipeline.sandboxed_slice import box_site
from local_factory.projects import (
    ProjectError,
    ProjectPaths,
    add_project,
    load_project,
    project_config,
    project_paths,
)
from local_factory.sandbox import Sandbox
from local_factory.sandbox.runtime import DockerRuntime, SandboxError

__all__ = ["Services", "build_parser", "default_services", "main"]

# What a command reports as `error: ...` (exit 1) instead of a traceback.
_FAILURES = (
    ProjectError,
    IntakeError,
    LedgerError,
    ConfigError,
    HoldoutError,
    SandboxError,
    GitError,
)
_TEXT_COMMANDS = frozenset({"intake", "answer"})  # the commands that take free text


@dataclass(frozen=True)
class Services:
    """What the CLI needs from the outside world: the model backend and the place work runs."""

    backend: Callable[[FactoryConfig, Path], AgentBackend]  # (config, transcripts directory)
    site: Callable[[FactoryConfig], AbstractContextManager[Site]]
    clock: Callable[[], datetime] = utc_now


def default_services() -> Services:
    """The real thing: the Claude API backend, and workers in the Docker box."""
    return Services(backend=_claude_backend, site=_box_site)


def _claude_backend(config: FactoryConfig, transcripts: Path) -> AgentBackend:
    from local_factory.agents import ClaudeAPIBackend  # imports the SDK: only when a model runs

    return ClaudeAPIBackend(config, transcript_dir=transcripts)


@contextmanager
def _box_site(config: FactoryConfig) -> Iterator[Site]:
    runtime = DockerRuntime(docker=config.sandbox.docker)
    try:
        yield box_site(Sandbox(runtime, config.sandbox))
    finally:
        runtime.cleanup()


# --------------------------------------------------------------------------- the parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="factory",
        description="local-factory: intake -> verified build -> evidence-bound release.",
        allow_abbrev=False,
    )
    parser.add_argument("--version", action="version", version=f"local-factory {__version__}")
    commands = parser.add_subparsers(dest="command", metavar="<command>")
    scoped = argparse.ArgumentParser(add_help=False)
    scoped.add_argument("--project", metavar="NAME", help="the project (when several exist)")

    project = commands.add_parser("project", help="manage projects", allow_abbrev=False)
    actions = project.add_subparsers(dest="action", metavar="<action>", required=True)
    add = actions.add_parser("add", help="adopt a git repository", allow_abbrev=False)
    add.add_argument("path", metavar="PATH", help="the repository to adopt")
    add.add_argument("--name", help="the project's name (default: [project] name)")
    add.add_argument("--branch", default="main", help="the branch to adopt (default: main)")

    intake = commands.add_parser(
        "intake", parents=[scoped], help="turn a request into a contract", allow_abbrev=False
    )
    intake.add_argument("text", nargs="?", metavar="TEXT", help="the request")
    intake.add_argument("--resume", metavar="TICKET", help="continue a ticket instead")

    asked = commands.add_parser(
        "questions", parents=[scoped], help="list a ticket's questions", allow_abbrev=False
    )
    asked.add_argument("ticket", metavar="TICKET")

    reply = commands.add_parser(
        "answer", parents=[scoped], help="answer one question", allow_abbrev=False
    )
    reply.add_argument("ticket", metavar="TICKET")
    reply.add_argument("question", metavar="QUESTION")
    reply.add_argument("text", nargs="?", metavar="TEXT", help="the answer")

    show = commands.add_parser(
        "show", parents=[scoped], help="show a ticket and its contract", allow_abbrev=False
    )
    show.add_argument("ticket", metavar="TICKET")
    show.add_argument("--holdout", action="store_true", help="also print the sealed probes")

    approving = commands.add_parser(
        "approve", parents=[scoped], help="freeze a proposed contract", allow_abbrev=False
    )
    approving.add_argument("ticket", metavar="TICKET")
    return parser


def _settle(parser: argparse.ArgumentParser, args: argparse.Namespace, extra: list[str]) -> None:
    """Finish parsing. argparse hands back a word that starts with a dash as unrecognized, and a
    free-text argument may well start with one (``factory answer T-0001 Q-1 --shout``): a single
    such word is the text, when the command takes one and has not got it yet."""
    if args.command in _TEXT_COMMANDS and args.text is None and len(extra) == 1:
        args.text = extra.pop()
    if extra:
        parser.error("unrecognized arguments: " + " ".join(extra))
    if args.command == "answer" and args.text is None:
        parser.error("the following arguments are required: TEXT")
    if args.command == "intake" and (args.text is None) == (args.resume is None):
        parser.error("give either TEXT or --resume TICKET")


# --------------------------------------------------------------------------- main


def main(argv: Sequence[str] | None = None, *, services: Services | None = None) -> int:
    parser = build_parser()
    args, extra = parser.parse_known_args(argv)
    if args.command is None:
        if extra:
            parser.error("unrecognized arguments: " + " ".join(extra))
        parser.print_help()
        return 0
    _settle(parser, args, extra)
    try:
        return _COMMANDS[args.command](
            _Context(load_factory_config(), services or default_services()), args
        )
    except _FAILURES as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


@dataclass(frozen=True)
class _Context:
    config: FactoryConfig
    services: Services


@dataclass(frozen=True)
class _Open:
    """A project's ledger and where the rest of its state lives."""

    ledger: Ledger
    paths: ProjectPaths


@contextmanager
def _open(ctx: _Context, name: str | None) -> Iterator[_Open]:
    project = load_project(ctx.config, name)
    paths = project_paths(ctx.config, project.name)
    with Ledger(paths.ledger, clock=ctx.services.clock) as ledger:
        yield _Open(ledger=ledger, paths=paths)


def _flag(args: argparse.Namespace) -> str:
    """``--project NAME`` for the commands this one tells the user to run next."""
    return f" --project {shlex.quote(args.project)}" if args.project else ""


# --------------------------------------------------------------------------- project add


def _project_add(ctx: _Context, args: argparse.Namespace) -> int:
    project = add_project(
        ctx.config,
        Path(args.path),
        name=args.name,
        branch=args.branch,
        clock=ctx.services.clock,
    )
    print(f"project {project.name} added")
    return 0


# --------------------------------------------------------------------------- intake and answer


def _intake(ctx: _Context, args: argparse.Namespace) -> int:
    with _open(ctx, args.project) as project:
        if args.resume is not None:
            if project.ledger.get_ticket(args.resume).status is TicketStatus.HALTED:
                project.ledger.resume(args.resume, actor=HUMAN)
            return _run(ctx, project, args, args.resume)
        ticket = receive(project.ledger, args.text)
        try:
            return _run(ctx, project, args, ticket.id)
        except _FAILURES:
            print(
                f"{ticket.id} was recorded; retry with: factory intake --resume {ticket.id}"
                f"{_flag(args)}",
                file=sys.stderr,
            )
            raise


def _run(ctx: _Context, project: _Open, args: argparse.Namespace, ticket_id: str) -> int:
    """Run intake on a ticket, in the site, and report where it ended."""
    paths = project.paths
    with ctx.services.site(ctx.config) as site:
        intake = Intake(
            site=site,
            ledger=project.ledger,
            repo=BareRepo.open(paths.repo),
            project=project_config(paths),
            backend=ctx.services.backend(ctx.config, paths.transcripts),
            holdout=HoldoutStore(paths.holdout),
            work_root=paths.work,
            clock=ctx.services.clock,
            factory_config=ctx.config,
        )
        outcome = asyncio.run(run_intake(intake, ticket_id))
    return _report(outcome, args, project.ledger)


def _report(outcome: IntakeOutcome, args: argparse.Namespace, ledger: Ledger) -> int:
    ticket_id = outcome.ticket.id
    print(f"{ticket_id}: {outcome.status}")
    if outcome.status == "needs_info":
        for question in outcome.questions:
            print(f"{question.id}: {question.text}")
            if question.why_blocking:
                print(f"    why it blocks: {question.why_blocking}")
        print(f'Answer with: factory answer {ticket_id} QUESTION "TEXT"{_flag(args)}')
        return 0
    if outcome.status == "contract_proposed":
        print(f"Review with: factory show {ticket_id} [--holdout]{_flag(args)}")
        print(f"Then approve with: factory approve {ticket_id}{_flag(args)}")
        return 0
    print(f"error: {_failure(outcome, ledger)}", file=sys.stderr)
    if outcome.status == "halted":
        print(f"Resume with: factory intake --resume {ticket_id}{_flag(args)}")
    return 1


def _failure(outcome: IntakeOutcome, ledger: Ledger) -> str:
    """Why intake stopped; for a role's error, which role (its ``intake_error`` event knows)."""
    if outcome.status != "agent_error":
        return outcome.error or ""
    failed = [e for e in ledger.events(outcome.ticket.id) if e.kind == "intake_error"]
    return f"{failed[-1].data['role']}: {outcome.error}"


def _answer(ctx: _Context, args: argparse.Namespace) -> int:
    with _open(ctx, args.project) as project:
        replied = answer(project.ledger, args.ticket, args.question, args.text)
        print(f"{replied.id} answered")
        waiting = open_questions(project.ledger, args.ticket)
        if waiting:
            print(f"{len(waiting)} question(s) still open: factory questions {args.ticket}")
            return 0
        return _run(ctx, project, args, args.ticket)


# --------------------------------------------------------------------------- read and approve


def _questions(ctx: _Context, args: argparse.Namespace) -> int:
    with _open(ctx, args.project) as project:
        asked = questions(project.ledger, args.ticket)
    if not asked:
        print("no questions")
    for question in asked:
        state = "open" if question.answer is None else "answered"
        print(f"{question.id} [{state}] {question.text}")
        if question.answer is not None:
            print(f"    {question.answer}")
    return 0


def _show(ctx: _Context, args: argparse.Namespace) -> int:
    with _open(ctx, args.project) as project:
        ticket = project.ledger.get_ticket(args.ticket)
        asked = questions(project.ledger, args.ticket)
        for line in _ticket_lines(ticket):
            print(line)
        if asked:
            print("questions:")
            for question in asked:
                reply = "(open)" if question.answer is None else question.answer
                print(f"  {question.id}: {question.text}\n      answer: {reply}")
        if args.holdout:
            record = HoldoutStore(project.paths.holdout).load(args.ticket)
            print("holdout (sealed, never shown to implementers):")
            if record is None:
                print("  none recorded")
            for probe in () if record is None else record.probes:
                print(_probe_text(probe))
    return 0


def _ticket_lines(ticket: Ticket) -> list[str]:
    lines = [
        f"{ticket.id}: {ticket.title}",
        f"status: {ticket.status.value}",
        f"type: {ticket.type.value}",
        f"risk: {ticket.risk.value}",
        "request:",
        *(f"  {line}" for line in ticket.request.splitlines()),
    ]
    contract = ticket.contract
    if contract is None:
        return lines
    lines += [f"contract (version {contract.version}): {contract.summary}", "criteria:"]
    for criterion in contract.criteria:
        lines.append(
            f"  {criterion.id} [{criterion.layer.value}, {criterion.kind.value}] {criterion.ears}"
        )
        lines += [f"      visible test: {test}" for test in criterion.tests]
        lines.append(f"      holdout probes: {criterion.holdout_probes}")
    for title, items in (
        ("non-functional requirements", contract.nfr),
        ("out of scope", contract.out_of_scope),
        ("assumptions", contract.assumptions),
    ):
        if items:
            lines += [f"{title}:", *(f"  - {item}" for item in items)]
    lines.append(f"migration impact: {contract.migration_impact.value}")
    if ticket.oracle_version is not None:
        lines.append(f"oracle version: {ticket.oracle_version}")
    if ticket.contract_hash is not None:
        lines.append(f"contract hash: {ticket.contract_hash}")
    return lines


def _probe_text(probe: HoldoutProbe) -> str:
    expect = probe.expect.model_dump(mode="json", exclude_none=True, exclude_defaults=True)
    lines = [f"  {probe.id} [{', '.join(probe.criteria)}] argv: {json.dumps(list(probe.argv))}"]
    if probe.stdin:
        lines.append(f"      stdin: {json.dumps(probe.stdin)}")
    lines.append(f"      expect: {json.dumps(expect, ensure_ascii=False)}")
    return "\n".join(lines)


def _approve(ctx: _Context, args: argparse.Namespace) -> int:
    with _open(ctx, args.project) as project:
        ticket = approve(
            project.ledger,
            BareRepo.open(project.paths.repo),
            HoldoutStore(project.paths.holdout),
            args.ticket,
        )
    print(f"{ticket.id}: approved")
    print(f"contract hash: {ticket.contract_hash}")
    return 0


_COMMANDS: dict[str, Callable[[_Context, argparse.Namespace], int]] = {
    "project": _project_add,
    "intake": _intake,
    "questions": _questions,
    "answer": _answer,
    "show": _show,
    "approve": _approve,
}


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
