"""Acceptance: the Phase 1a host vertical slice (docs/specs/phase-1a-slice.md). Protected."""

from __future__ import annotations

import asyncio
import shutil
from dataclasses import dataclass
from pathlib import Path

import pytest

from local_factory.agents.fake import (
    Action,
    EmitOutput,
    FakeBackend,
    RaiseHalt,
    RunCommand,
    WriteFile,
)
from local_factory.config import ProjectConfig, load_project_config
from local_factory.evidence.binding import check_binding, load_or_create_key, verify_signature
from local_factory.gitops.bare import BareRepo
from local_factory.hashing import content_hash
from local_factory.ledger import Ledger
from local_factory.models import (
    Contract,
    Criterion,
    HaltReason,
    RiskTier,
    TaskStatus,
    TestLayer,
    TicketStatus,
    TicketType,
)
from local_factory.pipeline.host_slice import SliceOutcome, run_host_slice

from .conftest import T0, Git

pytestmark = pytest.mark.integration

S = TicketStatus
FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "cli-target"
ORACLE = "sha256:" + "a" * 64

SHOUT_GREET = '''#!/usr/bin/env python3
"""A tiny polite greeter CLI."""
import argparse
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description="Print a polite greeting.")
    parser.add_argument("--name", required=True, help="Name of the person to greet.")
    parser.add_argument("--shout", action="store_true", help="Greet loudly.")
    args = parser.parse_args(argv)
    if not args.name:
        print("error: --name must not be empty", file=sys.stderr)
        return 2
    greeting = f"Hello, {args.name}!"
    print(greeting.upper() if args.shout else greeting)
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''

SHOUT_TEST = """import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class ShoutTest(unittest.TestCase):
    def test_shout(self):
        result = subprocess.run(
            [sys.executable, "greet.py", "--name", "Ada", "--shout"],
            cwd=ROOT, capture_output=True, text=True,
        )
        self.assertEqual(result.stdout, "HELLO, ADA!\\n")
"""

CONTRACT = Contract(
    version=1,
    summary="greet supports --shout",
    criteria=(
        Criterion(
            id="AC-1",
            ears="When --shout is given, greet shall print the greeting in upper case.",
            layer=TestLayer.E2E,
        ),
    ),
)

VERIFIER_PASS: list[Action] = [
    EmitOutput({"criteria": [{"id": "AC-1", "status": "pass", "evidence": "ok"}]})
]
VERIFIER_FAIL: list[Action] = [
    EmitOutput({"criteria": [{"id": "AC-1", "status": "fail", "feedback": "no"}]})
]
IMPLEMENT_SHOUT: list[Action] = [
    WriteFile("greet.py", SHOUT_GREET),
    WriteFile("tests/test_shout.py", SHOUT_TEST),
]


@dataclass
class World:
    ledger: Ledger
    repo: BareRepo
    project: ProjectConfig
    ticket_id: str
    main: str
    key: bytes
    work_root: Path

    def run(self, implementer: list[Action], verifier: list[Action] | None = None) -> SliceOutcome:
        backend = FakeBackend(
            {
                "implementer": implementer,
                "verifier": VERIFIER_PASS if verifier is None else verifier,
            }
        )
        return asyncio.run(
            run_host_slice(
                ledger=self.ledger,
                repo=self.repo,
                project=self.project,
                backend=backend,
                ticket_id=self.ticket_id,
                work_root=self.work_root,
                key=self.key,
                clock=lambda: T0,
            )
        )


@pytest.fixture
def world(tmp_path: Path, git: Git):
    target = tmp_path / "target"
    shutil.copytree(FIXTURE, target)
    git("init", "-q", cwd=target)
    git("add", "-A", cwd=target)
    git("commit", "-q", "-m", "target", cwd=target)

    state = tmp_path / "state"
    repo = BareRepo.init(state / "repo.git")
    repo.fetch_from(target, ["refs/heads/main:refs/heads/main"])
    with Ledger(state / "ledger.db") as ledger:
        ticket = ledger.create_ticket(
            title="shout",
            type=TicketType.FEATURE,
            risk=RiskTier.R1,
            request="--shout",
            actor="human",
        )
        ledger.transition(ticket.id, S.TRIAGED, actor="triage")
        ledger.transition(ticket.id, S.CONTRACT_PROPOSED, actor="spec")
        ledger.propose_contract(ticket.id, CONTRACT, actor="spec")
        ledger.set_oracle_version(ticket.id, ORACLE, actor="test-architect")
        ledger.freeze_contract(ticket.id, actor="human")
        yield World(
            ledger=ledger,
            repo=repo,
            project=load_project_config(target),
            ticket_id=ticket.id,
            main=repo.rev_parse("refs/heads/main"),
            key=load_or_create_key(state / "evidence.key"),
            work_root=tmp_path / "work",
        )


def transitions(world: World) -> list[TicketStatus | None]:
    return [e.to_status for e in world.ledger.events(world.ticket_id) if e.kind == "transition"]


# --------------------------------------------------------------------------- happy path


def test_happy_path_produces_bound_signed_evidence(world: World, git: Git) -> None:
    outcome = world.run(IMPLEMENT_SHOUT)

    assert outcome.status == "candidate_verified", outcome
    assert outcome.ticket.status is S.CANDIDATE_VERIFIED
    assert outcome.task.status is TaskStatus.ACCEPTED
    assert transitions(world)[-5:] == [
        S.PLANNED,
        S.BUILDING,
        S.VERIFYING,
        S.CONVERGED,
        S.CANDIDATE_VERIFIED,
    ]

    candidate = outcome.candidate_sha
    assert candidate is not None
    parents = git("rev-list", "--parents", "-n", "1", candidate, cwd=world.repo.path).split()[1:]
    assert parents == [world.main]
    assert b"--shout" in world.repo.read_file(candidate, "greet.py")
    assert world.repo.rev_parse("refs/heads/main") == world.main  # the slice never moves main

    evidence = outcome.evidence
    assert evidence is not None
    assert evidence.passed
    assert evidence.candidate_sha == candidate
    assert (
        check_binding(
            evidence,
            sha=candidate,
            tree_hash=world.repo.tree_hash(candidate),
            contract_hash=content_hash(CONTRACT),
            oracle_version=ORACLE,
        )
        == []
    )
    assert outcome.signature is not None
    assert verify_signature(evidence, outcome.signature, world.key)
    assert world.ledger.evidence_for(world.ticket_id, candidate) == evidence

    verdicts = world.ledger.verdicts(world.ticket_id)
    assert [v.round for v in verdicts] == [1, 2]
    assert verdicts[0].sha == outcome.task.head_sha
    assert verdicts[1].sha == candidate
    assert {g.name for g in verdicts[1].gates} == {"lint", "unit"}
    assert [r.role for r in world.ledger.runs(world.ticket_id)] == [
        "implementer",
        "verifier",
        "verifier",
    ]


# ------------------------------------------------------------------ the oracle is protected


def test_editing_a_protected_file_is_rejected(world: World) -> None:
    outcome = world.run([*IMPLEMENT_SHOUT, WriteFile("factory.toml", '[project]\nname = "x"\n')])
    assert (outcome.status, outcome.rejection) == ("rejected", "policy")
    assert [(v.rule, v.path) for v in outcome.violations] == [("protected_path", "factory.toml")]
    assert outcome.task.status is TaskStatus.REJECTED
    assert outcome.ticket.status is S.BUILDING
    assert world.repo.rev_parse(f"refs/heads/factory/{world.ticket_id}") == world.main
    assert outcome.evidence is None


def test_hostile_symlink_is_rejected_at_import(world: World) -> None:
    outcome = world.run([*IMPLEMENT_SHOUT, RunCommand(("ln", "-s", "/etc/passwd", "evil"))])
    assert (outcome.status, outcome.rejection) == ("rejected", "symlink_escape")
    assert outcome.task.status is TaskStatus.REJECTED


def test_doing_nothing_is_rejected(world: World) -> None:
    outcome = world.run([])
    assert (outcome.status, outcome.rejection) == ("rejected", "no_changes")


# ------------------------------------------------------------------ verification decides


def test_failing_gate_fails_verification(world: World) -> None:
    outcome = world.run([WriteFile("greet.py", "def broken(:\n")])
    assert outcome.status == "failed_verification"
    assert outcome.ticket.status is S.BUILDING
    assert outcome.verdict is not None
    assert not outcome.verdict.passed
    assert "gate:lint" in outcome.verdict.failing
    assert outcome.evidence is None


def test_failing_criterion_fails_verification(world: World) -> None:
    outcome = world.run(IMPLEMENT_SHOUT, verifier=VERIFIER_FAIL)
    assert outcome.status == "failed_verification"
    assert outcome.verdict is not None
    assert outcome.verdict.failing == ("criterion:AC-1",)


def test_broken_verifier_cannot_pass_a_verdict(world: World) -> None:
    outcome = world.run(IMPLEMENT_SHOUT, verifier=[EmitOutput({"nonsense": True})])
    assert outcome.status == "failed_verification"
    assert outcome.verdict is not None
    assert "criteria:none" in outcome.verdict.failing


# --------------------------------------------------------------------------- halts are state


def test_rate_limit_halts_and_resumes(world: World) -> None:
    outcome = world.run([RaiseHalt(HaltReason.RATE_LIMIT, resume_at=T0)])
    assert outcome.status == "halted"
    assert outcome.ticket.status is S.HALTED
    assert outcome.task.status is TaskStatus.HALTED
    halt = world.ledger.current_halt(world.ticket_id)
    assert halt is not None
    assert halt.reason is HaltReason.RATE_LIMIT
    assert world.ledger.resume(world.ticket_id, actor="scheduler").status is S.BUILDING
