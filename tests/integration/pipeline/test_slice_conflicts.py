"""Integration: integration and candidate conflicts, and a main that moved under the ticket."""

from __future__ import annotations

import asyncio

import pytest

from local_factory.agents import WriteFile
from local_factory.models import TaskStatus, TicketStatus
from local_factory.pipeline import SliceOutcome

from .conftest import (
    IMPLEMENT_SHOUT,
    TICKET_REF,
    VERIFIER_FAIL,
    VERIFIER_PASS,
    GatedBackend,
    World,
    greet_with,
)

pytestmark = pytest.mark.integration

T = TicketStatus
# Same output as the fixture's greet.py, different text on the same line: they conflict.
GREET_PERCENT = greet_with('print("Hello, %s!" % args.name)')
GREET_CONCAT = greet_with('print("Hello, " + args.name + "!")')
GREET_FORMAT = greet_with('print("Hello, {}!".format(args.name))')

RESERVED_TEST = """import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class ReservedTest(unittest.TestCase):
    def test_shout_is_reserved(self):
        result = subprocess.run(
            [sys.executable, "greet.py", "--name", "Ada", "--shout"],
            cwd=ROOT, capture_output=True, text=True,
        )
        self.assertNotEqual(result.returncode, 0)
"""


def test_concurrent_runs_on_one_ticket_conflict_on_merge(world: World) -> None:
    """task-1 is held in its agent run while task-2 (same base) merges first and verifies."""
    backend = GatedBackend(
        {
            "T-0001/task-1": [WriteFile("greet.py", GREET_PERCENT)],
            "T-0001/task-2": [WriteFile("greet.py", GREET_CONCAT)],
            "verifier": VERIFIER_PASS,
        },
        gated="T-0001/task-1",
    )

    async def scenario() -> tuple[SliceOutcome, SliceOutcome]:
        first = asyncio.create_task(world.start(backend))
        await backend.waiting.wait()
        second = await world.start(backend)
        backend.release.set()
        return await first, second

    first, second = asyncio.run(scenario())

    assert second.status == "candidate_verified", second
    assert second.task.base_sha == world.main
    assert first.task.base_sha == world.main
    assert (first.status, first.rejection) == ("rejected", "conflict")
    assert first.task.status is TaskStatus.REJECTED
    assert first.task.head_sha is None
    end = world.task_ends()[first.task.id]  # the ledger alone says why
    assert (end.status, end.reason, end.detail) == (
        TaskStatus.REJECTED,
        "conflict",
        {"conflicts": ["greet.py"]},
    )
    assert first.ticket.status is T.CANDIDATE_VERIFIED  # task-2's result stands
    assert world.repo.rev_parse(TICKET_REF) == second.task.head_sha
    assert world.repo.ref_exists("refs/factory/tasks/T-0001/task-1")  # imported, not merged
    world.assert_boundaries()


def test_a_candidate_conflict_with_a_moved_main_returns_to_building(world: World) -> None:
    world.repo.update_ref(TICKET_REF, world.main)  # the ticket branch was cut at the old main
    moved = world.advance_main({"greet.py": GREET_FORMAT})

    outcome = world.run_scripts([WriteFile("greet.py", GREET_CONCAT)])

    assert (outcome.status, outcome.rejection) == ("rejected", "candidate_conflict")
    assert outcome.ticket.status is T.BUILDING
    assert outcome.task.status is TaskStatus.ACCEPTED  # the task itself was fine
    assert outcome.verdict is not None
    assert (outcome.verdict.round, outcome.verdict.passed) == (1, True)
    assert (outcome.candidate_sha, outcome.evidence) == (None, None)
    assert world.transitions()[-3:] == [T.VERIFYING, T.CONVERGED, T.BUILDING]
    back = world.ledger.events(world.ticket_id)[-1]
    assert (back.kind, back.to_status) == ("transition", T.BUILDING)
    assert back.data == {"reason": "candidate_conflict", "detail": {"conflicts": ["greet.py"]}}
    assert world.task_ends() == {}  # the task merged cleanly; the ticket branch conflicts
    assert world.repo.rev_parse("refs/heads/main") == moved
    world.assert_boundaries()


def test_a_candidate_that_breaks_on_the_moved_main_fails_verification(world: World) -> None:
    """The ticket head passes; only the candidate (new main + ticket) breaks a gate."""
    first = world.run_scripts([WriteFile("NOTES.md", "draft\n")], verifier=VERIFIER_FAIL)
    assert first.status == "failed_verification"
    moved = world.advance_main({"tests/test_reserved.py": RESERVED_TEST})

    outcome = world.run_scripts(IMPLEMENT_SHOUT)

    assert outcome.status == "failed_verification"
    assert outcome.ticket.status is T.BUILDING
    assert outcome.task.status is TaskStatus.ACCEPTED
    candidate = outcome.candidate_sha
    assert candidate is not None
    assert world.parents(candidate) == [moved]
    verdicts = world.ledger.verdicts(world.ticket_id)
    assert [(v.round, v.passed) for v in verdicts] == [(1, False), (2, True), (3, False)]
    assert verdicts[1].sha == world.repo.rev_parse(TICKET_REF)
    assert outcome.verdict == verdicts[2]
    assert outcome.verdict.sha == candidate
    assert outcome.verdict.failing == ("gate:unit",)
    assert outcome.evidence is None
    assert world.ledger.evidence_for(world.ticket_id, candidate) is None
    assert world.transitions()[-3:] == [T.VERIFYING, T.CONVERGED, T.BUILDING]
    world.assert_boundaries()
