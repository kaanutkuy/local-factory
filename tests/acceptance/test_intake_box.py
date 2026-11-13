"""Gate 1 in the box: the same intake flow with every model run, the export and red-first in
network-less containers (docs/specs/phase-2b.md § Exit check). Protected; `container`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path

import pytest

from local_factory.agents.fake import EmitOutput, RunCommand, WriteFile
from local_factory.models import TicketStatus
from local_factory.oracle import oracle_ref
from local_factory.pipeline.intake import approve, receive, run_intake
from local_factory.pipeline.sandboxed_slice import box_site
from local_factory.sandbox import Sandbox
from tests.acceptance.intake_world import (
    REQUEST,
    SHOUT_ID,
    SHOUT_TEST,
    VACUOUS_ID,
    VACUOUS_TEST,
    ScriptedBackend,
    World,
    acceptance_path,
    happy_scripts,
    make_world,
    manifest,
    oracle_script,
)

pytestmark = [pytest.mark.container, pytest.mark.integration]


@pytest.fixture
def gate1(tmp_path: Path) -> Iterator[World]:
    with make_world(tmp_path) as world:
        yield world


def test_intake_in_the_box(gate1: World, sandbox: Sandbox) -> None:
    vacuous = oracle_script(
        files={acceptance_path(): SHOUT_TEST, acceptance_path(file="test_plain.py"): VACUOUS_TEST},
        out=manifest(visible=((SHOUT_ID, ("AC-1",)), (VACUOUS_ID, ("AC-1",)))),
    )
    scripts = happy_scripts()
    scripts["test-architect"].insert(0, vacuous)
    backend = ScriptedBackend(scripts)
    main = gate1.main
    ticket = receive(gate1.ledger, REQUEST)

    outcome = asyncio.run(run_intake(gate1.intake(backend, box_site(sandbox)), ticket.id))

    assert outcome.status == "contract_proposed", outcome
    assert [(c.id, c.red) for c in outcome.red_checks] == [(SHOUT_ID, True), ("HP-1", True)]
    rejected = [e.data for e in gate1.ledger.events(ticket.id) if e.kind == "oracle_rejected"]
    assert rejected == [{"attempt": 1, "problems": [f"{VACUOUS_ID}: passed on the base"]}]
    sha = gate1.repo.rev_parse(oracle_ref(ticket.id))
    assert [c.path for c in gate1.repo.diff(main, sha)] == [acceptance_path()]
    assert gate1.main == main
    # the box's trees never land on the host: only bundles do
    assert list(gate1.work_root.rglob("*.py")) == []

    approved = approve(gate1.ledger, gate1.repo, gate1.holdout, ticket.id)
    assert approved.status is TicketStatus.APPROVED


def test_tests_the_architect_ran_in_the_box_are_committed_without_caches(
    gate1: World, sandbox: Sandbox
) -> None:
    """What the first live run did: run the new tests in the box (leaving __pycache__ behind)."""
    run_them = (
        f"python3 -m unittest {acceptance_path()}; python3 -m unittest discover -s tests; true"
    )
    scripts = happy_scripts()
    scripts["test-architect"] = [
        [
            WriteFile(acceptance_path(), SHOUT_TEST),
            RunCommand(("sh", "-c", run_them)),
            EmitOutput(manifest()),
        ]
    ]
    ticket = receive(gate1.ledger, REQUEST)
    outcome = asyncio.run(
        run_intake(gate1.intake(ScriptedBackend(scripts), box_site(sandbox)), ticket.id)
    )
    assert outcome.status == "contract_proposed", outcome
    sha = gate1.repo.rev_parse(oracle_ref(ticket.id))
    assert [c.path for c in gate1.repo.diff(gate1.main, sha)] == [acceptance_path()]


def test_a_failing_role_in_the_box_is_an_agent_error(gate1: World, sandbox: Sandbox) -> None:
    scripts = happy_scripts()
    scripts["triage"] = [[EmitOutput({"title": "missing fields"})]]
    ticket = receive(gate1.ledger, REQUEST)
    outcome = asyncio.run(
        run_intake(gate1.intake(ScriptedBackend(scripts), box_site(sandbox)), ticket.id)
    )
    assert outcome.status == "agent_error"
    assert gate1.ledger.get_ticket(ticket.id).status is TicketStatus.RECEIVED
