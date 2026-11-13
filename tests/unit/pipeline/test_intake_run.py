"""Unit tests for how a run of intake treats a ticket and the model runs: what it refuses, what
the roles are told, models and limits, and every way a run can fail or halt.

The oracle loop is in test_intake_oracle.py; the happy path and the questions are in
tests/acceptance/test_intake.py.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import JsonValue

from local_factory.agents import AgentResult, AgentTask, RoleSpec, WorkerHandle
from local_factory.agents.backend import Capability
from local_factory.agents.fake import Action, EmitOutput, Fail, RaiseHalt
from local_factory.config import ConfigError, FactoryConfig
from local_factory.gitops import BareRepo, BundleLimits
from local_factory.ledger import TicketNotFound
from local_factory.models import Halt, HaltReason, TicketStatus, Tier
from local_factory.pipeline import IntakeError, IntakeOutcome
from local_factory.pipeline.flow import Site, Verification, Worker
from local_factory.pipeline.host_slice import host_site
from local_factory.pipeline.intake import (
    MAX_OPEN_TICKETS_SHOWN,
    approve,
    receive,
    run_intake,
)
from local_factory.sandbox.runtime import SandboxError
from tests.acceptance.intake_world import (
    REQUEST,
    ScriptedBackend,
    World,
    happy_scripts,
    make_world,
    spec_out,
    triage_out,
)

Scripts = dict[str, list[list[Action]]]


@pytest.fixture
def gate1(tmp_path: Path) -> Iterator[World]:
    with make_world(tmp_path) as world:
        yield world


def go(w: World, backend: ScriptedBackend, ticket_id: str, **changes: Any) -> IntakeOutcome:
    """Run intake on ``ticket_id``, with ``changes`` to the Intake's fields."""
    return asyncio.run(run_intake(replace(w.intake(backend), **changes), ticket_id))


@dataclass
class ProxySite:
    """The in-process host site, with a run limit and/or a failing ``check``."""

    limit_s: float | None = None
    broken: Exception | None = None
    inner: Site = field(default_factory=host_site)

    @property
    def run_limit_s(self) -> float | None:
        return self.limit_s

    @property
    def limits(self) -> BundleLimits:
        return self.inner.limits

    def check(self) -> None:
        if self.broken is not None:
            raise self.broken

    def image_digests(self) -> dict[str, str]:
        return self.inner.image_digests()

    def worker(self, bundle: Path, ref: str) -> AbstractAsyncContextManager[Worker]:
        return self.inner.worker(bundle, ref)

    def verification(
        self, repo: BareRepo, sha: str, ticket_dir: Path, *, round_no: int
    ) -> AbstractAsyncContextManager[Verification]:
        return self.inner.verification(repo, sha, ticket_dir, round_no=round_no)


STAGES = ["triage", "spec-writer", "ambiguity-critic", "test-architect"]


def rerun_scripts(backend: ScriptedBackend, role: str) -> None:
    """A retry starts again from the specification, so the roles before ``role`` run once more
    (triage does not: the ticket is already triaged)."""
    for earlier in STAGES[1 : STAGES.index(role)]:
        backend.add(earlier, *happy_scripts()[earlier])


@dataclass
class StubBackend:
    """Answers every run with ``reply(role)``, recording the roles it was asked to play."""

    reply: Callable[[RoleSpec], AgentResult]
    name: str = "fake"
    capabilities: frozenset[Capability] = frozenset()
    roles: list[str] = field(default_factory=list)

    async def run(self, role: RoleSpec, task: AgentTask, worker: WorkerHandle) -> AgentResult:
        self.roles.append(role.name)
        return self.reply(role)


# --------------------------------------------------------------------------- what intake continues


@pytest.mark.parametrize("status", [TicketStatus.APPROVED, TicketStatus.ABANDONED])
def test_a_ticket_past_intake_is_refused(gate1: World, status: TicketStatus) -> None:
    ticket = receive(gate1.ledger, REQUEST)
    backend = ScriptedBackend(happy_scripts())
    assert go(gate1, backend, ticket.id).status == "contract_proposed"
    if status is TicketStatus.APPROVED:
        approve(gate1.ledger, gate1.repo, gate1.holdout, ticket.id)
    else:
        gate1.ledger.transition(ticket.id, TicketStatus.ABANDONED, actor="human")
    calls = len(backend.calls)
    with pytest.raises(IntakeError, match=status.value):
        go(gate1, backend, ticket.id)
    assert len(backend.calls) == calls


def test_an_unknown_ticket_is_the_ledgers_error(gate1: World) -> None:
    with pytest.raises(TicketNotFound):
        go(gate1, ScriptedBackend({}), "T-0042")


def test_a_project_without_an_acceptance_command_changes_nothing(gate1: World) -> None:
    ticket = receive(gate1.ledger, REQUEST)
    backend = ScriptedBackend(happy_scripts())
    for commands in ({}, {"acceptance": "pytest tests/acceptance"}):
        project = gate1.project.model_copy(update={"commands": commands})
        with pytest.raises(IntakeError, match="acceptance"):
            go(gate1, backend, ticket.id, project=project)
    assert backend.calls == []
    assert gate1.ledger.get_ticket(ticket.id).status is TicketStatus.RECEIVED
    assert gate1.ledger.runs(ticket.id) == []


def test_the_site_is_checked_before_anything_changes(gate1: World) -> None:
    ticket = receive(gate1.ledger, REQUEST)
    backend = ScriptedBackend(happy_scripts())
    site = ProxySite(broken=SandboxError("docker is down"))
    with pytest.raises(SandboxError, match="docker is down"):
        asyncio.run(run_intake(gate1.intake(backend, site), ticket.id))
    assert backend.calls == [] and gate1.ledger.get_ticket(ticket.id).status.value == "received"


# --------------------------------------------------------------------------- models


def test_a_real_backend_needs_a_factory_config(gate1: World) -> None:
    ticket = receive(gate1.ledger, REQUEST)
    backend = ScriptedBackend(happy_scripts(), name="claude")
    with pytest.raises(IntakeError, match="factory config"):
        go(gate1, backend, ticket.id)
    assert backend.calls == []


def test_the_models_come_from_the_configured_tiers(gate1: World, tmp_path: Path) -> None:
    ticket = receive(gate1.ledger, REQUEST)
    backend = ScriptedBackend(happy_scripts(), name="claude")
    config = FactoryConfig(home=tmp_path / "home")
    assert go(gate1, backend, ticket.id, factory_config=config).status == "contract_proposed"
    runs = gate1.ledger.runs(ticket.id)
    assert [(r.role, r.backend, r.model, r.tier) for r in runs] == [
        ("triage", "claude", config.model_for("claude", Tier.STANDARD), Tier.STANDARD),
        ("spec-writer", "claude", config.model_for("claude", Tier.FRONTIER), Tier.FRONTIER),
        ("ambiguity-critic", "claude", config.model_for("claude", Tier.FRONTIER), Tier.FRONTIER),
        ("test-architect", "claude", config.model_for("claude", Tier.FRONTIER), Tier.FRONTIER),
    ]
    assert all(r.started_at <= r.finished_at and r.ticket_id == ticket.id for r in runs)
    assert all(r.task_id is None and r.usage.input_tokens > 0 for r in runs)


def test_an_unconfigured_model_is_the_configs_error(gate1: World, tmp_path: Path) -> None:
    ticket = receive(gate1.ledger, REQUEST)
    config = FactoryConfig(home=tmp_path / "home", tiers={"claude": {}})
    with pytest.raises(ConfigError):
        go(
            gate1,
            ScriptedBackend(happy_scripts(), name="claude"),
            ticket.id,
            factory_config=config,
        )
    assert gate1.ledger.runs(ticket.id) == []


def test_the_run_clock_stamps_the_run_records(gate1: World) -> None:
    ticks = iter(datetime(2026, 11, 13, 12, 0, s, tzinfo=UTC) for s in range(60))
    ticket = receive(gate1.ledger, REQUEST)
    go(gate1, ScriptedBackend(happy_scripts()), ticket.id, clock=lambda: next(ticks))
    first = gate1.ledger.runs(ticket.id)[0]
    assert (first.started_at.second, first.finished_at.second) == (0, 1)


# --------------------------------------------------------------------------- a run's limit


def test_a_run_that_outlives_the_sites_limit_is_an_agent_error(gate1: World) -> None:
    class Hangs:
        name = "fake"
        capabilities: frozenset[Capability] = frozenset()

        async def run(self, role: RoleSpec, task: AgentTask, worker: WorkerHandle) -> AgentResult:
            await asyncio.Event().wait()  # never finishes: the limit stops it
            raise AssertionError("unreachable")

    ticket = receive(gate1.ledger, REQUEST)
    intake = gate1.intake(Hangs(), ProxySite(limit_s=0.05))
    outcome = asyncio.run(run_intake(intake, ticket.id))
    assert outcome.status == "agent_error"
    assert outcome.error is not None and "timeout: the run exceeded 0.05s" in outcome.error
    [run] = gate1.ledger.runs(ticket.id)
    assert run.status == "error" and run.error is not None and "timeout" in run.error


# --------------------------------------------------------------------------- failures and halts


def stub_ok(output: dict[str, JsonValue] | None) -> Callable[[RoleSpec], AgentResult]:
    def reply(_role: RoleSpec) -> AgentResult:
        return AgentResult(status="ok", structured_output=output)

    return reply


@pytest.mark.parametrize(
    ("output", "needle"),
    [
        (None, "does not match TriageResult"),
        ({"title": "only a title"}, "type: Field required"),
        ({**triage_out(), "risk": "R9"}, "risk"),
    ],
    ids=["no output", "missing fields", "bad value"],
)
def test_a_result_that_does_not_validate_is_an_agent_error(
    gate1: World, output: dict[str, JsonValue] | None, needle: str
) -> None:
    ticket = receive(gate1.ledger, REQUEST)
    backend = StubBackend(stub_ok(output))
    outcome = asyncio.run(run_intake(gate1.intake(backend), ticket.id))
    assert outcome.status == "agent_error"
    assert outcome.error is not None and needle in outcome.error
    [event] = [e for e in gate1.ledger.events(ticket.id) if e.kind == "intake_error"]
    assert event.data["role"] == "triage" and needle in str(event.data["error"])
    assert gate1.ledger.get_ticket(ticket.id).status is TicketStatus.RECEIVED
    assert backend.roles == ["triage"]  # intake stops at the first unusable result


def test_a_long_error_is_clipped_in_the_ledger(gate1: World) -> None:
    ticket = receive(gate1.ledger, REQUEST)
    scripts = happy_scripts()
    scripts["triage"] = [[Fail("x" * 9_000)]]
    outcome = go(gate1, ScriptedBackend(scripts), ticket.id)
    assert outcome.error is not None and len(outcome.error) < 2_100
    [event] = [e for e in gate1.ledger.events(ticket.id) if e.kind == "intake_error"]
    assert len(str(event.data["error"])) == 2_000


@pytest.mark.parametrize("role", ["triage", "spec-writer", "ambiguity-critic", "test-architect"])
def test_every_role_can_fail_and_be_retried(gate1: World, role: str) -> None:
    scripts = happy_scripts()
    good = scripts[role][0]
    scripts[role] = [[Fail(f"{role} fell over")], good]
    ticket = receive(gate1.ledger, REQUEST)
    backend = ScriptedBackend(scripts)

    outcome = go(gate1, backend, ticket.id)
    assert outcome.status == "agent_error"
    assert outcome.error == f"{role} fell over"
    [event] = [e for e in gate1.ledger.events(ticket.id) if e.kind == "intake_error"]
    assert event.data["role"] == role
    assert gate1.ledger.get_ticket(ticket.id).contract is None
    assert gate1.ledger.runs(ticket.id)[-1].status == "error"

    rerun_scripts(backend, role)
    assert go(gate1, backend, ticket.id).status == "contract_proposed"
    assert gate1.refs("refs/factory/oracle-candidate/") == []


@pytest.mark.parametrize("role", ["triage", "spec-writer", "ambiguity-critic", "test-architect"])
def test_every_role_can_halt_the_ticket(gate1: World, role: str) -> None:
    scripts = happy_scripts()
    good = scripts[role][0]
    scripts[role] = [[RaiseHalt(HaltReason.RATE_LIMIT)], good]
    ticket = receive(gate1.ledger, REQUEST)
    backend = ScriptedBackend(scripts)

    outcome = go(gate1, backend, ticket.id)
    assert (outcome.status, outcome.error) == ("halted", "rate_limit")
    assert outcome.ticket.status is TicketStatus.HALTED
    assert gate1.ledger.runs(ticket.id)[-1].status == "halted"

    gate1.ledger.resume(ticket.id, actor="human")
    rerun_scripts(backend, role)
    assert go(gate1, backend, ticket.id).status == "contract_proposed"


def test_a_halts_detail_is_in_the_outcome(gate1: World) -> None:
    halt = Halt(reason=HaltReason.BILLING, detail="credit balance is too low")
    backend = StubBackend(lambda _role: AgentResult(status="halted", halt=halt))
    ticket = receive(gate1.ledger, REQUEST)
    outcome = asyncio.run(run_intake(gate1.intake(backend), ticket.id))
    assert (outcome.status, outcome.error) == ("halted", "billing: credit balance is too low")
    assert gate1.ledger.current_halt(ticket.id) == halt


# --------------------------------------------------------------------------- what roles read


def test_triage_sees_the_latest_open_tickets_with_their_requests_clipped(gate1: World) -> None:
    receive(gate1.ledger, "the oldest request is no longer shown")
    for number in range(MAX_OPEN_TICKETS_SHOWN - 1):
        receive(gate1.ledger, f"request number {number}")
    last = receive(gate1.ledger, "L" * 40 + "\n" + "tail  " * 200)
    ticket = receive(gate1.ledger, REQUEST)
    backend = ScriptedBackend(happy_scripts())
    go(gate1, backend, ticket.id)

    [triage] = backend.runs("triage")
    shown = triage.instructions
    assert "the oldest request" not in shown and "T-0001" not in shown
    assert f"{last.id}: {'L' * 40}" in shown
    assert shown.count("\n- T-") == MAX_OPEN_TICKETS_SHOWN
    clipped = [line for line in shown.splitlines() if line.startswith("  L" * 1)]
    assert len(clipped) == 1 and len(clipped[0].strip()) == 500 and clipped[0].endswith("...")
    assert ticket.id in shown.splitlines()[0]


def test_triage_with_no_other_tickets_says_so(gate1: World) -> None:
    ticket = receive(gate1.ledger, REQUEST)
    backend = ScriptedBackend(happy_scripts())
    go(gate1, backend, ticket.id)
    assert "no other open tickets" in backend.runs("triage")[0].instructions


def test_a_ticket_triaged_elsewhere_is_specified_without_a_triage_result(gate1: World) -> None:
    ticket = receive(gate1.ledger, REQUEST)
    gate1.ledger.transition(ticket.id, TicketStatus.TRIAGED, actor="human")
    backend = ScriptedBackend(happy_scripts())
    assert go(gate1, backend, ticket.id).status == "contract_proposed"
    assert "Triage result: none recorded." in backend.runs("spec-writer")[0].instructions
    assert backend.roles() == ["spec-writer", "ambiguity-critic", "test-architect"]


def test_a_previous_draft_is_revised_and_the_version_counts_up(gate1: World) -> None:
    scripts = happy_scripts()
    scripts["spec-writer"] = [[EmitOutput(spec_out(summary="First draft."))]]
    scripts["ambiguity-critic"] = [[Fail("critic fell over")], *scripts["ambiguity-critic"]]
    ticket = receive(gate1.ledger, REQUEST)
    backend = ScriptedBackend(scripts)
    assert go(gate1, backend, ticket.id).status == "agent_error"  # draft 1 recorded, then failure
    backend.add("spec-writer", [EmitOutput(spec_out(summary="Second draft."))])
    assert go(gate1, backend, ticket.id).status == "contract_proposed"

    first, second = backend.runs("spec-writer")
    assert "Specification draft" not in first.instructions
    assert "Specification draft, version 1" in second.instructions
    assert "First draft." in second.instructions and "Improve this draft" in second.instructions
    contract = gate1.ledger.get_ticket(ticket.id).contract
    assert contract is not None and (contract.version, contract.summary) == (2, "Second draft.")
    versions = [
        e.data["version"] for e in gate1.ledger.events(ticket.id) if e.kind == "spec_drafted"
    ]
    assert versions == [1, 2]


def test_nothing_runs_in_the_users_repository_or_moves_main(gate1: World) -> None:
    main = gate1.main
    ticket = receive(gate1.ledger, REQUEST)
    go(gate1, ScriptedBackend(happy_scripts()), ticket.id)
    assert gate1.main == main
    assert list(gate1.target.glob("tests/acceptance")) == []
