"""Acceptance: ledger state machine and store (docs/specs/phase-1a.md § ledger). Protected."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from local_factory.hashing import content_hash
from local_factory.ledger import (
    ContractFrozen,
    IllegalTransition,
    Ledger,
    LedgerError,
    TicketNotFound,
)
from local_factory.ledger.state_machine import (
    TERMINAL,
    TRANSITIONS,
    allowed_targets,
    check_transition,
)
from local_factory.models import (
    CriterionResult,
    CriterionStatus,
    Evidence,
    GateResult,
    GateStatus,
    Halt,
    HaltReason,
    RiskTier,
    RunRecord,
    TicketStatus,
    TicketType,
    Tier,
    Usage,
    Verdict,
)

from .conftest import T0, make_contract

S = TicketStatus
SHA_A = "a" * 40
SHA_B = "b" * 40
HASH = "sha256:" + "c" * 64


class Clock:
    def __init__(self) -> None:
        self.now = T0

    def __call__(self) -> datetime:
        self.now += timedelta(seconds=1)
        return self.now


@pytest.fixture
def ledger(tmp_path: Path):
    with Ledger(tmp_path / "ledger.db", clock=Clock()) as led:
        yield led


def new_ticket(led: Ledger, title: str = "tags"):
    return led.create_ticket(
        title=title, type=TicketType.FEATURE, risk=RiskTier.R1, request="add tags", actor="human"
    )


def to_contract_proposed(led: Ledger, ticket_id: str) -> None:
    led.transition(ticket_id, S.TRIAGED, actor="triage")
    led.transition(ticket_id, S.CONTRACT_PROPOSED, actor="spec")
    led.propose_contract(ticket_id, make_contract(2), actor="spec")
    led.set_oracle_version(ticket_id, HASH, actor="test-architect")


# --------------------------------------------------------------------------- state machine


def test_transition_table_matches_spec() -> None:
    assert {S.DONE, S.ROLLED_BACK, S.ABANDONED} == TERMINAL
    assert allowed_targets(S.RECEIVED) == {S.TRIAGED, S.NEEDS_INFO, S.ABANDONED}
    assert allowed_targets(S.VERIFYING) == {S.BUILDING, S.CONVERGED, S.BLOCKED, S.ABANDONED}
    assert allowed_targets(S.CANDIDATE_VERIFIED) == {S.MERGED, S.CONVERGED, S.ABANDONED}
    assert allowed_targets(S.MERGED) == {S.AWAITING_RELEASE_APPROVAL, S.DEPLOYING, S.DONE}
    assert allowed_targets(S.AWAITING_RELEASE_APPROVAL) == {S.DEPLOYING}
    assert allowed_targets(S.DEPLOYING) == {S.DONE, S.ROLLED_BACK}
    for status in TERMINAL:
        assert allowed_targets(status) == frozenset()
    for targets in TRANSITIONS.values():
        assert S.HALTED not in targets


def test_check_transition_rejects_skips() -> None:
    check_transition(S.RECEIVED, S.TRIAGED)
    with pytest.raises(IllegalTransition):
        check_transition(S.RECEIVED, S.BUILDING)
    with pytest.raises(IllegalTransition):
        check_transition(S.DONE, S.BUILDING)


# --------------------------------------------------------------------------- tickets


def test_ticket_ids_are_monotonic_and_survive_reopen(tmp_path: Path) -> None:
    path = tmp_path / "ledger.db"
    with Ledger(path, clock=Clock()) as led:
        assert new_ticket(led).id == "T-0001"
        assert new_ticket(led).id == "T-0002"
    with Ledger(path, clock=Clock()) as led:
        third = new_ticket(led)
        assert third.id == "T-0003"
        assert [t.id for t in led.list_tickets()] == ["T-0001", "T-0002", "T-0003"]
        assert led.get_ticket("T-0002").status is S.RECEIVED


def test_created_ticket_fields_and_event(ledger: Ledger) -> None:
    ticket = new_ticket(ledger)
    assert ticket.status is S.RECEIVED
    assert ticket.risk is RiskTier.R1
    assert ticket.created_at > T0
    events = ledger.events(ticket.id)
    assert [e.kind for e in events] == ["created"]
    assert events[0].actor == "human"


def test_unknown_ticket_raises(ledger: Ledger) -> None:
    with pytest.raises(TicketNotFound):
        ledger.get_ticket("T-9999")
    assert issubclass(TicketNotFound, LedgerError)


def test_transition_validates_and_records_event(ledger: Ledger) -> None:
    ticket = new_ticket(ledger)
    updated = ledger.transition(ticket.id, S.TRIAGED, actor="triage", reason="looks fine")
    assert updated.status is S.TRIAGED
    assert updated.updated_at > ticket.updated_at
    with pytest.raises(IllegalTransition):
        ledger.transition(ticket.id, S.MERGED, actor="rogue")
    assert ledger.get_ticket(ticket.id).status is S.TRIAGED
    last = ledger.events(ticket.id)[-1]
    assert (last.kind, last.from_status, last.to_status) == ("transition", S.RECEIVED, S.TRIAGED)


def test_list_tickets_filters_by_status(ledger: Ledger) -> None:
    a, b = new_ticket(ledger, "a"), new_ticket(ledger, "b")
    ledger.transition(b.id, S.TRIAGED, actor="triage")
    assert [t.id for t in ledger.list_tickets(S.RECEIVED)] == [a.id]
    assert [t.id for t in ledger.list_tickets(S.TRIAGED)] == [b.id]


# --------------------------------------------------------------------------- halts


def test_halt_and_resume_return_to_prior_status(ledger: Ledger) -> None:
    ticket = new_ticket(ledger)
    ledger.transition(ticket.id, S.TRIAGED, actor="triage")
    halt = Halt(reason=HaltReason.RATE_LIMIT, resume_at=T0 + timedelta(hours=1), detail="429")
    halted = ledger.halt(ticket.id, halt, actor="scheduler")
    assert halted.status is S.HALTED
    assert ledger.current_halt(ticket.id) == halt
    with pytest.raises(IllegalTransition):
        ledger.transition(ticket.id, S.CONTRACT_PROPOSED, actor="spec")
    with pytest.raises(LedgerError):
        ledger.halt(ticket.id, halt, actor="scheduler")
    resumed = ledger.resume(ticket.id, actor="scheduler")
    assert resumed.status is S.TRIAGED
    assert ledger.current_halt(ticket.id) is None
    kinds = [e.kind for e in ledger.events(ticket.id)]
    assert kinds[-2:] == ["halted", "resumed"]


def test_cannot_transition_to_halted_or_halt_terminal(ledger: Ledger) -> None:
    ticket = new_ticket(ledger)
    with pytest.raises(IllegalTransition):
        ledger.transition(ticket.id, S.HALTED, actor="x")
    ledger.transition(ticket.id, S.ABANDONED, actor="human")
    with pytest.raises(LedgerError):
        ledger.halt(ticket.id, Halt(reason=HaltReason.BUDGET), actor="x")
    with pytest.raises(LedgerError):
        ledger.resume(ticket.id, actor="x")


# --------------------------------------------------------------------------- contract


def test_freeze_contract_hashes_and_approves(ledger: Ledger) -> None:
    ticket = new_ticket(ledger)
    to_contract_proposed(ledger, ticket.id)
    frozen = ledger.freeze_contract(ticket.id, actor="human")
    assert frozen.status is S.APPROVED
    assert frozen.contract == make_contract(2)
    assert frozen.contract_hash == content_hash(make_contract(2))
    assert frozen.oracle_version == HASH
    kinds = [e.kind for e in ledger.events(ticket.id)]
    assert kinds[-2:] == ["contract_frozen", "transition"]


def test_frozen_contract_is_immutable(ledger: Ledger) -> None:
    ticket = new_ticket(ledger)
    to_contract_proposed(ledger, ticket.id)
    ledger.freeze_contract(ticket.id, actor="human")
    with pytest.raises(ContractFrozen):
        ledger.propose_contract(ticket.id, make_contract(3), actor="spec")
    with pytest.raises(ContractFrozen):
        ledger.set_oracle_version(ticket.id, "sha256:" + "d" * 64, actor="test-architect")
    assert ledger.get_ticket(ticket.id).contract == make_contract(2)


def test_freeze_requires_contract_oracle_and_status(ledger: Ledger) -> None:
    ticket = new_ticket(ledger)
    with pytest.raises(LedgerError):
        ledger.freeze_contract(ticket.id, actor="human")  # wrong status, no contract
    ledger.transition(ticket.id, S.TRIAGED, actor="triage")
    ledger.transition(ticket.id, S.CONTRACT_PROPOSED, actor="spec")
    ledger.propose_contract(ticket.id, make_contract(), actor="spec")
    with pytest.raises(LedgerError):
        ledger.freeze_contract(ticket.id, actor="human")  # no oracle version yet
    assert ledger.get_ticket(ticket.id).status is S.CONTRACT_PROPOSED


# --------------------------------------------------------------------------- append-only events


def test_events_are_append_only_in_the_database(tmp_path: Path) -> None:
    path = tmp_path / "ledger.db"
    with Ledger(path, clock=Clock()) as led:
        new_ticket(led)
    conn = sqlite3.connect(path)
    try:
        with pytest.raises(sqlite3.DatabaseError):
            conn.execute("UPDATE events SET actor = 'forged'")
        with pytest.raises(sqlite3.DatabaseError):
            conn.execute("DELETE FROM events")
    finally:
        conn.close()
    with Ledger(path, clock=Clock()) as led:
        assert [e.actor for e in led.events()] == ["human"]


def test_event_sequence_is_global_and_ordered(ledger: Ledger) -> None:
    a, b = new_ticket(ledger, "a"), new_ticket(ledger, "b")
    ledger.transition(a.id, S.TRIAGED, actor="triage")
    seqs = [e.seq for e in ledger.events()]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs) == 3
    assert [e.ticket_id for e in ledger.events(b.id)] == [b.id]


# ---------------------------------------------------------- tasks, runs, verdicts, evidence


def test_tasks_are_numbered_per_ticket_and_updatable(ledger: Ledger) -> None:
    a, b = new_ticket(ledger, "a"), new_ticket(ledger, "b")
    t1 = ledger.create_task(a.id, goal="g1", role="implementer", tier=Tier.STANDARD, actor="orch")
    t2 = ledger.create_task(a.id, goal="g2", role="implementer", tier=Tier.STANDARD, actor="orch")
    t3 = ledger.create_task(b.id, goal="g3", role="implementer", tier=Tier.FAST, actor="orch")
    assert (t1.id, t2.id, t3.id) == ("T-0001/task-1", "T-0001/task-2", "T-0002/task-1")
    updated = ledger.update_task(t1.id, actor="factory", base_sha=SHA_A, head_sha=SHA_B)
    assert (updated.base_sha, updated.head_sha) == (SHA_A, SHA_B)
    assert [t.id for t in ledger.list_tasks(a.id)] == [t1.id, t2.id]
    assert ledger.get_task(t3.id).tier is Tier.FAST


def test_runs_accumulate_cost(ledger: Ledger) -> None:
    ticket = new_ticket(ledger)
    for i, cost in enumerate((0.25, 1.5), start=1):
        ledger.record_run(
            RunRecord(
                id=f"run-{i}",
                ticket_id=ticket.id,
                role="implementer",
                backend="fake",
                model="m",
                tier=Tier.STANDARD,
                started_at=T0,
                finished_at=T0,
                status="ok",
                usage=Usage(input_tokens=10, output_tokens=5, cost_usd=cost),
            )
        )
    assert [r.id for r in ledger.runs(ticket.id)] == ["run-1", "run-2"]
    assert ledger.ticket_cost_usd(ticket.id) == pytest.approx(1.75)


def _verdict(ticket_id: str, sha: str, passed: bool) -> Verdict:
    return Verdict(
        ticket_id=ticket_id,
        round=1,
        sha=sha,
        gates=(GateResult(name="unit", status=GateStatus.PASS if passed else GateStatus.FAIL),),
        criteria=(CriterionResult(id="AC-1", status=CriterionStatus.PASS),),
        passed=passed,
        failing=() if passed else ("gate:unit",),
    )


def test_verdicts_and_evidence_round_trip(ledger: Ledger) -> None:
    ticket = new_ticket(ledger)
    verdict = _verdict(ticket.id, SHA_A, passed=True)
    ledger.record_verdict(verdict)
    assert ledger.verdicts(ticket.id) == [verdict]

    def evidence(created_offset: int) -> Evidence:
        return Evidence(
            ticket_id=ticket.id,
            candidate_sha=SHA_A,
            tree_hash=SHA_B,
            base_sha=SHA_B,
            contract_hash=HASH,
            oracle_version=HASH,
            factory_version="0.1.0",
            gates=verdict.gates,
            criteria=verdict.criteria,
            passed=True,
            created_at=T0 + timedelta(seconds=created_offset),
        )

    ledger.record_evidence(evidence(1), evidence_hash=HASH, signature="00")
    ledger.record_evidence(evidence(2), evidence_hash=HASH, signature="01")
    assert ledger.evidence_for(ticket.id, SHA_A) == evidence(2)
    assert ledger.evidence_for(ticket.id, SHA_B) is None
    kinds = [e.kind for e in ledger.events(ticket.id)]
    assert kinds.count("evidence_recorded") == 2 and "verdict_recorded" in kinds
