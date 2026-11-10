"""Unit tests for ticket creation, lookup, listing and transitions."""

from __future__ import annotations

import sqlite3
from itertools import pairwise
from pathlib import Path

import pytest
from pydantic import ValidationError

from local_factory.ledger import IllegalTransition, Ledger, LedgerError, TicketNotFound
from local_factory.ledger.state_machine import TERMINAL, allowed_targets
from local_factory.models import (
    Event,
    Halt,
    HaltReason,
    RiskTier,
    Ticket,
    TicketStatus,
    TicketType,
)

from .helpers import T0, FakeClock, new_ticket, walk_to

S = TicketStatus


# --------------------------------------------------------------------------- creation


def test_create_ticket_returns_the_stored_snapshot_and_records_an_event(
    ledger: Ledger, clock: FakeClock
) -> None:
    ticket = ledger.create_ticket(
        title="Tags", type=TicketType.BUG, risk=RiskTier.R2, request="please", actor="human"
    )
    assert ticket == ledger.get_ticket(ticket.id)
    assert (ticket.id, ticket.title, ticket.type, ticket.risk, ticket.request) == (
        "T-0001",
        "Tags",
        TicketType.BUG,
        RiskTier.R2,
        "please",
    )
    assert ticket.status is S.RECEIVED
    assert (ticket.contract, ticket.contract_hash, ticket.oracle_version) == (None, None, None)
    assert ticket.created_at == ticket.updated_at == clock.now
    assert ledger.events(ticket.id) == [
        Event(
            seq=1,
            ticket_id=ticket.id,
            at=clock.now,
            kind="created",
            actor="human",
            to_status=S.RECEIVED,
            data={"title": "Tags", "type": "bug", "risk": "R2"},
        )
    ]


def test_ticket_ids_are_numbered_past_four_digits_and_listed_numerically(
    db_path: Path, clock: FakeClock
) -> None:
    with Ledger(db_path, clock=clock) as led:
        seed = new_ticket(led)
    # Plant ticket number 9999 directly, so the next id needs five digits.
    planted = seed.model_copy(update={"id": "T-9999"})
    conn = sqlite3.connect(db_path)
    try:
        with conn:
            conn.execute(
                "INSERT INTO tickets (num, id, status, body) VALUES (?, ?, ?, ?)",
                (9999, planted.id, planted.status.value, planted.model_dump_json()),
            )
    finally:
        conn.close()
    with Ledger(db_path, clock=clock) as led:
        assert new_ticket(led).id == "T-10000"
        assert [t.id for t in led.list_tickets()] == ["T-0001", "T-9999", "T-10000"]


def test_invalid_ticket_fields_are_rejected_and_nothing_is_stored(ledger: Ledger) -> None:
    with pytest.raises(ValidationError):
        ledger.create_ticket(
            title="", type=TicketType.BUG, risk=RiskTier.R0, request="r", actor="human"
        )
    assert ledger.list_tickets() == []
    assert ledger.events() == []
    assert new_ticket(ledger).id == "T-0001"


def test_a_failing_event_rolls_back_the_whole_mutation(ledger: Ledger) -> None:
    with pytest.raises(ValidationError):
        ledger.create_ticket(
            title="t", type=TicketType.BUG, risk=RiskTier.R0, request="r", actor=""
        )
    assert ledger.list_tickets() == []
    assert ledger.events() == []
    ticket = new_ticket(ledger)
    with pytest.raises(ValidationError):
        ledger.transition(ticket.id, S.TRIAGED, actor="")
    assert ledger.get_ticket(ticket.id) == ticket
    assert [e.kind for e in ledger.events()] == ["created"]


# --------------------------------------------------------------------------- lookup & listing


def test_unknown_ticket_lookups_name_the_id(ledger: Ledger) -> None:
    with pytest.raises(TicketNotFound, match="T-0042"):
        ledger.get_ticket("T-0042")
    with pytest.raises(TicketNotFound):
        ledger.get_ticket("not-an-id")


def test_list_tickets_on_an_empty_ledger_and_with_no_matches(ledger: Ledger) -> None:
    assert ledger.list_tickets() == []
    new_ticket(ledger)
    assert ledger.list_tickets(S.DONE) == []


def test_list_tickets_reflects_the_current_status(ledger: Ledger) -> None:
    a, b, c = (new_ticket(ledger, name) for name in "abc")
    ledger.transition(a.id, S.TRIAGED, actor="triage")
    ledger.transition(c.id, S.TRIAGED, actor="triage")
    ledger.halt(c.id, Halt(reason=HaltReason.BUDGET), actor="scheduler")
    assert [t.id for t in ledger.list_tickets(S.TRIAGED)] == [a.id]
    assert [t.id for t in ledger.list_tickets(S.RECEIVED)] == [b.id]
    assert [t.id for t in ledger.list_tickets(S.HALTED)] == [c.id]
    assert all(isinstance(t, Ticket) for t in ledger.list_tickets())


# --------------------------------------------------------------------------- transitions


def test_transition_records_from_to_reason_and_timestamp(ledger: Ledger, clock: FakeClock) -> None:
    ticket = new_ticket(ledger)
    updated = ledger.transition(ticket.id, S.NEEDS_INFO, actor="triage", reason="which notes?")
    assert updated.status is S.NEEDS_INFO
    assert updated.updated_at == clock.now
    assert updated.created_at == ticket.created_at
    last = ledger.events(ticket.id)[-1]
    assert (last.kind, last.actor, last.at) == ("transition", "triage", clock.now)
    assert (last.from_status, last.to_status) == (S.RECEIVED, S.NEEDS_INFO)
    assert last.data == {"reason": "which notes?"}


def test_reason_defaults_to_empty(ledger: Ledger) -> None:
    ticket = new_ticket(ledger)
    ledger.transition(ticket.id, S.TRIAGED, actor="triage")
    assert ledger.events(ticket.id)[-1].data == {"reason": ""}


def test_transition_of_unknown_ticket(ledger: Ledger) -> None:
    with pytest.raises(TicketNotFound):
        ledger.transition("T-0001", S.TRIAGED, actor="triage")


def test_the_happy_path_reaches_done(ledger: Ledger) -> None:
    ticket = new_ticket(ledger)
    ledger.transition(ticket.id, S.TRIAGED, actor="triage")
    ledger.transition(ticket.id, S.CONTRACT_PROPOSED, actor="spec")
    walk_to(ledger, ticket.id, S.APPROVED)  # propose + oracle + freeze
    path = [
        S.PLANNED,
        S.BUILDING,
        S.VERIFYING,
        S.BUILDING,  # a failed verification round
        S.VERIFYING,
        S.CONVERGED,
        S.CANDIDATE_VERIFIED,
        S.MERGED,
        S.AWAITING_RELEASE_APPROVAL,
        S.DEPLOYING,
        S.DONE,
    ]
    for status in path:
        ledger.transition(ticket.id, status, actor="factory")
    assert ledger.get_ticket(ticket.id).status is S.DONE
    transitions = [
        (e.from_status, e.to_status) for e in ledger.events(ticket.id) if e.kind == "transition"
    ]
    expected_path = [S.RECEIVED, S.TRIAGED, S.CONTRACT_PROPOSED, S.APPROVED, *path]
    assert transitions == list(pairwise(expected_path))
    assert [e.kind for e in ledger.events(ticket.id)][:6] == [
        "created",
        "transition",
        "transition",
        "contract_proposed",
        "oracle_version_set",
        "contract_frozen",
    ]


@pytest.mark.parametrize("src", sorted(set(S) - {S.HALTED}))
def test_every_illegal_target_is_rejected_without_side_effects(
    ledger: Ledger, clock: FakeClock, src: TicketStatus
) -> None:
    ticket = walk_to(ledger, new_ticket(ledger).id, src)
    events_before, calls_before = ledger.events(), clock.calls
    for dst in set(S) - allowed_targets(src):
        with pytest.raises(IllegalTransition):
            ledger.transition(ticket.id, dst, actor="rogue")
    assert ledger.get_ticket(ticket.id) == ticket
    assert ledger.events() == events_before
    assert clock.calls == calls_before


@pytest.mark.parametrize("terminal", sorted(TERMINAL))
def test_terminal_tickets_accept_no_transition(ledger: Ledger, terminal: TicketStatus) -> None:
    ticket = walk_to(ledger, new_ticket(ledger).id, terminal)
    for dst in S:
        with pytest.raises(IllegalTransition):
            ledger.transition(ticket.id, dst, actor="rogue")


def test_transition_to_halted_points_at_halt(ledger: Ledger) -> None:
    ticket = new_ticket(ledger)
    with pytest.raises(IllegalTransition, match=r"halt\(\)"):
        ledger.transition(ticket.id, S.HALTED, actor="x")


def test_halted_ticket_must_be_resumed_even_to_its_prior_status(ledger: Ledger) -> None:
    ticket = walk_to(ledger, new_ticket(ledger).id, S.BUILDING)
    ledger.halt(ticket.id, Halt(reason=HaltReason.RATE_LIMIT), actor="scheduler")
    for dst in (S.BUILDING, S.VERIFYING, S.ABANDONED, S.HALTED):
        with pytest.raises(IllegalTransition, match=r"resume\(\)"):
            ledger.transition(ticket.id, dst, actor="x")
    assert ledger.get_ticket(ticket.id).status is S.HALTED


def test_approval_requires_freezing_the_contract(ledger: Ledger) -> None:
    ticket = walk_to(ledger, new_ticket(ledger).id, S.CONTRACT_PROPOSED)
    with pytest.raises(IllegalTransition, match=r"freeze_contract\(\)"):
        ledger.transition(ticket.id, S.APPROVED, actor="human")
    assert ledger.get_ticket(ticket.id).status is S.CONTRACT_PROPOSED


def test_blocked_ticket_can_be_re_approved_with_its_frozen_contract(ledger: Ledger) -> None:
    ticket = walk_to(ledger, new_ticket(ledger).id, S.BLOCKED)
    frozen_hash = ticket.contract_hash
    assert frozen_hash is not None
    ledger.transition(ticket.id, S.CONTRACT_PROPOSED, actor="human")
    approved = ledger.transition(ticket.id, S.APPROVED, actor="human", reason="unchanged")
    assert approved.status is S.APPROVED
    assert approved.contract_hash == frozen_hash


def test_illegal_transition_is_a_ledger_error(ledger: Ledger) -> None:
    ticket = new_ticket(ledger)
    with pytest.raises(LedgerError, match=f"{ticket.id}: illegal transition received -> merged"):
        ledger.transition(ticket.id, S.MERGED, actor="rogue")


# --------------------------------------------------------------------------- clock


def test_every_successful_mutation_reads_the_clock_exactly_once(
    ledger: Ledger, clock: FakeClock
) -> None:
    start = clock.calls
    ticket = new_ticket(ledger)
    assert clock.calls == start + 1
    ledger.transition(ticket.id, S.TRIAGED, actor="triage")
    assert clock.calls == start + 2
    ledger.halt(ticket.id, Halt(reason=HaltReason.BILLING), actor="scheduler")
    ledger.resume(ticket.id, actor="scheduler")
    assert clock.calls == start + 4
    events = ledger.events(ticket.id)
    assert [e.at for e in events] == sorted(e.at for e in events)
    assert events[0].at > T0
