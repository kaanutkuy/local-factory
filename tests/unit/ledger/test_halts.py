"""Unit tests for halt / resume."""

from __future__ import annotations

import sqlite3
from datetime import timedelta
from pathlib import Path

import pytest

from local_factory.ledger import IllegalTransition, Ledger, LedgerError, TicketNotFound
from local_factory.ledger.state_machine import TERMINAL
from local_factory.models import Halt, HaltReason, TicketStatus

from .helpers import T0, FakeClock, new_ticket, walk_to

S = TicketStatus
HALTABLE = sorted(set(S) - TERMINAL - {S.HALTED})


@pytest.mark.parametrize("status", HALTABLE)
def test_halt_and_resume_from_every_haltable_status(ledger: Ledger, status: TicketStatus) -> None:
    ticket = walk_to(ledger, new_ticket(ledger).id, status)
    halt = Halt(reason=HaltReason.RATE_LIMIT, resume_at=T0 + timedelta(hours=1), detail="429")
    halted = ledger.halt(ticket.id, halt, actor="scheduler")
    assert halted.status is S.HALTED
    assert ledger.current_halt(ticket.id) == halt
    resumed = ledger.resume(ticket.id, actor="scheduler")
    assert resumed.status is status
    assert ledger.current_halt(ticket.id) is None
    halted_event, resumed_event = ledger.events(ticket.id)[-2:]
    assert (halted_event.kind, halted_event.from_status, halted_event.to_status) == (
        "halted",
        status,
        S.HALTED,
    )
    assert (resumed_event.kind, resumed_event.from_status, resumed_event.to_status) == (
        "resumed",
        S.HALTED,
        status,
    )
    # The contract fields survive the round trip untouched.
    assert resumed.model_dump(exclude={"status", "updated_at"}) == ticket.model_dump(
        exclude={"status", "updated_at"}
    )


@pytest.mark.parametrize("terminal", sorted(TERMINAL))
def test_terminal_tickets_cannot_be_halted(ledger: Ledger, terminal: TicketStatus) -> None:
    ticket = walk_to(ledger, new_ticket(ledger).id, terminal)
    with pytest.raises(IllegalTransition, match="terminal"):
        ledger.halt(ticket.id, Halt(reason=HaltReason.BUDGET), actor="scheduler")
    assert ledger.current_halt(ticket.id) is None
    assert ledger.get_ticket(ticket.id).status is terminal


def test_halting_twice_is_rejected(ledger: Ledger) -> None:
    ticket = new_ticket(ledger)
    first = Halt(reason=HaltReason.BILLING)
    ledger.halt(ticket.id, first, actor="scheduler")
    with pytest.raises(IllegalTransition, match="already halted"):
        ledger.halt(ticket.id, Halt(reason=HaltReason.BUDGET), actor="scheduler")
    assert ledger.current_halt(ticket.id) == first


@pytest.mark.parametrize("status", [S.RECEIVED, S.BUILDING, S.DONE])
def test_resume_requires_a_halted_ticket(ledger: Ledger, status: TicketStatus) -> None:
    ticket = walk_to(ledger, new_ticket(ledger).id, status)
    with pytest.raises(IllegalTransition, match="not halted"):
        ledger.resume(ticket.id, actor="scheduler")


def test_halt_event_carries_the_halt(ledger: Ledger, clock: FakeClock) -> None:
    ticket = new_ticket(ledger)
    halt = Halt(reason=HaltReason.MAX_TURNS, detail="50 turns")
    halted = ledger.halt(ticket.id, halt, actor="scheduler")
    event = ledger.events(ticket.id)[-1]
    assert (event.actor, event.at, halted.updated_at) == ("scheduler", clock.now, clock.now)
    assert event.data == {"reason": "max_turns", "resume_at": None, "detail": "50 turns"}
    assert Halt.model_validate(event.data) == halt


def test_repeated_halts_track_the_latest(ledger: Ledger) -> None:
    ticket = walk_to(ledger, new_ticket(ledger).id, S.BUILDING)
    ledger.halt(ticket.id, Halt(reason=HaltReason.RATE_LIMIT), actor="scheduler")
    ledger.resume(ticket.id, actor="scheduler")
    ledger.transition(ticket.id, S.VERIFYING, actor="factory")
    second = Halt(reason=HaltReason.BUDGET, detail="cap")
    ledger.halt(ticket.id, second, actor="scheduler")
    assert ledger.current_halt(ticket.id) == second
    assert ledger.resume(ticket.id, actor="scheduler").status is S.VERIFYING
    kinds = [e.kind for e in ledger.events(ticket.id)]
    assert kinds.count("halted") == kinds.count("resumed") == 2


def test_halts_are_independent_per_ticket(ledger: Ledger) -> None:
    a, b = new_ticket(ledger, "a"), new_ticket(ledger, "b")
    ledger.halt(a.id, Halt(reason=HaltReason.BILLING), actor="scheduler")
    assert ledger.current_halt(b.id) is None
    ledger.halt(b.id, Halt(reason=HaltReason.BUDGET), actor="scheduler")
    ledger.resume(a.id, actor="scheduler")
    assert ledger.current_halt(a.id) is None
    assert ledger.current_halt(b.id) == Halt(reason=HaltReason.BUDGET)


def test_halt_survives_reopen(db_path: Path, clock: FakeClock) -> None:
    halt = Halt(reason=HaltReason.RATE_LIMIT, resume_at=T0 + timedelta(minutes=5))
    with Ledger(db_path, clock=clock) as led:
        ticket = walk_to(led, new_ticket(led).id, S.VERIFYING)
        led.halt(ticket.id, halt, actor="scheduler")
    with Ledger(db_path, clock=clock) as led:
        assert led.current_halt(ticket.id) == halt
        assert led.resume(ticket.id, actor="scheduler").status is S.VERIFYING


def test_halt_operations_on_unknown_ticket(ledger: Ledger) -> None:
    with pytest.raises(TicketNotFound):
        ledger.halt("T-0404", Halt(reason=HaltReason.BUDGET), actor="scheduler")
    with pytest.raises(TicketNotFound):
        ledger.resume("T-0404", actor="scheduler")
    with pytest.raises(TicketNotFound):
        ledger.current_halt("T-0404")


def test_resume_without_a_halt_record_is_reported(db_path: Path, clock: FakeClock) -> None:
    with Ledger(db_path, clock=clock) as led:
        ticket = new_ticket(led)
        led.halt(ticket.id, Halt(reason=HaltReason.BUDGET), actor="scheduler")
    conn = sqlite3.connect(db_path)
    try:
        with conn:
            conn.execute("DELETE FROM halts")
    finally:
        conn.close()
    with Ledger(db_path, clock=clock) as led:
        with pytest.raises(LedgerError, match="no open halt record"):
            led.resume(ticket.id, actor="scheduler")
        assert led.get_ticket(ticket.id).status is S.HALTED
