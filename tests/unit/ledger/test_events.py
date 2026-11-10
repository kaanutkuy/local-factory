"""Unit tests for the event log and the database-level append-only guarantees."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from local_factory.ledger import Ledger, TicketNotFound
from local_factory.models import TicketStatus

from .helpers import HASH, FakeClock, make_evidence, make_run, make_verdict, new_ticket

S = TicketStatus


@pytest.fixture
def populated(db_path: Path, clock: FakeClock) -> Iterator[sqlite3.Connection]:
    """A ledger with one row in every append-only table, opened by a foreign connection."""
    with Ledger(db_path, clock=clock) as led:
        ticket = new_ticket(led)
        led.record_run(make_run(ticket.id))
        led.record_verdict(make_verdict(ticket.id))
        led.record_evidence(make_evidence(ticket.id), evidence_hash=HASH, signature="00")
    conn = sqlite3.connect(db_path)
    try:
        yield conn
    finally:
        conn.close()


SNAPSHOT = {
    "events": "SELECT * FROM events ORDER BY seq",
    "runs": "SELECT * FROM runs ORDER BY seq",
    "verdicts": "SELECT * FROM verdicts ORDER BY seq",
    "evidence": "SELECT * FROM evidence ORDER BY seq",
}

TAMPERING = {
    "events": [
        "UPDATE events SET actor = 'forged'",
        "DELETE FROM events",
        "INSERT OR REPLACE INTO events (seq, ticket_id, at, kind, actor, data)"
        " VALUES (1, NULL, '2026-01-01T00:00:00Z', 'created', 'forged', '{}')",
        "INSERT INTO events (seq, ticket_id, at, kind, actor, data)"
        " VALUES (1, NULL, '2026-01-01T00:00:00Z', 'created', 'forged', '{}')"
        " ON CONFLICT (seq) DO UPDATE SET actor = 'forged'",
        "INSERT INTO events (ticket_id, at, kind, actor, data)"
        " VALUES (NULL, '2026-01-01T00:00:00Z', 'created', 'unsequenced', '{}')",
        "INSERT INTO events (seq, ticket_id, at, kind, actor, data)"
        " VALUES (999, NULL, '2026-01-01T00:00:00Z', 'created', 'gap', '{}')",
    ],
    "runs": [
        "UPDATE runs SET body = '{}'",
        "DELETE FROM runs",
        "INSERT OR REPLACE INTO runs (seq, id, ticket_id, body) VALUES (1, 'x', 'T-0001', '{}')",
        # A fresh seq but a duplicate id: REPLACE would silently delete the original run.
        "REPLACE INTO runs (seq, id, ticket_id, body) VALUES (2, 'run-1', 'T-0001', '{}')",
    ],
    "verdicts": [
        "UPDATE verdicts SET body = '{}'",
        "DELETE FROM verdicts",
        "INSERT OR REPLACE INTO verdicts (seq, ticket_id, sha, body)"
        " VALUES (1, 'T-0001', 'x', '{}')",
    ],
    "evidence": [
        "UPDATE evidence SET signature = 'ff'",
        "DELETE FROM evidence",
        "INSERT OR REPLACE INTO evidence (seq, ticket_id, sha, evidence_hash, signature, body)"
        " VALUES (1, 'T-0001', 'x', 'h', 's', '{}')",
    ],
}


@pytest.mark.parametrize(
    ("table", "sql"), [(table, sql) for table, stmts in TAMPERING.items() for sql in stmts]
)
def test_history_cannot_be_rewritten_from_outside(
    populated: sqlite3.Connection, table: str, sql: str
) -> None:
    before = populated.execute(SNAPSHOT[table]).fetchall()
    assert before
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        populated.execute(sql)
    populated.rollback()
    assert populated.execute(SNAPSHOT[table]).fetchall() == before


def test_a_foreign_append_at_the_next_seq_is_allowed_and_validated_on_read(
    populated: sqlite3.Connection, db_path: Path, clock: FakeClock
) -> None:
    (next_seq,) = populated.execute("SELECT MAX(seq) + 1 FROM events").fetchone()
    with populated:
        populated.execute(
            "INSERT INTO events (seq, ticket_id, at, kind, actor, data)"
            " VALUES (?, 'T-0001', '2026-02-01T00:00:00Z', 'note', 'ops', '{\"k\": 1}')",
            (next_seq,),
        )
    with Ledger(db_path, clock=clock) as led:
        last = led.events()[-1]
        assert (last.seq, last.kind, last.actor, last.data) == (next_seq, "note", "ops", {"k": 1})
        # The ledger keeps appending after the foreign event.
        led.transition("T-0001", S.TRIAGED, actor="triage")
        assert led.events()[-1].seq == next_seq + 1


def test_events_are_densely_numbered_and_filterable(ledger: Ledger) -> None:
    a, b = new_ticket(ledger, "a"), new_ticket(ledger, "b")
    ledger.transition(a.id, S.TRIAGED, actor="triage")
    ledger.transition(b.id, S.NEEDS_INFO, actor="triage")
    ledger.transition(a.id, S.ABANDONED, actor="human")
    everything = ledger.events()
    assert [e.seq for e in everything] == [1, 2, 3, 4, 5]
    assert [e.seq for e in ledger.events(a.id)] == [1, 3, 5]
    assert [e.seq for e in ledger.events(b.id)] == [2, 4]
    assert {e.ticket_id for e in everything} == {a.id, b.id}


def test_events_of_unknown_ticket(ledger: Ledger) -> None:
    assert ledger.events() == []
    with pytest.raises(TicketNotFound):
        ledger.events("T-0001")


def test_events_survive_reopen(db_path: Path, clock: FakeClock) -> None:
    with Ledger(db_path, clock=clock) as led:
        ticket = new_ticket(led)
        led.transition(ticket.id, S.TRIAGED, actor="triage", reason="ok")
        before = led.events()
    with Ledger(db_path, clock=clock) as led:
        assert led.events() == before
