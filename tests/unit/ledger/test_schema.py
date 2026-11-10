"""Unit tests for opening, reopening, closing and read-side validation."""

from __future__ import annotations

import sqlite3
from datetime import timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from local_factory.ledger import Ledger, LedgerError
from local_factory.ledger._schema import SCHEMA_VERSION
from local_factory.models import Halt, HaltReason, TaskStatus, TicketStatus, Tier

from .helpers import HASH, SHA_A, FakeClock, make_evidence, make_run, make_verdict, new_ticket


def _query(path: Path, sql: str) -> list[tuple[object, ...]]:
    conn = sqlite3.connect(path)
    try:
        return conn.execute(sql).fetchall()
    finally:
        conn.close()


def _execute(path: Path, sql: str) -> None:
    conn = sqlite3.connect(path)
    try:
        with conn:
            conn.execute(sql)
    finally:
        conn.close()


def test_new_ledger_uses_wal_and_records_the_schema_version(db_path: Path) -> None:
    with Ledger(db_path) as led:
        assert led.path == db_path
    assert _query(db_path, "PRAGMA journal_mode") == [("wal",)]
    assert _query(db_path, "PRAGMA user_version") == [(SCHEMA_VERSION,)]


def test_parent_directories_are_created(tmp_path: Path) -> None:
    path = tmp_path / "projects" / "notes" / "ledger.db"
    with Ledger(path) as led:
        new_ticket(led)
    assert path.is_file()


def test_default_clock_is_utc(db_path: Path) -> None:
    with Ledger(db_path) as led:
        ticket = new_ticket(led)
    assert ticket.created_at.utcoffset() == timedelta(0)


def test_reopen_preserves_every_kind_of_record(db_path: Path, clock: FakeClock) -> None:
    with Ledger(db_path, clock=clock) as led:
        ticket = new_ticket(led)
        task = led.create_task(ticket.id, goal="g", role="r", tier=Tier.APEX, actor="o")
        led.update_task(task.id, actor="factory", status=TaskStatus.SUBMITTED)
        led.record_run(make_run(ticket.id))
        led.record_verdict(make_verdict(ticket.id))
        led.record_evidence(make_evidence(ticket.id), evidence_hash=HASH, signature="00")
        led.halt(ticket.id, Halt(reason=HaltReason.BUDGET), actor="scheduler")
        snapshot = (
            led.list_tickets(),
            led.list_tasks(ticket.id),
            led.runs(ticket.id),
            led.verdicts(ticket.id),
            led.evidence_record(ticket.id, SHA_A),
            led.current_halt(ticket.id),
            led.events(),
        )
    with Ledger(db_path, clock=clock) as led:
        assert (
            led.list_tickets(),
            led.list_tasks(ticket.id),
            led.runs(ticket.id),
            led.verdicts(ticket.id),
            led.evidence_record(ticket.id, SHA_A),
            led.current_halt(ticket.id),
            led.events(),
        ) == snapshot
        assert led.resume(ticket.id, actor="scheduler").status is TicketStatus.RECEIVED


def test_unsupported_schema_version_is_refused(db_path: Path) -> None:
    with Ledger(db_path):
        pass
    _execute(db_path, f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    with pytest.raises(LedgerError, match="unsupported ledger schema version"):
        Ledger(db_path)


def test_a_file_that_is_not_a_database_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "ledger.db"
    path.write_text("definitely not sqlite, " * 10)
    with pytest.raises(LedgerError, match="not a ledger database"):
        Ledger(path)
    assert path.read_text().startswith("definitely not sqlite")


def test_close_is_idempotent_and_the_context_manager_closes(db_path: Path) -> None:
    with Ledger(db_path) as led:
        new_ticket(led)
    with pytest.raises(sqlite3.ProgrammingError):
        led.list_tickets()
    led.close()
    explicit = Ledger(db_path)
    explicit.close()
    explicit.close()


def test_rows_are_validated_on_read(db_path: Path, clock: FakeClock) -> None:
    with Ledger(db_path, clock=clock) as led:
        ticket = new_ticket(led)
        task = led.create_task(ticket.id, goal="g", role="r", tier=Tier.FAST, actor="o")
    _execute(db_path, "UPDATE tickets SET body = json_set(body, '$.status', 'teleported')")
    _execute(db_path, "UPDATE tasks SET body = json_set(body, '$.tier', 'ultra')")
    with Ledger(db_path, clock=clock) as led:
        with pytest.raises(ValidationError):
            led.get_ticket(ticket.id)
        with pytest.raises(ValidationError):
            led.list_tickets()
        with pytest.raises(ValidationError):
            led.get_task(task.id)
