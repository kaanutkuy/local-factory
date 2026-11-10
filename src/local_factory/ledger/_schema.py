"""SQLite schema for the ledger. Bump `SCHEMA_VERSION` (and add a migration) on any change.

`events`, `runs`, `verdicts` and `evidence` are append-only *in the database*: triggers abort any
UPDATE or DELETE, and any INSERT that does not append exactly at `max(seq) + 1`. The insert guard
matters because `INSERT OR REPLACE` / upserts would otherwise overwrite history: SQLite does not
fire DELETE triggers for REPLACE conflict resolution unless recursive triggers are enabled.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from local_factory.ledger.errors import LedgerError

SCHEMA_VERSION = 1

APPEND_ONLY_TABLES = ("events", "runs", "verdicts", "evidence")

_SCHEMA = """
BEGIN IMMEDIATE;

CREATE TABLE IF NOT EXISTS tickets (
    num INTEGER PRIMARY KEY,
    id TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL,
    body TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS tickets_by_status ON tickets (status, num);

CREATE TABLE IF NOT EXISTS halts (
    seq INTEGER PRIMARY KEY,
    ticket_id TEXT NOT NULL REFERENCES tickets (id),
    prior_status TEXT NOT NULL,
    body TEXT NOT NULL,
    halted_at TEXT NOT NULL,
    resumed_at TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS halts_one_open_per_ticket
    ON halts (ticket_id) WHERE resumed_at IS NULL;

CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    ticket_id TEXT NOT NULL REFERENCES tickets (id),
    num INTEGER NOT NULL,
    body TEXT NOT NULL,
    UNIQUE (ticket_id, num)
);

CREATE TABLE IF NOT EXISTS runs (
    seq INTEGER PRIMARY KEY,
    id TEXT NOT NULL UNIQUE,
    ticket_id TEXT NOT NULL REFERENCES tickets (id),
    body TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS runs_by_ticket ON runs (ticket_id, seq);

CREATE TABLE IF NOT EXISTS verdicts (
    seq INTEGER PRIMARY KEY,
    ticket_id TEXT NOT NULL REFERENCES tickets (id),
    sha TEXT NOT NULL,
    body TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS verdicts_by_ticket ON verdicts (ticket_id, seq);

CREATE TABLE IF NOT EXISTS evidence (
    seq INTEGER PRIMARY KEY,
    ticket_id TEXT NOT NULL REFERENCES tickets (id),
    sha TEXT NOT NULL,
    evidence_hash TEXT NOT NULL,
    signature TEXT NOT NULL,
    body TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS evidence_by_sha ON evidence (ticket_id, sha, seq);

CREATE TABLE IF NOT EXISTS events (
    seq INTEGER PRIMARY KEY,
    ticket_id TEXT REFERENCES tickets (id),
    at TEXT NOT NULL,
    kind TEXT NOT NULL,
    actor TEXT NOT NULL,
    from_status TEXT,
    to_status TEXT,
    data TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS events_by_ticket ON events (ticket_id, seq);

CREATE TRIGGER IF NOT EXISTS events_no_update BEFORE UPDATE ON events
BEGIN SELECT RAISE(ABORT, 'events is append-only'); END;
CREATE TRIGGER IF NOT EXISTS events_no_delete BEFORE DELETE ON events
BEGIN SELECT RAISE(ABORT, 'events is append-only'); END;
CREATE TRIGGER IF NOT EXISTS events_append_only BEFORE INSERT ON events
WHEN NEW.seq IS NULL OR NEW.seq IS NOT (SELECT COALESCE(MAX(seq), 0) + 1 FROM events)
BEGIN SELECT RAISE(ABORT, 'events is append-only: seq must be max(seq) + 1'); END;

CREATE TRIGGER IF NOT EXISTS runs_no_update BEFORE UPDATE ON runs
BEGIN SELECT RAISE(ABORT, 'runs is append-only'); END;
CREATE TRIGGER IF NOT EXISTS runs_no_delete BEFORE DELETE ON runs
BEGIN SELECT RAISE(ABORT, 'runs is append-only'); END;
CREATE TRIGGER IF NOT EXISTS runs_append_only BEFORE INSERT ON runs
WHEN NEW.seq IS NULL OR NEW.seq IS NOT (SELECT COALESCE(MAX(seq), 0) + 1 FROM runs)
    OR EXISTS (SELECT 1 FROM runs WHERE id = NEW.id)
BEGIN SELECT RAISE(ABORT, 'runs is append-only: seq must be max(seq) + 1, ids unique'); END;

CREATE TRIGGER IF NOT EXISTS verdicts_no_update BEFORE UPDATE ON verdicts
BEGIN SELECT RAISE(ABORT, 'verdicts is append-only'); END;
CREATE TRIGGER IF NOT EXISTS verdicts_no_delete BEFORE DELETE ON verdicts
BEGIN SELECT RAISE(ABORT, 'verdicts is append-only'); END;
CREATE TRIGGER IF NOT EXISTS verdicts_append_only BEFORE INSERT ON verdicts
WHEN NEW.seq IS NULL OR NEW.seq IS NOT (SELECT COALESCE(MAX(seq), 0) + 1 FROM verdicts)
BEGIN SELECT RAISE(ABORT, 'verdicts is append-only: seq must be max(seq) + 1'); END;

CREATE TRIGGER IF NOT EXISTS evidence_no_update BEFORE UPDATE ON evidence
BEGIN SELECT RAISE(ABORT, 'evidence is append-only'); END;
CREATE TRIGGER IF NOT EXISTS evidence_no_delete BEFORE DELETE ON evidence
BEGIN SELECT RAISE(ABORT, 'evidence is append-only'); END;
CREATE TRIGGER IF NOT EXISTS evidence_append_only BEFORE INSERT ON evidence
WHEN NEW.seq IS NULL OR NEW.seq IS NOT (SELECT COALESCE(MAX(seq), 0) + 1 FROM evidence)
BEGIN SELECT RAISE(ABORT, 'evidence is append-only: seq must be max(seq) + 1'); END;

PRAGMA user_version = 1;
COMMIT;
"""


def open_schema(conn: sqlite3.Connection, path: Path) -> None:
    """Enable WAL and foreign keys; create the schema in a new file, or check an existing one."""
    try:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA foreign_keys = ON")
        version = int(conn.execute("PRAGMA user_version").fetchone()[0])
    except sqlite3.DatabaseError as exc:
        raise LedgerError(f"{path}: not a ledger database ({exc})") from exc
    if version == 0:
        conn.executescript(_SCHEMA)
    elif version != SCHEMA_VERSION:
        raise LedgerError(
            f"{path}: unsupported ledger schema version {version} (expected {SCHEMA_VERSION})"
        )
