"""The ledger: the factory's single source of truth (SQLite, WAL, one connection).

Every mutating method runs in one `BEGIN IMMEDIATE` transaction, reads the injected clock exactly
once (after its preconditions hold) and appends at least one event. Models are stored as JSON
(`model_dump_json`) and validated on every read, so a corrupted row surfaces as an error instead
of a silently wrong snapshot.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from pydantic import BaseModel, JsonValue

from local_factory.hashing import canonical_json, content_hash, utc_now
from local_factory.ledger._schema import open_schema
from local_factory.ledger.errors import (
    ContractFrozen,
    IllegalTransition,
    LedgerError,
    TaskNotFound,
    TicketNotFound,
)
from local_factory.ledger.state_machine import TERMINAL, check_transition
from local_factory.models import (
    HASH_TAG_PATTERN,
    Contract,
    Event,
    Evidence,
    Halt,
    RiskTier,
    RunRecord,
    Task,
    TaskStatus,
    Ticket,
    TicketStatus,
    TicketType,
    Tier,
    Verdict,
)

__all__ = ["SYSTEM_ACTOR", "Clock", "EvidenceRecord", "Ledger"]

Clock = Callable[[], datetime]

# Actor for events appended by record_run / record_verdict / record_evidence, which take no actor:
# the factory itself records these facts.
SYSTEM_ACTOR = "factory"

_SIGNATURE_PATTERN = re.compile(r"[0-9a-f]+")

# Event kinds the ledger writes itself. `record_event` refuses them, so each keeps meaning exactly
# what the ledger says it means: nobody else can append a `transition` that moved nothing, or a
# `contract_frozen` for a contract that was never frozen.
RESERVED_EVENT_KINDS = frozenset(
    {
        "created",
        "transition",
        "contract_proposed",
        "oracle_version_set",
        "contract_frozen",
        "halted",
        "resumed",
        "task_created",
        "task_updated",
        "run_recorded",
        "verdict_recorded",
        "evidence_recorded",
        "triaged",
    }
)

_PROPOSABLE = frozenset({TicketStatus.TRIAGED, TicketStatus.NEEDS_INFO})

_NEXT_SEQ = {
    "events": "SELECT COALESCE(MAX(seq), 0) + 1 FROM events",
    "runs": "SELECT COALESCE(MAX(seq), 0) + 1 FROM runs",
    "verdicts": "SELECT COALESCE(MAX(seq), 0) + 1 FROM verdicts",
    "evidence": "SELECT COALESCE(MAX(seq), 0) + 1 FROM evidence",
}


@dataclass(frozen=True)
class EvidenceRecord:
    """Stored evidence plus the proofs it was recorded with (release checks re-verify them)."""

    evidence: Evidence
    evidence_hash: str
    signature: str


class Ledger:
    """Tickets, tasks, halts, runs, verdicts, evidence and the append-only event log."""

    def __init__(self, path: Path, *, clock: Clock = utc_now) -> None:
        self.path = path
        self._clock = clock
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, isolation_level=None)
        try:
            self._conn.row_factory = sqlite3.Row
            open_schema(self._conn, path)
        except BaseException:
            self._conn.close()
            raise

    # ------------------------------------------------------------------ lifecycle

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Close the connection. Idempotent."""
        self._conn.close()

    # ------------------------------------------------------------------ tickets

    def create_ticket(
        self, *, title: str, type: TicketType, risk: RiskTier, request: str, actor: str
    ) -> Ticket:
        with self._transaction():
            num = self._scalar("SELECT COALESCE(MAX(num), 0) + 1 FROM tickets")
            now = self._clock()
            ticket = Ticket(
                id=f"T-{num:04d}",
                title=title,
                type=type,
                risk=risk,
                status=TicketStatus.RECEIVED,
                request=request,
                created_at=now,
                updated_at=now,
            )
            self._conn.execute(
                "INSERT INTO tickets (num, id, status, body) VALUES (?, ?, ?, ?)",
                (num, ticket.id, ticket.status.value, _to_json(ticket)),
            )
            self._append_event(
                at=now,
                kind="created",
                actor=actor,
                ticket_id=ticket.id,
                to_status=ticket.status,
                data={"title": ticket.title, "type": ticket.type.value, "risk": ticket.risk.value},
            )
        return ticket

    def get_ticket(self, ticket_id: str) -> Ticket:
        return self._load_ticket(ticket_id)

    def list_tickets(self, status: TicketStatus | None = None) -> list[Ticket]:
        """All tickets (or those in `status`), ordered by id number."""
        if status is None:
            rows = self._conn.execute("SELECT body FROM tickets ORDER BY num").fetchall()
        else:
            rows = self._conn.execute(
                "SELECT body FROM tickets WHERE status = ? ORDER BY num", (status.value,)
            ).fetchall()
        return [Ticket.model_validate_json(row["body"]) for row in rows]

    def transition(
        self,
        ticket_id: str,
        to: TicketStatus,
        *,
        actor: str,
        reason: str = "",
        data: Mapping[str, JsonValue] | None = None,
    ) -> Ticket:
        """Move a ticket along the state machine.

        The event's data is `{"reason": reason}`, plus `"detail": {...data}` when `data` is given
        (e.g. a candidate conflict's paths), so the ledger alone explains the change.

        Rejected with `IllegalTransition`: anything not in the table, `to=HALTED` (use `halt()`),
        any transition of a HALTED ticket (use `resume()`), and `to=APPROVED` while the contract
        is not frozen (use `freeze_contract()`, which approves atomically with the freeze).
        """
        with self._transaction():
            ticket = self._load_ticket(ticket_id)
            if ticket.status is TicketStatus.HALTED:
                raise IllegalTransition(f"{ticket_id} is halted; resume() it before transitioning")
            if to is TicketStatus.HALTED:
                raise IllegalTransition(f"{ticket_id}: use halt() to halt a ticket")
            if to is TicketStatus.APPROVED and ticket.contract_hash is None:
                raise IllegalTransition(
                    f"{ticket_id}: approve with freeze_contract(), which freezes the contract"
                )
            self._check(ticket, to)
            return self._apply_transition(
                ticket, to, actor=actor, reason=reason, data=data, now=self._clock()
            )

    def triage(
        self,
        ticket_id: str,
        *,
        title: str,
        type: TicketType,
        risk: RiskTier,
        actor: str,
        data: Mapping[str, JsonValue] | None = None,
    ) -> Ticket:
        """RECEIVED -> TRIAGED with the triage result: title, type and risk set, a `triaged`
        event (`data` plus `title`, `type` and `risk`) and the transition, in one transaction."""
        with self._transaction():
            ticket = self._load_ticket(ticket_id)
            if ticket.status is not TicketStatus.RECEIVED:
                raise IllegalTransition(
                    f"{ticket_id}: only a received ticket can be triaged "
                    f"(status is {ticket.status})"
                )
            now = self._clock()
            triaged = self._save_ticket(ticket, now, title=title, type=type, risk=risk)
            self._append_event(
                at=now,
                kind="triaged",
                actor=actor,
                ticket_id=ticket_id,
                data={
                    **(data or {}),
                    "title": triaged.title,
                    "type": triaged.type.value,
                    "risk": triaged.risk.value,
                },
            )
            return self._apply_transition(
                triaged, TicketStatus.TRIAGED, actor=actor, reason="triaged", now=now
            )

    # ------------------------------------------------------------------ contract

    def propose_contract(self, ticket_id: str, contract: Contract, *, actor: str) -> Ticket:
        """Store (or replace) the proposed contract. Does not change the status."""
        with self._transaction():
            ticket = self._load_ticket(ticket_id)
            _ensure_not_frozen(ticket)
            return self._store_contract(ticket, contract, now=self._clock(), actor=actor)

    def set_oracle_version(self, ticket_id: str, version: str, *, actor: str) -> Ticket:
        with self._transaction():
            ticket = self._load_ticket(ticket_id)
            _ensure_not_frozen(ticket)
            return self._store_oracle_version(ticket, version, now=self._clock(), actor=actor)

    def propose(
        self, ticket_id: str, contract: Contract, oracle_version: str, *, actor: str
    ) -> Ticket:
        """Gate 1's proposal: store the contract and its oracle version and move the ticket to
        CONTRACT_PROPOSED, in one transaction.

        Only from TRIAGED or NEEDS_INFO, and only while the contract is not frozen. The events
        are the ones :meth:`propose_contract` and :meth:`set_oracle_version` write, followed by
        the transition.
        """
        with self._transaction():
            ticket = self._load_ticket(ticket_id)
            _ensure_not_frozen(ticket)
            if ticket.status not in _PROPOSABLE:
                raise IllegalTransition(
                    f"{ticket_id}: a contract can only be proposed in "
                    f"{' or '.join(sorted(_PROPOSABLE))} (status is {ticket.status})"
                )
            self._check(ticket, TicketStatus.CONTRACT_PROPOSED)
            now = self._clock()
            ticket = self._store_contract(ticket, contract, now=now, actor=actor)
            ticket = self._store_oracle_version(ticket, oracle_version, now=now, actor=actor)
            return self._apply_transition(
                ticket,
                TicketStatus.CONTRACT_PROPOSED,
                actor=actor,
                reason="contract proposed",
                now=now,
            )

    def freeze_contract(self, ticket_id: str, *, actor: str) -> Ticket:
        """Freeze contract + oracle version (hash recorded) and approve the ticket, atomically."""
        with self._transaction():
            ticket = self._load_ticket(ticket_id)
            _ensure_not_frozen(ticket)
            if ticket.status is not TicketStatus.CONTRACT_PROPOSED:
                raise LedgerError(
                    f"{ticket_id}: a contract can only be frozen in status "
                    f"{TicketStatus.CONTRACT_PROPOSED} (status is {ticket.status})"
                )
            if ticket.contract is None:
                raise LedgerError(f"{ticket_id}: no contract proposed; propose_contract() first")
            if ticket.oracle_version is None:
                raise LedgerError(f"{ticket_id}: no oracle version; set_oracle_version() first")
            self._check(ticket, TicketStatus.APPROVED)
            now = self._clock()
            contract_hash = content_hash(ticket.contract)
            frozen = self._save_ticket(ticket, now, contract_hash=contract_hash)
            self._append_event(
                at=now,
                kind="contract_frozen",
                actor=actor,
                ticket_id=ticket_id,
                data={
                    "version": ticket.contract.version,
                    "contract_hash": contract_hash,
                    "oracle_version": ticket.oracle_version,
                },
            )
            return self._apply_transition(
                frozen, TicketStatus.APPROVED, actor=actor, reason="contract frozen", now=now
            )

    # ------------------------------------------------------------------ halts

    def halt(self, ticket_id: str, halt: Halt, *, actor: str) -> Ticket:
        """Any non-terminal, non-halted status -> HALTED, remembering the prior status."""
        with self._transaction():
            ticket = self._load_ticket(ticket_id)
            if ticket.status is TicketStatus.HALTED:
                raise IllegalTransition(f"{ticket_id} is already halted")
            if ticket.status in TERMINAL:
                raise IllegalTransition(f"{ticket_id} is {ticket.status} (terminal); cannot halt")
            now = self._clock()
            self._conn.execute(
                "INSERT INTO halts (ticket_id, prior_status, body, halted_at) VALUES (?, ?, ?, ?)",
                (ticket_id, ticket.status.value, _to_json(halt), now.isoformat()),
            )
            updated = self._save_ticket(ticket, now, status=TicketStatus.HALTED)
            self._append_event(
                at=now,
                kind="halted",
                actor=actor,
                ticket_id=ticket_id,
                from_status=ticket.status,
                to_status=TicketStatus.HALTED,
                data=halt.model_dump(mode="json"),
            )
        return updated

    def resume(self, ticket_id: str, *, actor: str) -> Ticket:
        """HALTED -> exactly the status the ticket was halted in."""
        with self._transaction():
            ticket = self._load_ticket(ticket_id)
            if ticket.status is not TicketStatus.HALTED:
                raise IllegalTransition(f"{ticket_id} is {ticket.status}, not halted")
            row = self._open_halt(ticket_id)
            if row is None:
                raise LedgerError(f"{ticket_id} is halted but has no open halt record")
            prior = TicketStatus(row["prior_status"])
            now = self._clock()
            self._conn.execute(
                "UPDATE halts SET resumed_at = ? WHERE seq = ?", (now.isoformat(), row["seq"])
            )
            updated = self._save_ticket(ticket, now, status=prior)
            self._append_event(
                at=now,
                kind="resumed",
                actor=actor,
                ticket_id=ticket_id,
                from_status=TicketStatus.HALTED,
                to_status=prior,
            )
        return updated

    def current_halt(self, ticket_id: str) -> Halt | None:
        self._load_ticket(ticket_id)
        row = self._open_halt(ticket_id)
        return None if row is None else Halt.model_validate_json(row["body"])

    # ------------------------------------------------------------------ events

    def record_event(
        self,
        ticket_id: str,
        kind: str,
        *,
        actor: str,
        data: Mapping[str, JsonValue] | None = None,
    ) -> Event:
        """Append an event of the caller's own kind to an existing ticket's log.

        The ledger changes nothing else. Kinds the ledger writes itself
        (:data:`RESERVED_EVENT_KINDS`) are refused with `ValueError`, so they keep meaning what the
        ledger says; an unknown ticket is `TicketNotFound`.
        """
        if kind in RESERVED_EVENT_KINDS:
            raise ValueError(f"event kind {kind!r} is written by the ledger itself")
        with self._transaction():
            self._load_ticket(ticket_id)
            return self._append_event(
                at=self._clock(), kind=kind, actor=actor, ticket_id=ticket_id, data=dict(data or {})
            )

    def events(self, ticket_id: str | None = None) -> list[Event]:
        """All events (or one ticket's), ordered by the global sequence number."""
        if ticket_id is None:
            rows = self._conn.execute("SELECT * FROM events ORDER BY seq").fetchall()
        else:
            self._load_ticket(ticket_id)
            rows = self._conn.execute(
                "SELECT * FROM events WHERE ticket_id = ? ORDER BY seq", (ticket_id,)
            ).fetchall()
        return [_event_from_row(row) for row in rows]

    # ------------------------------------------------------------------ tasks

    def create_task(
        self,
        ticket_id: str,
        *,
        goal: str,
        role: str,
        tier: Tier,
        criteria_served: Sequence[str] = (),
        depends_on: Sequence[str] = (),
        actor: str,
    ) -> Task:
        """Create a PENDING task. Dependencies must be existing tasks of the same ticket."""
        with self._transaction():
            self._load_ticket(ticket_id)
            num = self._scalar(
                "SELECT COALESCE(MAX(num), 0) + 1 FROM tasks WHERE ticket_id = ?", (ticket_id,)
            )
            task = Task(
                id=f"{ticket_id}/task-{num}",
                ticket_id=ticket_id,
                goal=goal,
                criteria_served=criteria_served,
                depends_on=depends_on,
                role=role,
                tier=tier,
            )
            for dep in task.depends_on:
                if self._load_task(dep).ticket_id != ticket_id:
                    raise LedgerError(f"{task.id}: dependency {dep} belongs to another ticket")
            now = self._clock()
            self._conn.execute(
                "INSERT INTO tasks (id, ticket_id, num, body) VALUES (?, ?, ?, ?)",
                (task.id, ticket_id, num, _to_json(task)),
            )
            self._append_event(
                at=now,
                kind="task_created",
                actor=actor,
                ticket_id=ticket_id,
                data={
                    "task_id": task.id,
                    "role": task.role,
                    "tier": task.tier.value,
                    "depends_on": list(task.depends_on),
                },
            )
        return task

    def get_task(self, task_id: str) -> Task:
        return self._load_task(task_id)

    def list_tasks(self, ticket_id: str) -> list[Task]:
        """A ticket's tasks in creation order."""
        self._load_ticket(ticket_id)
        rows = self._conn.execute(
            "SELECT body FROM tasks WHERE ticket_id = ? ORDER BY num", (ticket_id,)
        ).fetchall()
        return [Task.model_validate_json(row["body"]) for row in rows]

    def update_task(
        self,
        task_id: str,
        *,
        actor: str,
        status: TaskStatus | None = None,
        base_sha: str | None = None,
        head_sha: str | None = None,
        reason: str = "",
        data: Mapping[str, JsonValue] | None = None,
    ) -> Task:
        """Set the given fields (None means unchanged). At least one field is required.

        `reason` and `data` (e.g. a rejection's violations) go into the event as
        `{"reason": ..., "detail": {...}}`, so the ledger alone explains the change.
        """
        changes = {
            name: value
            for name, value in (("status", status), ("base_sha", base_sha), ("head_sha", head_sha))
            if value is not None
        }
        with self._transaction():
            task = self._load_task(task_id)
            if not changes:
                raise LedgerError(f"{task_id}: update_task() needs status, base_sha or head_sha")
            updated = Task.model_validate({**task.model_dump(), **changes})
            now = self._clock()
            self._conn.execute(
                "UPDATE tasks SET body = ? WHERE id = ?", (_to_json(updated), task_id)
            )
            dumped = updated.model_dump(mode="json")
            self._append_event(
                at=now,
                kind="task_updated",
                actor=actor,
                ticket_id=task.ticket_id,
                data={
                    "task_id": task_id,
                    "changes": {name: dumped[name] for name in changes},
                    "reason": reason,
                    "detail": dict(data or {}),
                },
            )
        return updated

    # ------------------------------------------------------------------ runs

    def record_run(self, run: RunRecord) -> None:
        with self._transaction():
            self._load_ticket(run.ticket_id)
            if run.task_id is not None and self._load_task(run.task_id).ticket_id != run.ticket_id:
                raise LedgerError(f"run {run.id}: task {run.task_id} is not in {run.ticket_id}")
            if self._conn.execute("SELECT 1 FROM runs WHERE id = ?", (run.id,)).fetchone():
                raise LedgerError(f"run {run.id} is already recorded")
            now = self._clock()
            self._conn.execute(
                "INSERT INTO runs (seq, id, ticket_id, body) VALUES (?, ?, ?, ?)",
                (self._next_seq("runs"), run.id, run.ticket_id, _to_json(run)),
            )
            self._append_event(
                at=now,
                kind="run_recorded",
                actor=SYSTEM_ACTOR,
                ticket_id=run.ticket_id,
                data={
                    "run_id": run.id,
                    "task_id": run.task_id,
                    "role": run.role,
                    "status": run.status,
                    "error": run.error,
                    "cost_usd": run.usage.cost_usd,
                },
            )

    def runs(self, ticket_id: str) -> list[RunRecord]:
        """A ticket's runs in recording order."""
        self._load_ticket(ticket_id)
        rows = self._conn.execute(
            "SELECT body FROM runs WHERE ticket_id = ? ORDER BY seq", (ticket_id,)
        ).fetchall()
        return [RunRecord.model_validate_json(row["body"]) for row in rows]

    def ticket_cost_usd(self, ticket_id: str) -> float:
        return math.fsum(run.usage.cost_usd for run in self.runs(ticket_id))

    # ------------------------------------------------------------------ verdicts & evidence

    def record_verdict(self, verdict: Verdict) -> None:
        with self._transaction():
            self._load_ticket(verdict.ticket_id)
            now = self._clock()
            self._conn.execute(
                "INSERT INTO verdicts (seq, ticket_id, sha, body) VALUES (?, ?, ?, ?)",
                (
                    self._next_seq("verdicts"),
                    verdict.ticket_id,
                    verdict.sha,
                    _to_json(verdict),
                ),
            )
            self._append_event(
                at=now,
                kind="verdict_recorded",
                actor=SYSTEM_ACTOR,
                ticket_id=verdict.ticket_id,
                data={
                    "round": verdict.round,
                    "sha": verdict.sha,
                    "passed": verdict.passed,
                    "failing": list(verdict.failing),
                },
            )

    def verdicts(self, ticket_id: str) -> list[Verdict]:
        """A ticket's verdicts in recording order."""
        self._load_ticket(ticket_id)
        rows = self._conn.execute(
            "SELECT body FROM verdicts WHERE ticket_id = ? ORDER BY seq", (ticket_id,)
        ).fetchall()
        return [Verdict.model_validate_json(row["body"]) for row in rows]

    def record_evidence(self, evidence: Evidence, *, evidence_hash: str, signature: str) -> None:
        """Store signed evidence. `evidence_hash` is a `sha256:` tag; `signature` lowercase hex."""
        if not re.fullmatch(HASH_TAG_PATTERN, evidence_hash):
            raise ValueError(f"evidence_hash must match {HASH_TAG_PATTERN}: {evidence_hash!r}")
        if not _SIGNATURE_PATTERN.fullmatch(signature):
            raise ValueError(f"signature must be lowercase hex: {signature!r}")
        with self._transaction():
            self._load_ticket(evidence.ticket_id)
            now = self._clock()
            self._conn.execute(
                "INSERT INTO evidence (seq, ticket_id, sha, evidence_hash, signature, body)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (
                    self._next_seq("evidence"),
                    evidence.ticket_id,
                    evidence.candidate_sha,
                    evidence_hash,
                    signature,
                    _to_json(evidence),
                ),
            )
            self._append_event(
                at=now,
                kind="evidence_recorded",
                actor=SYSTEM_ACTOR,
                ticket_id=evidence.ticket_id,
                data={
                    "sha": evidence.candidate_sha,
                    "evidence_hash": evidence_hash,
                    "passed": evidence.passed,
                },
            )

    def evidence_for(self, ticket_id: str, sha: str) -> Evidence | None:
        """The most recently recorded evidence for candidate `sha`, or None."""
        record = self.evidence_record(ticket_id, sha)
        return None if record is None else record.evidence

    def evidence_record(self, ticket_id: str, sha: str) -> EvidenceRecord | None:
        """Like `evidence_for`, with the hash and signature it was recorded with."""
        self._load_ticket(ticket_id)
        row = self._conn.execute(
            "SELECT body, evidence_hash, signature FROM evidence"
            " WHERE ticket_id = ? AND sha = ? ORDER BY seq DESC LIMIT 1",
            (ticket_id, sha),
        ).fetchone()
        if row is None:
            return None
        return EvidenceRecord(
            evidence=Evidence.model_validate_json(row["body"]),
            evidence_hash=row["evidence_hash"],
            signature=row["signature"],
        )

    # ------------------------------------------------------------------ internals

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise
        self._conn.execute("COMMIT")

    def _scalar(self, sql: str, params: Sequence[object] = ()) -> int:
        return int(self._conn.execute(sql, params).fetchone()[0])

    def _next_seq(self, table: str) -> int:
        # Append-only tables must grow at exactly max(seq) + 1 (enforced by triggers).
        return self._scalar(_NEXT_SEQ[table])

    def _load_ticket(self, ticket_id: str) -> Ticket:
        row = self._conn.execute("SELECT body FROM tickets WHERE id = ?", (ticket_id,)).fetchone()
        if row is None:
            raise TicketNotFound(f"unknown ticket {ticket_id!r}")
        return Ticket.model_validate_json(row["body"])

    def _save_ticket(self, ticket: Ticket, now: datetime, **changes: Any) -> Ticket:
        """Validate `ticket` with `changes` applied and `updated_at=now`, and persist it."""
        updated = Ticket.model_validate({**ticket.model_dump(), **changes, "updated_at": now})
        self._conn.execute(
            "UPDATE tickets SET status = ?, body = ? WHERE id = ?",
            (updated.status.value, _to_json(updated), updated.id),
        )
        return updated

    def _load_task(self, task_id: str) -> Task:
        row = self._conn.execute("SELECT body FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if row is None:
            raise TaskNotFound(f"unknown task {task_id!r}")
        return Task.model_validate_json(row["body"])

    def _open_halt(self, ticket_id: str) -> sqlite3.Row | None:
        row: sqlite3.Row | None = self._conn.execute(
            "SELECT seq, prior_status, body FROM halts WHERE ticket_id = ? AND resumed_at IS NULL",
            (ticket_id,),
        ).fetchone()
        return row

    @staticmethod
    def _check(ticket: Ticket, to: TicketStatus) -> None:
        try:
            check_transition(ticket.status, to)
        except IllegalTransition as exc:
            raise IllegalTransition(f"{ticket.id}: {exc}") from exc

    def _apply_transition(
        self,
        ticket: Ticket,
        to: TicketStatus,
        *,
        actor: str,
        reason: str,
        now: datetime,
        data: Mapping[str, JsonValue] | None = None,
    ) -> Ticket:
        updated = self._save_ticket(ticket, now, status=to)
        event_data: dict[str, JsonValue] = {"reason": reason}
        if data is not None:
            event_data["detail"] = dict(data)  # a copy: the caller's mapping is not kept
        self._append_event(
            at=now,
            kind="transition",
            actor=actor,
            ticket_id=ticket.id,
            from_status=ticket.status,
            to_status=to,
            data=event_data,
        )
        return updated

    def _store_contract(
        self, ticket: Ticket, contract: Contract, *, now: datetime, actor: str
    ) -> Ticket:
        updated = self._save_ticket(ticket, now, contract=contract)
        self._append_event(
            at=now,
            kind="contract_proposed",
            actor=actor,
            ticket_id=ticket.id,
            data={"version": contract.version, "contract_hash": content_hash(contract)},
        )
        return updated

    def _store_oracle_version(
        self, ticket: Ticket, version: str, *, now: datetime, actor: str
    ) -> Ticket:
        updated = self._save_ticket(ticket, now, oracle_version=version)
        self._append_event(
            at=now,
            kind="oracle_version_set",
            actor=actor,
            ticket_id=ticket.id,
            data={"oracle_version": version},
        )
        return updated

    def _append_event(
        self,
        *,
        at: datetime,
        kind: str,
        actor: str,
        ticket_id: str | None,
        from_status: TicketStatus | None = None,
        to_status: TicketStatus | None = None,
        data: dict[str, JsonValue] | None = None,
    ) -> Event:
        # Validate the complete event before writing it; a failure aborts the whole transaction.
        event = Event(
            seq=self._next_seq("events"),
            ticket_id=ticket_id,
            at=at,
            kind=kind,
            actor=actor,
            from_status=from_status,
            to_status=to_status,
            data=data or {},
        )
        row = event.model_dump(mode="json")
        self._conn.execute(
            "INSERT INTO events (seq, ticket_id, at, kind, actor, from_status, to_status, data)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                row["seq"],
                row["ticket_id"],
                row["at"],
                row["kind"],
                row["actor"],
                row["from_status"],
                row["to_status"],
                canonical_json(row["data"]).decode(),
            ),
        )
        return event


def _to_json(model: BaseModel) -> str:
    """`model_dump_json`, refusing values that cannot be read back.

    Pydantic writes NaN and +-inf as `null`, which then fails validation on every later read; in
    an append-only table that would make the record (and its whole ticket's list) unreadable for
    good. `canonical_json` raises ValueError for them instead.
    """
    canonical_json(model)
    return model.model_dump_json()


def _ensure_not_frozen(ticket: Ticket) -> None:
    if ticket.contract_hash is not None:
        raise ContractFrozen(
            f"{ticket.id}: the contract is frozen ({ticket.contract_hash}); "
            "contract and oracle version can no longer change"
        )


def _event_from_row(row: sqlite3.Row) -> Event:
    return Event.model_validate(
        {
            "seq": row["seq"],
            "ticket_id": row["ticket_id"],
            "at": row["at"],
            "kind": row["kind"],
            "actor": row["actor"],
            "from_status": row["from_status"],
            "to_status": row["to_status"],
            "data": json.loads(row["data"]),
        }
    )
