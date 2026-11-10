"""Unit tests for runs, cost accounting, verdicts and evidence."""

from __future__ import annotations

import dataclasses
import math
import sqlite3
from pathlib import Path

import pytest

from local_factory.ledger import (
    EvidenceRecord,
    Ledger,
    LedgerError,
    TaskNotFound,
    TicketNotFound,
)
from local_factory.ledger.store import SYSTEM_ACTOR
from local_factory.models import GateResult, GateStatus, Tier, Usage

from .helpers import (
    HASH,
    HASH_2,
    SHA_A,
    SHA_B,
    FakeClock,
    make_evidence,
    make_run,
    make_verdict,
    new_ticket,
)

# --------------------------------------------------------------------------- runs


def test_runs_round_trip_every_field(ledger: Ledger, clock: FakeClock) -> None:
    ticket = new_ticket(ledger)
    task = ledger.create_task(ticket.id, goal="g", role="implementer", tier=Tier.FAST, actor="o")
    run = make_run(ticket.id, task_id=task.id, cost=0.25).model_copy(
        update={"session_id": "sess-1", "transcript_path": Path("runs/run-1.jsonl")}
    )
    ledger.record_run(run)
    assert ledger.runs(ticket.id) == [run]
    event = ledger.events(ticket.id)[-1]
    assert (event.kind, event.actor, event.at) == ("run_recorded", SYSTEM_ACTOR, clock.now)
    assert event.data == {
        "run_id": "run-1",
        "task_id": task.id,
        "role": "implementer",
        "status": "ok",
        "error": None,
        "cost_usd": 0.25,
    }


def test_a_failed_run_keeps_its_error_in_the_record_and_the_event(
    db_path: Path, clock: FakeClock
) -> None:
    with Ledger(db_path, clock=clock) as led:
        ticket = new_ticket(led)
        failed = make_run(ticket.id).model_copy(
            update={"status": "error", "error": "RunCommand exited 2: pytest -q"}
        )
        led.record_run(failed)
    conn = sqlite3.connect(db_path)
    try:
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            conn.execute("UPDATE runs SET body = json_set(body, '$.error', NULL)")
    finally:
        conn.close()
    with Ledger(db_path, clock=clock) as led:
        assert led.runs(ticket.id) == [failed]
        event = led.events(ticket.id)[-1]
        assert (event.data["status"], event.data["error"]) == (
            "error",
            "RunCommand exited 2: pytest -q",
        )


def test_runs_are_per_ticket_in_recording_order(ledger: Ledger) -> None:
    a, b = new_ticket(ledger, "a"), new_ticket(ledger, "b")
    ledger.record_run(make_run(a.id, "run-z", cost=1.0))
    ledger.record_run(make_run(b.id, "run-y", cost=2.0))
    ledger.record_run(make_run(a.id, "run-a", cost=0.5))
    assert [r.id for r in ledger.runs(a.id)] == ["run-z", "run-a"]
    assert ledger.ticket_cost_usd(a.id) == pytest.approx(1.5)
    assert ledger.ticket_cost_usd(b.id) == pytest.approx(2.0)


def test_cost_of_a_ticket_without_runs_is_zero(ledger: Ledger) -> None:
    ticket = new_ticket(ledger)
    assert ledger.runs(ticket.id) == []
    assert ledger.ticket_cost_usd(ticket.id) == 0.0


def test_cost_sums_without_float_drift(ledger: Ledger) -> None:
    ticket = new_ticket(ledger)
    for i in range(10):
        ledger.record_run(make_run(ticket.id, f"run-{i}", cost=0.1))
    assert ledger.ticket_cost_usd(ticket.id) == 1.0


def test_record_run_rejects_unknown_or_foreign_references(ledger: Ledger) -> None:
    a, b = new_ticket(ledger, "a"), new_ticket(ledger, "b")
    foreign = ledger.create_task(b.id, goal="g", role="r", tier=Tier.FAST, actor="o")
    with pytest.raises(TicketNotFound):
        ledger.record_run(make_run("T-0404"))
    with pytest.raises(TaskNotFound):
        ledger.record_run(make_run(a.id, task_id=f"{a.id}/task-1"))
    with pytest.raises(LedgerError, match="not in T-0001"):
        ledger.record_run(make_run(a.id, task_id=foreign.id))
    assert ledger.runs(a.id) == []


def test_run_ids_are_unique(ledger: Ledger) -> None:
    a, b = new_ticket(ledger, "a"), new_ticket(ledger, "b")
    ledger.record_run(make_run(a.id, "run-1", cost=1.0))
    events = ledger.events()
    with pytest.raises(LedgerError, match="already recorded"):
        ledger.record_run(make_run(b.id, "run-1", cost=9.0))
    assert ledger.events() == events
    assert ledger.ticket_cost_usd(b.id) == 0.0


def test_run_queries_on_unknown_ticket(ledger: Ledger) -> None:
    with pytest.raises(TicketNotFound):
        ledger.runs("T-0404")
    with pytest.raises(TicketNotFound):
        ledger.ticket_cost_usd("T-0404")


# --------------------------------------------------------------------------- verdicts


def test_verdicts_round_trip_in_order(ledger: Ledger, clock: FakeClock) -> None:
    a, b = new_ticket(ledger, "a"), new_ticket(ledger, "b")
    first = make_verdict(a.id, round=1, passed=False)
    second = make_verdict(a.id, round=2, sha=SHA_B)
    ledger.record_verdict(first)
    ledger.record_verdict(make_verdict(b.id))
    ledger.record_verdict(second)
    assert ledger.verdicts(a.id) == [first, second]
    assert ledger.verdicts(b.id) == [make_verdict(b.id)]
    failing_event = ledger.events(a.id)[-2]
    assert (failing_event.kind, failing_event.actor) == ("verdict_recorded", SYSTEM_ACTOR)
    assert failing_event.data == {
        "round": 1,
        "sha": SHA_A,
        "passed": False,
        "failing": ["gate:unit"],
    }
    assert ledger.events(a.id)[-1].at == clock.now


def test_verdict_operations_on_unknown_ticket(ledger: Ledger) -> None:
    with pytest.raises(TicketNotFound):
        ledger.record_verdict(make_verdict("T-0404"))
    with pytest.raises(TicketNotFound):
        ledger.verdicts("T-0404")
    assert ledger.events() == []


def test_no_verdicts_yet(ledger: Ledger) -> None:
    assert ledger.verdicts(new_ticket(ledger).id) == []


# --------------------------------------------------------------------------- evidence


def test_evidence_for_returns_the_most_recently_recorded(ledger: Ledger) -> None:
    ticket = new_ticket(ledger)
    later_created = make_evidence(ticket.id, offset_s=60)
    earlier_created = make_evidence(ticket.id, offset_s=1)
    ledger.record_evidence(later_created, evidence_hash=HASH, signature="aa")
    ledger.record_evidence(earlier_created, evidence_hash=HASH_2, signature="bb")
    # "Latest" is recording order (the ledger's clock), not the evidence's own created_at.
    assert ledger.evidence_for(ticket.id, SHA_A) == earlier_created
    event = ledger.events(ticket.id)[-1]
    assert (event.kind, event.actor) == ("evidence_recorded", SYSTEM_ACTOR)
    assert event.data == {"sha": SHA_A, "evidence_hash": HASH_2, "passed": True}


def test_evidence_is_scoped_to_ticket_and_sha(ledger: Ledger) -> None:
    a, b = new_ticket(ledger, "a"), new_ticket(ledger, "b")
    ledger.record_evidence(make_evidence(a.id, sha=SHA_A), evidence_hash=HASH, signature="00")
    ledger.record_evidence(make_evidence(b.id, sha=SHA_B), evidence_hash=HASH, signature="00")
    assert ledger.evidence_for(a.id, SHA_A) == make_evidence(a.id, sha=SHA_A)
    assert ledger.evidence_for(a.id, SHA_B) is None
    assert ledger.evidence_for(b.id, SHA_A) is None


@pytest.mark.parametrize(
    ("evidence_hash", "signature", "message"),
    [
        ("c" * 64, "00", "evidence_hash"),
        ("sha256:" + "C" * 64, "00", "evidence_hash"),
        (HASH, "", "signature"),
        (HASH, "ABCD", "signature"),
        (HASH, "zz", "signature"),
    ],
)
def test_record_evidence_validates_hash_and_signature(
    ledger: Ledger, evidence_hash: str, signature: str, message: str
) -> None:
    ticket = new_ticket(ledger)
    with pytest.raises(ValueError, match=message):
        ledger.record_evidence(
            make_evidence(ticket.id), evidence_hash=evidence_hash, signature=signature
        )
    assert ledger.evidence_for(ticket.id, SHA_A) is None
    assert [e.kind for e in ledger.events()] == ["created"]


def test_evidence_operations_on_unknown_ticket(ledger: Ledger) -> None:
    with pytest.raises(TicketNotFound):
        ledger.record_evidence(make_evidence("T-0404"), evidence_hash=HASH, signature="00")
    with pytest.raises(TicketNotFound):
        ledger.evidence_for("T-0404", SHA_A)


# --------------------------------------------------------------------------- evidence records


def test_evidence_record_carries_the_proofs_of_the_latest_recording(ledger: Ledger) -> None:
    ticket = new_ticket(ledger)
    first, second = make_evidence(ticket.id, offset_s=1), make_evidence(ticket.id, offset_s=2)
    ledger.record_evidence(first, evidence_hash=HASH, signature="aa")
    ledger.record_evidence(second, evidence_hash=HASH_2, signature="bb")
    record = ledger.evidence_record(ticket.id, SHA_A)
    assert record == EvidenceRecord(evidence=second, evidence_hash=HASH_2, signature="bb")
    assert ledger.evidence_for(ticket.id, SHA_A) == second


def test_evidence_record_is_scoped_to_ticket_and_sha(ledger: Ledger) -> None:
    a, b = new_ticket(ledger, "a"), new_ticket(ledger, "b")
    ledger.record_evidence(make_evidence(a.id), evidence_hash=HASH, signature="0f")
    assert ledger.evidence_record(a.id, SHA_B) is None
    assert ledger.evidence_record(b.id, SHA_A) is None
    record = ledger.evidence_record(a.id, SHA_A)
    assert record is not None
    assert (record.evidence_hash, record.signature) == (HASH, "0f")


def test_evidence_record_of_unknown_ticket(ledger: Ledger) -> None:
    with pytest.raises(TicketNotFound):
        ledger.evidence_record("T-0404", SHA_A)


def test_evidence_record_is_immutable(ledger: Ledger) -> None:
    ticket = new_ticket(ledger)
    ledger.record_evidence(make_evidence(ticket.id), evidence_hash=HASH, signature="00")
    record = ledger.evidence_record(ticket.id, SHA_A)
    assert record is not None
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.signature = "ff"  # type: ignore[misc] # asserting the frozen dataclass rejects it


# --------------------------------------------------------------------------- non-finite floats


def test_a_nan_gate_metric_is_refused_instead_of_poisoning_the_verdicts(ledger: Ledger) -> None:
    ticket = new_ticket(ledger)
    good = make_verdict(ticket.id)
    ledger.record_verdict(good)
    nan_gate = GateResult.model_construct(
        name="unit", status=GateStatus.PASS, metrics={"coverage": math.nan}
    )
    poisoned = good.model_copy(update={"round": 2, "gates": (nan_gate,)})
    with pytest.raises(ValueError, match="JSON compliant"):
        ledger.record_verdict(poisoned)
    assert ledger.verdicts(ticket.id) == [good]
    assert [e.kind for e in ledger.events(ticket.id)] == ["created", "verdict_recorded"]


@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
def test_non_finite_run_cost_is_refused(ledger: Ledger, value: float) -> None:
    ticket = new_ticket(ledger)
    run = make_run(ticket.id).model_copy(update={"usage": Usage.model_construct(cost_usd=value)})
    with pytest.raises(ValueError, match="JSON compliant"):
        ledger.record_run(run)
    assert ledger.runs(ticket.id) == []
    assert ledger.ticket_cost_usd(ticket.id) == 0.0


def test_non_finite_evidence_is_refused(ledger: Ledger) -> None:
    ticket = new_ticket(ledger)
    slow_gate = GateResult.model_construct(name="e2e", status=GateStatus.PASS, duration_s=math.inf)
    evidence = make_evidence(ticket.id).model_copy(update={"gates": (slow_gate,)})
    with pytest.raises(ValueError, match="JSON compliant"):
        ledger.record_evidence(evidence, evidence_hash=HASH, signature="00")
    assert ledger.evidence_record(ticket.id, SHA_A) is None
