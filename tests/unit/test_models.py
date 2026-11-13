"""Unit tests for the shared domain models (local_factory.models)."""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from local_factory.models import (
    MODE_EXECUTABLE,
    MODE_GITLINK,
    MODE_REGULAR,
    MODE_SYMLINK,
    AddedLine,
    AggregateResult,
    ChangedFile,
    ChangeKind,
    Contract,
    Criterion,
    CriterionResult,
    CriterionStatus,
    Event,
    Evidence,
    GateResult,
    GateStatus,
    Halt,
    HaltReason,
    MigrationImpact,
    RiskTier,
    RunRecord,
    Task,
    TaskStatus,
    TestLayer,
    Ticket,
    TicketStatus,
    TicketType,
    Tier,
    Usage,
    Verdict,
    Violation,
    Waiver,
)

NOW = datetime(2026, 1, 1, tzinfo=UTC)
SHA1 = "a" * 40
SHA256 = "b" * 64
HASH = "sha256:" + "c" * 64


def criterion(id: str = "AC-1", **kwargs: Any) -> Criterion:
    return Criterion.model_validate(
        {"id": id, "ears": "When x, shall y.", "layer": "unit", **kwargs}
    )


def ticket(**kwargs: Any) -> Ticket:
    data: dict[str, Any] = {
        "id": "T-0001",
        "title": "t",
        "type": "feature",
        "risk": "R1",
        "status": "received",
        "request": "r",
        "created_at": NOW,
        "updated_at": NOW,
    }
    return Ticket.model_validate({**data, **kwargs})


def task(**kwargs: Any) -> Task:
    data: dict[str, Any] = {
        "id": "T-0001/task-1",
        "ticket_id": "T-0001",
        "goal": "g",
        "role": "implementer",
        "tier": "fast",
    }
    return Task.model_validate({**data, **kwargs})


def waiver(**kwargs: Any) -> Waiver:
    data: dict[str, Any] = {
        "id": "W-1",
        "gate": "lint",
        "scope": "src/app.py:42",
        "reason": "generated code",
        "requested_by": "implementer",
    }
    return Waiver.model_validate({**data, **kwargs})


# --------------------------------------------------------------------------- frozen base


def test_models_are_frozen() -> None:
    t = ticket()
    with pytest.raises(ValidationError):
        t.title = "changed"  # type: ignore[misc] # asserting the frozen config rejects assignment
    assert t.title == "t"


def test_models_forbid_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="extra"):
        ticket(colour="red")


def test_mutable_defaults_are_not_shared() -> None:
    a, b = GateResult(name="unit", status=GateStatus.PASS), GateResult(name="e2e", status="pass")
    a.metrics["coverage"] = 0.9
    assert b.metrics == {}
    assert Event(seq=1, at=NOW, kind="k", actor="a").data == {}


# --------------------------------------------------------------------------- contract


def test_contract_defaults() -> None:
    contract = Contract(version=1, summary="s", criteria=(criterion(),))
    assert contract.migration_impact is MigrationImpact.NONE
    assert (contract.nfr, contract.out_of_scope, contract.assumptions) == ((), (), ())


def test_contract_rejects_duplicate_criterion_ids() -> None:
    with pytest.raises(ValidationError, match="duplicate criterion ids"):
        Contract(version=1, summary="s", criteria=(criterion("AC-1"), criterion("AC-1")))


def test_contract_accepts_distinct_criterion_ids() -> None:
    contract = Contract(version=2, summary="s", criteria=(criterion("AC-1"), criterion("AC-2")))
    assert [c.id for c in contract.criteria] == ["AC-1", "AC-2"]


@pytest.mark.parametrize(
    "overrides",
    [{"version": 0}, {"summary": ""}, {"criteria": ()}, {"migration_impact": "drop-everything"}],
)
def test_contract_field_constraints(overrides: dict[str, Any]) -> None:
    data: dict[str, Any] = {"version": 1, "summary": "s", "criteria": [criterion()], **overrides}
    with pytest.raises(ValidationError):
        Contract.model_validate(data)


@pytest.mark.parametrize("bad_id", ["AC-", "ac-1", "AC-1a", "AC1", " AC-1", "AC-1\n"])
def test_criterion_id_pattern(bad_id: str) -> None:
    with pytest.raises(ValidationError):
        criterion(bad_id)


def test_criterion_constraints_and_defaults() -> None:
    assert criterion("AC-123").tests == ()
    assert criterion().holdout_probes == 0
    with pytest.raises(ValidationError):
        criterion(holdout_probes=-1)
    with pytest.raises(ValidationError):
        criterion(ears="")
    with pytest.raises(ValidationError):
        criterion(layer="smoke")


def test_contract_json_round_trip() -> None:
    contract = Contract(
        version=3,
        summary="s",
        criteria=(criterion("AC-1", tests=["tests/test_x.py::test_y"], holdout_probes=2),),
        nfr=("p95 < 200ms",),
        migration_impact=MigrationImpact.EXPAND_ONLY,
    )
    assert Contract.model_validate_json(contract.model_dump_json()) == contract


# --------------------------------------------------------------------------- tickets & tasks


@pytest.mark.parametrize("good", ["T-0001", "T-9999", "T-10000", "T-123456"])
def test_ticket_id_accepts_four_or_more_digits(good: str) -> None:
    assert ticket(id=good).id == good


@pytest.mark.parametrize("bad", ["T-001", "T-0001\n", "t-0001", "T0001", "T-00a1", "T-0001/x"])
def test_ticket_id_rejects_malformed(bad: str) -> None:
    with pytest.raises(ValidationError):
        ticket(id=bad)


def test_id_patterns_only_accept_ascii_digits() -> None:
    """KNOWN CONTRACT BUG (reported): `\\d` matches any Unicode digit in pydantic's regex engine.

    Ids end up in paths, refs and CLI arguments; the patterns should use `[0-9]`.
    """
    builders: dict[str, Callable[[], object]] = {
        "T-\u0661\u0662\u0663\u0664": lambda: ticket(id="T-\u0661\u0662\u0663\u0664"),
        "T-0001/task-\u0663": lambda: task(id="T-0001/task-\u0663"),
        "AC-\u0967": lambda: criterion("AC-\u0967"),
        "W-\uff11": lambda: waiver(id="W-\uff11"),
    }
    accepted = []
    for value, build in builders.items():
        try:
            build()
        except ValidationError:
            continue
        accepted.append(value)
    assert accepted == [], f"non-ASCII digits accepted in ids: {accepted}"


def test_float_fields_reject_nan_and_infinity() -> None:
    """KNOWN CONTRACT BUG (reported): float fields accept NaN and +-inf.

    `canonical_json` now rejects them (so such a model cannot be content-hashed) and
    `model_dump_json` writes them as null (so it cannot be read back). Proposed fix:
    `allow_inf_nan=False` in `FrozenModel.model_config`.
    """
    builders: dict[str, Callable[[], object]] = {
        "Usage.cost_usd=inf": lambda: Usage(cost_usd=math.inf),
        "GateResult.duration_s=inf": lambda: GateResult(
            name="unit", status=GateStatus.PASS, duration_s=math.inf
        ),
        "GateResult.metrics=nan": lambda: GateResult(
            name="unit", status=GateStatus.PASS, metrics={"x": math.nan}
        ),
    }
    accepted = []
    for case, build in builders.items():
        try:
            build()
        except ValidationError:
            continue
        accepted.append(case)
    assert accepted == [], f"non-finite floats accepted: {accepted}"


@pytest.mark.parametrize("field", ["contract_hash", "oracle_version"])
def test_ticket_hash_fields_require_sha256_tags(field: str) -> None:
    assert getattr(ticket(**{field: HASH}), field) == HASH
    for bad in ("c" * 64, "sha256:" + "c" * 63, "sha256:" + "C" * 64, "sha1:" + "c" * 40):
        with pytest.raises(ValidationError):
            ticket(**{field: bad})


def test_ticket_requires_a_title_and_known_enums() -> None:
    with pytest.raises(ValidationError):
        ticket(title="")
    with pytest.raises(ValidationError):
        ticket(type="epic")
    with pytest.raises(ValidationError):
        ticket(risk="R3")
    with pytest.raises(ValidationError):
        ticket(status="halted-ish")
    assert ticket(request="").request == ""


def test_ticket_json_round_trip_with_contract() -> None:
    t = ticket(
        contract=Contract(version=1, summary="s", criteria=(criterion(),)),
        contract_hash=HASH,
        oracle_version=HASH,
        status=TicketStatus.APPROVED,
    )
    assert Ticket.model_validate_json(t.model_dump_json()) == t


def test_task_defaults_and_patterns() -> None:
    t = task()
    assert (t.status, t.base_sha, t.head_sha, t.criteria_served, t.depends_on) == (
        TaskStatus.PENDING,
        None,
        None,
        (),
        (),
    )
    assert task(id="T-0001/task-12").id == "T-0001/task-12"
    for bad in ("T-0001/task-", "T-0001/task-1/x", "T-0001", "task-1"):
        with pytest.raises(ValidationError):
            task(id=bad)
    with pytest.raises(ValidationError):
        task(ticket_id="T-1")


@pytest.mark.parametrize("good", [SHA1, SHA256])
def test_sha_fields_accept_sha1_and_sha256_object_names(good: str) -> None:
    assert task(base_sha=good, head_sha=good).head_sha == good


@pytest.mark.parametrize(
    "bad", ["a" * 39, "a" * 41, "a" * 63, "a" * 65, "A" * 40, "g" * 40, SHA1 + "\n", ""]
)
def test_sha_fields_reject_other_strings(bad: str) -> None:
    with pytest.raises(ValidationError):
        task(head_sha=bad)


# --------------------------------------------------------------------------- halts, usage, runs


def test_halt_defaults() -> None:
    halt = Halt(reason=HaltReason.RATE_LIMIT)
    assert (halt.resume_at, halt.detail) == (None, "")
    with pytest.raises(ValidationError):
        Halt.model_validate({"reason": "tired"})


def test_usage_defaults_and_non_negativity() -> None:
    assert Usage() == Usage(
        input_tokens=0, output_tokens=0, cache_read_tokens=0, cache_write_tokens=0, cost_usd=0.0
    )
    for field in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens"):
        with pytest.raises(ValidationError):
            Usage.model_validate({field: -1})
    with pytest.raises(ValidationError):
        Usage(cost_usd=-0.01)


def run_record(**kwargs: Any) -> RunRecord:
    data: dict[str, Any] = {
        "id": "run-1",
        "ticket_id": "T-0001",
        "role": "implementer",
        "backend": "fake",
        "model": "m",
        "tier": "standard",
        "started_at": NOW,
        "finished_at": NOW,
        "status": "ok",
    }
    return RunRecord.model_validate({**data, **kwargs})


def test_run_record_round_trip_and_constraints() -> None:
    run = run_record(
        task_id="T-0001/task-2",
        session_id="s",
        transcript_path=Path("runs/x.jsonl"),
        usage=Usage(input_tokens=1, cost_usd=0.5),
        status="error",
        error="worker exited 137",
    )
    assert RunRecord.model_validate_json(run.model_dump_json()) == run
    assert run_record().usage == Usage()
    assert run_record().error is None
    with pytest.raises(ValidationError):
        run_record(status="crashed")
    with pytest.raises(ValidationError):
        run_record(id="")
    with pytest.raises(ValidationError):
        run_record(task_id="T-0001")


# --------------------------------------------------------------------------- verification


def test_gate_result_constraints() -> None:
    gate = GateResult(name="unit", status=GateStatus.FAIL, exit_code=1)
    assert (gate.duration_s, gate.output_tail) == (0.0, "")
    with pytest.raises(ValidationError):
        GateResult(name="", status=GateStatus.PASS)
    with pytest.raises(ValidationError):
        GateResult(name="unit", status=GateStatus.PASS, duration_s=-1.0)


def test_criterion_result_constraints() -> None:
    result = CriterionResult(id="AC-2", status=CriterionStatus.INCONCLUSIVE)
    assert (result.evidence, result.feedback, result.holdout_hint) == ("", "", None)
    with pytest.raises(ValidationError):
        CriterionResult(id="AC-x", status=CriterionStatus.PASS)


@pytest.mark.parametrize(
    ("approved", "adjudicated_by", "effective"),
    [
        (False, None, False),
        (False, "verifier", False),  # adjudicated but rejected
        (True, None, False),  # approved by nobody
        (True, "implementer", False),  # self-approval
        (True, "verifier", True),
    ],
)
def test_waiver_is_effective_only_when_approved_by_someone_else(
    approved: bool, adjudicated_by: str | None, effective: bool
) -> None:
    assert waiver(approved=approved, adjudicated_by=adjudicated_by).is_effective is effective


def test_waiver_defaults_and_constraints() -> None:
    w = waiver()
    assert (w.adjudicated_by, w.approved, w.is_effective) == (None, False, False)
    for field in ("gate", "scope", "reason", "requested_by"):
        with pytest.raises(ValidationError):
            waiver(**{field: ""})
    for bad in ("W-", "W1", "w-1", "WV-1"):
        with pytest.raises(ValidationError):
            waiver(id=bad)


def test_verdict_and_aggregate_round_trip() -> None:
    verdict = Verdict(
        ticket_id="T-0001",
        round=2,
        sha=SHA1,
        gates=(GateResult(name="unit", status=GateStatus.WAIVED, metrics={"cov": 0.9}),),
        criteria=(CriterionResult(id="AC-1", status=CriterionStatus.FAIL, feedback="f"),),
        waivers=(waiver(approved=True, adjudicated_by="verifier"),),
        passed=False,
        failing=("criterion:AC-1",),
    )
    assert Verdict.model_validate_json(verdict.model_dump_json()) == verdict
    assert AggregateResult(passed=True, failing=()).failing == ()
    with pytest.raises(ValidationError):
        verdict.model_validate({**verdict.model_dump(), "round": 0})


# --------------------------------------------------------------------------- git & policy


def test_change_kinds_and_modes() -> None:
    assert [k.value for k in ChangeKind] == ["A", "M", "D", "R", "T"]
    assert (MODE_REGULAR, MODE_EXECUTABLE, MODE_SYMLINK, MODE_GITLINK) == (
        "100644",
        "100755",
        "120000",
        "160000",
    )


def test_changed_file_defaults_and_constraints() -> None:
    change = ChangedFile(path="a.py", kind=ChangeKind.ADDED, mode=MODE_REGULAR, size=3)
    assert (change.old_path, change.old_mode, change.is_binary, change.symlink_target) == (
        None,
        None,
        False,
        None,
    )
    assert change.added_lines == ()
    with pytest.raises(ValidationError):
        ChangedFile(path="", kind=ChangeKind.ADDED)
    with pytest.raises(ValidationError):
        ChangedFile(path="a", kind=ChangeKind.ADDED, size=-1)
    with pytest.raises(ValidationError):
        AddedLine(line_no=0, text="x")


def test_violation_constraints() -> None:
    assert Violation(rule="gitlink", path="vendor/x", message="m").line is None
    with pytest.raises(ValidationError):
        Violation(rule="", path="p", message="m")
    with pytest.raises(ValidationError):
        Violation(rule="r", path="p", message="")


# --------------------------------------------------------------------------- events & evidence


def test_event_constraints() -> None:
    event = Event(seq=1, at=NOW, kind="created", actor="human", to_status=TicketStatus.RECEIVED)
    assert (event.ticket_id, event.from_status) == (None, None)
    with pytest.raises(ValidationError):
        Event(seq=0, at=NOW, kind="k", actor="a")
    with pytest.raises(ValidationError):
        Event(seq=1, at=NOW, kind="", actor="a")
    with pytest.raises(ValidationError):
        Event(seq=1, at=NOW, kind="k", actor="")
    with pytest.raises(ValidationError):
        Event(seq=1, at=NOW, kind="k", actor="a", ticket_id="T-1")


def evidence(**kwargs: Any) -> Evidence:
    data: dict[str, Any] = {
        "ticket_id": "T-0001",
        "candidate_sha": SHA1,
        "tree_hash": SHA1,
        "base_sha": SHA256,
        "contract_hash": HASH,
        "oracle_version": HASH,
        "factory_version": "0.1.0",
        "gates": [],
        "criteria": [],
        "passed": True,
        "created_at": NOW,
    }
    return Evidence.model_validate({**data, **kwargs})


def test_evidence_schema_version_is_pinned() -> None:
    assert evidence().schema_version == 1
    with pytest.raises(ValidationError):
        evidence(schema_version=2)


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("candidate_sha", "x"),
        ("tree_hash", "A" * 40),
        ("base_sha", ""),
        ("contract_hash", SHA256),
        ("oracle_version", "sha256:"),
        ("factory_version", ""),
        ("ticket_id", "T-01"),
    ],
)
def test_evidence_field_constraints(field: str, bad: str) -> None:
    with pytest.raises(ValidationError):
        evidence(**{field: bad})


def test_evidence_round_trip() -> None:
    e = evidence(image_digests={"worker": "sha256:" + "e" * 64}, models={"implementer": "m"})
    assert Evidence.model_validate_json(e.model_dump_json()) == e


# --------------------------------------------------------------------------- enums


def test_enum_values() -> None:
    assert [t.value for t in TicketType] == ["bug", "feature", "chore"]
    assert [r.value for r in RiskTier] == ["R0", "R1", "R2"]
    assert [t.value for t in Tier] == ["fast", "standard", "frontier", "apex"]
    assert {h.value for h in HaltReason} == {
        "rate_limit",
        "billing",
        "budget",
        "max_turns",
        "auth",
        "unavailable",
    }
    assert len(TicketStatus) == 18


def test_test_layer_is_not_collected_by_pytest() -> None:
    assert TestLayer.__test__ is False
    assert [layer.value for layer in TestLayer] == ["unit", "integration", "e2e"]
