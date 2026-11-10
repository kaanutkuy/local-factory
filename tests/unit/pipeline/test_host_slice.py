"""Unit tests for the host slice: roles, criterion reconciliation, models and preconditions.

Precondition failures must happen before the slice changes anything, so these run against a
real ledger in tmp_path and a repository path that does not exist (git is never reached).
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from local_factory.agents import FakeBackend
from local_factory.config import ConfigError, FactoryConfig, ProjectConfig
from local_factory.gitops import BareRepo
from local_factory.ledger import Ledger, TicketNotFound
from local_factory.models import (
    Contract,
    Criterion,
    CriterionResult,
    CriterionStatus,
    Halt,
    HaltReason,
    RiskTier,
    TestLayer,
    Ticket,
    TicketStatus,
    TicketType,
    Tier,
)
from local_factory.pipeline import (
    IMPLEMENTER_ROLE,
    VERIFIER_ROLE,
    SliceError,
    SliceOutcome,
    VerifierReport,
    reconcile_criteria,
    run_host_slice,
)
from local_factory.pipeline.host_slice import FAKE_MODEL, UNREPORTED, _frozen, _models

T0 = datetime(2026, 1, 1, tzinfo=UTC)
ORACLE = "sha256:" + "a" * 64
CONTRACT = Contract(
    version=1,
    summary="three criteria",
    criteria=tuple(
        Criterion(id=f"AC-{i}", ears=f"When {i}, the system shall {i}.", layer=TestLayer.UNIT)
        for i in (1, 2, 3)
    ),
)
PASS, FAIL, INCONCLUSIVE = CriterionStatus.PASS, CriterionStatus.FAIL, CriterionStatus.INCONCLUSIVE


def result(cid: str, status: CriterionStatus, feedback: str = "") -> CriterionResult:
    return CriterionResult(id=cid, status=status, feedback=feedback)


# --------------------------------------------------------------------------- roles


def test_roles_match_the_spec() -> None:
    assert (IMPLEMENTER_ROLE.name, IMPLEMENTER_ROLE.tier) == ("implementer", Tier.STANDARD)
    assert IMPLEMENTER_ROLE.output_schema is None
    assert (VERIFIER_ROLE.name, VERIFIER_ROLE.tier) == ("verifier", Tier.FRONTIER)
    assert VERIFIER_ROLE.output_schema is VerifierReport
    assert IMPLEMENTER_ROLE.prompt
    assert VERIFIER_ROLE.prompt


def test_verifier_report_parses_criterion_results() -> None:
    parsed = VerifierReport.model_validate(
        {"criteria": [{"id": "AC-1", "status": "pass", "evidence": "ok"}]}
    )
    assert parsed.criteria == [CriterionResult(id="AC-1", status=PASS, evidence="ok")]


# --------------------------------------------------------------------------- reconciliation


def test_reconcile_follows_contract_order() -> None:
    reported = [result("AC-3", PASS), result("AC-1", FAIL), result("AC-2", PASS)]
    assert [(r.id, r.status) for r in reconcile_criteria(reported, CONTRACT)] == [
        ("AC-1", FAIL),
        ("AC-2", PASS),
        ("AC-3", PASS),
    ]


def test_reconcile_marks_unreported_criteria_inconclusive() -> None:
    reconciled = reconcile_criteria([result("AC-2", PASS)], CONTRACT)
    assert [(r.id, r.status) for r in reconciled] == [
        ("AC-1", INCONCLUSIVE),
        ("AC-2", PASS),
        ("AC-3", INCONCLUSIVE),
    ]
    assert reconciled[0].feedback == UNREPORTED


def test_reconcile_lets_a_duplicate_failure_win() -> None:
    reported = [
        result("AC-1", PASS, "first"),
        result("AC-1", FAIL, "second"),
        result("AC-1", INCONCLUSIVE, "third"),
        result("AC-2", PASS, "only pass"),
        result("AC-2", PASS, "still pass"),
        result("AC-3", INCONCLUSIVE, "unsure"),
        result("AC-3", PASS, "sure"),
    ]
    assert [r.feedback for r in reconcile_criteria(reported, CONTRACT)] == [
        "second",
        "only pass",
        "unsure",
    ]


def test_reconcile_drops_criteria_outside_the_contract() -> None:
    reported = [result("AC-1", PASS), result("AC-2", PASS), result("AC-3", PASS)]
    reconciled = reconcile_criteria([*reported, result("AC-9", FAIL)], CONTRACT)
    assert reconciled == tuple(reported)


# --------------------------------------------------------------------------- models


def test_the_fake_backend_runs_the_fake_model() -> None:
    assert _models("fake", None) == {"implementer": FAKE_MODEL, "verifier": FAKE_MODEL}


def test_other_backends_use_the_configured_tier_models(tmp_path: Path) -> None:
    config = FactoryConfig(home=tmp_path)
    assert _models("claude", config) == {
        "implementer": config.model_for("claude", Tier.STANDARD),
        "verifier": config.model_for("claude", Tier.FRONTIER),
    }


def test_other_backends_need_a_config() -> None:
    with pytest.raises(SliceError, match="'claude' needs a factory config"):
        _models("claude", None)


def test_an_unconfigured_backend_is_a_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        _models("nobody", FactoryConfig(home=tmp_path))


def test_a_ticket_without_a_frozen_contract_is_refused() -> None:
    ticket = Ticket(
        id="T-0001",
        title="t",
        type=TicketType.BUG,
        risk=RiskTier.R0,
        status=TicketStatus.APPROVED,
        request="r",
        contract=CONTRACT,
        created_at=T0,
        updated_at=T0,
    )
    with pytest.raises(SliceError, match="no frozen contract"):
        _frozen(ticket)


# --------------------------------------------------------------------------- preconditions


@pytest.fixture
def ledger(tmp_path: Path) -> Iterator[Ledger]:
    with Ledger(tmp_path / "ledger.db", clock=lambda: T0) as opened:
        yield opened


def approved(ledger: Ledger) -> str:
    ticket = ledger.create_ticket(
        title="t", type=TicketType.FEATURE, risk=RiskTier.R1, request="r", actor="human"
    )
    ledger.transition(ticket.id, TicketStatus.TRIAGED, actor="triage")
    ledger.transition(ticket.id, TicketStatus.CONTRACT_PROPOSED, actor="spec")
    ledger.propose_contract(ticket.id, CONTRACT, actor="spec")
    ledger.set_oracle_version(ticket.id, ORACLE, actor="test-architect")
    ledger.freeze_contract(ticket.id, actor="human")
    return ticket.id


def start(
    ledger: Ledger,
    tmp_path: Path,
    ticket_id: str,
    *,
    backend: FakeBackend | None = None,
    key: bytes = b"k" * 32,
) -> SliceOutcome:
    return asyncio.run(
        run_host_slice(
            ledger=ledger,
            repo=BareRepo(tmp_path / "no-such-repo.git"),
            project=ProjectConfig(name="p"),
            backend=backend or FakeBackend({}),
            ticket_id=ticket_id,
            work_root=tmp_path / "work",
            key=key,
            clock=lambda: T0,
        )
    )


class NamedBackend(FakeBackend):
    name = "claude"


def assert_untouched(ledger: Ledger, tmp_path: Path, events_before: int) -> None:
    assert len(ledger.events()) == events_before
    assert not (tmp_path / "work").exists()


def test_an_unknown_ticket_is_not_found(ledger: Ledger, tmp_path: Path) -> None:
    with pytest.raises(TicketNotFound):
        start(ledger, tmp_path, "T-0042")
    assert_untouched(ledger, tmp_path, 0)


@pytest.mark.parametrize("status", [TicketStatus.RECEIVED, TicketStatus.TRIAGED])
def test_a_ticket_before_approval_is_refused(
    ledger: Ledger, tmp_path: Path, status: TicketStatus
) -> None:
    ticket = ledger.create_ticket(
        title="t", type=TicketType.FEATURE, risk=RiskTier.R1, request="r", actor="human"
    )
    if status is TicketStatus.TRIAGED:
        ledger.transition(ticket.id, status, actor="triage")
    before = len(ledger.events())
    backend = FakeBackend({})
    with pytest.raises(SliceError, match=f"{ticket.id} is {status.value}"):
        start(ledger, tmp_path, ticket.id, backend=backend)
    assert backend.calls == []
    assert_untouched(ledger, tmp_path, before)


@pytest.mark.parametrize(
    "status", [TicketStatus.PLANNED, TicketStatus.VERIFYING, TicketStatus.HALTED]
)
def test_a_ticket_that_is_not_approved_or_building_is_refused(
    ledger: Ledger, tmp_path: Path, status: TicketStatus
) -> None:
    ticket_id = approved(ledger)
    ledger.transition(ticket_id, TicketStatus.PLANNED, actor="x")
    if status is TicketStatus.VERIFYING:
        ledger.transition(ticket_id, TicketStatus.BUILDING, actor="x")
        ledger.transition(ticket_id, TicketStatus.VERIFYING, actor="x")
    if status is TicketStatus.HALTED:
        ledger.halt(ticket_id, Halt(reason=HaltReason.BUDGET), actor="x")
    before = len(ledger.events())
    with pytest.raises(SliceError, match="needs approved or building"):
        start(ledger, tmp_path, ticket_id)
    assert_untouched(ledger, tmp_path, before)


def test_an_empty_key_is_refused(ledger: Ledger, tmp_path: Path) -> None:
    ticket_id = approved(ledger)
    before = len(ledger.events())
    with pytest.raises(ValueError, match="key must not be empty"):
        start(ledger, tmp_path, ticket_id, key=b"")
    assert_untouched(ledger, tmp_path, before)


def test_an_unconfigured_real_backend_is_refused(ledger: Ledger, tmp_path: Path) -> None:
    ticket_id = approved(ledger)
    before = len(ledger.events())
    with pytest.raises(SliceError, match="needs a factory config"):
        start(ledger, tmp_path, ticket_id, backend=NamedBackend({}))
    assert_untouched(ledger, tmp_path, before)
