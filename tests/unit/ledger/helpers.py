"""Shared helpers for the ledger unit tests."""

from __future__ import annotations

from collections import deque
from datetime import UTC, datetime, timedelta

from local_factory.ledger import Ledger
from local_factory.ledger.state_machine import allowed_targets
from local_factory.models import (
    Contract,
    Criterion,
    CriterionResult,
    CriterionStatus,
    Evidence,
    GateResult,
    GateStatus,
    RiskTier,
    RunRecord,
    TestLayer,
    Ticket,
    TicketStatus,
    TicketType,
    Tier,
    Usage,
    Verdict,
)

T0 = datetime(2026, 1, 1, tzinfo=UTC)
HASH = "sha256:" + "c" * 64
HASH_2 = "sha256:" + "d" * 64
SHA_A = "a" * 40
SHA_B = "b" * 40
SHA_C = "c" * 64  # sha256 object format


class FakeClock:
    """Deterministic clock: every call advances one second and is counted."""

    def __init__(self, start: datetime = T0) -> None:
        self.now = start
        self.calls = 0

    def __call__(self) -> datetime:
        self.calls += 1
        self.now += timedelta(seconds=1)
        return self.now


def make_contract(n: int = 1, *, version: int = 1, summary: str = "tag notes") -> Contract:
    return Contract(
        version=version,
        summary=summary,
        criteria=tuple(
            Criterion(id=f"AC-{i}", ears=f"When {i}, the system shall {i}.", layer=TestLayer.UNIT)
            for i in range(1, n + 1)
        ),
    )


def new_ticket(led: Ledger, title: str = "tags") -> Ticket:
    return led.create_ticket(
        title=title, type=TicketType.FEATURE, risk=RiskTier.R1, request="add tags", actor="human"
    )


def path_to(
    target: TicketStatus, start: TicketStatus = TicketStatus.RECEIVED
) -> list[TicketStatus]:
    """Shortest list of statuses (excluding `start`) leading from `start` to `target`."""
    previous: dict[TicketStatus, TicketStatus | None] = {start: None}
    queue = deque([start])
    while queue:
        status = queue.popleft()
        if status is target:
            steps: list[TicketStatus] = []
            node: TicketStatus | None = status
            while node is not None and node is not start:
                steps.append(node)
                node = previous[node]
            return steps[::-1]
        for nxt in sorted(allowed_targets(status)):
            if nxt not in previous:
                previous[nxt] = status
                queue.append(nxt)
    raise AssertionError(f"{target} is unreachable from {start}")


def walk_to(led: Ledger, ticket_id: str, target: TicketStatus) -> Ticket:
    """Drive a ticket to `target` through legal operations (freezing the contract on approval)."""
    ticket = led.get_ticket(ticket_id)
    for step in path_to(target, ticket.status):
        if step is TicketStatus.APPROVED and ticket.contract_hash is None:
            if ticket.contract is None:
                led.propose_contract(ticket_id, make_contract(), actor="spec")
            if ticket.oracle_version is None:
                led.set_oracle_version(ticket_id, HASH, actor="test-architect")
            ticket = led.freeze_contract(ticket_id, actor="human")
        else:
            ticket = led.transition(ticket_id, step, actor="walker")
    assert ticket.status is target
    return ticket


def make_run(
    ticket_id: str, run_id: str = "run-1", *, cost: float = 0.5, task_id: str | None = None
) -> RunRecord:
    return RunRecord(
        id=run_id,
        ticket_id=ticket_id,
        task_id=task_id,
        role="implementer",
        backend="fake",
        model="m",
        tier=Tier.STANDARD,
        started_at=T0,
        finished_at=T0 + timedelta(minutes=1),
        status="ok",
        usage=Usage(input_tokens=10, output_tokens=5, cost_usd=cost),
    )


def make_verdict(
    ticket_id: str, *, sha: str = SHA_A, round: int = 1, passed: bool = True
) -> Verdict:
    return Verdict(
        ticket_id=ticket_id,
        round=round,
        sha=sha,
        gates=(GateResult(name="unit", status=GateStatus.PASS if passed else GateStatus.FAIL),),
        criteria=(CriterionResult(id="AC-1", status=CriterionStatus.PASS),),
        passed=passed,
        failing=() if passed else ("gate:unit",),
    )


def make_evidence(ticket_id: str, *, sha: str = SHA_A, offset_s: int = 0) -> Evidence:
    verdict = make_verdict(ticket_id, sha=sha)
    return Evidence(
        ticket_id=ticket_id,
        candidate_sha=sha,
        tree_hash=SHA_B,
        base_sha=SHA_B,
        contract_hash=HASH,
        oracle_version=HASH,
        factory_version="0.1.0",
        gates=verdict.gates,
        criteria=verdict.criteria,
        passed=True,
        created_at=T0 + timedelta(seconds=offset_s),
    )
