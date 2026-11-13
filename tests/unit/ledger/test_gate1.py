"""Unit tests for the ledger's Gate 1 methods: record_event, triage and propose.

Every method is one transaction that reads the clock once after its preconditions hold; a refusal
changes nothing, not even the event log.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import JsonValue, ValidationError

from local_factory.ledger import (
    ContractFrozen,
    IllegalTransition,
    Ledger,
    LedgerError,
    TicketNotFound,
)
from local_factory.ledger.store import RESERVED_EVENT_KINDS
from local_factory.models import (
    Halt,
    HaltReason,
    RiskTier,
    TaskStatus,
    Ticket,
    TicketStatus,
    TicketType,
    Tier,
)

from .helpers import (
    HASH,
    HASH_2,
    FakeClock,
    make_contract,
    make_evidence,
    make_run,
    make_verdict,
    new_ticket,
    walk_to,
)

S = TicketStatus


def kinds(led: Ledger, ticket_id: str) -> list[str]:
    return [event.kind for event in led.events(ticket_id)]


def triage(led: Ledger, ticket_id: str, **overrides: Any) -> Ticket:
    fields: dict[str, Any] = {
        "title": "Add tags",
        "type": TicketType.FEATURE,
        "risk": RiskTier.R1,
        "actor": "factory",
    }
    return led.triage(ticket_id, **{**fields, **overrides})


# --------------------------------------------------------------------------- record_event


def test_the_reserved_kinds_are_exactly_the_kinds_the_ledger_writes(ledger: Ledger) -> None:
    ticket = new_ticket(ledger)
    triage(ledger, ticket.id)
    walk_to(ledger, ticket.id, S.BUILDING)
    task = ledger.create_task(
        ticket.id, goal="g", role="implementer", tier=Tier.STANDARD, actor="o"
    )
    ledger.update_task(task.id, status=TaskStatus.RUNNING, actor="o")
    ledger.record_run(make_run(ticket.id))
    ledger.record_verdict(make_verdict(ticket.id))
    ledger.record_evidence(make_evidence(ticket.id), evidence_hash=HASH, signature="00")
    ledger.halt(ticket.id, Halt(reason=HaltReason.BUDGET, detail="x"), actor="o")
    ledger.resume(ticket.id, actor="o")
    written = set(kinds(ledger, ticket.id))
    assert written == RESERVED_EVENT_KINDS


def test_a_reserved_kind_is_refused_before_the_ledger_is_touched(
    ledger: Ledger, clock: FakeClock
) -> None:
    ticket = new_ticket(ledger)
    before, calls = ledger.events(), clock.calls
    for kind in sorted(RESERVED_EVENT_KINDS):
        with pytest.raises(ValueError, match=f"'{kind}'"):
            ledger.record_event(ticket.id, kind, actor="factory")
    with pytest.raises(ValueError, match="'transition'"):  # even for a ticket that is not there
        ledger.record_event("T-0099", "transition", actor="factory")
    assert (ledger.events(), clock.calls) == (before, calls)


def test_record_event_appends_one_event_and_changes_nothing_else(
    ledger: Ledger, clock: FakeClock
) -> None:
    ticket = new_ticket(ledger)
    calls = clock.calls
    data: dict[str, JsonValue] = {"blocking": [{"text": "q"}], "n": 1}
    event = ledger.record_event(ticket.id, "critique", actor="factory", data=data)
    assert clock.calls == calls + 1
    assert (event.seq, event.at) == (ledger.events()[-1].seq, clock.now)
    assert (event.from_status, event.to_status) == (None, None)
    assert ledger.get_ticket(ticket.id) == ticket
    data["n"] = 2  # the stored data is a copy
    assert ledger.events(ticket.id)[-1].data == {"blocking": [{"text": "q"}], "n": 1}


def test_record_event_works_on_every_kind_of_ticket_state(ledger: Ledger) -> None:
    ticket = new_ticket(ledger)
    walk_to(ledger, ticket.id, S.ABANDONED)
    ledger.record_event(ticket.id, "note", actor="human")
    assert kinds(ledger, ticket.id)[-1] == "note"


def test_an_unknown_ticket_is_not_found_and_the_clock_untouched(
    ledger: Ledger, clock: FakeClock
) -> None:
    with pytest.raises(TicketNotFound):
        ledger.record_event("T-0099", "note", actor="factory")
    assert clock.calls == 0


@pytest.mark.parametrize(
    "bad",
    [{"kind": ""}, {"actor": ""}, {"data": {"x": float("nan")}}, {"data": {"x": {1, 2}}}],
    ids=["no-kind", "no-actor", "nan", "set"],
)
def test_an_event_that_cannot_be_stored_leaves_the_log_unchanged(
    ledger: Ledger, bad: dict[str, Any]
) -> None:
    ticket = new_ticket(ledger)
    before = ledger.events()
    args: dict[str, Any] = {"kind": "note", "actor": "factory", **bad}
    with pytest.raises((ValueError, TypeError, ValidationError)):
        ledger.record_event(ticket.id, args.pop("kind"), **args)
    assert ledger.events() == before
    ledger.record_event(ticket.id, "note", actor="factory")  # and the ledger still works


# --------------------------------------------------------------------------- triage


def test_triage_is_one_transaction_with_two_events_and_one_clock_read(
    ledger: Ledger, clock: FakeClock
) -> None:
    ticket = new_ticket(ledger)
    calls = clock.calls
    triaged = triage(ledger, ticket.id, type=TicketType.BUG, risk=RiskTier.R2, title="Fix it")
    assert clock.calls == calls + 1
    assert (triaged.title, triaged.type, triaged.risk, triaged.status) == (
        "Fix it",
        TicketType.BUG,
        RiskTier.R2,
        S.TRIAGED,
    )
    assert triaged.updated_at == clock.now and triaged.created_at == ticket.created_at
    assert ledger.get_ticket(ticket.id) == triaged
    assert kinds(ledger, ticket.id) == ["created", "triaged", "transition"]
    transition = ledger.events(ticket.id)[-1]
    assert (transition.from_status, transition.to_status) == (S.RECEIVED, S.TRIAGED)
    assert transition.actor == "factory" and transition.data == {"reason": "triaged"}


def test_the_factory_s_fields_win_over_the_given_data(ledger: Ledger) -> None:
    ticket = new_ticket(ledger)
    given = {"title": "forged", "risk": "R0", "size": "M", "areas": ["a"]}
    triage(ledger, ticket.id, data=given)
    [event] = [e for e in ledger.events(ticket.id) if e.kind == "triaged"]
    assert event.data == {
        "title": "Add tags",
        "type": "feature",
        "risk": "R1",
        "size": "M",
        "areas": ["a"],
    }
    assert given["title"] == "forged"  # the caller's mapping is untouched


@pytest.mark.parametrize("status", [S.TRIAGED, S.NEEDS_INFO, S.CONTRACT_PROPOSED, S.ABANDONED])
def test_only_a_received_ticket_is_triaged(
    ledger: Ledger, clock: FakeClock, status: TicketStatus
) -> None:
    ticket = new_ticket(ledger)
    walk_to(ledger, ticket.id, status)
    before, calls = ledger.events(), clock.calls
    with pytest.raises(IllegalTransition, match="only a received ticket"):
        triage(ledger, ticket.id)
    assert (ledger.events(), clock.calls) == (before, calls)


def test_a_halted_ticket_is_not_triaged(ledger: Ledger) -> None:
    ticket = new_ticket(ledger)
    ledger.halt(ticket.id, Halt(reason=HaltReason.BUDGET, detail="x"), actor="o")
    with pytest.raises(IllegalTransition):
        triage(ledger, ticket.id)


def test_an_unknown_ticket_cannot_be_triaged(ledger: Ledger) -> None:
    with pytest.raises(TicketNotFound):
        triage(ledger, "T-0099")


def test_an_invalid_triage_changes_nothing(ledger: Ledger) -> None:
    ticket = new_ticket(ledger)
    before = ledger.events()
    with pytest.raises(ValidationError):
        triage(ledger, ticket.id, title="")
    assert ledger.get_ticket(ticket.id) == ticket
    assert ledger.events() == before


# --------------------------------------------------------------------------- propose


def triaged_ticket(led: Ledger) -> str:
    ticket = new_ticket(led)
    triage(led, ticket.id)
    return ticket.id


def test_propose_writes_the_same_events_as_the_separate_calls_then_transitions(
    ledger: Ledger, clock: FakeClock
) -> None:
    ticket_id = triaged_ticket(ledger)
    contract = make_contract(2)
    calls = clock.calls
    proposed = ledger.propose(ticket_id, contract, HASH, actor="factory")
    assert clock.calls == calls + 1  # one clock read for the whole proposal
    assert (proposed.status, proposed.contract, proposed.oracle_version) == (
        S.CONTRACT_PROPOSED,
        contract,
        HASH,
    )
    assert proposed.contract_hash is None  # proposed, not frozen
    assert ledger.get_ticket(ticket_id) == proposed
    tail = ledger.events(ticket_id)[-3:]
    assert [e.kind for e in tail] == ["contract_proposed", "oracle_version_set", "transition"]
    assert len({e.at for e in tail}) == 1
    assert tail[2].data == {"reason": "contract proposed"}
    assert (tail[2].from_status, tail[2].to_status) == (S.TRIAGED, S.CONTRACT_PROPOSED)

    twin = Ledger(ledger.path.with_name("twin.db"), clock=FakeClock())
    with twin:
        other = new_ticket(twin)
        triage(twin, other.id)
        twin.propose_contract(other.id, contract, actor="factory")
        twin.set_oracle_version(other.id, HASH, actor="factory")
        separate = [(e.kind, e.actor, e.data) for e in twin.events(other.id)[-2:]]
    assert separate == [(e.kind, e.actor, e.data) for e in tail[:2]]


def test_propose_from_needs_info_replaces_an_earlier_proposal_after_a_bounce(
    ledger: Ledger,
) -> None:
    ticket_id = triaged_ticket(ledger)
    ledger.propose(ticket_id, make_contract(1), HASH, actor="factory")
    ledger.transition(ticket_id, S.NEEDS_INFO, actor="human", reason="more questions")
    again = ledger.propose(ticket_id, make_contract(2, version=2), HASH_2, actor="factory")
    assert (again.contract, again.oracle_version) == (make_contract(2, version=2), HASH_2)
    assert again.status is S.CONTRACT_PROPOSED


@pytest.mark.parametrize("status", [S.RECEIVED, S.CONTRACT_PROPOSED, S.ABANDONED])
def test_propose_only_from_triaged_or_needs_info(
    ledger: Ledger, clock: FakeClock, status: TicketStatus
) -> None:
    ticket = new_ticket(ledger)
    walk_to(ledger, ticket.id, status)
    snapshot, before, calls = ledger.get_ticket(ticket.id), ledger.events(), clock.calls
    with pytest.raises(IllegalTransition, match="needs_info or triaged"):
        ledger.propose(ticket.id, make_contract(), HASH, actor="factory")
    assert (ledger.get_ticket(ticket.id), ledger.events(), clock.calls) == (
        snapshot,
        before,
        calls,
    )


def test_propose_is_refused_once_the_contract_is_frozen(ledger: Ledger) -> None:
    ticket = new_ticket(ledger)
    walk_to(ledger, ticket.id, S.APPROVED)
    with pytest.raises(ContractFrozen):
        ledger.propose(ticket.id, make_contract(), HASH_2, actor="factory")


def test_a_halted_ticket_cannot_be_proposed_for(ledger: Ledger) -> None:
    ticket_id = triaged_ticket(ledger)
    ledger.halt(ticket_id, Halt(reason=HaltReason.BUDGET, detail="x"), actor="o")
    with pytest.raises(LedgerError):
        ledger.propose(ticket_id, make_contract(), HASH, actor="factory")


def test_a_proposal_that_does_not_validate_is_stored_nowhere(ledger: Ledger) -> None:
    ticket_id = triaged_ticket(ledger)
    before = ledger.get_ticket(ticket_id), ledger.events()
    with pytest.raises(ValidationError):  # the contract is fine; the oracle version is not
        ledger.propose(ticket_id, make_contract(), "not-a-hash", actor="factory")
    assert (ledger.get_ticket(ticket_id), ledger.events()) == before


def test_an_unknown_ticket_cannot_be_proposed_for(ledger: Ledger) -> None:
    with pytest.raises(TicketNotFound):
        ledger.propose("T-0099", make_contract(), HASH, actor="factory")
