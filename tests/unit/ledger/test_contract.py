"""Unit tests for contract proposal, oracle version and freezing."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from pydantic import ValidationError

from local_factory.hashing import content_hash
from local_factory.ledger import ContractFrozen, Ledger, LedgerError, TicketNotFound
from local_factory.models import Halt, HaltReason, TicketStatus

from .helpers import HASH, HASH_2, FakeClock, make_contract, new_ticket, walk_to

S = TicketStatus


def _proposed(ledger: Ledger, *, contract: bool = True, oracle: bool = True) -> str:
    ticket = walk_to(ledger, new_ticket(ledger).id, S.CONTRACT_PROPOSED)
    if contract:
        ledger.propose_contract(ticket.id, make_contract(2), actor="spec")
    if oracle:
        ledger.set_oracle_version(ticket.id, HASH, actor="test-architect")
    return ticket.id


# --------------------------------------------------------------------------- before freezing


def test_propose_contract_stores_it_without_changing_status(
    ledger: Ledger, clock: FakeClock
) -> None:
    ticket_id = walk_to(ledger, new_ticket(ledger).id, S.CONTRACT_PROPOSED).id
    contract = make_contract(3)
    updated = ledger.propose_contract(ticket_id, contract, actor="spec")
    assert updated.contract == contract
    assert updated.status is S.CONTRACT_PROPOSED
    assert updated.contract_hash is None
    assert updated.updated_at == clock.now
    assert ledger.get_ticket(ticket_id) == updated
    event = ledger.events(ticket_id)[-1]
    assert (event.kind, event.actor, event.from_status, event.to_status) == (
        "contract_proposed",
        "spec",
        None,
        None,
    )
    assert event.data == {"version": 1, "contract_hash": content_hash(contract)}


def test_a_proposed_contract_can_be_replaced_until_frozen(ledger: Ledger) -> None:
    ticket_id = _proposed(ledger)
    revised = make_contract(1, version=2, summary="tag notes, revised")
    assert ledger.propose_contract(ticket_id, revised, actor="spec").contract == revised
    frozen = ledger.freeze_contract(ticket_id, actor="human")
    assert frozen.contract == revised
    assert frozen.contract_hash == content_hash(revised)


def test_propose_contract_is_not_tied_to_a_status(ledger: Ledger) -> None:
    ticket = new_ticket(ledger)
    updated = ledger.propose_contract(ticket.id, make_contract(), actor="spec")
    assert updated.status is S.RECEIVED
    assert updated.contract == make_contract()


def test_set_oracle_version_stores_and_records_it(ledger: Ledger) -> None:
    ticket = new_ticket(ledger)
    assert ledger.set_oracle_version(ticket.id, HASH, actor="ta").oracle_version == HASH
    assert ledger.set_oracle_version(ticket.id, HASH_2, actor="ta").oracle_version == HASH_2
    event = ledger.events(ticket.id)[-1]
    assert (event.kind, event.actor, event.data) == (
        "oracle_version_set",
        "ta",
        {"oracle_version": HASH_2},
    )


@pytest.mark.parametrize(
    "bad", ["", "c" * 64, "sha256:" + "C" * 64, "sha256:abc", "md5:" + "c" * 64]
)
def test_set_oracle_version_rejects_malformed_tags(ledger: Ledger, bad: str) -> None:
    ticket = new_ticket(ledger)
    with pytest.raises(ValidationError):
        ledger.set_oracle_version(ticket.id, bad, actor="ta")
    assert ledger.get_ticket(ticket.id).oracle_version is None
    assert [e.kind for e in ledger.events(ticket.id)] == ["created"]


@pytest.mark.parametrize(
    "operation",
    [
        lambda led: led.propose_contract("T-0404", make_contract(), actor="spec"),
        lambda led: led.set_oracle_version("T-0404", HASH, actor="ta"),
        lambda led: led.freeze_contract("T-0404", actor="human"),
    ],
    ids=["propose", "oracle", "freeze"],
)
def test_contract_operations_on_unknown_ticket(
    ledger: Ledger, operation: Callable[[Ledger], object]
) -> None:
    with pytest.raises(TicketNotFound):
        operation(ledger)


# --------------------------------------------------------------------------- freezing


def test_freeze_appends_frozen_then_transition_at_one_timestamp(
    ledger: Ledger, clock: FakeClock
) -> None:
    ticket_id = _proposed(ledger)
    calls = clock.calls
    frozen = ledger.freeze_contract(ticket_id, actor="human")
    assert clock.calls == calls + 1
    assert frozen.updated_at == clock.now
    assert frozen.contract_hash == content_hash(make_contract(2))
    frozen_event, transition = ledger.events(ticket_id)[-2:]
    assert frozen_event.kind == "contract_frozen"
    assert frozen_event.data == {
        "version": 1,
        "contract_hash": content_hash(make_contract(2)),
        "oracle_version": HASH,
    }
    assert (transition.kind, transition.from_status, transition.to_status) == (
        "transition",
        S.CONTRACT_PROPOSED,
        S.APPROVED,
    )
    assert transition.data == {"reason": "contract frozen"}  # no structured detail
    assert frozen_event.at == transition.at == clock.now
    assert frozen_event.actor == transition.actor == "human"
    assert transition.seq == frozen_event.seq + 1


def test_freeze_requires_contract_proposed_status(ledger: Ledger) -> None:
    ticket = new_ticket(ledger)
    ledger.propose_contract(ticket.id, make_contract(), actor="spec")
    ledger.set_oracle_version(ticket.id, HASH, actor="ta")
    with pytest.raises(LedgerError, match="contract_proposed") as excinfo:
        ledger.freeze_contract(ticket.id, actor="human")
    assert not isinstance(excinfo.value, ContractFrozen)
    assert ledger.get_ticket(ticket.id).contract_hash is None


def test_freeze_requires_a_contract(ledger: Ledger) -> None:
    ticket_id = _proposed(ledger, contract=False)
    with pytest.raises(LedgerError, match=r"propose_contract\(\)"):
        ledger.freeze_contract(ticket_id, actor="human")


def test_freeze_requires_an_oracle_version(ledger: Ledger) -> None:
    ticket_id = _proposed(ledger, oracle=False)
    with pytest.raises(LedgerError, match=r"set_oracle_version\(\)"):
        ledger.freeze_contract(ticket_id, actor="human")


def test_freeze_is_rejected_while_halted(ledger: Ledger) -> None:
    ticket_id = _proposed(ledger)
    ledger.halt(ticket_id, Halt(reason=HaltReason.RATE_LIMIT), actor="scheduler")
    with pytest.raises(LedgerError):
        ledger.freeze_contract(ticket_id, actor="human")
    assert ledger.get_ticket(ticket_id).contract_hash is None


def test_rejected_freeze_has_no_side_effects(ledger: Ledger, clock: FakeClock) -> None:
    ticket_id = _proposed(ledger, oracle=False)
    before, events, calls = ledger.get_ticket(ticket_id), ledger.events(), clock.calls
    with pytest.raises(LedgerError):
        ledger.freeze_contract(ticket_id, actor="human")
    assert (ledger.get_ticket(ticket_id), ledger.events(), clock.calls) == (before, events, calls)


def test_freeze_with_invalid_actor_rolls_back_hash_and_status(ledger: Ledger) -> None:
    ticket_id = _proposed(ledger)
    before = ledger.get_ticket(ticket_id)
    with pytest.raises(ValidationError):
        ledger.freeze_contract(ticket_id, actor="")
    assert ledger.get_ticket(ticket_id) == before


# --------------------------------------------------------------------------- after freezing


def test_after_freezing_contract_and_oracle_are_immutable(ledger: Ledger) -> None:
    ticket_id = _proposed(ledger)
    frozen = ledger.freeze_contract(ticket_id, actor="human")
    events = ledger.events()
    with pytest.raises(ContractFrozen, match=ticket_id):
        ledger.propose_contract(ticket_id, make_contract(1), actor="spec")
    with pytest.raises(ContractFrozen):
        ledger.set_oracle_version(ticket_id, HASH_2, actor="ta")
    with pytest.raises(ContractFrozen):
        ledger.freeze_contract(ticket_id, actor="human")
    assert ledger.get_ticket(ticket_id) == frozen
    assert ledger.events() == events


def test_contract_stays_frozen_through_later_statuses(ledger: Ledger) -> None:
    ticket = walk_to(ledger, new_ticket(ledger).id, S.BLOCKED)
    ledger.transition(ticket.id, S.CONTRACT_PROPOSED, actor="human")
    with pytest.raises(ContractFrozen):
        ledger.propose_contract(ticket.id, make_contract(4), actor="spec")
    with pytest.raises(ContractFrozen):
        ledger.freeze_contract(ticket.id, actor="human")
    assert ledger.get_ticket(ticket.id).contract == make_contract()
