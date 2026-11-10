"""Unit tests for the ticket state machine."""

from __future__ import annotations

from collections import deque
from typing import cast

import pytest

import local_factory.ledger as ledger_pkg
from local_factory.ledger.state_machine import (
    TERMINAL,
    TRANSITIONS,
    IllegalTransition,
    allowed_targets,
    check_transition,
)
from local_factory.models import TicketStatus

S = TicketStatus

# Transcribed independently from docs/specs/phase-1a.md § ledger.
SPEC_TABLE: dict[TicketStatus, set[TicketStatus]] = {
    S.RECEIVED: {S.TRIAGED, S.NEEDS_INFO, S.ABANDONED},
    S.TRIAGED: {S.CONTRACT_PROPOSED, S.NEEDS_INFO, S.ABANDONED},
    S.NEEDS_INFO: {S.TRIAGED, S.CONTRACT_PROPOSED, S.ABANDONED},
    S.CONTRACT_PROPOSED: {S.APPROVED, S.NEEDS_INFO, S.ABANDONED},
    S.APPROVED: {S.PLANNED, S.ABANDONED},
    S.PLANNED: {S.BUILDING, S.ABANDONED},
    S.BUILDING: {S.VERIFYING, S.BLOCKED, S.ABANDONED},
    S.VERIFYING: {S.BUILDING, S.CONVERGED, S.BLOCKED, S.ABANDONED},
    S.CONVERGED: {S.CANDIDATE_VERIFIED, S.BUILDING, S.ABANDONED},
    S.CANDIDATE_VERIFIED: {S.MERGED, S.CONVERGED, S.ABANDONED},
    S.MERGED: {S.AWAITING_RELEASE_APPROVAL, S.DEPLOYING, S.DONE},
    S.AWAITING_RELEASE_APPROVAL: {S.DEPLOYING},
    S.DEPLOYING: {S.DONE, S.ROLLED_BACK},
    S.BLOCKED: {S.BUILDING, S.CONTRACT_PROPOSED, S.ABANDONED},
    S.DONE: set(),
    S.ROLLED_BACK: set(),
    S.ABANDONED: set(),
}


def _reachable(start: TicketStatus) -> set[TicketStatus]:
    seen, queue = {start}, deque([start])
    while queue:
        for nxt in allowed_targets(queue.popleft()):
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return seen


def test_table_is_exactly_the_spec_table() -> None:
    assert {src: set(dsts) for src, dsts in TRANSITIONS.items()} == SPEC_TABLE
    assert all(isinstance(dsts, frozenset) for dsts in TRANSITIONS.values())


def test_every_status_but_halted_is_a_key() -> None:
    assert set(TRANSITIONS) == set(S) - {S.HALTED}


def test_halted_is_neither_a_source_nor_a_target() -> None:
    assert S.HALTED not in TRANSITIONS
    assert allowed_targets(S.HALTED) == frozenset()
    assert all(S.HALTED not in dsts for dsts in TRANSITIONS.values())


def test_exactly_the_terminal_statuses_are_dead_ends() -> None:
    assert frozenset({S.DONE, S.ROLLED_BACK, S.ABANDONED}) == TERMINAL
    for status in set(S) - {S.HALTED}:
        assert (allowed_targets(status) == frozenset()) == (status in TERMINAL), status


def test_every_status_is_reachable_and_every_ticket_can_finish() -> None:
    assert _reachable(S.RECEIVED) == set(S) - {S.HALTED}
    for status in set(S) - {S.HALTED}:
        assert _reachable(status) & TERMINAL, f"{status} cannot reach a terminal status"


def test_check_transition_is_exhaustively_consistent_with_the_table() -> None:
    for src in S:
        for dst in S:
            if dst in allowed_targets(src):
                check_transition(src, dst)
            else:
                with pytest.raises(IllegalTransition):
                    check_transition(src, dst)


@pytest.mark.parametrize(
    ("src", "dst", "hint"),
    [
        (S.RECEIVED, S.BUILDING, "allowed: abandoned, needs_info, triaged"),
        (S.RECEIVED, S.RECEIVED, "allowed: abandoned, needs_info, triaged"),
        (S.BUILDING, S.HALTED, "only via halt()"),
        (S.HALTED, S.BUILDING, "must be resume()d"),
        (S.DONE, S.BUILDING, "done is terminal"),
        (S.ABANDONED, S.RECEIVED, "abandoned is terminal"),
    ],
)
def test_rejections_name_the_transition_and_say_what_to_do(
    src: TicketStatus, dst: TicketStatus, hint: str
) -> None:
    with pytest.raises(IllegalTransition) as excinfo:
        check_transition(src, dst)
    message = str(excinfo.value)
    assert f"{src} -> {dst}" in message
    assert hint in message


def test_transition_table_is_read_only() -> None:
    mutable_view = cast("dict[TicketStatus, frozenset[TicketStatus]]", TRANSITIONS)
    with pytest.raises(TypeError):
        mutable_view[S.DONE] = frozenset({S.RECEIVED})
    assert allowed_targets(S.DONE) == frozenset()


def test_state_machine_error_is_the_public_ledger_error() -> None:
    assert IllegalTransition is ledger_pkg.IllegalTransition
    assert issubclass(IllegalTransition, ledger_pkg.LedgerError)
