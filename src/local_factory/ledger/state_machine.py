"""The ticket state machine (docs/specs/phase-1a.md § ledger).

HALTED is deliberately absent from the table: it is entered only via `Ledger.halt()` and left
only via `Ledger.resume()`, which returns the ticket to exactly the status it was halted in.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from local_factory.ledger.errors import IllegalTransition
from local_factory.models import TicketStatus

__all__ = ["TERMINAL", "TRANSITIONS", "IllegalTransition", "allowed_targets", "check_transition"]

_S = TicketStatus

TERMINAL: frozenset[TicketStatus] = frozenset({_S.DONE, _S.ROLLED_BACK, _S.ABANDONED})

TRANSITIONS: Mapping[TicketStatus, frozenset[TicketStatus]] = MappingProxyType(
    {
        _S.RECEIVED: frozenset({_S.TRIAGED, _S.NEEDS_INFO, _S.ABANDONED}),
        _S.TRIAGED: frozenset({_S.CONTRACT_PROPOSED, _S.NEEDS_INFO, _S.ABANDONED}),
        _S.NEEDS_INFO: frozenset({_S.TRIAGED, _S.CONTRACT_PROPOSED, _S.ABANDONED}),
        _S.CONTRACT_PROPOSED: frozenset({_S.APPROVED, _S.NEEDS_INFO, _S.ABANDONED}),
        _S.APPROVED: frozenset({_S.PLANNED, _S.ABANDONED}),
        _S.PLANNED: frozenset({_S.BUILDING, _S.ABANDONED}),
        _S.BUILDING: frozenset({_S.VERIFYING, _S.BLOCKED, _S.ABANDONED}),
        _S.VERIFYING: frozenset({_S.BUILDING, _S.CONVERGED, _S.BLOCKED, _S.ABANDONED}),
        _S.CONVERGED: frozenset({_S.CANDIDATE_VERIFIED, _S.BUILDING, _S.ABANDONED}),
        _S.CANDIDATE_VERIFIED: frozenset({_S.MERGED, _S.CONVERGED, _S.ABANDONED}),
        _S.MERGED: frozenset({_S.AWAITING_RELEASE_APPROVAL, _S.DEPLOYING, _S.DONE}),
        _S.AWAITING_RELEASE_APPROVAL: frozenset({_S.DEPLOYING}),
        _S.DEPLOYING: frozenset({_S.DONE, _S.ROLLED_BACK}),
        _S.BLOCKED: frozenset({_S.BUILDING, _S.CONTRACT_PROPOSED, _S.ABANDONED}),
        _S.DONE: frozenset(),
        _S.ROLLED_BACK: frozenset(),
        _S.ABANDONED: frozenset(),
    }
)


def allowed_targets(status: TicketStatus) -> frozenset[TicketStatus]:
    """Statuses reachable from `status` in one `transition()` (empty for terminal and HALTED)."""
    return TRANSITIONS.get(status, frozenset())


def check_transition(src: TicketStatus, dst: TicketStatus) -> None:
    """Raise `IllegalTransition` unless `src -> dst` is in the transition table."""
    if dst in allowed_targets(src):
        return
    if dst is _S.HALTED:
        hint = "tickets enter halted only via halt()"
    elif src is _S.HALTED:
        hint = "a halted ticket must be resume()d before it can transition"
    elif src in TERMINAL:
        hint = f"{src} is terminal"
    else:
        hint = "allowed: " + ", ".join(sorted(allowed_targets(src)))
    raise IllegalTransition(f"illegal transition {src} -> {dst} ({hint})")
