"""Ledger errors. Every error the ledger raises on an invalid operation is a `LedgerError`.

The subclass names are fixed by the phase-1a spec (and its acceptance tests), hence the N818
suppressions: they read as domain conditions rather than carrying an `Error` suffix.
"""

from __future__ import annotations


class LedgerError(Exception):
    """An operation the ledger refuses: unknown records, illegal changes, unmet preconditions."""


class IllegalTransition(LedgerError):  # noqa: N818 - name fixed by docs/specs/phase-1a.md
    """A ticket status change that the state machine (or halt/resume rules) does not allow."""


class TicketNotFound(LedgerError):  # noqa: N818 - name fixed by docs/specs/phase-1a.md
    """No ticket with the given id exists in this ledger."""


class TaskNotFound(LedgerError):  # noqa: N818 - name fixed by docs/specs/phase-1a.md
    """No task with the given id exists in this ledger."""


class ContractFrozen(LedgerError):  # noqa: N818 - name fixed by docs/specs/phase-1a.md
    """The ticket's contract (and oracle version) are frozen and can no longer change."""
