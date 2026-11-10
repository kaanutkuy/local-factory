"""The public surface of local_factory.ledger."""

from __future__ import annotations

import pytest

import local_factory.ledger as ledger_pkg
from local_factory.ledger import (
    ContractFrozen,
    EvidenceRecord,
    IllegalTransition,
    Ledger,
    LedgerError,
    TaskNotFound,
    TicketNotFound,
)
from local_factory.ledger.store import EvidenceRecord as StoreEvidenceRecord
from local_factory.ledger.store import Ledger as StoreLedger


def test_package_re_exports_ledger_and_all_errors() -> None:
    assert set(ledger_pkg.__all__) == {
        "Ledger",
        "LedgerError",
        "IllegalTransition",
        "TicketNotFound",
        "TaskNotFound",
        "ContractFrozen",
        "EvidenceRecord",
    }
    assert Ledger is StoreLedger
    assert EvidenceRecord is StoreEvidenceRecord


@pytest.mark.parametrize("error", [IllegalTransition, TicketNotFound, TaskNotFound, ContractFrozen])
def test_every_specific_error_is_a_ledger_error(error: type[LedgerError]) -> None:
    assert issubclass(error, LedgerError)
    assert error is not LedgerError


def test_ledger_error_is_a_plain_exception() -> None:
    assert issubclass(LedgerError, Exception)
    assert not issubclass(LedgerError, ValueError)
