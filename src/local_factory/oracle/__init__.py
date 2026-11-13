"""The oracle: the independent tests that define "done" before any code is written (Gate 1).

Visible acceptance tests are committed for the implementers; black-box probes are sealed in the
holdout store and run by the factory alone. See docs/specs/phase-2b.md § oracle.
"""

from __future__ import annotations

from local_factory.hashing import content_hash
from local_factory.models import Holdout
from local_factory.oracle._ids import check_ticket_id
from local_factory.oracle.holdout import HoldoutError, HoldoutStore
from local_factory.oracle.policy import (
    MAX_ORACLE_FILE_BYTES,
    MAX_ORACLE_FILES,
    ORACLE_BINARY,
    ORACLE_FILES,
    ORACLE_PATH,
    ORACLE_SIZE,
    ORACLE_SYMLINK,
    acceptance_dir,
    check_oracle_changes,
)
from local_factory.oracle.probes import (
    PROBE_SCRIPT,
    PROBE_TIMEOUT_S,
    judge_probe,
    probe_argv,
    run_probe,
)
from local_factory.oracle.redfirst import (
    NO_TESTS_RAN_EXIT,
    RED_TEST_TIMEOUT_S,
    judge_visible,
    red_first,
    render_acceptance,
    split_test_id,
)

__all__ = [
    "MAX_ORACLE_FILES",
    "MAX_ORACLE_FILE_BYTES",
    "NO_TESTS_RAN_EXIT",
    "ORACLE_BINARY",
    "ORACLE_FILES",
    "ORACLE_PATH",
    "ORACLE_REF_PREFIX",
    "ORACLE_SIZE",
    "ORACLE_SYMLINK",
    "PROBE_SCRIPT",
    "PROBE_TIMEOUT_S",
    "RED_TEST_TIMEOUT_S",
    "HoldoutError",
    "HoldoutStore",
    "acceptance_dir",
    "check_oracle_changes",
    "judge_probe",
    "judge_visible",
    "oracle_ref",
    "oracle_version",
    "probe_argv",
    "red_first",
    "render_acceptance",
    "run_probe",
    "split_test_id",
]

ORACLE_REF_PREFIX = "refs/factory/oracle/"


def oracle_ref(ticket_id: str) -> str:
    """The ref that keeps a ticket's accepted oracle commit: ``refs/factory/oracle/T-0001``."""
    return f"{ORACLE_REF_PREFIX}{check_ticket_id(ticket_id)}"


def oracle_version(holdout: Holdout) -> str:
    """The version the ledger freezes for a ticket's oracle: the hash of its holdout record.

    The record binds the oracle commit (``oracle_sha``) and every probe with its expectation, so
    changing either after approval changes the version.
    """
    return content_hash(holdout)
