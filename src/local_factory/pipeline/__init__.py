"""Pipelines: the Phase 1a modules composed into end-to-end flows driven by the ledger.

One flow (:mod:`.flow`), two places to run it: :func:`run_host_slice` (plain directories on the
host, in-process; not a security boundary) and :func:`run_sandboxed_slice` (every worker-side step
in the box).
"""

from __future__ import annotations

from local_factory.pipeline.flow import (
    IMPLEMENTER_ROLE,
    VERIFIER_ROLE,
    SliceError,
    SliceOutcome,
    SliceStatus,
    VerifierReport,
    reconcile_criteria,
)
from local_factory.pipeline.host_slice import run_host_slice
from local_factory.pipeline.sandboxed_slice import run_sandboxed_slice

__all__ = [
    "IMPLEMENTER_ROLE",
    "VERIFIER_ROLE",
    "SliceError",
    "SliceOutcome",
    "SliceStatus",
    "VerifierReport",
    "reconcile_criteria",
    "run_host_slice",
    "run_sandboxed_slice",
]
