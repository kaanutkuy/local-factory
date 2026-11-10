"""Pipelines: the Phase 1a modules composed into end-to-end flows driven by the ledger."""

from __future__ import annotations

from local_factory.pipeline.host_slice import (
    IMPLEMENTER_ROLE,
    VERIFIER_ROLE,
    SliceError,
    SliceOutcome,
    SliceStatus,
    VerifierReport,
    reconcile_criteria,
    run_host_slice,
)

__all__ = [
    "IMPLEMENTER_ROLE",
    "VERIFIER_ROLE",
    "SliceError",
    "SliceOutcome",
    "SliceStatus",
    "VerifierReport",
    "reconcile_criteria",
    "run_host_slice",
]
