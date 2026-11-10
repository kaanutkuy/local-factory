"""Evidence binding: build, hash, HMAC-sign and check evidence against an exact candidate."""

from __future__ import annotations

from local_factory.evidence.binding import (
    KEY_BYTES,
    EvidenceKeyError,
    build_evidence,
    check_binding,
    evidence_hash,
    load_or_create_key,
    sign,
    verify_signature,
)

__all__ = [
    "KEY_BYTES",
    "EvidenceKeyError",
    "build_evidence",
    "check_binding",
    "evidence_hash",
    "load_or_create_key",
    "sign",
    "verify_signature",
]
