"""Canonical serialization and content hashing. Orchestrator-owned."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

from pydantic import BaseModel, JsonValue


def utc_now() -> datetime:
    return datetime.now(UTC)


def canonical_json(data: BaseModel | JsonValue) -> bytes:
    """Deterministic JSON bytes: sorted keys, no insignificant whitespace, UTF-8.

    NaN and infinities are rejected (ValueError): they are not JSON and would make hashes of
    otherwise-equal content diverge across implementations.
    """
    if isinstance(data, BaseModel):
        data = data.model_dump(mode="json")
    return json.dumps(
        data, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def sha256_tag(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def content_hash(data: BaseModel | JsonValue) -> str:
    """`sha256:<hex>` over the canonical JSON of `data`."""
    return sha256_tag(canonical_json(data))
