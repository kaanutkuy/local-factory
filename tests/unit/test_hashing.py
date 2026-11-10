"""Unit tests for canonical serialization and content hashing (local_factory.hashing)."""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import UTC, datetime

import pytest
from pydantic import JsonValue

from local_factory.hashing import canonical_json, content_hash, sha256_tag, utc_now
from local_factory.models import (
    HASH_TAG_PATTERN,
    Contract,
    Criterion,
    GateResult,
    GateStatus,
    Halt,
    HaltReason,
    TestLayer,
)

EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def contract(summary: str = "s") -> Contract:
    return Contract(
        version=1,
        summary=summary,
        criteria=(Criterion(id="AC-1", ears="When x, shall y.", layer=TestLayer.UNIT),),
    )


def test_utc_now_is_timezone_aware_utc() -> None:
    before = datetime.now(UTC)
    now = utc_now()
    assert now.tzinfo is UTC
    assert before <= now <= datetime.now(UTC)


def test_canonical_json_sorts_keys_recursively_without_whitespace() -> None:
    data: JsonValue = {"b": 1, "a": {"z": [3, {"y": None, "x": True}], "c": "d"}}
    assert canonical_json(data) == b'{"a":{"c":"d","z":[3,{"x":true,"y":null}]},"b":1}'


def test_canonical_json_is_independent_of_insertion_order() -> None:
    assert canonical_json({"a": 1, "b": 2}) == canonical_json({"b": 2, "a": 1})


def test_canonical_json_keeps_unicode_as_utf8() -> None:
    encoded = canonical_json({"name": "café ✓"})
    assert encoded == '{"name":"café ✓"}'.encode()
    assert b"\\u" not in encoded


def test_canonical_json_of_scalars_and_lists() -> None:
    assert canonical_json("x") == b'"x"'
    assert canonical_json(1.5) == b"1.5"
    assert canonical_json(None) == b"null"
    assert canonical_json([2, 1]) == b"[2,1]"  # list order is significant


def test_canonical_json_of_a_model_uses_its_json_mode_dump() -> None:
    halt = Halt(reason=HaltReason.BUDGET, resume_at=datetime(2026, 1, 1, tzinfo=UTC), detail="d")
    assert canonical_json(halt) == canonical_json(halt.model_dump(mode="json"))
    assert json.loads(canonical_json(halt)) == {
        "detail": "d",
        "reason": "budget",
        "resume_at": "2026-01-01T00:00:00Z",
    }


def test_sha256_tag_format_and_known_vector() -> None:
    assert sha256_tag(b"") == "sha256:" + EMPTY_SHA256
    assert sha256_tag(b"abc") == "sha256:" + hashlib.sha256(b"abc").hexdigest()
    assert re.fullmatch(HASH_TAG_PATTERN, sha256_tag(b"anything"))


def test_content_hash_is_the_tag_of_the_canonical_json() -> None:
    data: JsonValue = {"k": [1, 2, 3]}
    assert content_hash(data) == sha256_tag(canonical_json(data))
    assert content_hash(contract()) == sha256_tag(canonical_json(contract()))


def test_content_hash_is_deterministic_and_content_sensitive() -> None:
    assert content_hash(contract()) == content_hash(contract())
    assert content_hash(contract()) != content_hash(contract("other"))
    assert content_hash(contract()) == content_hash(contract().model_dump(mode="json"))
    assert content_hash({"a": 1}) != content_hash({"a": "1"})
    assert content_hash(contract()) == content_hash(
        Contract.model_validate_json(contract().model_dump_json())
    )


def test_content_hash_pinned_value() -> None:
    # Guards against accidental changes to the canonical form (which would invalidate every
    # frozen contract hash and evidence binding).
    assert content_hash({"b": [1, "é"], "a": None}) == sha256_tag('{"a":null,"b":[1,"é"]}'.encode())


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_floats_are_rejected_wherever_they_appear(value: float) -> None:
    cases: list[JsonValue] = [value, [1.0, value], {"outer": {"inner": value}}]
    for data in cases:
        with pytest.raises(ValueError, match="JSON compliant"):
            canonical_json(data)
        with pytest.raises(ValueError, match="JSON compliant"):
            content_hash(data)


def test_non_finite_floats_inside_models_are_rejected() -> None:
    gate = GateResult.model_construct(name="unit", status=GateStatus.PASS, metrics={"x": math.nan})
    with pytest.raises(ValueError, match="JSON compliant"):
        content_hash(gate)


def test_large_finite_floats_are_fine() -> None:
    assert canonical_json({"x": 1e308, "y": -0.0}) == b'{"x":1e+308,"y":-0.0}'
