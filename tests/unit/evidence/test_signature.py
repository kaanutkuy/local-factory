"""Unit tests for sign and verify_signature (HMAC-SHA256, constant-time comparison)."""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Callable

import pytest

from local_factory.evidence.binding import sign, verify_signature
from local_factory.hashing import canonical_json

from .conftest import KEY, make_evidence


def test_signature_is_hex_hmac_sha256_of_the_canonical_json() -> None:
    ev = make_evidence()
    expected = hmac.new(KEY, canonical_json(ev), hashlib.sha256).hexdigest()
    assert sign(ev, KEY) == expected
    assert len(expected) == 64


def test_signature_is_deterministic_and_key_dependent() -> None:
    ev = make_evidence()
    assert sign(ev, KEY) == sign(make_evidence(), KEY)
    assert sign(ev, KEY) != sign(ev, KEY[::-1])


def test_valid_signature_verifies() -> None:
    ev = make_evidence()
    assert verify_signature(ev, sign(ev, KEY), KEY)


def test_uppercase_hex_encodes_the_same_signature() -> None:
    ev = make_evidence()
    assert verify_signature(ev, sign(ev, KEY).upper(), KEY)


def test_signature_does_not_transfer_to_other_evidence_or_keys() -> None:
    ev = make_evidence()
    signature = sign(ev, KEY)
    assert not verify_signature(make_evidence(passed=False), signature, KEY)
    assert not verify_signature(make_evidence(factory_version="0.2.0"), signature, KEY)
    assert not verify_signature(ev, signature, b"\x00" * 32)


def test_any_flipped_digit_is_rejected() -> None:
    ev = make_evidence()
    signature = sign(ev, KEY)
    for index in (0, 31, 63):
        flipped = "0" if signature[index] != "0" else "1"
        tampered = signature[:index] + flipped + signature[index + 1 :]
        assert not verify_signature(ev, tampered, KEY)


@pytest.mark.parametrize(
    "transform",
    [
        lambda _: "",
        lambda _: "not-hex",
        lambda s: s[:-1],
        lambda s: s + "0",
        lambda s: s[:-2],
        lambda s: s[:32] + " " + s[33:],
        lambda s: " " + s[1:],
        lambda s: s[:-1] + "\n",
        lambda s: s[:-1] + "é",
        lambda s: "g" + s[1:],
        lambda s: "0x" + s[2:],
    ],
)
def test_malformed_signatures_are_invalid_not_errors(transform: Callable[[str], str]) -> None:
    ev = make_evidence()
    assert verify_signature(ev, transform(sign(ev, KEY)), KEY) is False


def test_verification_uses_constant_time_comparison(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[bytes, bytes]] = []
    real = hmac.compare_digest

    def spy(a: bytes, b: bytes) -> bool:
        calls.append((a, b))
        return real(a, b)

    monkeypatch.setattr(hmac, "compare_digest", spy)
    ev = make_evidence()
    assert verify_signature(ev, sign(ev, KEY), KEY)
    assert len(calls) == 1


def test_empty_key_is_refused() -> None:
    ev = make_evidence()
    with pytest.raises(ValueError, match="must not be empty"):
        sign(ev, b"")
    with pytest.raises(ValueError, match="must not be empty"):
        verify_signature(ev, "0" * 64, b"")
