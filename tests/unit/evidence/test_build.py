"""Unit tests for build_evidence and evidence_hash."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from types import MappingProxyType
from typing import Any

import pytest
from pydantic import ValidationError

from local_factory.evidence.binding import build_evidence, evidence_hash
from local_factory.hashing import canonical_json, content_hash
from local_factory.models import Evidence

from .conftest import BASE, CONTRACT, ORACLE, SHA, T0, TREE, make_evidence, make_verdict

# --------------------------------------------------------------------------- build_evidence


def test_build_copies_every_verdict_field_and_anchor() -> None:
    verdict = make_verdict()
    ev = make_evidence()
    assert ev == Evidence(
        ticket_id="T-0001",
        candidate_sha=SHA,
        tree_hash=TREE,
        base_sha=BASE,
        contract_hash=CONTRACT,
        oracle_version=ORACLE,
        factory_version="0.1.0",
        image_digests={"verifier": "sha256:" + "6" * 64},
        models={"implementer": "fake"},
        gates=verdict.gates,
        criteria=verdict.criteria,
        waivers=verdict.waivers,
        passed=True,
        created_at=T0,
    )
    assert ev.schema_version == 1


def test_build_records_a_failed_verdict_as_not_passed() -> None:
    ev = make_evidence(passed=False)
    assert ev.passed is False
    assert ev.gates == make_verdict(passed=False).gates


def test_build_accepts_a_failed_verdict_without_failing_reasons() -> None:
    assert make_evidence(verdict=make_verdict(passed=False, failing=())).passed is False


def test_optional_mappings_default_to_empty() -> None:
    ev = make_evidence(image_digests=None, models=None)
    assert (ev.image_digests, ev.models) == ({}, {})


def test_mappings_are_copied_not_aliased() -> None:
    models = {"implementer": "fake"}
    ev = make_evidence(models=models, image_digests=MappingProxyType({"app": "sha256:x"}))
    models["implementer"] = "changed"
    assert ev.models == {"implementer": "fake"}
    assert ev.image_digests == {"app": "sha256:x"}


def test_clock_is_called_once() -> None:
    calls: list[datetime] = []

    def clock() -> datetime:
        calls.append(T0)
        return T0

    assert make_evidence(clock=clock).created_at == T0
    assert len(calls) == 1


def test_default_clock_is_current_utc_time() -> None:
    before = datetime.now(UTC)
    fields: dict[str, Any] = {
        "ticket_id": "T-0001",
        "verdict": make_verdict(),
        "tree_hash": TREE,
        "base_sha": BASE,
        "contract_hash": CONTRACT,
        "oracle_version": ORACLE,
        "factory_version": "0.1.0",
    }
    created = build_evidence(**fields).created_at
    assert created.tzinfo is not None
    assert before - timedelta(seconds=1) <= created <= datetime.now(UTC) + timedelta(seconds=1)


def test_ticket_mismatch_names_both_tickets() -> None:
    with pytest.raises(ValueError, match=r"ticket T-0001, not ticket T-0002"):
        make_evidence(ticket_id="T-0002")


def test_verdict_claiming_a_pass_with_failures_is_refused() -> None:
    inconsistent = make_verdict(passed=True, failing=("gate:unit",))
    with pytest.raises(ValueError, match=r"inconsistent verdict for ticket T-0001"):
        make_evidence(verdict=inconsistent)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tree_hash", "not-a-sha"),
        ("base_sha", "A" * 40),
        ("contract_hash", "4" * 64),
        ("oracle_version", "sha256:short"),
        ("factory_version", ""),
    ],
)
def test_malformed_anchors_are_rejected(field: str, value: str) -> None:
    overrides: dict[str, Any] = {field: value}
    with pytest.raises(ValidationError):
        make_evidence(**overrides)


# --------------------------------------------------------------------------- evidence_hash


def test_hash_is_sha256_of_the_canonical_json() -> None:
    ev = make_evidence()
    expected = "sha256:" + hashlib.sha256(canonical_json(ev)).hexdigest()
    assert evidence_hash(ev) == expected == content_hash(ev)


def test_hash_ignores_mapping_insertion_order() -> None:
    first = make_evidence(models={"a": "m1", "b": "m2"})
    second = make_evidence(models={"b": "m2", "a": "m1"})
    assert evidence_hash(first) == evidence_hash(second)


@pytest.mark.parametrize(
    "overrides",
    [
        {"tree_hash": "9" * 40},
        {"base_sha": "9" * 40},
        {"contract_hash": "sha256:" + "9" * 64},
        {"oracle_version": "sha256:" + "9" * 64},
        {"factory_version": "0.1.1"},
        {"models": {"implementer": "other"}},
        {"image_digests": {}},
        {"clock": lambda: T0 + timedelta(microseconds=1)},
        {"verdict": make_verdict(sha="9" * 40)},
        {"verdict": make_verdict(waivers=())},
    ],
)
def test_hash_changes_with_any_field(overrides: dict[str, Any]) -> None:
    assert evidence_hash(make_evidence(**overrides)) != evidence_hash(make_evidence())


def test_hash_survives_a_json_round_trip() -> None:
    ev = make_evidence()
    assert evidence_hash(Evidence.model_validate_json(ev.model_dump_json())) == evidence_hash(ev)
