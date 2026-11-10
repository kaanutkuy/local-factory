"""Acceptance: evidence building, hashing, signing, binding (docs/specs/phase-1a.md § evidence).

Protected.
"""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from local_factory.evidence.binding import (
    EvidenceKeyError,
    build_evidence,
    check_binding,
    evidence_hash,
    load_or_create_key,
    sign,
    verify_signature,
)
from local_factory.models import (
    CriterionResult,
    CriterionStatus,
    GateResult,
    GateStatus,
    Verdict,
    Waiver,
)

from .conftest import T0

SHA = "1" * 40
TREE = "2" * 40
BASE = "3" * 40
CONTRACT = "sha256:" + "4" * 64
ORACLE = "sha256:" + "5" * 64


def verdict(passed: bool = True) -> Verdict:
    return Verdict(
        ticket_id="T-0001",
        round=1,
        sha=SHA,
        gates=(GateResult(name="unit", status=GateStatus.PASS if passed else GateStatus.FAIL),),
        criteria=(CriterionResult(id="AC-1", status=CriterionStatus.PASS),),
        waivers=(
            Waiver(
                id="W-1",
                gate="mutation",
                scope="m:1",
                reason="equivalent",
                requested_by="implementer",
                adjudicated_by="verifier",
                approved=True,
            ),
        ),
        passed=passed,
        failing=() if passed else ("gate:unit",),
    )


def evidence(passed: bool = True):
    return build_evidence(
        ticket_id="T-0001",
        verdict=verdict(passed),
        tree_hash=TREE,
        base_sha=BASE,
        contract_hash=CONTRACT,
        oracle_version=ORACLE,
        factory_version="0.1.0",
        image_digests={"verifier": "sha256:" + "6" * 64},
        models={"implementer": "fake"},
        clock=lambda: T0,
    )


def test_build_evidence_copies_the_verdict() -> None:
    ev = evidence()
    v = verdict()
    assert ev.candidate_sha == v.sha == SHA
    assert (ev.gates, ev.criteria, ev.waivers, ev.passed) == (v.gates, v.criteria, v.waivers, True)
    assert (ev.tree_hash, ev.base_sha, ev.contract_hash, ev.oracle_version) == (
        TREE,
        BASE,
        CONTRACT,
        ORACLE,
    )
    assert ev.created_at == T0
    assert ev.models == {"implementer": "fake"}


def test_evidence_hash_is_deterministic_and_content_sensitive() -> None:
    assert evidence_hash(evidence()) == evidence_hash(evidence())
    assert evidence_hash(evidence()).startswith("sha256:")
    assert evidence_hash(evidence()) != evidence_hash(evidence(passed=False))


def test_sign_and_verify(tmp_path: Path) -> None:
    key = load_or_create_key(tmp_path / "keys" / "evidence.key")
    ev = evidence()
    signature = sign(ev, key)
    assert verify_signature(ev, signature, key)
    assert not verify_signature(evidence(passed=False), signature, key)
    assert not verify_signature(ev, signature, b"\x00" * 32)
    assert not verify_signature(ev, "not-hex", key)


def test_key_file_is_private_and_stable(tmp_path: Path) -> None:
    path = tmp_path / "keys" / "evidence.key"
    key = load_or_create_key(path)
    assert len(key) == 32
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert load_or_create_key(path) == key


def test_key_file_with_loose_permissions_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "evidence.key"
    load_or_create_key(path)
    path.chmod(0o644)
    with pytest.raises(EvidenceKeyError):
        load_or_create_key(path)


def test_binding_checks_every_anchor() -> None:
    ev = evidence()
    assert (
        check_binding(ev, sha=SHA, tree_hash=TREE, contract_hash=CONTRACT, oracle_version=ORACLE)
        == []
    )
    problems = check_binding(
        ev,
        sha="9" * 40,
        tree_hash="8" * 40,
        contract_hash="sha256:" + "7" * 64,
        oracle_version="sha256:" + "0" * 64,
    )
    assert len(problems) == 4
    joined = " ".join(problems)
    for word in ("sha", "tree", "contract", "oracle"):
        assert word in joined


def test_binding_reports_failed_evidence() -> None:
    problems = check_binding(
        evidence(passed=False),
        sha=SHA,
        tree_hash=TREE,
        contract_hash=CONTRACT,
        oracle_version=ORACLE,
    )
    assert len(problems) == 1
    assert "pass" in problems[0]


def test_build_evidence_requires_consistent_inputs() -> None:
    with pytest.raises(ValueError, match="ticket"):
        build_evidence(
            ticket_id="T-0002",  # verdict belongs to T-0001
            verdict=verdict(),
            tree_hash=TREE,
            base_sha=BASE,
            contract_hash=CONTRACT,
            oracle_version=ORACLE,
            factory_version="0.1.0",
        )
