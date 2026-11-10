"""Unit tests for check_binding: every anchor is compared, and failed evidence never binds."""

from __future__ import annotations

import pytest

from local_factory.evidence.binding import check_binding
from local_factory.models import Evidence

from .conftest import CONTRACT, ORACLE, SHA, TREE, make_evidence, make_verdict

ANCHORS = {"sha": SHA, "tree_hash": TREE, "contract_hash": CONTRACT, "oracle_version": ORACLE}
OTHER = {
    "sha": "9" * 40,
    "tree_hash": "8" * 40,
    "contract_hash": "sha256:" + "7" * 64,
    "oracle_version": "sha256:" + "0" * 64,
}
LABELS = {
    "sha": "candidate sha",
    "tree_hash": "tree hash",
    "contract_hash": "contract hash",
    "oracle_version": "oracle version",
}


def check(evidence: Evidence, **overrides: str) -> list[str]:
    anchors = {**ANCHORS, **overrides}
    return check_binding(
        evidence,
        sha=anchors["sha"],
        tree_hash=anchors["tree_hash"],
        contract_hash=anchors["contract_hash"],
        oracle_version=anchors["oracle_version"],
    )


def test_matching_passed_evidence_is_bound() -> None:
    assert check(make_evidence()) == []


@pytest.mark.parametrize("anchor", list(ANCHORS))
def test_each_anchor_mismatch_is_reported_with_both_values(anchor: str) -> None:
    problems = check(make_evidence(), **{anchor: OTHER[anchor]})
    assert problems == [
        f"{LABELS[anchor]} mismatch: evidence has {ANCHORS[anchor]}, candidate has {OTHER[anchor]}"
    ]


def test_all_simultaneous_mismatches_are_reported_in_order() -> None:
    problems = check(make_evidence(passed=False), **OTHER)
    assert len(problems) == 5
    for problem, label in zip(problems, LABELS.values(), strict=False):
        assert problem.startswith(f"{label} mismatch")
    assert "did not pass" in problems[-1]


def test_two_mismatches_report_exactly_those_two() -> None:
    problems = check(make_evidence(), tree_hash=OTHER["tree_hash"], sha=OTHER["sha"])
    assert [p.split(" mismatch")[0] for p in problems] == ["candidate sha", "tree hash"]


def test_failed_evidence_never_binds_even_when_anchors_match() -> None:
    assert check(make_evidence(passed=False)) == [
        "evidence records a verification that did not pass (passed=false)"
    ]


def test_comparison_is_exact() -> None:
    sha = "abcdef" + "0" * 34
    evidence = make_evidence(verdict=make_verdict(sha=sha))
    assert check(evidence, sha=sha) == []
    problems = check(evidence, sha=sha.upper(), contract_hash=CONTRACT + " ")
    assert [p.split(" mismatch")[0] for p in problems] == ["candidate sha", "contract hash"]


def test_base_sha_is_not_a_binding_anchor() -> None:
    other_base = make_evidence(base_sha="9" * 40)
    assert check(other_base) == []
