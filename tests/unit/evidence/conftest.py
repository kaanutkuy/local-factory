"""Shared factories for the evidence unit tests."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from local_factory.evidence.binding import build_evidence
from local_factory.models import (
    CriterionResult,
    CriterionStatus,
    Evidence,
    GateResult,
    GateStatus,
    Verdict,
    Waiver,
)

T0 = datetime(2026, 1, 1, tzinfo=UTC)
SHA = "1" * 40
TREE = "2" * 40
BASE = "3" * 40
CONTRACT = "sha256:" + "4" * 64
ORACLE = "sha256:" + "5" * 64
KEY = bytes(range(32))


def make_verdict(*, passed: bool = True, **overrides: Any) -> Verdict:
    fields: dict[str, Any] = {
        "ticket_id": "T-0001",
        "round": 2,
        "sha": SHA,
        "gates": (
            GateResult(name="lint", status=GateStatus.PASS),
            GateResult(name="unit", status=GateStatus.PASS if passed else GateStatus.FAIL),
        ),
        "criteria": (CriterionResult(id="AC-1", status=CriterionStatus.PASS),),
        "waivers": (
            Waiver(
                id="W-1",
                gate="mutation",
                scope="m:1",
                reason="equivalent mutant",
                requested_by="implementer",
                adjudicated_by="verifier",
                approved=True,
            ),
        ),
        "passed": passed,
        "failing": () if passed else ("gate:unit",),
    }
    return Verdict.model_validate({**fields, **overrides})


def make_evidence(*, passed: bool = True, **overrides: Any) -> Evidence:
    fields: dict[str, Any] = {
        "ticket_id": "T-0001",
        "verdict": make_verdict(passed=passed),
        "tree_hash": TREE,
        "base_sha": BASE,
        "contract_hash": CONTRACT,
        "oracle_version": ORACLE,
        "factory_version": "0.1.0",
        "image_digests": {"verifier": "sha256:" + "6" * 64},
        "models": {"implementer": "fake"},
        "clock": lambda: T0,
    }
    return build_evidence(**{**fields, **overrides})
