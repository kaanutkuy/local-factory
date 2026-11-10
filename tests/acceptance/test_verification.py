"""Acceptance: gate runner and min-aggregation (docs/specs/phase-1a.md § verification).

Protected.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from local_factory.config import ProjectConfig
from local_factory.models import (
    CriterionResult,
    CriterionStatus,
    GateResult,
    GateStatus,
    Waiver,
)
from local_factory.verification.aggregate import aggregate, make_verdict
from local_factory.verification.gates import (
    DEFAULT_GATE_ORDER,
    GateSpec,
    gates_from_project,
    run_gate,
    run_gates,
)

G = GateStatus
C = CriterionStatus
SHA = "e" * 40


def gate(name: str, status: GateStatus = G.PASS) -> GateResult:
    return GateResult(name=name, status=status)


def crit(cid: str, status: CriterionStatus = C.PASS) -> CriterionResult:
    return CriterionResult(id=cid, status=status)


def waiver(gate_name: str, *, approved: bool = True, by: str | None = "verifier") -> Waiver:
    return Waiver(
        id="W-1",
        gate=gate_name,
        scope="mutant:42",
        reason="equivalent mutant",
        requested_by="implementer",
        adjudicated_by=by,
        approved=approved,
    )


# --------------------------------------------------------------------------- gate specs


def test_gates_from_project_follow_default_order_and_skip_missing() -> None:
    assert DEFAULT_GATE_ORDER == ("lint", "typecheck", "unit", "integration", "e2e")
    config = ProjectConfig(name="p", commands={"unit": "u", "lint": "l", "serve": "s"})
    specs = gates_from_project(config)
    assert [(s.name, s.command) for s in specs] == [("lint", "l"), ("unit", "u")]
    assert [s.name for s in gates_from_project(config, ["unit"])] == ["unit"]


# --------------------------------------------------------------------------- running gates


def test_passing_gate(tmp_path: Path) -> None:
    result = run_gate(GateSpec(name="lint", command="echo all good"), cwd=tmp_path)
    assert (result.name, result.status, result.exit_code) == ("lint", G.PASS, 0)
    assert "all good" in result.output_tail
    assert result.duration_s >= 0


def test_failing_gate_keeps_exit_code_and_stderr(tmp_path: Path) -> None:
    result = run_gate(GateSpec(name="unit", command="echo boom >&2; exit 3"), cwd=tmp_path)
    assert (result.status, result.exit_code) == (G.FAIL, 3)
    assert "boom" in result.output_tail


def test_missing_command_fails(tmp_path: Path) -> None:
    result = run_gate(GateSpec(name="unit", command="definitely-not-a-command-xyz"), cwd=tmp_path)
    assert result.status is G.FAIL
    assert result.exit_code == 127


def test_gate_runs_in_cwd_with_env(tmp_path: Path) -> None:
    (tmp_path / "marker.txt").write_text("here\n")
    result = run_gate(
        GateSpec(name="unit", command='cat marker.txt && echo "$GATE_VAR"'),
        cwd=tmp_path,
        env={"GATE_VAR": "from-env"},
    )
    assert result.status is G.PASS
    assert "here" in result.output_tail and "from-env" in result.output_tail


def test_output_tail_is_bounded(tmp_path: Path) -> None:
    result = run_gate(
        GateSpec(name="unit", command="printf 'a%.0s' $(seq 1 500); printf END"),
        cwd=tmp_path,
        tail_bytes=10,
    )
    assert len(result.output_tail) <= 10
    assert result.output_tail.endswith("END")


def test_timeout_kills_the_whole_process_group(tmp_path: Path) -> None:
    started = time.monotonic()
    result = run_gate(
        GateSpec(name="e2e", command="sleep 30 & sleep 30; wait", timeout_s=0.5), cwd=tmp_path
    )
    assert result.status is G.ERROR
    assert time.monotonic() - started < 10


def test_run_gates_fail_fast_skips_without_executing(tmp_path: Path) -> None:
    specs = [
        GateSpec(name="lint", command="true"),
        GateSpec(name="unit", command="false"),
        GateSpec(name="e2e", command="touch ran.txt"),
    ]
    results = run_gates(specs, cwd=tmp_path, fail_fast=True)
    assert [(r.name, r.status) for r in results] == [
        ("lint", G.PASS),
        ("unit", G.FAIL),
        ("e2e", G.SKIPPED),
    ]
    assert not (tmp_path / "ran.txt").exists()
    results = run_gates(specs, cwd=tmp_path)
    assert [r.status for r in results] == [G.PASS, G.FAIL, G.PASS]


# --------------------------------------------------------------------------- min-aggregation


def test_all_passing_passes() -> None:
    result = aggregate([gate("lint"), gate("unit")], [crit("AC-1"), crit("AC-2")])
    assert result.passed and result.failing == ()


def test_vacuous_verdicts_fail() -> None:
    assert aggregate([], [crit("AC-1")]).failing == ("gates:none",)
    assert aggregate([gate("unit")], []).failing == ("criteria:none",)
    assert aggregate([gate("unit")], [], require_criteria=False).passed


def test_one_failure_fails_everything_in_order() -> None:
    result = aggregate(
        [gate("lint"), gate("unit", G.FAIL), gate("e2e", G.ERROR), gate("mut", G.SKIPPED)],
        [crit("AC-1"), crit("AC-2", C.INCONCLUSIVE), crit("AC-3", C.FAIL)],
    )
    assert not result.passed
    assert result.failing == (
        "gate:unit",
        "gate:e2e",
        "gate:mut:skipped",
        "criterion:AC-2",
        "criterion:AC-3",
    )


def test_waived_gate_needs_an_effective_waiver() -> None:
    gates = [gate("mutation", G.WAIVED)]
    crits = [crit("AC-1")]
    assert aggregate(gates, crits).failing == ("gate:mutation:unwaived",)
    assert aggregate(gates, crits, [waiver("mutation", approved=False)]).failing == (
        "gate:mutation:unwaived",
    )
    assert aggregate(gates, crits, [waiver("mutation", by=None)]).failing == (
        "gate:mutation:unwaived",
    )
    assert aggregate(gates, crits, [waiver("mutation", by="implementer")]).failing == (
        "gate:mutation:unwaived",
    )
    assert aggregate(gates, crits, [waiver("coverage")]).failing == ("gate:mutation:unwaived",)
    assert aggregate(gates, crits, [waiver("mutation")]).passed


def test_duplicates_are_rejected() -> None:
    with pytest.raises(ValueError, match="unit"):
        aggregate([gate("unit"), gate("unit")], [crit("AC-1")])
    with pytest.raises(ValueError, match="AC-1"):
        aggregate([gate("unit")], [crit("AC-1"), crit("AC-1")])


def test_make_verdict_applies_aggregation() -> None:
    verdict = make_verdict(
        ticket_id="T-0001",
        round=2,
        sha=SHA,
        gates=[gate("unit", G.FAIL)],
        criteria=[crit("AC-1")],
    )
    assert (verdict.ticket_id, verdict.round, verdict.sha) == ("T-0001", 2, SHA)
    assert not verdict.passed
    assert verdict.failing == ("gate:unit",)
    assert verdict.gates == (gate("unit", G.FAIL),)
    passing = make_verdict(
        ticket_id="T-0001", round=3, sha=SHA, gates=[gate("unit")], criteria=[crit("AC-1")]
    )
    assert passing.passed and passing.failing == ()
