from __future__ import annotations

import pytest
from pydantic import ValidationError

from local_factory.models import (
    CriterionResult,
    CriterionStatus,
    GateResult,
    GateStatus,
    Waiver,
)
from local_factory.verification.aggregate import aggregate, make_verdict

G = GateStatus
C = CriterionStatus
SHA = "a" * 40


def gate(name: str, status: GateStatus = G.PASS) -> GateResult:
    return GateResult(name=name, status=status)


def crit(cid: str, status: CriterionStatus = C.PASS) -> CriterionResult:
    return CriterionResult(id=cid, status=status)


def waiver(
    gate_name: str,
    *,
    wid: str = "W-1",
    approved: bool = True,
    requested_by: str = "implementer",
    adjudicated_by: str | None = "verifier",
) -> Waiver:
    return Waiver(
        id=wid,
        gate=gate_name,
        scope="mutant:7",
        reason="equivalent mutant",
        requested_by=requested_by,
        adjudicated_by=adjudicated_by,
        approved=approved,
    )


def test_both_none_markers_come_first_in_fixed_order() -> None:
    assert aggregate([], []).failing == ("gates:none", "criteria:none")
    assert aggregate([], [], require_criteria=False).failing == ("gates:none",)


def test_none_markers_precede_failing_criteria() -> None:
    result = aggregate([], [crit("AC-1", C.FAIL)])
    assert result.failing == ("gates:none", "criterion:AC-1")
    assert not result.passed


def test_failing_entries_keep_input_order_not_sorted_order() -> None:
    result = aggregate(
        [gate("zeta", G.FAIL), gate("alpha", G.ERROR)],
        [crit("AC-9", C.FAIL), crit("AC-1", C.INCONCLUSIVE)],
    )
    assert result.failing == ("gate:zeta", "gate:alpha", "criterion:AC-9", "criterion:AC-1")


def test_require_criteria_false_still_judges_given_criteria() -> None:
    result = aggregate([gate("unit")], [crit("AC-1", C.FAIL)], require_criteria=False)
    assert result.failing == ("criterion:AC-1",)


def test_mixed_waivers_only_lift_waived_gates_with_an_effective_waiver() -> None:
    gates = [
        gate("mutation", G.WAIVED),  # effective waiver -> lifted
        gate("coverage", G.WAIVED),  # waiver not approved -> unwaived
        gate("flake", G.WAIVED),  # self-adjudicated -> unwaived
        gate("unit", G.FAIL),  # an effective waiver never lifts a FAIL
        gate("e2e", G.SKIPPED),  # ... nor a SKIPPED gate
        gate("lint"),
    ]
    waivers = [
        waiver("mutation", wid="W-1"),
        waiver("coverage", wid="W-2", approved=False),
        waiver("flake", wid="W-3", requested_by="verifier", adjudicated_by="verifier"),
        waiver("unit", wid="W-4"),
        waiver("e2e", wid="W-5"),
    ]
    result = aggregate(gates, [crit("AC-1")], waivers)
    assert result.failing == (
        "gate:coverage:unwaived",
        "gate:flake:unwaived",
        "gate:unit",
        "gate:e2e:skipped",
    )


def test_one_effective_waiver_among_ineffective_ones_is_enough() -> None:
    waivers = [
        waiver("mutation", wid="W-1", approved=False),
        waiver("mutation", wid="W-2", adjudicated_by=None),
        waiver("mutation", wid="W-3"),
    ]
    assert aggregate([gate("mutation", G.WAIVED)], [crit("AC-1")], waivers).passed


def test_waivers_match_the_gate_name_exactly() -> None:
    gates = [gate("mutation", G.WAIVED)]
    for other in ("Mutation", "mutation ", "mut"):
        assert aggregate(gates, [crit("AC-1")], [waiver(other)]).failing == (
            "gate:mutation:unwaived",
        )


def test_waivers_do_not_affect_criteria() -> None:
    result = aggregate([gate("unit")], [crit("AC-1", C.FAIL)], [waiver("AC-1")])
    assert result.failing == ("criterion:AC-1",)


def test_duplicates_are_rejected_even_when_everything_passes() -> None:
    with pytest.raises(ValueError, match=r"duplicate gate names: lint, unit"):
        aggregate([gate("unit"), gate("lint"), gate("unit"), gate("lint")], [crit("AC-1")])
    with pytest.raises(ValueError, match=r"duplicate criterion ids: AC-2"):
        aggregate([gate("unit")], [crit("AC-2"), crit("AC-1"), crit("AC-2", C.FAIL)])


def test_make_verdict_carries_every_input_including_rejected_waivers() -> None:
    waivers = [waiver("mutation", wid="W-1"), waiver("coverage", wid="W-2", approved=False)]
    verdict = make_verdict(
        ticket_id="T-0042",
        round=1,
        sha=SHA,
        gates=[gate("unit"), gate("mutation", G.WAIVED)],
        criteria=[crit("AC-1")],
        waivers=waivers,
    )
    assert verdict.passed and verdict.failing == ()
    assert verdict.waivers == tuple(waivers)
    assert verdict.gates == (gate("unit"), gate("mutation", G.WAIVED))
    assert verdict.criteria == (crit("AC-1"),)


def test_make_verdict_requires_criteria() -> None:
    verdict = make_verdict(ticket_id="T-0001", round=1, sha=SHA, gates=[gate("unit")], criteria=[])
    assert not verdict.passed
    assert verdict.failing == ("criteria:none",)


def test_make_verdict_validates_its_fields() -> None:
    with pytest.raises(ValidationError):
        make_verdict(ticket_id="T-1", round=1, sha=SHA, gates=[gate("u")], criteria=[crit("AC-1")])
    with pytest.raises(ValidationError):
        make_verdict(
            ticket_id="T-0001", round=0, sha=SHA, gates=[gate("u")], criteria=[crit("AC-1")]
        )
    with pytest.raises(ValidationError):
        make_verdict(
            ticket_id="T-0001", round=1, sha="xyz", gates=[gate("u")], criteria=[crit("AC-1")]
        )
