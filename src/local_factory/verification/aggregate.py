"""Min-aggregation of gate and criterion results (docs/specs/phase-1a.md § verification).

A verdict passes only if *every* deterministic gate passed (or was waived by an effective,
verifier-adjudicated waiver) and every acceptance criterion passed. Nothing - in particular no LLM
judgment - can lift a failing gate (PLAN §5.5, §6.4). Vacuous inputs fail.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence

from local_factory.models import (
    AggregateResult,
    CriterionResult,
    CriterionStatus,
    GateResult,
    GateStatus,
    Verdict,
    Waiver,
)

NO_GATES = "gates:none"
NO_CRITERIA = "criteria:none"


def aggregate(
    gates: Sequence[GateResult],
    criteria: Sequence[CriterionResult],
    waivers: Sequence[Waiver] = (),
    *,
    require_criteria: bool = True,
) -> AggregateResult:
    """``failing`` lists none-markers, then failing gates and criteria, each in input order."""
    gates, criteria = tuple(gates), tuple(criteria)
    _reject_duplicates("gate names", (g.name for g in gates))
    _reject_duplicates("criterion ids", (c.id for c in criteria))
    waived = frozenset(w.gate for w in waivers if w.is_effective)

    failing: list[str] = []
    if not gates:
        failing.append(NO_GATES)
    if require_criteria and not criteria:
        failing.append(NO_CRITERIA)
    for gate in gates:
        marker = _gate_marker(gate, waived)
        if marker is not None:
            failing.append(marker)
    failing.extend(f"criterion:{c.id}" for c in criteria if c.status is not CriterionStatus.PASS)
    return AggregateResult(passed=not failing, failing=tuple(failing))


def make_verdict(
    *,
    ticket_id: str,
    round: int,
    sha: str,
    gates: Sequence[GateResult],
    criteria: Sequence[CriterionResult],
    waivers: Sequence[Waiver] = (),
) -> Verdict:
    """A verdict whose ``passed``/``failing`` come from :func:`aggregate` (criteria required)."""
    result = aggregate(gates, criteria, waivers)
    return Verdict(
        ticket_id=ticket_id,
        round=round,
        sha=sha,
        gates=tuple(gates),
        criteria=tuple(criteria),
        waivers=tuple(waivers),
        passed=result.passed,
        failing=result.failing,
    )


def _gate_marker(gate: GateResult, waived: frozenset[str]) -> str | None:
    if gate.status is GateStatus.PASS:
        return None
    if gate.status is GateStatus.SKIPPED:
        return f"gate:{gate.name}:skipped"
    if gate.status is GateStatus.WAIVED:
        return None if gate.name in waived else f"gate:{gate.name}:unwaived"
    return f"gate:{gate.name}"  # FAIL, ERROR - and anything unknown fails closed


def _reject_duplicates(what: str, keys: Iterable[str]) -> None:
    duplicates = sorted(key for key, count in Counter(keys).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate {what}: {', '.join(duplicates)}")
