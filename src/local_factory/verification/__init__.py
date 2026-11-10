"""Verification: the deterministic gate runner and min-aggregation into verdicts."""

from __future__ import annotations

from local_factory.verification.aggregate import aggregate, make_verdict
from local_factory.verification.gates import (
    DEFAULT_GATE_ORDER,
    GateSpec,
    gates_from_project,
    run_gate,
    run_gates,
)

__all__ = [
    "DEFAULT_GATE_ORDER",
    "GateSpec",
    "aggregate",
    "gates_from_project",
    "make_verdict",
    "run_gate",
    "run_gates",
]
