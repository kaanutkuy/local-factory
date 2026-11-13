"""Pipelines: the Phase 1a modules composed into end-to-end flows driven by the ledger.

One flow (:mod:`.flow`), two places to run it: :func:`run_host_slice` (plain directories on the
host, in-process; not a security boundary) and :func:`run_sandboxed_slice` (every worker-side step
in the box); :func:`host_site` and :func:`box_site` are those two places. Gate 1 (:mod:`.intake`)
runs on the same sites.

The intake names are loaded lazily (PEP 562): :mod:`.intake` imports ``local_factory.oracle``,
which imports :mod:`.flow` and so this package, and the oracle must stay importable first.
``from local_factory.pipeline import run_intake`` works as usual.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from local_factory.pipeline.flow import (
    IMPLEMENTER_ROLE,
    VERIFIER_ROLE,
    SliceError,
    SliceOutcome,
    SliceStatus,
    VerifierReport,
    reconcile_criteria,
)
from local_factory.pipeline.host_slice import host_site, run_host_slice
from local_factory.pipeline.sandboxed_slice import box_site, run_sandboxed_slice

_INTAKE_EXPORTS = (
    "Intake",
    "IntakeError",
    "IntakeOutcome",
    "IntakeStatus",
    "answer",
    "approve",
    "open_questions",
    "questions",
    "receive",
    "run_intake",
)

__all__ = [
    "IMPLEMENTER_ROLE",
    "VERIFIER_ROLE",
    "Intake",
    "IntakeError",
    "IntakeOutcome",
    "IntakeStatus",
    "SliceError",
    "SliceOutcome",
    "SliceStatus",
    "VerifierReport",
    "answer",
    "approve",
    "box_site",
    "host_site",
    "open_questions",
    "questions",
    "receive",
    "reconcile_criteria",
    "run_host_slice",
    "run_intake",
    "run_sandboxed_slice",
]


if TYPE_CHECKING:
    from local_factory.pipeline.intake import (
        Intake,
        IntakeError,
        IntakeOutcome,
        IntakeStatus,
        answer,
        approve,
        open_questions,
        questions,
        receive,
        run_intake,
    )
else:
    # Hidden from mypy (it skips this branch), so mypy knows exactly the names imported above and
    # rejects any other `from local_factory.pipeline import ...`.

    def __getattr__(name: str) -> object:
        if name not in _INTAKE_EXPORTS:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
        value = getattr(importlib.import_module(f"{__name__}.intake"), name)
        globals()[name] = value  # resolve each name once
        return value

    def __dir__() -> list[str]:
        return sorted({*globals(), *__all__})
