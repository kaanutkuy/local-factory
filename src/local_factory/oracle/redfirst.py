"""Red-first: every new test and probe must fail on the code it is about to constrain.

A visible acceptance test that already passes on the base proves nothing about a *change*, and
neither does a probe the app already satisfies (docs/specs/phase-2b.md § oracle). The checks of a
*keep* criterion are the opposite: they guard behaviour the code has today, so they must pass on
the base. This module runs and judges every check the same way (red, green, or neither) and marks
the keep ones; the caller decides with :attr:`RedCheck.ok`. The visible tests run on the oracle
commit (main plus the new test files), one gate each, in one verification. Every probe runs on
main in a verification of its own, so one probe cannot leave anything behind for the next.
"""

from __future__ import annotations

import re
import shlex
from collections.abc import Collection, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from local_factory.gitops import BareRepo
from local_factory.models import GateResult, GateStatus, HoldoutProbe, RedCheck
from local_factory.oracle.probes import judge_probe, run_probe
from local_factory.verification import GateSpec

if TYPE_CHECKING:  # not at runtime: pipeline imports the oracle
    from local_factory.pipeline.flow import Site

__all__ = [
    "NO_TESTS_RAN_EXIT",
    "RED_TEST_TIMEOUT_S",
    "judge_visible",
    "red_first",
    "render_acceptance",
    "split_test_id",
]

RED_TEST_TIMEOUT_S = 120.0
NO_TESTS_RAN_EXIT = 5  # pytest and unittest (3.12+) agree: "no test ran"
_OUTPUT_CHARS = 2_000  # the tail of a run kept in a RedCheck
_REASON_CHARS = 300  # the tail of an unclean run kept in its reason
_PLACEHOLDER = re.compile(r"\{path\}|\{name\}")


def split_test_id(test_id: str) -> tuple[str, str]:
    """``"path::name"`` -> ``(path, name)``, split on the first ``::``."""
    path, separator, name = test_id.partition("::")
    if not separator:
        raise ValueError(f"not a test id (expected path::name): {test_id!r}")
    return path, name


def render_acceptance(template: str, test_id: str) -> str:
    """The acceptance command for one test: ``{path}`` and ``{name}`` become shell-quoted parts.

    ``{path}`` is required, ``{name}`` optional. Only those two placeholders are replaced, in one
    pass over the template, so a path or a name that looks like a placeholder (or like shell) is
    never interpreted: each part is quoted and the replacement text is never scanned again.
    """
    if "{path}" not in template:
        raise ValueError(f"the acceptance command must contain {{path}}: {template!r}")
    path, name = split_test_id(test_id)
    parts = {"{path}": shlex.quote(path), "{name}": shlex.quote(name)}
    return _PLACEHOLDER.sub(lambda match: parts[match.group()], template)


def judge_visible(test_id: str, result: GateResult, *, keep: bool = False) -> RedCheck:
    """Red: the test ran and failed on the base. Green: it ran and passed. Neither: it did not
    run cleanly. ``keep`` only marks the check (a keep check is ``ok`` when green)."""
    tail = result.output_tail[-_OUTPUT_CHARS:]
    red = green = False
    if result.status is GateStatus.FAIL and result.exit_code == NO_TESTS_RAN_EXIT:
        reason = f"no test ran (exit {NO_TESTS_RAN_EXIT})"
    elif result.status is GateStatus.FAIL:
        red, reason = True, f"failed on the base (exit {result.exit_code})"
    elif result.status is GateStatus.PASS:
        green, reason = True, "passed on the base"
    else:
        reason = "did not run cleanly: " + result.output_tail[-_REASON_CHARS:]
    return RedCheck(
        kind="visible", id=test_id, red=red, green=green, keep=keep, reason=reason, output=tail
    )


async def red_first(
    *,
    site: Site,
    repo: BareRepo,
    oracle_sha: str,
    base_sha: str,
    visible: Sequence[str],
    probes: Sequence[HoldoutProbe],
    acceptance: str,
    ticket_dir: Path,
    keep: Collection[str] = (),
) -> tuple[RedCheck, ...]:
    """Run every visible test on ``oracle_sha`` and every probe on ``base_sha``.

    Returns one :class:`RedCheck` per test, then per probe, in the order given. ``keep`` holds the
    ids (test ids and probe ids) of the checks that guard keep criteria: those checks come back
    with ``keep=True``; nothing else about running or judging them differs. Raises ``ValueError``
    (before running anything) if ``acceptance`` or a test id is malformed.
    """
    specs = [
        (
            test_id,
            GateSpec(
                name=f"red-first {test_id}",
                command=render_acceptance(acceptance, test_id),
                timeout_s=RED_TEST_TIMEOUT_S,
            ),
        )
        for test_id in visible
    ]
    checks: list[RedCheck] = []
    round_no = 1
    if specs:
        async with site.verification(repo, oracle_sha, ticket_dir, round_no=round_no) as tree:
            for test_id, spec in specs:
                # one gate at a time: the gate runner stops after a failing gate, and here every
                # gate is expected to fail
                [result] = await tree.gates([spec])
                checks.append(judge_visible(test_id, result, keep=test_id in keep))
    for probe in probes:
        round_no += 1
        async with site.verification(repo, base_sha, ticket_dir, round_no=round_no) as tree:
            outcome = await run_probe(tree.handle, probe)
        passed, reason = judge_probe(probe, outcome)
        checks.append(
            RedCheck(
                kind="holdout",
                id=probe.id,
                red=not passed and not outcome.timed_out,
                green=passed,
                keep=probe.id in keep,
                reason=reason,
                output=outcome.output[-_OUTPUT_CHARS:],
            )
        )
    return tuple(checks)
