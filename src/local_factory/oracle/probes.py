"""Running a black-box probe (docs/specs/phase-2b.md § oracle).

A probe is data: an ``argv`` and a ``stdin`` that run, without a shell, at the root of an app's
tree, and an expectation. Only the argv and the stdin ever go into a box; the expectation stays
on the host and is compared here, against the output that comes back. A verdict's reason says
*that* something differed, never *what* was expected, so feedback cannot leak the answer.
"""

from __future__ import annotations

from local_factory.agents import WorkerHandle
from local_factory.agents.tools import CommandResult
from local_factory.models import ProbeDraft

__all__ = ["PROBE_SCRIPT", "PROBE_TIMEOUT_S", "judge_probe", "probe_argv", "run_probe"]

PROBE_TIMEOUT_S = 30.0
# $0 = "probe", $1 = the stdin text, the rest = the argv. The program's stdin is the text, its
# stderr is discarded, so the command's output is the program's stdout alone.
PROBE_SCRIPT = 'printf %s "$1" | { shift; exec "$@"; } 2>/dev/null'


def probe_argv(probe: ProbeDraft) -> list[str]:
    """The command that runs ``probe``: a fixed wrapper, with the probe's data as arguments."""
    return ["sh", "-c", PROBE_SCRIPT, "probe", probe.stdin, *probe.argv]


async def run_probe(handle: WorkerHandle, probe: ProbeDraft) -> CommandResult:
    """Run ``probe`` in ``handle``'s tree (its root is the cwd); only argv and stdin go in."""
    return await handle.tools().run_command(probe_argv(probe), timeout_s=PROBE_TIMEOUT_S)


def judge_probe(probe: ProbeDraft, result: CommandResult) -> tuple[bool, str]:
    """``(passed, reason)``: does ``result`` meet the probe's expectation?

    Trailing newlines are ignored on both sides. The reason names the kinds of mismatch only,
    never an expected value.
    """
    if result.timed_out:
        return False, f"timed out after {PROBE_TIMEOUT_S:g}s"
    expect = probe.expect
    actual = result.output.rstrip("\n")
    mismatches: list[str] = []
    if expect.exit_code is not None and result.exit_code != expect.exit_code:
        mismatches.append(f"exit code {result.exit_code} differs")
    if expect.stdout is not None and actual != expect.stdout.rstrip("\n"):
        mismatches.append("stdout differs")
    if any(fragment.rstrip("\n") not in actual for fragment in expect.stdout_contains):
        mismatches.append("stdout lacks an expected fragment")
    if mismatches:
        return False, "failed on the base: " + "; ".join(mismatches)
    return True, "passed on the base"
