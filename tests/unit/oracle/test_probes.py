"""Unit tests for probes: what goes into the box, and what a verdict may say."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path

import pytest

from local_factory.agents import WorkerHandle
from local_factory.agents.tools import CommandResult
from local_factory.models import ProbeDraft, ProbeExpectation
from local_factory.oracle import PROBE_SCRIPT, PROBE_TIMEOUT_S, judge_probe, probe_argv, run_probe

EXPECTED = "SEALED-EXPECTED-OUTPUT"


class RecordingExecutor:
    """A ToolExecutor that records the commands it is asked to run."""

    def __init__(self, result: CommandResult) -> None:
        self.result = result
        self.commands: list[tuple[list[str], float]] = []

    async def write_file(self, path: str, content: str) -> None:
        raise AssertionError("a probe writes no files")

    async def read_file(self, path: str) -> str:
        raise AssertionError("a probe reads no files")

    async def delete_file(self, path: str) -> None:
        raise AssertionError("a probe deletes no files")

    async def run_command(self, argv: Sequence[str], *, timeout_s: float) -> CommandResult:
        self.commands.append((list(argv), timeout_s))
        return self.result


def draft(**expect: object) -> ProbeDraft:
    return ProbeDraft(
        criteria=("AC-1",),
        argv=("python3", "greet.py", "--name", "Ada"),
        stdin="line one\nline two",
        expect=ProbeExpectation.model_validate(expect or {"stdout": EXPECTED, "exit_code": 7}),
    )


def test_only_the_argv_and_the_stdin_go_into_the_box(tmp_path: Path) -> None:
    executor = RecordingExecutor(CommandResult(exit_code=0, output="out"))
    handle = WorkerHandle(work_dir=tmp_path, executor=executor)
    result = asyncio.run(
        run_probe(handle, draft(stdout=EXPECTED, stdout_contains=[EXPECTED + "2"]))
    )
    assert result == executor.result
    [(argv, timeout)] = executor.commands
    assert argv == [
        "sh",
        "-c",
        PROBE_SCRIPT,
        "probe",
        "line one\nline two",
        "python3",
        "greet.py",
        "--name",
        "Ada",
    ]
    assert timeout == PROBE_TIMEOUT_S
    assert EXPECTED not in "".join(argv)  # the expectation stays on the host


def test_no_stdin_is_an_empty_argument() -> None:
    bare = ProbeDraft(criteria=("AC-1",), argv=("true",), expect=ProbeExpectation(exit_code=0))
    assert probe_argv(bare) == ["sh", "-c", PROBE_SCRIPT, "probe", "", "true"]


def test_the_script_is_a_fixed_string_that_never_expands_probe_data() -> None:
    hostile = ProbeDraft(
        criteria=("AC-1",),
        argv=("echo", "$(touch pwned)", "; touch pwned"),
        stdin="$(touch pwned) `touch pwned`",
        expect=ProbeExpectation(exit_code=0),
    )
    argv = probe_argv(hostile)
    assert argv[:3] == ["sh", "-c", PROBE_SCRIPT]
    assert "pwned" not in PROBE_SCRIPT


def test_a_probe_with_a_missing_program_exits_127_with_no_output(tmp_path: Path) -> None:
    missing = ProbeDraft(
        criteria=("AC-1",), argv=("no-such-program-xyz",), expect=ProbeExpectation(exit_code=0)
    )
    result = asyncio.run(run_probe(WorkerHandle(work_dir=tmp_path), missing))
    assert (result.exit_code, result.output) == (127, "")


def test_a_probe_that_hangs_times_out(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import local_factory.oracle.probes as probes

    monkeypatch.setattr(probes, "PROBE_TIMEOUT_S", 0.3)
    slow = ProbeDraft(
        criteria=("AC-1",), argv=("sleep", "30"), expect=ProbeExpectation(exit_code=0)
    )
    result = asyncio.run(run_probe(WorkerHandle(work_dir=tmp_path), slow))
    assert result.timed_out and result.exit_code is None
    assert judge_probe(slow, result) == (False, "timed out after 0.3s")


@pytest.mark.parametrize(
    ("output", "passed"),
    [("a\nb\n\n\n", True), ("a\nb", True), ("a\nb \n", False), ("\na\nb", False)],
)
def test_trailing_newlines_are_ignored_on_both_sides(output: str, passed: bool) -> None:
    result = CommandResult(exit_code=0, output=output)
    assert judge_probe(draft(stdout="a\nb\n"), result)[0] is passed


def test_fragments_may_end_in_a_newline_too() -> None:
    result = CommandResult(exit_code=0, output="alpha\nbeta\n")
    assert judge_probe(draft(stdout_contains=["beta\n", "alpha"]), result)[0]


def test_every_given_expectation_must_hold() -> None:
    expect = draft(exit_code=0, stdout="ok", stdout_contains=["o"])
    assert judge_probe(expect, CommandResult(exit_code=0, output="ok")) == (
        True,
        "passed on the base",
    )
    assert judge_probe(expect, CommandResult(exit_code=1, output="ok"))[1] == (
        "failed on the base: exit code 1 differs"
    )


def test_a_killed_program_is_a_mismatch_not_a_crash() -> None:
    result = CommandResult(exit_code=-9, output="")
    assert judge_probe(draft(exit_code=0), result) == (
        False,
        "failed on the base: exit code -9 differs",
    )


def test_no_reason_ever_contains_an_expected_value() -> None:
    expect = draft(exit_code=41, stdout=EXPECTED, stdout_contains=[EXPECTED + "-FRAGMENT"])
    for result in (
        CommandResult(exit_code=0, output="nope"),
        CommandResult(exit_code=41, output=EXPECTED[:-1]),
        CommandResult(exit_code=None, output="", timed_out=True),
    ):
        _, reason = judge_probe(expect, result)
        assert EXPECTED not in reason and "41" not in reason
