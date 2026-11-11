"""Unit tests for FakeBackend acting only through its worker's tools (WorkerHandle.executor)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from local_factory.agents.backend import WorkerHandle
from local_factory.agents.fake import (
    DEFAULT_COMMAND_TIMEOUT_S,
    ERROR_TAIL_CHARS,
    Action,
    DeleteFile,
    EmitOutput,
    Fail,
    FakeBackend,
    RaiseHalt,
    RunCommand,
    WriteFile,
)
from local_factory.agents.tools import CommandResult, ToolError, ToolExecutor
from local_factory.models import HaltReason

from .conftest import run


class Recorder:
    """A ToolExecutor that records every call and touches nothing."""

    def __init__(
        self,
        results: Sequence[CommandResult] = (),
        fail_on: str | None = None,
        error: BaseException | None = None,
    ) -> None:
        self.calls: list[tuple[str, object]] = []
        self._results = list(results)
        self._fail_on = fail_on
        self._error = error or ToolError("refused by the box")

    def _record(self, kind: str, detail: object) -> None:
        self.calls.append((kind, detail))
        if kind == self._fail_on:
            raise self._error

    async def write_file(self, path: str, content: str) -> None:
        self._record("write", (path, content))

    async def read_file(self, path: str) -> str:
        self._record("read", path)
        return ""

    async def delete_file(self, path: str) -> None:
        self._record("delete", path)

    async def run_command(self, argv: Sequence[str], *, timeout_s: float) -> CommandResult:
        self._record("run", (tuple(argv), timeout_s))
        return self._results.pop(0) if self._results else CommandResult(exit_code=0, output="")


IN_BOX = Path("/nonexistent/in-the-box/work")  # never looked at on the host


def boxed(recorder: Recorder) -> WorkerHandle:
    return WorkerHandle(work_dir=IN_BOX, executor=recorder)


def run_boxed(
    recorder: Recorder, *actions: Action, command_timeout_s: float = DEFAULT_COMMAND_TIMEOUT_S
) -> str | None:
    """Run `actions` against `recorder`; the resulting error (None if ok)."""
    backend = FakeBackend({"implementer": actions}, command_timeout_s=command_timeout_s)
    return run(backend, boxed(recorder)).error


def test_recorder_is_a_tool_executor() -> None:
    assert isinstance(Recorder(), ToolExecutor)


def test_every_side_effect_goes_through_the_executor_in_order() -> None:
    recorder = Recorder()
    error = run_boxed(
        recorder,
        WriteFile("a/b.py", "x = 1\n"),
        EmitOutput({"ignored": True}),
        RunCommand(("make", "test")),
        RunCommand(("sleep", "1"), timeout_s=2.5),
        DeleteFile("old.py"),
        command_timeout_s=60.0,
    )
    assert error is None
    assert recorder.calls == [
        ("write", ("a/b.py", "x = 1\n")),
        ("run", (("make", "test"), 60.0)),
        ("run", (("sleep", "1"), 2.5)),
        ("delete", "old.py"),
    ]


def test_with_an_executor_the_host_work_dir_is_never_checked_or_created() -> None:
    assert not IN_BOX.exists()
    recorder = Recorder()
    result = run(FakeBackend({"implementer": [WriteFile("f.txt", "x")]}), boxed(recorder))
    assert result.status == "ok", result.error
    assert recorder.calls == [("write", ("f.txt", "x"))]
    assert not IN_BOX.exists()


def test_scripts_without_side_effects_make_no_calls() -> None:
    recorder = Recorder()
    backend = FakeBackend({"implementer": [EmitOutput({"x": 1}), RaiseHalt(HaltReason.BUDGET)]})
    assert run(backend, boxed(recorder)).status == "halted"
    assert run_boxed(recorder, Fail("nope")) == "nope"
    assert recorder.calls == []


def test_default_command_timeout_is_passed_to_the_executor() -> None:
    recorder = Recorder()
    assert run_boxed(recorder, RunCommand(("true",))) is None
    assert recorder.calls == [("run", (("true",), DEFAULT_COMMAND_TIMEOUT_S))]


@pytest.mark.parametrize(
    ("kind", "action", "description"),
    [
        ("write", WriteFile("../x", "y"), "WriteFile '../x'"),
        ("delete", DeleteFile("gone.txt"), "DeleteFile 'gone.txt'"),
        ("run", RunCommand(("ls", "-l")), "RunCommand ls -l"),
    ],
)
def test_tool_errors_become_action_errors_and_stop_the_script(
    kind: str, action: Action, description: str
) -> None:
    recorder = Recorder(fail_on=kind)
    first: Action = DeleteFile("first.txt") if kind == "write" else WriteFile("first.txt", "1")
    error = run_boxed(recorder, first, action, RunCommand(("never",)), WriteFile("never", "2"))
    assert error == f"action 2 ({description}): refused by the box"
    assert len(recorder.calls) == 2


def test_factory_failures_are_not_blamed_on_the_worker() -> None:
    """Anything but ToolError (e.g. the sandbox runtime failing) propagates out of run()."""

    class RuntimeDownError(Exception):
        pass

    recorder = Recorder(fail_on="write", error=RuntimeDownError("docker daemon is not running"))
    with pytest.raises(RuntimeDownError, match="docker daemon"):
        run_boxed(recorder, WriteFile("a.txt", "x"))


def test_timed_out_command_is_an_error_with_its_output_tail() -> None:
    recorder = Recorder([CommandResult(exit_code=None, output="  progress 99%\n", timed_out=True)])
    error = run_boxed(recorder, RunCommand(("build",), timeout_s=7.0), WriteFile("after", "x"))
    assert error == "action 1 (RunCommand build): timed out after 7s and was killed: progress 99%"
    assert len(recorder.calls) == 1


def test_silent_timeout_reports_the_timeout_only() -> None:
    recorder = Recorder([CommandResult(exit_code=None, output="", timed_out=True)])
    error = run_boxed(recorder, RunCommand(("hang",)), command_timeout_s=1.5)
    assert error == "action 1 (RunCommand hang): timed out after 1.5s and was killed"


def test_failing_command_reports_exit_status_and_bounded_output_tail() -> None:
    output = "x" * (ERROR_TAIL_CHARS * 2) + "END\n"
    recorder = Recorder([CommandResult(exit_code=137, output=output)])
    error = run_boxed(recorder, RunCommand(("oom",)))
    assert error is not None
    assert error.startswith("action 1 (RunCommand oom): exit status 137: xxx")
    assert error.endswith("END")
    assert len(error.split(": ", 2)[2]) == ERROR_TAIL_CHARS


def test_successful_commands_continue_the_script() -> None:
    recorder = Recorder([CommandResult(exit_code=0, output="noise"), CommandResult(1, "bad")])
    error = run_boxed(recorder, RunCommand(("a",)), RunCommand(("b",)), RunCommand(("c",)))
    assert error == "action 2 (RunCommand b): exit status 1: bad"
    assert [call[0] for call in recorder.calls] == ["run", "run"]
