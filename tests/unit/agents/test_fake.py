"""Unit tests for FakeBackend script lookup, execution order, commands, halts and failures."""

from __future__ import annotations

import asyncio
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from local_factory.agents.backend import (
    AgentBackend,
    AgentResult,
    AgentTask,
    Capability,
    RoleSpec,
    WorkerHandle,
)
from local_factory.agents.fake import (
    DEFAULT_USAGE,
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
from local_factory.models import Halt, HaltReason, Tier, Usage

from .conftest import ROLE, TASK, run, run_script

PY = sys.executable


def py(code: str) -> RunCommand:
    return RunCommand((PY, "-c", code))


# --------------------------------------------------------------------------- identity


def test_identity_and_protocol() -> None:
    backend = FakeBackend({})
    assert isinstance(backend, AgentBackend)
    assert backend.name == "fake"
    assert backend.capabilities == frozenset({Capability.STRUCTURED_OUTPUT})
    assert backend.calls == []


# --------------------------------------------------------------------------- construction


@pytest.mark.parametrize("bogus", [object(), "WriteFile", ("a.txt", "x"), None])
def test_non_action_items_are_rejected_at_construction(bogus: Any) -> None:
    with pytest.raises(TypeError, match="not a FakeBackend action"):
        FakeBackend({"implementer": [WriteFile("a", "x"), bogus]})


def test_a_string_is_not_a_script() -> None:
    script: Any = "WriteFile"
    with pytest.raises(TypeError, match="'implementer'"):
        FakeBackend({"implementer": script})


def test_scripts_are_snapshotted_at_construction(worker: WorkerHandle) -> None:
    script: list[Action] = [WriteFile("a.txt", "x")]
    backend = FakeBackend({"implementer": script})
    script.append(Fail("added later"))
    assert run(backend, worker).status == "ok"


# --------------------------------------------------------------------------- lookup


def test_empty_script_succeeds_with_nothing_emitted(worker: WorkerHandle) -> None:
    result = run_script(worker)
    assert result == AgentResult(status="ok", usage=DEFAULT_USAGE)
    assert list(worker.work_dir.iterdir()) == []


def test_default_usage() -> None:
    assert Usage(input_tokens=100, output_tokens=50) == DEFAULT_USAGE


def test_task_without_id_uses_the_role_script(worker: WorkerHandle) -> None:
    backend = FakeBackend({"implementer": [WriteFile("who.txt", "role")]})
    untracked = AgentTask(ticket_id="T-0001", instructions="x")
    assert run(backend, worker, task=untracked).status == "ok"
    assert (worker.work_dir / "who.txt").read_text() == "role"


def test_task_script_is_used_even_when_the_role_has_none(worker: WorkerHandle) -> None:
    backend = FakeBackend({"T-0001/task-1": [WriteFile("who.txt", "task")]})
    reviewer = RoleSpec(name="reviewer", prompt="p", tier=Tier.FAST)
    assert run(backend, worker, role=reviewer).status == "ok"
    assert (worker.work_dir / "who.txt").read_text() == "task"


def test_missing_script_error_names_task_and_role(worker: WorkerHandle) -> None:
    result = run(FakeBackend({"T-0001/task-2": [], "planner": []}), worker)
    assert result.status == "error"
    assert result.error == "no script for task 'T-0001/task-1' or role 'implementer'"
    assert result.usage == DEFAULT_USAGE


def test_every_call_is_recorded_in_order(worker: WorkerHandle) -> None:
    backend = FakeBackend({"implementer": []})
    other_role = RoleSpec(name="nobody", prompt="p", tier=Tier.APEX)
    other_task = AgentTask(ticket_id="T-0002", instructions="y")
    run(backend, worker)
    run(backend, worker, role=other_role, task=other_task)
    assert backend.calls == [(ROLE, TASK), (other_role, other_task)]


def test_a_script_can_be_run_repeatedly(worker: WorkerHandle) -> None:
    backend = FakeBackend({"implementer": [WriteFile("n.txt", "1")]})
    assert [run(backend, worker).status for _ in range(3)] == ["ok", "ok", "ok"]
    assert len(backend.calls) == 3


# --------------------------------------------------------------------------- worker directory


def test_missing_worker_directory_is_an_error_and_is_not_created(tmp_path: Path) -> None:
    missing = WorkerHandle(work_dir=tmp_path / "missing")
    result = run_script(missing, WriteFile("a.txt", "x"))
    assert result.status == "error"
    assert result.error is not None
    assert "does not exist" in result.error
    assert not missing.work_dir.exists()


def test_worker_path_that_is_a_file_is_an_error(tmp_path: Path) -> None:
    file = tmp_path / "file"
    file.write_text("")
    assert run_script(WorkerHandle(work_dir=file), WriteFile("a.txt", "x")).status == "error"


def test_relative_worker_directory_is_resolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "rel").mkdir()
    monkeypatch.chdir(tmp_path)
    result = run_script(WorkerHandle(work_dir=Path("rel")), WriteFile("a.txt", "x"))
    assert result.status == "ok", result.error
    assert (tmp_path / "rel" / "a.txt").read_text() == "x"


def test_symlinked_worker_directory_is_followed(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    (tmp_path / "alias").symlink_to(real)
    result = run_script(WorkerHandle(work_dir=tmp_path / "alias"), WriteFile("a.txt", "x"))
    assert result.status == "ok", result.error
    assert (real / "a.txt").read_text() == "x"


# --------------------------------------------------------------------------- ordering & stopping


def test_actions_see_the_effects_of_earlier_actions(worker: WorkerHandle) -> None:
    result = run_script(
        worker,
        WriteFile("in.txt", "abc"),
        py("open('out.txt', 'w').write(open('in.txt').read().upper())"),
        DeleteFile("in.txt"),
    )
    assert result.status == "ok", result.error
    assert (worker.work_dir / "out.txt").read_text() == "ABC"
    assert not (worker.work_dir / "in.txt").exists()


def test_failed_action_stops_the_script_but_keeps_earlier_effects(worker: WorkerHandle) -> None:
    result = run_script(
        worker, WriteFile("before.txt", "x"), DeleteFile("missing.txt"), WriteFile("after.txt", "y")
    )
    assert result.status == "error"
    assert result.error == "action 2 (DeleteFile 'missing.txt'): no such file"
    assert (worker.work_dir / "before.txt").exists()
    assert not (worker.work_dir / "after.txt").exists()


def test_action_errors_take_precedence_over_output_validation(worker: WorkerHandle) -> None:
    result = run_script(worker, EmitOutput({"x": 1}), Fail("gave up"))
    assert (result.status, result.error, result.structured_output) == ("error", "gave up", None)


# --------------------------------------------------------------------------- RunCommand


def test_command_runs_in_the_worker_directory(worker: WorkerHandle) -> None:
    result = run_script(worker, py("import os; open('cwd.txt', 'w').write(os.getcwd())"))
    assert result.status == "ok", result.error
    assert Path((worker.work_dir / "cwd.txt").read_text()) == worker.work_dir.resolve()


def test_command_arguments_are_not_interpreted_by_a_shell(worker: WorkerHandle) -> None:
    args = ("$HOME", ";", "rm -rf .", "`id`", "*")
    code = "import sys; open('argv.txt', 'w').write(repr(sys.argv[1:]))"
    result = run_script(worker, RunCommand((PY, "-c", code, *args)))
    assert result.status == "ok", result.error
    assert (worker.work_dir / "argv.txt").read_text() == repr(list(args))


def test_command_stdin_is_empty(worker: WorkerHandle) -> None:
    result = run_script(worker, py("import sys; raise SystemExit(len(sys.stdin.read()))"))
    assert result.status == "ok", result.error


def test_command_output_is_captured_not_leaked(
    worker: WorkerHandle, capfd: pytest.CaptureFixture[str]
) -> None:
    result = run_script(worker, py("import sys; print('out'); print('err', file=sys.stderr)"))
    assert result.status == "ok", result.error
    captured = capfd.readouterr()
    assert (captured.out, captured.err) == ("", "")


def test_failing_command_reports_status_and_stderr_tail(worker: WorkerHandle) -> None:
    result = run_script(
        worker, py("import sys; print('noise'); sys.stderr.write('boom\\n'); raise SystemExit(3)")
    )
    assert result.status == "error"
    assert result.error is not None
    assert result.error.startswith("action 1 (RunCommand ")
    assert result.error.endswith("): exit status 3: boom")


def test_failing_command_falls_back_to_stdout(worker: WorkerHandle) -> None:
    result = run_script(worker, py("print('only stdout'); raise SystemExit(2)"))
    assert result.error is not None
    assert result.error.endswith("exit status 2: only stdout")


def test_silent_failing_command_reports_status_only(worker: WorkerHandle) -> None:
    result = run_script(worker, py("raise SystemExit(5)"))
    assert result.error is not None
    assert result.error.endswith("): exit status 5")


def test_long_command_output_is_truncated_to_its_tail(worker: WorkerHandle) -> None:
    noisy = (
        f"import sys; sys.stderr.write('x' * {ERROR_TAIL_CHARS * 3} + 'END'); raise SystemExit(1)"
    )
    result = run_script(worker, py(noisy))
    assert result.error is not None
    assert result.error.endswith("xEND")
    assert len(result.error) < ERROR_TAIL_CHARS + 200


def test_missing_binary_is_an_error(worker: WorkerHandle) -> None:
    result = run_script(
        worker, RunCommand(("definitely-not-a-real-binary-xyz", "--flag")), WriteFile("a", "x")
    )
    assert result.status == "error"
    assert result.error is not None
    assert "FileNotFoundError" in result.error
    assert "definitely-not-a-real-binary-xyz --flag" in result.error
    assert not (worker.work_dir / "a").exists()


def test_non_executable_command_is_an_error(worker: WorkerHandle) -> None:
    script = worker.work_dir / "script.sh"
    script.write_text("#!/bin/sh\nexit 0\n")
    script.chmod(0o644)
    result = run_script(worker, RunCommand((str(script),)))
    assert result.status == "error"
    assert result.error is not None
    assert "PermissionError" in result.error


def test_empty_argv_is_an_error(worker: WorkerHandle) -> None:
    result = run_script(worker, RunCommand(()))
    assert result.error == "action 1 (RunCommand ): empty argv"


def test_concurrent_runs_do_not_interfere(tmp_path: Path) -> None:
    workers = [WorkerHandle(work_dir=tmp_path / f"w{i}") for i in range(3)]
    for handle in workers:
        handle.work_dir.mkdir()
    backend = FakeBackend({"implementer": [py("open('done.txt', 'w').write('ok')")]})

    async def run_all() -> list[AgentResult]:
        return list(await asyncio.gather(*(backend.run(ROLE, TASK, w) for w in workers)))

    results = asyncio.run(run_all())
    assert [r.status for r in results] == ["ok", "ok", "ok"]
    assert all((w.work_dir / "done.txt").read_text() == "ok" for w in workers)
    assert len(backend.calls) == 3


# --------------------------------------------------------------------------- RaiseHalt / Fail


@pytest.mark.parametrize("reason", list(HaltReason))
def test_halt_reports_the_reason(worker: WorkerHandle, reason: HaltReason) -> None:
    result = run_script(worker, RaiseHalt(reason))
    assert result == AgentResult(status="halted", halt=Halt(reason=reason), usage=DEFAULT_USAGE)


def test_halt_discards_earlier_output_and_skips_later_actions(worker: WorkerHandle) -> None:
    resume = datetime(2026, 1, 1, 5, tzinfo=UTC)
    result = run_script(
        worker,
        EmitOutput({"x": 1}),
        RaiseHalt(HaltReason.RATE_LIMIT, resume_at=resume),
        WriteFile("after.txt", "x"),
    )
    assert result.halt == Halt(reason=HaltReason.RATE_LIMIT, resume_at=resume)
    assert result.structured_output is None
    assert not (worker.work_dir / "after.txt").exists()


def test_fail_without_a_message_still_reports_an_error(worker: WorkerHandle) -> None:
    result = run_script(worker, Fail(""))
    assert (result.status, result.error) == ("error", "scripted failure (no message)")


# --------------------------------------------------------------------------- usage


@pytest.mark.parametrize(
    "script",
    [
        [],
        [Fail("x")],
        [RaiseHalt(HaltReason.BUDGET)],
        [DeleteFile("missing")],
        [EmitOutput({"a": 1})],
    ],
)
def test_every_result_carries_the_configured_usage(
    worker: WorkerHandle, script: list[Action]
) -> None:
    usage = Usage(input_tokens=1, output_tokens=2, cache_write_tokens=3, cost_usd=0.25)
    assert run(FakeBackend({"implementer": script}, usage=usage), worker).usage == usage
    no_script = FakeBackend({}, usage=usage)
    assert run(no_script, worker).usage == usage
