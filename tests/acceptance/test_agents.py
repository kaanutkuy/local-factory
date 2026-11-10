"""Acceptance: backend contract and FakeBackend (docs/specs/phase-1a.md § agents). Protected."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from local_factory.agents.backend import (
    AgentBackend,
    AgentResult,
    AgentTask,
    Capability,
    RoleSpec,
    WorkerHandle,
)
from local_factory.agents.fake import (
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

from .conftest import T0

IMPLEMENTER = RoleSpec(name="implementer", prompt="build it", tier=Tier.STANDARD)
TASK = AgentTask(ticket_id="T-0001", task_id="T-0001/task-1", instructions="do the thing")


class Report(BaseModel):
    summary: str
    files_changed: int


def run(
    backend: FakeBackend, worker: WorkerHandle, role: RoleSpec = IMPLEMENTER, task: AgentTask = TASK
):
    return asyncio.run(backend.run(role, task, worker))


@pytest.fixture
def worker(tmp_path: Path) -> WorkerHandle:
    work = tmp_path / "work"
    work.mkdir()
    return WorkerHandle(work_dir=work)


def test_fake_backend_satisfies_the_protocol() -> None:
    backend: AgentBackend = FakeBackend({})
    assert backend.name == "fake"
    assert Capability.STRUCTURED_OUTPUT in backend.capabilities


def test_actions_run_in_order_inside_the_worker(worker: WorkerHandle) -> None:
    backend = FakeBackend(
        {
            "T-0001/task-1": [
                WriteFile("src/app.py", "print('hi')\n"),
                WriteFile("old.txt", "x"),
                DeleteFile("old.txt"),
                RunCommand((sys.executable, "-c", "open('ran.txt', 'w').write('ok')")),
            ]
        },
        usage=Usage(input_tokens=7, output_tokens=3, cost_usd=0.01),
    )
    result = run(backend, worker)
    assert result.status == "ok", result.error
    assert (worker.work_dir / "src" / "app.py").read_text() == "print('hi')\n"
    assert not (worker.work_dir / "old.txt").exists()
    assert (worker.work_dir / "ran.txt").read_text() == "ok"
    assert result.usage == Usage(input_tokens=7, output_tokens=3, cost_usd=0.01)
    assert backend.calls == [(IMPLEMENTER, TASK)]


def test_task_script_takes_precedence_over_role_script(worker: WorkerHandle) -> None:
    backend = FakeBackend(
        {
            "implementer": [WriteFile("who.txt", "role")],
            "T-0001/task-1": [WriteFile("who.txt", "task")],
        }
    )
    run(backend, worker)
    assert (worker.work_dir / "who.txt").read_text() == "task"
    other = AgentTask(ticket_id="T-0001", task_id="T-0001/task-9", instructions="x")
    run(backend, worker, task=other)
    assert (worker.work_dir / "who.txt").read_text() == "role"


def test_missing_script_is_an_error(worker: WorkerHandle) -> None:
    result = run(FakeBackend({}), worker)
    assert result.status == "error"
    assert result.error and "no script" in result.error


@pytest.mark.parametrize("path", ["../escape.txt", "/tmp/escape.txt", "a/../../escape.txt"])
def test_paths_cannot_escape_the_worker(worker: WorkerHandle, path: str) -> None:
    result = run(FakeBackend({"implementer": [WriteFile(path, "x")]}), worker)
    assert result.status == "error"
    assert not (worker.work_dir.parent / "escape.txt").exists()


def test_symlinks_cannot_escape_the_worker(worker: WorkerHandle, tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (worker.work_dir / "link").symlink_to(outside)
    result = run(FakeBackend({"implementer": [WriteFile("link/pwned.txt", "x")]}), worker)
    assert result.status == "error"
    assert not (outside / "pwned.txt").exists()


def test_failing_command_is_an_error(worker: WorkerHandle) -> None:
    script: list[Action] = [
        RunCommand((sys.executable, "-c", "raise SystemExit(4)")),
        WriteFile("after.txt", "x"),
    ]
    result = run(FakeBackend({"implementer": script}), worker)
    assert result.status == "error"
    assert not (worker.work_dir / "after.txt").exists()


def test_halt_stops_execution(worker: WorkerHandle) -> None:
    resume = T0.replace(hour=5)
    script: list[Action] = [
        RaiseHalt(HaltReason.RATE_LIMIT, resume_at=resume),
        WriteFile("after.txt", "x"),
    ]
    result = run(FakeBackend({"implementer": script}), worker)
    assert result.status == "halted"
    assert result.halt == Halt(reason=HaltReason.RATE_LIMIT, resume_at=resume)
    assert not (worker.work_dir / "after.txt").exists()


def test_fail_stops_execution(worker: WorkerHandle) -> None:
    result = run(FakeBackend({"implementer": [Fail("model refused"), WriteFile("a", "x")]}), worker)
    assert (result.status, result.error) == ("error", "model refused")
    assert not (worker.work_dir / "a").exists()


def test_structured_output_is_validated_against_the_role_schema(worker: WorkerHandle) -> None:
    role = RoleSpec(name="implementer", prompt="p", tier=Tier.STANDARD, output_schema=Report)
    ok = run(
        FakeBackend({"implementer": [EmitOutput({"summary": "done", "files_changed": 2})]}),
        worker,
        role=role,
    )
    assert ok.status == "ok"
    assert ok.structured_output == {"summary": "done", "files_changed": 2}

    bad = run(FakeBackend({"implementer": [EmitOutput({"summary": "done"})]}), worker, role=role)
    assert bad.status == "error"

    missing = run(FakeBackend({"implementer": [WriteFile("a", "x")]}), worker, role=role)
    assert missing.status == "error"

    untyped = run(FakeBackend({"implementer": [EmitOutput({"anything": [1, 2]})]}), worker)
    assert untyped.structured_output == {"anything": [1, 2]}


def test_agent_result_invariants() -> None:
    with pytest.raises(ValidationError):
        AgentResult(status="halted")
    with pytest.raises(ValidationError):
        AgentResult(status="ok", halt=Halt(reason=HaltReason.BUDGET))
    with pytest.raises(ValidationError):
        AgentResult(status="error")
    assert AgentResult(status="error", error="boom").error == "boom"
