"""SandboxExecutor: one container per tool call, built and interpreted correctly (no Docker)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from local_factory.agents.tools import CommandResult, ToolError, ToolExecutor
from local_factory.config import SandboxConfig
from local_factory.sandbox import SandboxExecutor
from local_factory.sandbox.protocol import LF_ENTRY, MAX_DETAIL
from local_factory.sandbox.runtime import ContainerResult, DockerRuntime, SandboxError
from local_factory.sandbox.spec import BindMount, VolumeMount
from local_factory.sandbox.workspace import TaskWorkspace

from .conftest import FakeRuntime, result


@pytest.fixture
def tools(
    runtime: DockerRuntime, config: SandboxConfig, workspace: TaskWorkspace, factory_src: Path
) -> SandboxExecutor:
    return SandboxExecutor(runtime, config, workspace, factory_src=factory_src)


def test_is_a_tool_executor(tools: SandboxExecutor) -> None:
    assert isinstance(tools, ToolExecutor)


def test_every_call_is_one_box_on_the_workspace(
    tools: SandboxExecutor, fake: FakeRuntime, config: SandboxConfig, factory_src: Path
) -> None:
    fake.push(result(0))
    asyncio.run(tools.delete_file("old.py"))
    (run,) = fake.runs
    spec = run.spec
    assert spec.argv == (LF_ENTRY, "delete", "--root", "/work", "--path", "old.py")
    assert spec.volumes == (VolumeMount("lf-lftest-volume0000", "/work", read_only=False),)
    assert spec.binds == (BindMount(factory_src, "/opt/local-factory/local_factory"),)
    assert spec.binds[0].read_only
    assert spec.tmpfs == ()
    assert (spec.workdir, spec.interactive, spec.env) == ("/work", False, {})
    assert (spec.image, spec.uid, spec.memory_mb) == (config.image, config.uid, config.memory_mb)
    assert run.kwargs["timeout_s"] == config.command_timeout_s
    assert run.kwargs.get("output", "head") == "head"


def test_write_sends_the_content_on_stdin(tools: SandboxExecutor, fake: FakeRuntime) -> None:
    fake.push(result(0))
    asyncio.run(tools.write_file("src/new/mod.py", "VALUE = 'ünïcode'\n"))
    run = fake.last
    assert run.spec.argv == (LF_ENTRY, "write", "--root", "/work", "--path", "src/new/mod.py")
    assert run.spec.interactive
    assert run.kwargs["stdin"] == "VALUE = 'ünïcode'\n".encode()


def test_write_refuses_unencodable_content(tools: SandboxExecutor) -> None:
    with pytest.raises(ToolError, match="UTF-8"):
        asyncio.run(tools.write_file("a.txt", "lone \udc80 surrogate"))


def test_nul_in_a_path_is_refused_before_any_box(tools: SandboxExecutor) -> None:
    with pytest.raises(ToolError, match="NUL"):
        asyncio.run(tools.write_file("a\x00b", "x"))
    with pytest.raises(ToolError, match="NUL"):
        asyncio.run(tools.read_file("a\x00b"))
    with pytest.raises(ToolError, match="NUL"):
        asyncio.run(tools.delete_file("a\x00b"))


def test_read_returns_text_and_is_capped(
    tools: SandboxExecutor, fake: FakeRuntime, config: SandboxConfig
) -> None:
    fake.push(result(0, stdout=b"caf\xc3\xa9 \xff\n"))
    assert asyncio.run(tools.read_file("README.md")) == "café �\n"
    limit = str(config.max_output_bytes)
    assert fake.last.spec.argv[-2:] == ("--max-bytes", limit)
    assert fake.last.kwargs["max_stdout_bytes"] == config.max_output_bytes
    fake.push(result(None, stdout=b"x" * 10, output_exceeded=True))
    with pytest.raises(ToolError, match=f"more than {limit} bytes"):
        asyncio.run(tools.read_file("big.txt"))


def test_a_refusal_carries_the_boxes_reason(tools: SandboxExecutor, fake: FakeRuntime) -> None:
    fake.push(result(2, stderr="noise\nerror: path escapes the worker directory\n"))
    with pytest.raises(ToolError) as refused:
        asyncio.run(tools.read_file("../x"))
    assert str(refused.value) == "path escapes the worker directory"


@pytest.mark.parametrize(
    ("box", "message"),
    [
        (result(1, stderr="Traceback ...\nKeyError: 'x'\n"), "exited with status 1: Traceback"),
        (result(137), "the box exited with status 137"),
        (result(137, oom_killed=True), "ran out of memory"),
        (result(None, timed_out=True), "timed out after 60s"),
    ],
)
def test_other_box_failures_are_tool_errors(
    tools: SandboxExecutor, fake: FakeRuntime, box: ContainerResult, message: str
) -> None:
    fake.push(box)
    with pytest.raises(ToolError, match=message) as failed:
        asyncio.run(tools.delete_file("x"))
    assert len(str(failed.value)) <= MAX_DETAIL


def test_tool_error_messages_are_bounded(tools: SandboxExecutor, fake: FakeRuntime) -> None:
    fake.push(result(5, stderr="e" * 50_000))
    with pytest.raises(ToolError) as failed:
        asyncio.run(tools.delete_file("x"))
    assert len(str(failed.value)) <= MAX_DETAIL


def test_factory_failures_propagate(tools: SandboxExecutor, fake: FakeRuntime) -> None:
    fake.push(SandboxError("docker is not running"))
    with pytest.raises(SandboxError, match="docker is not running"):
        asyncio.run(tools.write_file("a.txt", "x"))
    fake.push(SandboxError("image missing"))
    with pytest.raises(SandboxError):
        asyncio.run(tools.run_command(["true"], timeout_s=5))


# --------------------------------------------------------------------------- run_command


def test_run_command_execs_argv_in_tail_mode(
    tools: SandboxExecutor, fake: FakeRuntime, config: SandboxConfig
) -> None:
    fake.push(result(5, stdout=b"to-out\nto-err\n", stderr="ignored"))
    got = asyncio.run(tools.run_command(["sh", "-c", "exit 5", "--", "-x"], timeout_s=7.5))
    assert got == CommandResult(exit_code=5, output="to-out\nto-err\n", timed_out=False)
    run = fake.last
    assert run.spec.argv == (LF_ENTRY, "exec", "--", "sh", "-c", "exit 5", "--", "-x")
    assert (run.spec.workdir, run.spec.interactive) == ("/work", False)
    assert run.kwargs["timeout_s"] == 7.5
    assert run.kwargs["output"] == "tail"
    assert run.kwargs["max_stdout_bytes"] == config.max_output_bytes + 1


def test_run_command_reports_timeouts_and_oom(tools: SandboxExecutor, fake: FakeRuntime) -> None:
    fake.push(result(None, stdout=b"partial\n", timed_out=True))
    slow = asyncio.run(tools.run_command(("sleep", "60"), timeout_s=1.5))
    assert (slow.exit_code, slow.timed_out, slow.ok) == (None, True, False)
    assert slow.output.startswith("partial\n")
    assert "[local-factory] killed: timed out after 1.5s" in slow.output
    fake.push(result(137, stdout=b"allocating\n", oom_killed=True))
    hog = asyncio.run(tools.run_command(["python3", "-c", "..."], timeout_s=5))
    assert (hog.exit_code, hog.timed_out) == (137, False)
    assert hog.output.endswith("\n[local-factory] killed: out of memory\n")


def test_run_command_output_is_a_bounded_tail(
    tools: SandboxExecutor, fake: FakeRuntime, config: SandboxConfig
) -> None:
    limit = config.max_output_bytes
    # What a tail-mode runtime returns for long output: the last limit + 1 bytes.
    for lead in (b"", b"x"):  # the cut falls inside an "é", or right before one
        data = lead + ("é" * limit).encode() + b"TAIL"
        fake.push(result(0, stdout=data[-(limit + 1) :]))
        loud = asyncio.run(tools.run_command(["yes"], timeout_s=5))
        assert len(loud.output) <= limit
        assert "�" not in loud.output
        assert loud.output.endswith("éTAIL")


@pytest.mark.parametrize(
    ("argv", "timeout_s", "error"),
    [
        ([], 5.0, ToolError),
        (["echo", "a\x00b"], 5.0, ToolError),
        ("ls -la", 5.0, TypeError),
        (["true"], 0.0, ValueError),
        (["true"], float("inf"), ValueError),
        (["true"], float("nan"), ValueError),
    ],
)
def test_run_command_rejects_bad_requests_before_any_box(
    tools: SandboxExecutor, argv: list[str], timeout_s: float, error: type[Exception]
) -> None:
    with pytest.raises(error):
        asyncio.run(tools.run_command(argv, timeout_s=timeout_s))
