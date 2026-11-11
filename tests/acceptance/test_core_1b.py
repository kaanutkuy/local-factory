"""Acceptance: Phase 1b core changes — sandbox config, structured transitions, the bundle cap, the
tool executor contract and FakeBackend acting only through it (docs/specs/phase-1b.md). Protected.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Sequence
from pathlib import Path

import pytest
from pydantic import ValidationError

from local_factory.agents.backend import AgentTask, RoleSpec, WorkerHandle
from local_factory.agents.fake import DeleteFile, FakeBackend, RunCommand, WriteFile
from local_factory.agents.local import LocalExecutor, delete_file, read_file, write_file
from local_factory.agents.tools import CommandResult, ToolError, ToolExecutor
from local_factory.config import (
    ConfigError,
    FactoryConfig,
    SandboxConfig,
    load_factory_config,
)
from local_factory.gitops.bare import BareRepo
from local_factory.gitops.bundles import BundleLimits, BundleRejected, import_task_bundle
from local_factory.ledger import Ledger
from local_factory.models import RiskTier, TicketStatus, TicketType, Tier

IMPLEMENTER = RoleSpec(name="implementer", prompt="build it", tier=Tier.STANDARD)
TASK = AgentTask(ticket_id="T-0001", task_id="T-0001/task-1", instructions="do the thing")


# --------------------------------------------------------------------------- sandbox config


def test_sandbox_config_defaults_never_root_and_bounded() -> None:
    config = FactoryConfig(home=Path("/nonexistent")).sandbox
    assert config == SandboxConfig()
    assert config.image == "local-factory/sandbox:v1"
    assert config.uid >= 1 and config.gid >= 1
    assert config.max_bundle_bytes == 100_000_000
    assert config.cache_mounts == ()


@pytest.mark.parametrize(
    "bad",
    [
        {"uid": 0},
        {"gid": 0},
        {"memory_mb": 8},
        {"cpus": 0},
        {"cache_mounts": [{"source": "relative/dir", "target": "/cache"}]},
        {"cache_mounts": [{"source": "/srv/cache", "target": "cache"}]},
        {"cache_mounts": [{"source": "/srv/cache", "target": "/work/cache"}]},
        {"cache_mounts": [{"source": "/srv/cache", "target": "/opt/local-factory"}]},
        {"cache_mounts": [{"source": "/srv/a,b", "target": "/cache"}]},
        {"cache_mounts": [{"source": "/srv/cache", "target": "/cache,x"}]},
        {
            "cache_mounts": [
                {"source": "/srv/a", "target": "/cache"},
                {"source": "/srv/b", "target": "/cache"},
            ]
        },
        {"gate_env": {"not an identifier": "x"}},
        {"network": "bridge"},  # there is no setting that grants network access
    ],
)
def test_sandbox_config_rejects_unsafe_values(bad: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        SandboxConfig.model_validate(bad)


def test_sandbox_section_in_factory_config(tmp_path: Path) -> None:
    (tmp_path / "config.toml").write_text(
        "[sandbox]\nmemory_mb = 512\n"
        'cache_mounts = [{source = "/srv/uv", target = "/cache/uv"}]\n'
        'gate_env = {UV_CACHE_DIR = "/cache/uv"}\n'
    )
    config = load_factory_config(tmp_path).sandbox
    assert config.memory_mb == 512
    assert [(str(m.source), m.target) for m in config.cache_mounts] == [("/srv/uv", "/cache/uv")]
    assert config.gate_env == {"UV_CACHE_DIR": "/cache/uv"}


def test_unknown_sandbox_key_is_an_error(tmp_path: Path) -> None:
    (tmp_path / "config.toml").write_text("[sandbox]\nprivileged = true\n")
    with pytest.raises(ConfigError, match="privileged"):
        load_factory_config(tmp_path)


# --------------------------------------------------------------------------- ledger


def test_transition_data_is_structured_and_optional(tmp_path: Path) -> None:
    with Ledger(tmp_path / "ledger.db") as ledger:
        ticket = ledger.create_ticket(
            title="t", type=TicketType.FEATURE, risk=RiskTier.R1, request="r", actor="human"
        )
        ledger.transition(ticket.id, TicketStatus.TRIAGED, actor="triage", reason="plain")
        ledger.transition(
            ticket.id,
            TicketStatus.CONTRACT_PROPOSED,
            actor="spec",
            reason="with detail",
            data={"conflicts": ["greet.py"], "n": 1},
        )
        events = [e for e in ledger.events(ticket.id) if e.kind == "transition"]
    assert events[0].data == {"reason": "plain"}
    assert events[1].data == {
        "reason": "with detail",
        "detail": {"conflicts": ["greet.py"], "n": 1},
    }


# --------------------------------------------------------------------------- bundle cap


def test_bundle_limits_have_a_size_cap() -> None:
    assert BundleLimits().max_bundle_bytes == 100_000_000


def test_oversized_bundle_is_rejected_before_anything_else(tmp_path: Path, seed_repo: Path) -> None:
    repo = BareRepo.init(tmp_path / "repo.git")
    repo.fetch_from(seed_repo, ["refs/heads/main:refs/heads/main"])
    main = repo.rev_parse("refs/heads/main")
    garbage = tmp_path / "task.bundle"
    garbage.write_bytes(b"x" * 2048)  # not a bundle at all: size is checked first
    with pytest.raises(BundleRejected) as rejected:
        import_task_bundle(
            repo,
            garbage,
            base_sha=main,
            dest_ref="refs/factory/tasks/T-0001/task-1",
            limits=BundleLimits(max_bundle_bytes=1024),
        )
    assert rejected.value.code == "bundle_too_large"
    assert not repo.ref_exists("refs/factory/tasks/T-0001/task-1")
    with pytest.raises(BundleRejected) as small_garbage:
        import_task_bundle(
            repo,
            garbage,
            base_sha=main,
            dest_ref="refs/factory/tasks/T-0001/task-1",
            limits=BundleLimits(max_bundle_bytes=4096),
        )
    assert small_garbage.value.code == "invalid_bundle"


# --------------------------------------------------------------------------- local tools


def test_sync_helpers_confine_to_the_root(tmp_path: Path) -> None:
    root = tmp_path / "work"
    root.mkdir()
    write_file(root, "pkg/mod.py", "x = 1\n")
    assert read_file(root, "pkg/mod.py", max_bytes=100) == b"x = 1\n"
    with pytest.raises(ToolError):
        read_file(root, "pkg/mod.py", max_bytes=3)
    (tmp_path / "outside.txt").write_text("secret")
    (root / "link").symlink_to(tmp_path / "outside.txt")
    for bad in ("/etc/passwd", "../outside.txt", "link", "pkg/../../outside.txt", "a\x00b"):
        with pytest.raises(ToolError):
            read_file(root, bad, max_bytes=100)
        with pytest.raises(ToolError):
            write_file(root, bad, "pwned")
    assert (tmp_path / "outside.txt").read_text() == "secret"
    delete_file(root, "pkg/mod.py")
    assert not (root / "pkg" / "mod.py").exists()
    assert (tmp_path / "outside.txt").exists()
    with pytest.raises(ToolError):
        delete_file(root, "pkg")  # a directory
    with pytest.raises(ToolError):
        delete_file(root, "missing.txt")


def test_local_executor_is_a_tool_executor(tmp_path: Path) -> None:
    executor = LocalExecutor(tmp_path)
    assert isinstance(executor, ToolExecutor)

    async def scenario() -> tuple[str, CommandResult, CommandResult]:
        await executor.write_file("a.txt", "hello\n")
        text = await executor.read_file("a.txt")
        ok = await executor.run_command(["sh", "-c", "cat a.txt; echo oops >&2"], timeout_s=30)
        failing = await executor.run_command(["sh", "-c", "exit 3"], timeout_s=30)
        await executor.delete_file("a.txt")
        return text, ok, failing

    text, ok, failing = asyncio.run(scenario())
    assert text == "hello\n"
    assert ok.ok and ok.exit_code == 0 and not ok.timed_out
    assert "hello" in ok.output and "oops" in ok.output  # stdout and stderr interleaved
    assert (failing.exit_code, failing.ok) == (3, False)
    assert not (tmp_path / "a.txt").exists()


def test_local_executor_kills_on_timeout_and_bounds_output(tmp_path: Path) -> None:
    executor = LocalExecutor(tmp_path, max_output_bytes=1000)

    async def scenario() -> tuple[CommandResult, CommandResult]:
        slow = await executor.run_command(["sleep", "30"], timeout_s=0.5)
        loud = await executor.run_command(
            ["python3", "-c", "print('a' * 50000 + 'END')"], timeout_s=30
        )
        return slow, loud

    slow, loud = asyncio.run(scenario())
    assert slow.timed_out and slow.exit_code is None and not slow.ok
    assert len(loud.output) <= 1000 and loud.output.rstrip().endswith("END")  # the tail is kept


def test_local_executor_does_not_leak_the_host_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LF_CANARY_SECRET", "canary-6b1f")
    result = asyncio.run(LocalExecutor(tmp_path).run_command(["env"], timeout_s=30))
    assert result.ok
    assert "canary-6b1f" not in result.output


# ------------------------------------------------------------------ FakeBackend uses its hands


class RecordingExecutor:
    """A ToolExecutor that records calls and touches nothing."""

    def __init__(self, result: CommandResult | None = None, error: str | None = None) -> None:
        self.calls: list[tuple[str, object]] = []
        self.result = result or CommandResult(exit_code=0, output="")
        self.error = error

    async def write_file(self, path: str, content: str) -> None:
        self.calls.append(("write", (path, content)))
        if self.error:
            raise ToolError(self.error)

    async def read_file(self, path: str) -> str:
        self.calls.append(("read", path))
        return ""

    async def delete_file(self, path: str) -> None:
        self.calls.append(("delete", path))

    async def run_command(self, argv: Sequence[str], *, timeout_s: float) -> CommandResult:
        self.calls.append(("run", (tuple(argv), timeout_s)))
        return self.result


def test_worker_handle_tools_default_to_a_local_executor(tmp_path: Path) -> None:
    assert isinstance(WorkerHandle(work_dir=tmp_path).tools(), LocalExecutor)
    recorder = RecordingExecutor()
    assert WorkerHandle(work_dir=tmp_path, executor=recorder).tools() is recorder


def test_fake_backend_acts_only_through_the_executor(tmp_path: Path) -> None:
    recorder = RecordingExecutor()
    in_box = Path("/work")  # a path inside the box; it need not exist on the host
    assert not in_box.exists() or in_box.is_dir()
    backend = FakeBackend(
        {
            "implementer": [
                WriteFile("app.py", "print(1)\n"),
                RunCommand(("python3", "app.py")),
                RunCommand(("make",), timeout_s=7.0),
                DeleteFile("old.py"),
            ]
        },
        command_timeout_s=42.0,
    )
    result = asyncio.run(
        backend.run(IMPLEMENTER, TASK, WorkerHandle(work_dir=in_box, executor=recorder))
    )
    assert result.status == "ok", result.error
    assert recorder.calls == [
        ("write", ("app.py", "print(1)\n")),
        ("run", (("python3", "app.py"), 42.0)),
        ("run", (("make",), 7.0)),
        ("delete", "old.py"),
    ]
    assert not (tmp_path / "app.py").exists()


def test_fake_backend_reports_command_timeouts_and_tool_errors(tmp_path: Path) -> None:
    script = {"implementer": [RunCommand(("sleep", "100"))]}
    timed_out = RecordingExecutor(CommandResult(exit_code=None, output="", timed_out=True))
    result = asyncio.run(
        FakeBackend(script).run(
            IMPLEMENTER, TASK, WorkerHandle(work_dir=tmp_path, executor=timed_out)
        )
    )
    assert result.status == "error"
    assert result.error is not None and "timed out" in result.error

    refused = RecordingExecutor(error="path escapes the worker directory")
    result = asyncio.run(
        FakeBackend({"implementer": [WriteFile("../x", "y")]}).run(
            IMPLEMENTER, TASK, WorkerHandle(work_dir=tmp_path, executor=refused)
        )
    )
    assert result.status == "error"
    assert result.error is not None
    assert result.error.startswith("action 1 (")
    assert "path escapes the worker directory" in result.error


def test_fake_backend_default_command_timeout() -> None:
    recorder = RecordingExecutor()
    asyncio.run(
        FakeBackend({"implementer": [RunCommand(("true",))]}).run(
            IMPLEMENTER, TASK, WorkerHandle(work_dir=Path(os.devnull), executor=recorder)
        )
    )
    assert recorder.calls == [("run", (("true",), 300.0))]
