"""Acceptance: the Sandbox facade — workspaces, the in-box ToolExecutor and task export
(docs/specs/phase-1b.md § box). Protected. Needs Docker and the sandbox image.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

import local_factory
from local_factory.agents.tools import ToolError, ToolExecutor
from local_factory.config import SandboxConfig
from local_factory.gitops.bundles import import_task_bundle
from local_factory.sandbox import ExportOutcome, Sandbox
from local_factory.sandbox.runtime import DockerRuntime, SandboxError

from .conftest import BOX_IMAGE, Git, Seeded, docker_objects, small_sandbox_config

pytestmark = [pytest.mark.container, pytest.mark.integration]


def test_the_box_never_runs_as_the_host_user(docker_runtime: DockerRuntime) -> None:
    with pytest.raises(SandboxError):
        Sandbox(docker_runtime, small_sandbox_config(uid=os.getuid()))
    with pytest.raises(SandboxError):
        Sandbox(docker_runtime, small_sandbox_config(gid=os.getgid()))


def test_factory_source_must_be_world_readable(
    docker_runtime: DockerRuntime, box_config: SandboxConfig, tmp_path: Path
) -> None:
    src = tmp_path / "local_factory"
    src.mkdir()
    (src / "__init__.py").write_text("")
    (src / "__init__.py").chmod(0o600)
    with pytest.raises(SandboxError):
        Sandbox(docker_runtime, box_config, factory_src=src)


def test_defaults_and_image(sandbox: Sandbox) -> None:
    assert sandbox.factory_src.resolve() == Path(local_factory.__file__).resolve().parent
    image = sandbox.check()
    assert image.startswith("sha256:")
    assert sandbox.image_id() == image
    assert sandbox.config.image == BOX_IMAGE


def test_materialized_workspace_and_tools(sandbox: Sandbox, seeded: Seeded) -> None:
    async def scenario() -> None:
        with sandbox.workspace() as ws:
            await sandbox.materialize(ws, bundle=seeded.bundle, ref=seeded.ref)
            tools = sandbox.executor(ws)
            assert isinstance(tools, ToolExecutor)
            assert await tools.read_file("README.md") == "# seed\n"
            await tools.write_file("src/new/mod.py", "VALUE = 'ünïcode'\n")
            assert await tools.read_file("src/new/mod.py") == "VALUE = 'ünïcode'\n"
            log = await tools.run_command(["git", "log", "--format=%H"], timeout_s=60)
            assert log.ok and seeded.main in log.output
            branch = await tools.run_command(["git", "branch", "--show-current"], timeout_s=60)
            assert branch.output.strip() == "task"
            await tools.delete_file("README.md")
            gone = await tools.run_command(["test", "-e", "README.md"], timeout_s=60)
            assert gone.exit_code == 1

    asyncio.run(scenario())
    assert docker_objects(sandbox.runtime.instance) == ([], [])


def test_tools_are_confined_to_the_workspace(sandbox: Sandbox, seeded: Seeded) -> None:
    async def scenario() -> None:
        with sandbox.workspace() as ws:
            await sandbox.materialize(ws, bundle=seeded.bundle, ref=seeded.ref)
            tools = sandbox.executor(ws)
            await tools.run_command(["ln", "-s", "/etc/hostname", "link"], timeout_s=60)
            for bad in ("../escape.txt", "/etc/passwd", "link", "src/../../x"):
                with pytest.raises(ToolError):
                    await tools.read_file(bad)
                with pytest.raises(ToolError):
                    await tools.write_file(bad, "x")
            with pytest.raises(ToolError):
                await tools.delete_file("missing.txt")
            with pytest.raises(ToolError):
                await tools.delete_file("src")
            with pytest.raises(ToolError):
                await tools.run_command([], timeout_s=60)

    asyncio.run(scenario())


def test_commands_report_exit_codes_output_and_timeouts(
    sandbox: Sandbox, seeded: Seeded, box_config: SandboxConfig
) -> None:
    async def scenario() -> None:
        with sandbox.workspace() as ws:
            await sandbox.materialize(ws, bundle=seeded.bundle, ref=seeded.ref)
            tools = sandbox.executor(ws)
            mixed = await tools.run_command(
                ["sh", "-c", "echo to-out; echo to-err >&2; exit 5"], timeout_s=60
            )
            assert (mixed.exit_code, mixed.timed_out, mixed.ok) == (5, False, False)
            assert "to-out" in mixed.output and "to-err" in mixed.output
            missing = await tools.run_command(["no-such-tool"], timeout_s=60)
            assert missing.exit_code == 127
            slow = await tools.run_command(["sleep", "60"], timeout_s=1.5)
            assert slow.timed_out and slow.exit_code is None
            loud = await tools.run_command(
                ["python3", "-c", "print('z' * 500000 + 'TAIL')"], timeout_s=60
            )
            assert loud.ok
            assert len(loud.output) <= box_config.max_output_bytes
            assert loud.output.rstrip().endswith("TAIL")
            await tools.run_command(
                ["python3", "-c", "open('big.txt', 'w').write('b' * 200000)"], timeout_s=60
            )
            with pytest.raises(ToolError):
                await tools.read_file("big.txt")  # over max_output_bytes

    asyncio.run(scenario())


def test_a_broken_base_bundle_is_a_factory_error(sandbox: Sandbox, tmp_path: Path) -> None:
    junk = tmp_path / "junk.bundle"
    junk.write_bytes(b"not a bundle\n")

    async def scenario() -> None:
        with sandbox.workspace() as ws, pytest.raises(SandboxError):
            await sandbox.materialize(ws, bundle=junk, ref="refs/heads/factory-base-x")

    asyncio.run(scenario())


# --------------------------------------------------------------------------- export


def export_after(
    sandbox: Sandbox, seeded: Seeded, out: Path, *commands: list[str]
) -> ExportOutcome:
    async def scenario() -> ExportOutcome:
        with sandbox.workspace() as ws:
            await sandbox.materialize(ws, bundle=seeded.bundle, ref=seeded.ref)
            tools = sandbox.executor(ws)
            for argv in commands:
                await tools.run_command(argv, timeout_s=60)
            return await sandbox.export(
                ws, base_sha=seeded.main, message="T-0001/task-1: change", out=out
            )

    return asyncio.run(scenario())


def test_export_produces_a_bundle_the_host_can_import(
    sandbox: Sandbox, seeded: Seeded, tmp_path: Path, git: Git
) -> None:
    out = tmp_path / "out" / "task.bundle"
    out.parent.mkdir()
    outcome = export_after(sandbox, seeded, out, ["sh", "-c", "echo new > feature.txt"])
    assert outcome.status == "exported", outcome
    assert out.stat().st_size > 0
    assert out.stat().st_mode & 0o777 == 0o600
    imported = import_task_bundle(
        seeded.repo, out, base_sha=seeded.main, dest_ref="refs/factory/tasks/T-0001/task-1"
    )
    assert seeded.repo.read_file(imported.head_sha, "feature.txt") == b"new\n"
    assert [c.path for c in imported.changes] == ["feature.txt"]
    subject = git("log", "-1", "--format=%s", imported.head_sha, cwd=seeded.repo.path)
    assert subject == "T-0001/task-1: change"  # the harness commit


def test_export_reports_no_changes(sandbox: Sandbox, seeded: Seeded, tmp_path: Path) -> None:
    out = tmp_path / "task.bundle"
    outcome = export_after(sandbox, seeded, out)
    assert outcome.status == "no_changes"
    assert not out.exists()


def test_a_worker_that_breaks_its_repo_cannot_crash_export(
    sandbox: Sandbox, seeded: Seeded, tmp_path: Path
) -> None:
    out = tmp_path / "task.bundle"
    outcome = export_after(sandbox, seeded, out, ["rm", "-rf", ".git"], ["touch", "x"])
    assert outcome.status == "worker_git_error"
    assert outcome.detail
    assert len(outcome.detail) <= 2000
    assert not out.exists()

    out2 = tmp_path / "task2.bundle"
    corrupt = ["sh", "-c", "rm -rf .git/objects && mkdir .git/objects && echo y > y"]
    assert export_after(sandbox, seeded, out2, corrupt).status == "worker_git_error"
    assert not out2.exists()


def test_export_enforces_the_bundle_cap(
    docker_runtime: DockerRuntime, seeded: Seeded, tmp_path: Path
) -> None:
    tight = Sandbox(docker_runtime, small_sandbox_config(max_bundle_bytes=20_000))
    out = tmp_path / "task.bundle"
    outcome = export_after(
        tight, seeded, out, ["sh", "-c", "head -c 300000 /dev/urandom > noise.bin"]
    )
    assert outcome.status == "bundle_too_large"
    assert not out.exists()


def test_export_never_overwrites(sandbox: Sandbox, seeded: Seeded, tmp_path: Path) -> None:
    out = tmp_path / "task.bundle"
    out.write_bytes(b"precious")
    with pytest.raises(FileExistsError):
        export_after(sandbox, seeded, out, ["touch", "x"])
    assert out.read_bytes() == b"precious"
