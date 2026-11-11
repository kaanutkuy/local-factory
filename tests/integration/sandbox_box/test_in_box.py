"""The box for real: lf-entry in the image, the tools, export and gates (Docker + sandbox image).

These complement the protected acceptance tests with the box's own edge cases. Each test gets a
fresh DockerRuntime whose objects must all be gone when it ends (see conftest).
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path

import pytest

from local_factory.agents.tools import CommandResult, ToolError
from local_factory.gitops.bare import BareRepo
from local_factory.gitops.bundles import import_task_bundle
from local_factory.models import GateStatus
from local_factory.sandbox import ExportOutcome, Sandbox
from local_factory.sandbox.executor import factory_mount
from local_factory.sandbox.protocol import EXIT_USAGE, LF_ENTRY
from local_factory.sandbox.runtime import DockerRuntime, SandboxError
from local_factory.sandbox.spec import ContainerSpec
from local_factory.verification.gates import GateSpec

from .conftest import DEST_REF, Bundle, box_config, last_error, leftovers

pytestmark = [pytest.mark.container, pytest.mark.integration]


def commands(
    sandbox: Sandbox, bundle: Bundle, *argvs: Sequence[str], timeout_s: float = 60.0
) -> list[CommandResult]:
    async def scenario() -> list[CommandResult]:
        with sandbox.workspace() as ws:
            await sandbox.materialize(ws, bundle=bundle.path, ref=bundle.ref)
            tools = sandbox.executor(ws)
            return [await tools.run_command(argv, timeout_s=timeout_s) for argv in argvs]

    return asyncio.run(scenario())


def export_after(
    sandbox: Sandbox, bundle: Bundle, base: str, out: Path, *argvs: Sequence[str]
) -> ExportOutcome:
    async def scenario() -> ExportOutcome:
        with sandbox.workspace() as ws:
            await sandbox.materialize(ws, bundle=bundle.path, ref=bundle.ref)
            tools = sandbox.executor(ws)
            for argv in argvs:
                done = await tools.run_command(argv, timeout_s=60)
                assert done.ok, done.output
            return await sandbox.export(ws, base_sha=base, message="T-0001/task-1: x", out=out)

    return asyncio.run(scenario())


def sh(script: str) -> list[str]:
    return ["sh", "-c", script]


# --------------------------------------------------------------------------- lf-entry itself


def test_lf_entry_runs_isolated_on_python_312(sandbox: Sandbox, bundle: Bundle) -> None:
    version, flags = commands(
        sandbox,
        bundle,
        ["python3", "-c", "import sys; print(sys.version_info[:2])"],
        sh("head -1 /usr/local/bin/lf-entry"),
    )
    assert version.output.strip() == "(3, 12)"
    assert "python3 -I" in flags.output

    spec = ContainerSpec.from_config(
        sandbox.config,
        argv=(LF_ENTRY, "frobnicate"),
        binds=(factory_mount(sandbox.factory_src),),
    )
    usage = asyncio.run(sandbox.runtime.run(spec, timeout_s=60))
    assert usage.exit_code == EXIT_USAGE
    assert "unknown command 'frobnicate'" in last_error(usage.stderr)


def test_commands_see_a_normal_process(sandbox: Sandbox, bundle: Bundle) -> None:
    pipe, where, missing = commands(
        sandbox,
        bundle,
        sh("yes | head -n 1"),  # `yes` must die of SIGPIPE quietly, not see EPIPE
        sh("id -u; pwd; echo $HOME"),
        ["lf-no-such-tool"],
    )
    assert (pipe.exit_code, pipe.output) == (0, "y\n")
    assert where.output.split() == [str(sandbox.config.uid), "/work", "/home/sandbox"]
    assert missing.exit_code == 127
    assert "error: cannot run 'lf-no-such-tool'" in missing.output


def test_exec_of_a_non_executable_file_is_126(sandbox: Sandbox, bundle: Bundle) -> None:
    async def scenario() -> CommandResult:
        with sandbox.workspace() as ws:
            await sandbox.materialize(ws, bundle=bundle.path, ref=bundle.ref)
            tools = sandbox.executor(ws)
            await tools.write_file("script.sh", "#!/bin/sh\necho hi\n")
            return await tools.run_command(["./script.sh"], timeout_s=60)

    result = asyncio.run(scenario())
    assert result.exit_code == 126
    assert "error: cannot run './script.sh'" in result.output


# --------------------------------------------------------------------------- tools


def test_file_tools_round_trip_and_refuse(sandbox: Sandbox, bundle: Bundle) -> None:
    big = "line ünïcode\n" * 20_000  # ~300 kB on stdin: more than one pipe buffer

    async def scenario() -> None:
        with sandbox.workspace() as ws:
            await sandbox.materialize(ws, bundle=bundle.path, ref=bundle.ref)
            tools = sandbox.executor(ws)
            await tools.write_file("big.txt", big)
            size = await tools.run_command(["wc", "-c", "big.txt"], timeout_s=60)
            assert size.output.split()[0] == str(len(big.encode()))
            await tools.run_command(sh("printf 'caf\\303\\251 \\377\\n' > raw.txt"), timeout_s=60)
            assert await tools.read_file("raw.txt") == "café �\n"
            with pytest.raises(ToolError, match="no such file"):
                await tools.read_file("missing.txt")
            await tools.run_command(["mkfifo", "pipe"], timeout_s=60)
            with pytest.raises(ToolError, match="not a regular file"):
                await tools.read_file("pipe")  # never blocks on a planted FIFO

    asyncio.run(scenario())


def test_a_full_workspace_is_a_tool_error(docker_runtime: DockerRuntime, bundle: Bundle) -> None:
    sandbox = Sandbox(docker_runtime, box_config(work_size_mb=16))

    async def scenario() -> None:
        with sandbox.workspace() as ws:
            await sandbox.materialize(ws, bundle=bundle.path, ref=bundle.ref)
            with pytest.raises(ToolError, match="No space left on device"):
                await sandbox.executor(ws).write_file("huge.txt", "x" * (20 * 1024 * 1024))

    asyncio.run(scenario())


def test_materializing_twice_is_a_factory_error(sandbox: Sandbox, bundle: Bundle) -> None:
    async def scenario() -> None:
        with sandbox.workspace() as ws:
            await sandbox.materialize(ws, bundle=bundle.path, ref=bundle.ref)
            with pytest.raises(SandboxError, match="not empty"):
                await sandbox.materialize(ws, bundle=bundle.path, ref=bundle.ref)

    asyncio.run(scenario())
    assert leftovers(sandbox.runtime.instance) == ([], [])


# --------------------------------------------------------------------------- export


def test_export_ignores_worker_hooks_and_config(
    sandbox: Sandbox, bundle: Bundle, base: str, repo: BareRepo, tmp_path: Path
) -> None:
    out = tmp_path / "task.bundle"
    outcome = export_after(
        sandbox,
        bundle,
        base,
        out,
        sh(
            "printf '#!/bin/sh\\nexit 1\\n' > .git/hooks/pre-commit"
            " && chmod +x .git/hooks/pre-commit"
            " && git config core.hooksPath .git/hooks && git config commit.gpgSign true"
            " && echo feature > feature.txt"
        ),
    )
    assert outcome == ExportOutcome("exported")
    imported = import_task_bundle(repo, out, base_sha=base, dest_ref=DEST_REF)
    assert repo.read_file(imported.head_sha, "feature.txt") == b"feature\n"


def test_hostile_filters_make_export_fail_not_raise(
    docker_runtime: DockerRuntime, bundle: Bundle, base: str, tmp_path: Path
) -> None:
    sandbox = Sandbox(docker_runtime, box_config(command_timeout_s=5.0))
    filter_setup = (
        "echo '* filter=evil' > .gitattributes && git config filter.evil.required true"
        " && echo change > change.txt && git config filter.evil.clean "
    )
    failing = tmp_path / "failing.bundle"
    outcome = export_after(sandbox, bundle, base, failing, sh(filter_setup + "false"))
    assert outcome.status == "worker_git_error"
    assert "clean filter" in outcome.detail
    assert not failing.exists()

    hanging = tmp_path / "hanging.bundle"
    outcome = export_after(sandbox, bundle, base, hanging, sh(filter_setup + "'sleep 60'"))
    assert outcome.status == "worker_git_error"
    assert "timed out after 5s" in outcome.detail
    assert not hanging.exists()


# --------------------------------------------------------------------------- gates


def test_a_tree_that_cannot_be_materialized_is_a_gate_error(
    sandbox: Sandbox, tmp_path: Path
) -> None:
    junk = tmp_path / "junk.bundle"
    junk.write_bytes(b"not a bundle\n")
    (result,) = asyncio.run(
        sandbox.run_gates([GateSpec("unit", "echo ran")], bundle=junk, ref="refs/heads/x")
    )
    assert (result.status, result.exit_code) == (GateStatus.ERROR, None)
    assert "[local-factory] gate 'unit' could not be started: materialize:" in result.output_tail
    assert "ran" not in result.output_tail.replace("could not", "")


def test_gates_run_as_the_box_user_and_report_signals_and_oom(
    sandbox: Sandbox, bundle: Bundle
) -> None:
    results = asyncio.run(
        sandbox.run_gates(
            [
                GateSpec("where", "id -u; pwd; echo $HOME; git branch --show-current"),
                GateSpec("killed", "kill -9 $$"),
                GateSpec("hog", "python3 -c 'x = bytearray(1024 * 1024 * 1024)'"),
            ],
            bundle=bundle.path,
            ref=bundle.ref,
        )
    )
    where, killed, hog = results
    assert where.status is GateStatus.PASS
    assert where.output_tail.split() == [str(sandbox.config.uid), "/work", "/home/sandbox", "task"]
    assert (killed.status, killed.exit_code) == (GateStatus.FAIL, 137)
    assert hog.status is GateStatus.FAIL
    assert "[local-factory] gate 'hog' killed: out of memory" in hog.output_tail
