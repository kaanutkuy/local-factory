"""DockerRuntime and TaskWorkspace against real Docker: behaviour beyond the acceptance suite."""

from __future__ import annotations

import asyncio
import dataclasses
import io
import os
import secrets
import subprocess
from pathlib import Path
from typing import Any

import pytest

from local_factory.config import SandboxConfig
from local_factory.sandbox.runtime import ContainerResult, DockerRuntime, SandboxError
from local_factory.sandbox.spec import BOX_WORK, BindMount, ContainerSpec, VolumeMount
from local_factory.sandbox.workspace import TaskWorkspace

from .conftest import leftovers

pytestmark = [pytest.mark.container, pytest.mark.integration]


def box(config: SandboxConfig, *argv: str, **kwargs: Any) -> ContainerSpec:
    return ContainerSpec.from_config(config, argv=argv, **kwargs)


def run(runtime: DockerRuntime, spec: ContainerSpec, **kwargs: Any) -> ContainerResult:
    kwargs.setdefault("timeout_s", 60.0)
    return asyncio.run(runtime.run(spec, **kwargs))


def volume_exists(name: str) -> bool:
    probe = subprocess.run(["docker", "volume", "inspect", name], capture_output=True, check=False)
    return probe.returncode == 0


# --------------------------------------------------------------------------- run


def test_a_missing_command_is_the_box_s_result_not_an_error(
    runtime: DockerRuntime, config: SandboxConfig
) -> None:
    result = run(runtime, box(config, "no-such-command-lf"))
    assert result.exit_code == 127
    assert leftovers(runtime.instance) == ([], [])


def test_death_by_signal_is_an_exit_code(runtime: DockerRuntime, config: SandboxConfig) -> None:
    result = run(runtime, box(config, "sh", "-c", "kill -TERM $$"))
    assert result.exit_code == 128 + 15


def test_large_stdin_round_trips(runtime: DockerRuntime, config: SandboxConfig) -> None:
    data = os.urandom(3_000_000)
    result = run(
        runtime, box(config, "cat", interactive=True), stdin=data, max_stdout_bytes=4_000_000
    )
    assert (result.exit_code, result.output_exceeded) == (0, False)
    assert result.stdout == data


def test_stdin_for_a_box_that_never_reads_it(runtime: DockerRuntime, config: SandboxConfig) -> None:
    result = run(runtime, box(config, "true", interactive=True), stdin=b"x" * 8_000_000)
    assert result.exit_code == 0


def test_stderr_keeps_only_a_bounded_tail(runtime: DockerRuntime, config: SandboxConfig) -> None:
    script = "import sys; sys.stderr.write('e' * 200000 + 'END')"
    result = run(runtime, box(config, "python3", "-c", script), stderr_tail_bytes=1_000)
    assert result.exit_code == 0
    assert len(result.stderr) <= 1_000
    assert result.stderr.endswith("END")


def test_a_capped_sink_holds_exactly_the_cap(runtime: DockerRuntime, config: SandboxConfig) -> None:
    sink = io.BytesIO()
    result = run(runtime, box(config, "yes"), stdout_sink=sink, max_stdout_bytes=50_000)
    assert result.output_exceeded
    assert len(sink.getvalue()) == 50_000
    assert sink.getvalue().startswith(b"y\ny\n")


def test_a_tiny_timeout_is_a_timeout(runtime: DockerRuntime, config: SandboxConfig) -> None:
    result = run(runtime, box(config, "sleep", "60"), timeout_s=0.001)
    assert (result.timed_out, result.exit_code) == (True, None)
    assert leftovers(runtime.instance) == ([], [])


def test_runs_can_overlap(runtime: DockerRuntime, config: SandboxConfig) -> None:
    async def scenario() -> list[ContainerResult]:
        return await asyncio.gather(
            *(
                runtime.run(box(config, "sh", "-c", f"sleep 0.5; echo {n}"), timeout_s=60.0)
                for n in range(4)
            )
        )

    results = asyncio.run(scenario())
    assert [r.stdout for r in results] == [b"0\n", b"1\n", b"2\n", b"3\n"]
    assert leftovers(runtime.instance) == ([], [])


def test_a_container_that_cannot_start_is_a_factory_error(
    runtime: DockerRuntime, config: SandboxConfig, tmp_path: Path
) -> None:
    afile = tmp_path / "file.txt"
    afile.write_text("not a directory\n")
    afile.chmod(0o644)
    spec = box(config, "true", binds=(BindMount(afile, "/usr"),))  # a file over a directory
    with pytest.raises(SandboxError, match="could not be started"):
        run(runtime, spec)
    assert leftovers(runtime.instance) == ([], [])


def test_a_missing_volume_is_refused_and_never_created(
    runtime: DockerRuntime, config: SandboxConfig
) -> None:
    ghost = f"lf-{runtime.instance}-{secrets.token_hex(6)}"
    with pytest.raises(SandboxError, match=ghost):
        run(runtime, box(config, "true", volumes=(VolumeMount(ghost, BOX_WORK),)))
    with pytest.raises(SandboxError, match=ghost):
        runtime.start_detached(box(config, "true", volumes=(VolumeMount(ghost, BOX_WORK),)))
    assert not volume_exists(ghost)  # docker create would have made a disk-backed one
    assert leftovers(runtime.instance) == ([], [])


def test_a_volume_the_factory_did_not_create_is_refused(
    runtime: DockerRuntime, config: SandboxConfig
) -> None:
    foreign = f"lf-foreign-{secrets.token_hex(6)}"
    subprocess.run(["docker", "volume", "create", foreign], capture_output=True, check=True)
    try:
        spec = box(config, "true", volumes=(VolumeMount(foreign, BOX_WORK),))
        with pytest.raises(SandboxError, match="not created by the factory"):
            run(runtime, spec)
    finally:
        subprocess.run(["docker", "volume", "rm", "-f", foreign], capture_output=True, check=True)
    assert leftovers(runtime.instance) == ([], [])


def test_a_detached_box_with_a_missing_image_leaves_nothing(
    runtime: DockerRuntime, config: SandboxConfig
) -> None:
    missing = config.model_copy(update={"image": "local-factory/no-such-image:v0"})
    with pytest.raises(SandboxError, match="no-such-image"):
        runtime.start_detached(box(missing, "sleep", "infinity"))
    assert leftovers(runtime.instance) == ([], [])


# --------------------------------------------------------------------------- workspaces


def test_closing_a_workspace_destroys_its_data(
    runtime: DockerRuntime, config: SandboxConfig
) -> None:
    with TaskWorkspace(runtime, config) as first:
        volume = first.volume
        write = box(config, "sh", "-c", "echo secret > /work/f", volumes=(first.mount(),))
        assert run(runtime, write).exit_code == 0
    assert not volume_exists(volume)
    with TaskWorkspace(runtime, config) as second:
        listing = run(runtime, box(config, "ls", "-A", "/work", volumes=(second.mount(),)))
        assert listing.stdout == b""


def test_a_workspace_whose_holder_cannot_start_leaves_nothing(
    runtime: DockerRuntime, config: SandboxConfig
) -> None:
    missing = config.model_copy(update={"image": "local-factory/no-such-image:v0"})
    ws = TaskWorkspace(runtime, missing)
    with pytest.raises(SandboxError, match="no-such-image"):
        ws.open()
    assert leftovers(runtime.instance) == ([], [])


def test_cleanup_removes_an_open_workspace_and_close_still_works(
    runtime: DockerRuntime, config: SandboxConfig
) -> None:
    ws = TaskWorkspace(runtime, config).open()
    containers, volumes = runtime.owned()
    assert (len(containers), volumes) == (1, [ws.volume])
    runtime.cleanup()
    assert leftovers(runtime.instance) == ([], [])
    ws.close()  # its holder and volume are already gone: still fine


def test_a_read_only_workspace_mount_cannot_be_written(
    runtime: DockerRuntime, config: SandboxConfig
) -> None:
    with TaskWorkspace(runtime, config) as ws:
        seed = box(config, "sh", "-c", "echo keep > /work/f", volumes=(ws.mount(),))
        assert run(runtime, seed).exit_code == 0
        tamper = box(
            config,
            "sh",
            "-c",
            "rm -f /work/f; echo x > /work/f; cat /work/f",
            volumes=(ws.mount(read_only=True),),
        )
        result = run(runtime, tamper)
        assert result.stdout == b"keep\n"
        assert "Read-only file system" in result.stderr


@pytest.mark.parametrize("uid", [10001, 20000])
def test_the_workspace_root_belongs_to_the_box_user_whatever_its_uid(
    runtime: DockerRuntime, config: SandboxConfig, uid: int
) -> None:
    # Without volume-nocopy, docker copies the image's /work (10001:10001 0755) onto the live
    # workspace when the next container is created: 0755, and unwritable for any other uid.
    config = config.model_copy(update={"uid": uid, "gid": uid})
    with TaskWorkspace(runtime, config) as ws:
        for _ in range(2):
            probe = box(config, "sh", "-c", "touch /work/x && stat -c '%u %g %a' /work")
            result = run(runtime, box(config, "true"))  # another container in between
            assert result.exit_code == 0
            result = run(runtime, dataclasses.replace(probe, volumes=(ws.mount(),)))
            assert (result.exit_code, result.stdout.split()) == (
                0,
                [str(uid).encode(), str(uid).encode(), b"700"],
            )
