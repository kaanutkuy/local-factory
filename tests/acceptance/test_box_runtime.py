"""Acceptance: the Docker runtime and task workspaces (docs/specs/phase-1b.md § runtime). Protected.

Needs Docker and the sandbox image (scripts/build-images).
"""

from __future__ import annotations

import asyncio
import io
import subprocess
from pathlib import Path
from typing import Any

import pytest

from local_factory.config import SandboxConfig
from local_factory.sandbox.runtime import ContainerResult, DockerRuntime, SandboxError, docker_env
from local_factory.sandbox.spec import BindMount, ContainerSpec
from local_factory.sandbox.workspace import TaskWorkspace

from .conftest import BOX_IMAGE, docker_objects

pytestmark = pytest.mark.container


def box(config: SandboxConfig, *argv: str, **kwargs: Any) -> ContainerSpec:
    return ContainerSpec.from_config(config, argv=argv, **kwargs)


def run(runtime: DockerRuntime, spec: ContainerSpec, **kwargs: Any) -> ContainerResult:
    kwargs.setdefault("timeout_s", 60.0)
    return asyncio.run(runtime.run(spec, **kwargs))


def test_docker_env_passes_only_what_the_cli_needs() -> None:
    env = docker_env(
        {
            "PATH": "/usr/bin",
            "HOME": "/home/u",
            "DOCKER_HOST": "unix:///run/user/1000/docker.sock",
            "DOCKER_CONTEXT": "default",
            "ANTHROPIC_API_KEY": "sk-secret",
            "GITHUB_TOKEN": "ghp-secret",
            "AWS_SECRET_ACCESS_KEY": "aws-secret",
            "LD_PRELOAD": "/tmp/evil.so",
        }
    )
    assert env == {
        "PATH": "/usr/bin",
        "HOME": "/home/u",
        "DOCKER_HOST": "unix:///run/user/1000/docker.sock",
        "DOCKER_CONTEXT": "default",
    }


def test_check_reports_the_image_id(docker_runtime: DockerRuntime) -> None:
    expected = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", BOX_IMAGE],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert docker_runtime.check(BOX_IMAGE) == expected
    assert docker_runtime.image_id(BOX_IMAGE) == expected
    assert expected.startswith("sha256:")


def test_a_missing_image_is_a_factory_error(
    docker_runtime: DockerRuntime, box_config: SandboxConfig
) -> None:
    with pytest.raises(SandboxError, match="scripts/build-images"):
        docker_runtime.check("local-factory/no-such-image:v0")
    missing = box(box_config.model_copy(update={"image": "local-factory/no-such-image:v0"}), "true")
    with pytest.raises(SandboxError):
        run(docker_runtime, missing)


def test_run_reports_exit_code_and_output(
    docker_runtime: DockerRuntime, box_config: SandboxConfig
) -> None:
    result = run(docker_runtime, box(box_config, "sh", "-c", "echo out; echo err >&2; exit 7"))
    assert result.exit_code == 7
    assert result.stdout == b"out\n"
    assert "err" in result.stderr
    assert not (result.timed_out or result.oom_killed or result.output_exceeded)
    assert result.duration_s > 0
    assert docker_objects(docker_runtime.instance) == ([], [])  # removed after the run


def test_the_box_runs_as_the_configured_user(
    docker_runtime: DockerRuntime, box_config: SandboxConfig
) -> None:
    result = run(docker_runtime, box(box_config, "id", "-u"))
    assert result.stdout.decode().strip() == str(box_config.uid)


def test_timeout_kills_and_removes(
    docker_runtime: DockerRuntime, box_config: SandboxConfig
) -> None:
    result = run(docker_runtime, box(box_config, "sleep", "60"), timeout_s=1.5)
    assert result.timed_out
    assert result.exit_code is None
    assert result.duration_s < 30
    assert docker_objects(docker_runtime.instance) == ([], [])


def test_head_mode_kills_on_too_much_output(
    docker_runtime: DockerRuntime, box_config: SandboxConfig
) -> None:
    result = run(docker_runtime, box(box_config, "yes"), max_stdout_bytes=10_000)
    assert result.output_exceeded
    assert result.exit_code is None
    assert not result.timed_out
    assert len(result.stdout) <= 10_000
    assert result.stdout.startswith(b"y\ny\n")


def test_tail_mode_keeps_the_end(docker_runtime: DockerRuntime, box_config: SandboxConfig) -> None:
    script = "import sys; sys.stdout.write('x' * 200000 + 'THE-END')"
    result = run(
        docker_runtime,
        box(box_config, "python3", "-c", script),
        max_stdout_bytes=5_000,
        output="tail",
    )
    assert result.exit_code == 0
    assert not result.output_exceeded
    assert len(result.stdout) <= 5_000
    assert result.stdout.endswith(b"THE-END")


def test_sink_mode_streams_and_caps(
    docker_runtime: DockerRuntime, box_config: SandboxConfig
) -> None:
    sink = io.BytesIO()
    ok = run(
        docker_runtime,
        box(box_config, "python3", "-c", "import sys; sys.stdout.buffer.write(b'B' * 300000)"),
        stdout_sink=sink,
        max_stdout_bytes=1_000_000,
    )
    assert (ok.exit_code, ok.output_exceeded, ok.stdout) == (0, False, b"")
    assert sink.getvalue() == b"B" * 300_000

    capped = run(
        docker_runtime,
        box(box_config, "yes"),
        stdout_sink=io.BytesIO(),
        max_stdout_bytes=50_000,
    )
    assert capped.output_exceeded and capped.exit_code is None


def test_stdin_reaches_an_interactive_box(
    docker_runtime: DockerRuntime, box_config: SandboxConfig
) -> None:
    result = run(docker_runtime, box(box_config, "cat", interactive=True), stdin=b"hello box\n")
    assert (result.exit_code, result.stdout) == (0, b"hello box\n")


def test_inconsistent_run_options_are_rejected(
    docker_runtime: DockerRuntime, box_config: SandboxConfig
) -> None:
    with pytest.raises(ValueError, match="interactive"):
        run(docker_runtime, box(box_config, "cat"), stdin=b"x")
    with pytest.raises(ValueError, match="tail"):
        run(docker_runtime, box(box_config, "true"), stdout_sink=io.BytesIO(), output="tail")


def test_out_of_memory_is_reported(
    docker_runtime: DockerRuntime, box_config: SandboxConfig
) -> None:
    hog = "x = bytearray(1024 * 1024 * 1024)"  # 1 GiB in a 256 MiB box
    result = run(docker_runtime, box(box_config, "python3", "-c", hog))
    assert result.exit_code != 0
    assert result.oom_killed


def test_cancelling_a_run_still_removes_the_container(
    docker_runtime: DockerRuntime, box_config: SandboxConfig
) -> None:
    async def scenario() -> None:
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(
                docker_runtime.run(box(box_config, "sleep", "60"), timeout_s=60.0), timeout=3.0
            )

    asyncio.run(scenario())
    assert docker_objects(docker_runtime.instance) == ([], [])


def test_host_environment_never_reaches_the_box(
    docker_runtime: DockerRuntime, box_config: SandboxConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LF_CANARY_ENV", "canary-0d9e5c")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "canary-0d9e5c-key")
    result = run(docker_runtime, box(box_config, "env", env={"GIVEN": "yes"}))
    assert result.exit_code == 0
    assert b"canary-0d9e5c" not in result.stdout
    assert b"GIVEN=yes" in result.stdout
    assert b"HOME=/home/sandbox" in result.stdout


def test_read_only_bind_mounts(
    docker_runtime: DockerRuntime, box_config: SandboxConfig, tmp_path: Path
) -> None:
    data = tmp_path / "input.txt"
    data.write_text("from the factory\n")
    data.chmod(0o644)
    spec = box(
        box_config,
        "sh",
        "-c",
        "cat /in/input.txt; echo tamper >> /in/input.txt && echo WROTE",
        binds=(BindMount(data, "/in/input.txt"),),
    )
    result = run(docker_runtime, spec)
    assert b"from the factory" in result.stdout
    assert b"WROTE" not in result.stdout
    assert data.read_text() == "from the factory\n"


def test_detached_containers_and_volumes_are_owned_and_cleaned(
    docker_runtime: DockerRuntime, box_config: SandboxConfig
) -> None:
    name = docker_runtime.start_detached(box(box_config, "sleep", "infinity"))
    volume = docker_runtime.create_volume(size_mb=8, uid=box_config.uid, gid=box_config.gid)
    containers, volumes = docker_runtime.owned()
    assert name in containers
    assert volume in volumes
    assert name.startswith(f"lf-{docker_runtime.instance}-")
    docker_runtime.remove_container(name)
    docker_runtime.remove_container(name)  # idempotent
    docker_runtime.remove_volume(volume)
    docker_runtime.remove_volume(volume)  # idempotent
    assert docker_objects(docker_runtime.instance) == ([], [])

    docker_runtime.start_detached(box(box_config, "sleep", "infinity"))
    docker_runtime.create_volume(size_mb=8, uid=box_config.uid, gid=box_config.gid)
    docker_runtime.cleanup()
    assert docker_objects(docker_runtime.instance) == ([], [])


def test_instances_do_not_see_each_other(
    docker_runtime: DockerRuntime, box_config: SandboxConfig
) -> None:
    other = DockerRuntime()
    assert other.instance != docker_runtime.instance
    try:
        name = other.start_detached(box(box_config, "sleep", "infinity"))
        assert name not in docker_runtime.owned()[0]
        docker_runtime.cleanup()  # must not touch the other instance's objects
        assert name in other.owned()[0]
    finally:
        other.cleanup()
    assert docker_objects(other.instance) == ([], [])


# --------------------------------------------------------------------------- workspaces


def test_workspace_persists_between_boxes_and_is_bounded(
    docker_runtime: DockerRuntime, box_config: SandboxConfig
) -> None:
    with TaskWorkspace(docker_runtime, box_config) as ws:
        mount = ws.mount()
        assert (mount.target, mount.read_only, mount.volume) == ("/work", False, ws.volume)
        write = box(box_config, "sh", "-c", "echo kept > /work/f", volumes=(mount,))
        assert run(docker_runtime, write).exit_code == 0
        read = box(box_config, "cat", "/work/f", volumes=(ws.mount(read_only=True),))
        assert run(docker_runtime, read).stdout == b"kept\n"
        ro = box(box_config, "sh", "-c", "echo x > /work/g", volumes=(ws.mount(read_only=True),))
        assert run(docker_runtime, ro).exit_code != 0
        fill = box(
            box_config,
            "sh",
            "-c",
            f"head -c {(box_config.work_size_mb + 16) * 1024 * 1024} /dev/zero > /work/big",
            volumes=(mount,),
        )
        assert run(docker_runtime, fill).exit_code != 0  # the quota holds
        perms = box(box_config, "stat", "-c", "%u %a", "/work", volumes=(mount,))
        assert run(docker_runtime, perms).stdout.decode().split() == [str(box_config.uid), "700"]
        containers, volumes = docker_objects(docker_runtime.instance)
        assert len(containers) == 1  # the holder
        assert volumes == [ws.volume]
    assert docker_objects(docker_runtime.instance) == ([], [])


def test_workspaces_are_separate(docker_runtime: DockerRuntime, box_config: SandboxConfig) -> None:
    with (
        TaskWorkspace(docker_runtime, box_config) as one,
        TaskWorkspace(docker_runtime, box_config) as two,
    ):
        assert one.volume != two.volume
        mine = box(box_config, "sh", "-c", "echo one > /work/f", volumes=(one.mount(),))
        assert run(docker_runtime, mine).exit_code == 0
        theirs = box(box_config, "ls", "-A", "/work", volumes=(two.mount(),))
        assert run(docker_runtime, theirs).stdout == b""


def test_workspace_open_and_close_are_idempotent(
    docker_runtime: DockerRuntime, box_config: SandboxConfig
) -> None:
    ws = TaskWorkspace(docker_runtime, box_config)
    assert ws.open() is ws
    volume = ws.volume
    assert ws.open() is ws and ws.volume == volume
    ws.close()
    ws.close()
    assert docker_objects(docker_runtime.instance) == ([], [])
