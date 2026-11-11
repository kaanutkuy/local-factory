"""Unit tests for DockerRuntime against a fake docker CLI (no Docker needed).

The fake (see conftest.py) logs every call, so these tests check exactly what the runtime asks
docker to do, and how it reacts when docker fails or answers nonsense.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import re
import threading
from pathlib import Path
from typing import Any

import pytest

from local_factory.config import SandboxConfig
from local_factory.sandbox import runtime as runtime_module
from local_factory.sandbox.runtime import (
    ContainerResult,
    DockerRuntime,
    SandboxError,
    _Background,
    _State,
    _Stdout,
    _stream,
    _Tail,
    _write_all,
    docker_env,
)
from local_factory.sandbox.spec import INSTANCE_LABEL, LABEL, ContainerSpec, VolumeMount

from .conftest import FakeDocker, MakeFake

CONFIG = SandboxConfig()
SPEC = ContainerSpec.from_config(CONFIG, argv=("true",))
INTERACTIVE = ContainerSpec.from_config(CONFIG, argv=("cat",), interactive=True)
NAME = "lf-lftest-0123456789ab"


def runtime_for(fake: FakeDocker) -> DockerRuntime:
    return DockerRuntime(docker=str(fake.path), instance="lftest")


def run(runtime: DockerRuntime, spec: ContainerSpec = SPEC, **kwargs: Any) -> ContainerResult:
    kwargs.setdefault("timeout_s", 30.0)
    return asyncio.run(runtime.run(spec, **kwargs))


def state(**fields: object) -> str:
    base: dict[str, object] = {"Status": "exited", "ExitCode": 0, "OOMKilled": False, "Error": ""}
    return json.dumps({**base, **fields}) + "\n"


# --------------------------------------------------------------------------- environment


def test_docker_env_reads_the_process_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOCKER_CONFIG", "/etc/docker-cli")
    monkeypatch.setenv("DOCKER_TLS_VERIFY", "1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")
    monkeypatch.delenv("DOCKER_HOST", raising=False)
    env = docker_env()
    assert env["DOCKER_CONFIG"] == "/etc/docker-cli"
    assert env["DOCKER_TLS_VERIFY"] == "1"
    assert "DOCKER_HOST" not in env
    assert set(env) <= set(runtime_module.DOCKER_ENV_KEYS)


def test_docker_env_omits_what_is_not_set() -> None:
    assert docker_env({"DOCKER_CERT_PATH": "/certs", "SHELL": "/bin/sh"}) == {
        "DOCKER_CERT_PATH": "/certs"
    }


def test_the_cli_gets_only_the_docker_environment(
    fake_docker: MakeFake, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dump = tmp_path / "env.json"
    fake = fake_docker()
    script = fake.path.read_text().replace(
        "args = sys.argv[1:]",
        f"args = sys.argv[1:]\nopen({str(dump)!r}, 'w').write(json.dumps(dict(os.environ)))\n"
        f"open({str(dump)!r} + '.cwd', 'w').write(os.getcwd())",
    )
    fake.path.write_text(script)
    monkeypatch.setenv("LF_CANARY", "canary")
    runtime_for(fake).check("img")
    env = json.loads(dump.read_text())
    assert "LF_CANARY" not in env
    assert set(env) - {"PWD", "SHLVL", "_", "LC_CTYPE"} <= set(runtime_module.DOCKER_ENV_KEYS)
    assert Path(str(dump) + ".cwd").read_text() == "/"


# --------------------------------------------------------------------------- construction


def test_default_instances_are_random_and_well_formed() -> None:
    one, two = DockerRuntime(), DockerRuntime()
    assert re.fullmatch(r"lf[0-9a-f]{12}", one.instance)
    assert one.instance != two.instance
    assert one.docker == "docker"
    assert repr(one) == f"DockerRuntime(docker='docker', instance={one.instance!r})"


@pytest.mark.parametrize("instance", ["", "Upper", "-lead", "has space", "a_b", 7])
def test_invalid_instances_are_refused(instance: object) -> None:
    with pytest.raises(ValueError, match="instance"):
        DockerRuntime(instance=instance)  # type: ignore[arg-type]  # the point: bad values


@pytest.mark.parametrize("docker", ["", "dock\x00er", None])
def test_invalid_docker_clis_are_refused(docker: object) -> None:
    with pytest.raises(ValueError, match="docker"):
        DockerRuntime(docker=docker)  # type: ignore[arg-type]  # the point: bad values


def test_a_relative_docker_path_is_anchored_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert DockerRuntime(docker="bin/docker").docker == str(tmp_path / "bin" / "docker")


# --------------------------------------------------------------------------- images


def test_check_asks_the_daemon_then_the_image(fake_docker: MakeFake) -> None:
    fake = fake_docker()
    assert runtime_for(fake).check("local-factory/sandbox:v1") == "sha256:" + "a" * 64
    assert fake.calls() == [
        ["version", "--format", "{{.Server.Version}}"],
        ["image", "inspect", "--format", "{{.Id}}", "local-factory/sandbox:v1"],
    ]


def test_an_unreachable_daemon_is_reported(fake_docker: MakeFake) -> None:
    fake = fake_docker({"version": {"exit": 1, "stderr": "Cannot connect to the Docker daemon"}})
    with pytest.raises(SandboxError, match=r"reach the Docker daemon.*Cannot connect"):
        runtime_for(fake).check("img")
    assert fake.commands() == ["version"]


def test_a_missing_docker_cli_is_reported(tmp_path: Path) -> None:
    runtime = DockerRuntime(docker=str(tmp_path / "no-such-docker"))
    with pytest.raises(SandboxError, match="cannot run the docker CLI"):
        runtime.check("img")


def test_a_missing_image_names_the_fix(fake_docker: MakeFake) -> None:
    fake = fake_docker({"image inspect": {"exit": 1, "stderr": "No such image: img:v0"}})
    with pytest.raises(SandboxError, match=r"'img:v0'.*scripts/build-images.*No such image"):
        runtime_for(fake).image_id("img:v0")


def test_an_invalid_image_reference_is_refused_without_asking_docker(
    fake_docker: MakeFake,
) -> None:
    fake = fake_docker()
    with pytest.raises(SandboxError, match="invalid sandbox image reference"):
        runtime_for(fake).image_id("--privileged")
    assert fake.calls() == []


def test_a_strange_image_id_is_refused(fake_docker: MakeFake) -> None:
    fake = fake_docker({"image inspect": {"stdout": "sha256:short\n"}})
    with pytest.raises(SandboxError, match="unexpected id"):
        runtime_for(fake).image_id("img")


def test_a_hanging_cli_is_cut_off(fake_docker: MakeFake, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime_module, "CLI_TIMEOUT_S", 0.3)
    fake = fake_docker({"version": {"exec": ["sleep", "30"]}})
    with pytest.raises(SandboxError, match=r"did not finish within 0\.3s"):
        runtime_for(fake).check("img")


def test_long_docker_errors_are_bounded(fake_docker: MakeFake) -> None:
    fake = fake_docker({"version": {"exit": 1, "stderr": "x" * 10_000 + "THE-END"}})
    with pytest.raises(SandboxError) as caught:
        runtime_for(fake).check("img")
    message = str(caught.value)
    assert message.endswith("THE-END")
    assert len(message) < runtime_module.MESSAGE_CHARS + 200


# --------------------------------------------------------------------------- run: options


@pytest.mark.parametrize(
    ("spec", "kwargs", "error", "match"),
    [
        (SPEC, {"stdin": b"x"}, ValueError, "interactive"),
        (INTERACTIVE, {"stdin": "text"}, TypeError, "bytes"),
        (SPEC, {"stdout_sink": io.BytesIO(), "output": "tail"}, ValueError, "tail"),
        (SPEC, {"output": "middle"}, ValueError, "output"),
        (SPEC, {"timeout_s": 0}, ValueError, "timeout_s"),
        (SPEC, {"timeout_s": float("nan")}, ValueError, "timeout_s"),
        (SPEC, {"timeout_s": float("inf")}, ValueError, "timeout_s"),
        (SPEC, {"timeout_s": True}, ValueError, "timeout_s"),
        (SPEC, {"timeout_s": "5"}, ValueError, "timeout_s"),
        (SPEC, {"max_stdout_bytes": -1}, ValueError, "max_stdout_bytes"),
        (SPEC, {"max_stdout_bytes": True}, ValueError, "max_stdout_bytes"),
        (SPEC, {"stderr_tail_bytes": 1.5}, ValueError, "stderr_tail_bytes"),
    ],
)
def test_inconsistent_options_are_refused_before_docker_runs(
    fake_docker: MakeFake,
    spec: ContainerSpec,
    kwargs: dict[str, Any],
    error: type[Exception],
    match: str,
) -> None:
    fake = fake_docker()
    with pytest.raises(error, match=match):
        run(runtime_for(fake), spec, **kwargs)
    assert fake.calls() == []


# --------------------------------------------------------------------------- run: flow


def test_run_creates_attaches_inspects_and_removes(fake_docker: MakeFake) -> None:
    fake = fake_docker({"start --attach": {"stdout": "out\n", "stderr": "err\n", "exit": 1}})
    result = run(runtime_for(fake))
    # The exit code is docker inspect's, not the attach process's.
    assert (result.exit_code, result.stdout, result.stderr) == (3, b"out\n", "err\n")
    assert not (result.timed_out or result.oom_killed or result.output_exceeded)
    calls = fake.calls()
    assert [c[0] for c in calls] == ["create", "start", "inspect", "rm"]
    name = calls[0][calls[0].index("--name") + 1]
    assert re.fullmatch(r"lf-lftest-[0-9a-f]{12}", name)
    assert (f"{INSTANCE_LABEL}=lftest") in calls[0]
    assert calls[1] == ["start", "--attach", name]
    assert calls[2] == ["inspect", "--type", "container", "--format", "{{json .State}}", name]
    assert calls[3] == ["rm", "--force", "--volumes", name]


def test_out_of_memory_is_reported(fake_docker: MakeFake) -> None:
    fake = fake_docker({"inspect": {"stdout": state(ExitCode=137, OOMKilled=True)}})
    result = run(runtime_for(fake))
    assert (result.exit_code, result.oom_killed) == (137, True)


def test_stdin_is_written_then_closed(fake_docker: MakeFake, tmp_path: Path) -> None:
    received = tmp_path / "stdin"
    fake = fake_docker({"start --attach": {"stdin_to": str(received)}})
    run(runtime_for(fake), INTERACTIVE, stdin=b"payload\x00\xff")
    assert received.read_bytes() == b"payload\x00\xff"
    assert fake.calls()[1][:3] == ["start", "--attach", "--interactive"]
    run(runtime_for(fake), INTERACTIVE)  # no stdin: an immediate EOF
    assert received.read_bytes() == b""


def test_a_box_that_ignores_its_stdin_is_fine(fake_docker: MakeFake) -> None:
    fake = fake_docker({"start --attach": {}})  # exits without reading
    result = run(runtime_for(fake), INTERACTIVE, stdin=b"z" * 4_000_000)
    assert result.exit_code == 3


def test_a_create_failure_is_a_factory_error_and_removal_still_runs(
    fake_docker: MakeFake,
) -> None:
    fake = fake_docker({"create": {"exit": 125, "stderr": "No such image: img"}})
    with pytest.raises(SandboxError, match=r"create a container.*No such image"):
        run(runtime_for(fake))
    assert fake.commands() == ["create", "rm"]


def test_an_attach_process_that_cannot_start_is_a_factory_error(
    fake_docker: MakeFake, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def refuse(*args: object, **kwargs: object) -> None:
        raise OSError("exec format error")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", refuse)
    fake = fake_docker()
    with pytest.raises(SandboxError, match="exec format error"):
        run(runtime_for(fake))
    assert fake.commands() == ["create", "rm"]


@pytest.mark.parametrize(
    "output",
    ["not json\n", "[]\n", '{"Status": "exited"}\n', state(ExitCode=True), state(Error=None)],
)
def test_nonsense_from_inspect_is_a_factory_error(fake_docker: MakeFake, output: str) -> None:
    fake = fake_docker({"inspect": {"stdout": output}})
    with pytest.raises(SandboxError, match="unexpected docker inspect output"):
        run(runtime_for(fake))
    assert fake.commands()[-1] == "rm"


def test_a_failed_start_is_a_factory_error(fake_docker: MakeFake) -> None:
    fake = fake_docker(
        {
            "start --attach": {"stderr": "Error response from daemon: OCI runtime", "exit": 1},
            "inspect": {"stdout": state(Status="created", ExitCode=127, Error="mount failed")},
        }
    )
    with pytest.raises(SandboxError, match="could not be started: mount failed"):
        run(runtime_for(fake))


def test_a_container_left_running_is_a_factory_error(fake_docker: MakeFake) -> None:
    fake = fake_docker(
        {
            "start --attach": {"stderr": "lost connection", "exit": 1},
            "inspect": {"stdout": state(Status="running")},
        }
    )
    with pytest.raises(SandboxError, match=r"did not run to completion.*lost connection"):
        run(runtime_for(fake))


def test_a_timeout_kills_even_a_box_that_never_started(fake_docker: MakeFake) -> None:
    fake = fake_docker(
        {
            "start --attach": {"stdout": "partial\n", "exec": ["sleep", "30"]},
            "inspect": {"stdout": state(Status="created")},
        }
    )
    result = run(runtime_for(fake), timeout_s=0.3)
    assert (result.timed_out, result.exit_code, result.stdout) == (True, None, b"partial\n")
    assert 0.3 <= result.duration_s < 10
    stop = fake.calls()[2]
    assert stop[:5] == ["stop", "--signal", "KILL", "--timeout", "0"]
    assert fake.commands() == ["create", "start", "stop", "inspect", "rm"]


def test_head_mode_stops_reading_and_kills(fake_docker: MakeFake) -> None:
    fake = fake_docker(
        {"start --attach": {"stdout": "y\n", "repeat": 100_000, "exec": ["sleep", "30"]}}
    )
    result = run(runtime_for(fake), max_stdout_bytes=1_000)
    assert (result.output_exceeded, result.exit_code, result.timed_out) == (True, None, False)
    assert result.stdout == b"y\n" * 500
    assert "stop" in fake.commands()


class _Failing(io.BytesIO):
    def write(self, data: Any) -> int:
        raise OSError(28, "No space left on device")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_stdout_bytes": 1_000},
        {"max_stdout_bytes": 1_000, "stdout_sink": io.BytesIO()},
        {"stdout_sink": _Failing()},
        {"timeout_s": 0.5},
        {"timeout_s": 0.5, "output": "tail"},
    ],
)
def test_the_attach_process_is_fully_reaped_on_every_path(
    fake_docker: MakeFake, monkeypatch: pytest.MonkeyPatch, kwargs: dict[str, Any]
) -> None:
    # Process.wait() returns once the exit is seen; a pipe nobody read to EOF (a head cap, a
    # failing sink, a timeout) would outlive the run and be torn down after its loop is gone.
    spawned: list[asyncio.subprocess.Process] = []
    real_spawn = asyncio.create_subprocess_exec

    async def spawn(*args: Any, **options: Any) -> asyncio.subprocess.Process:
        proc = await real_spawn(*args, **options)
        spawned.append(proc)
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    fake = fake_docker(
        {"start --attach": {"stdout": "y\n", "repeat": 400_000, "exec": ["sleep", "30"]}}
    )

    async def scenario() -> None:
        with contextlib.suppress(SandboxError):
            await runtime_for(fake).run(SPEC, **{"timeout_s": 30.0, **kwargs})
        (proc,) = spawned
        assert proc.returncode is not None
        assert proc.stdout is not None and proc.stdout.at_eof()
        assert proc.stderr is not None and proc.stderr.at_eof()

    asyncio.run(scenario())


def test_head_mode_at_exactly_the_cap_is_not_exceeded(fake_docker: MakeFake) -> None:
    fake = fake_docker({"start --attach": {"stdout": "abcd"}})
    result = run(runtime_for(fake), max_stdout_bytes=4)
    assert (result.output_exceeded, result.stdout, result.exit_code) == (False, b"abcd", 3)


def test_tail_mode_keeps_the_end(fake_docker: MakeFake) -> None:
    fake = fake_docker({"start --attach": {"stdout": "x" * 200_000 + "END"}})
    result = run(runtime_for(fake), output="tail", max_stdout_bytes=10)
    assert (result.stdout, result.output_exceeded) == (b"x" * 7 + b"END", False)


def test_the_sink_gets_at_most_the_cap(fake_docker: MakeFake) -> None:
    fake = fake_docker({"start --attach": {"stdout": "B" * 300, "exec": ["sleep", "30"]}})
    sink = io.BytesIO()
    result = run(runtime_for(fake), stdout_sink=sink, max_stdout_bytes=100)
    assert (result.output_exceeded, result.stdout) == (True, b"")
    assert sink.getvalue() == b"B" * 100


def test_a_failing_sink_is_a_factory_error(fake_docker: MakeFake) -> None:
    class Full(io.BytesIO):
        def write(self, data: Any) -> int:
            raise OSError(28, "No space left on device")

    fake = fake_docker()
    with pytest.raises(SandboxError, match="No space left"):
        run(runtime_for(fake), stdout_sink=Full())
    assert fake.commands()[-1] == "rm"


def test_stderr_keeps_a_bounded_tail(fake_docker: MakeFake) -> None:
    fake = fake_docker({"start --attach": {"stderr": "é" * 1_000 + "END"}})
    result = run(runtime_for(fake), stderr_tail_bytes=101)
    assert result.stderr == "é" * 49 + "END"  # the half character at the cut is dropped


def test_a_failed_removal_after_success_is_reported(
    fake_docker: MakeFake, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = fake_docker({"rm": {"exit": 1, "stderr": "device busy"}, "ps": {"stdout": NAME}})
    runtime = runtime_for(fake)
    monkeypatch.setattr(runtime, "_new_name", lambda: NAME)
    with pytest.raises(SandboxError, match=f"cannot remove the container {NAME}: device busy"):
        run(runtime)


def test_a_failed_removal_does_not_hide_the_original_error(
    fake_docker: MakeFake, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = fake_docker(
        {
            "create": {"exit": 1, "stderr": "the real problem"},
            "rm": {"exit": 1, "stderr": "device busy"},
            "ps": {"stdout": NAME},
        }
    )
    runtime = runtime_for(fake)
    monkeypatch.setattr(runtime, "_new_name", lambda: NAME)
    with pytest.raises(SandboxError, match="the real problem"):
        run(runtime)


def test_a_second_cancellation_still_waits_for_the_removal(
    fake_docker: MakeFake, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = fake_docker({"start --attach": {"exec": ["sleep", "30"]}})
    runtime = runtime_for(fake)
    created, removing, release = threading.Event(), threading.Event(), threading.Event()
    real_create, real_remove = runtime._create, runtime._remove_container

    def create(spec: ContainerSpec, args: list[str]) -> None:
        real_create(spec, args)
        created.set()

    def remove(name: str) -> None:
        removing.set()
        release.wait(30)
        real_remove(name)

    monkeypatch.setattr(runtime, "_create", create)
    monkeypatch.setattr(runtime, "_remove_container", remove)

    async def scenario() -> None:
        task = asyncio.create_task(runtime.run(SPEC, timeout_s=60.0))
        assert await asyncio.to_thread(created.wait, 30)
        task.cancel()
        assert await asyncio.to_thread(removing.wait, 30)
        task.cancel()  # again, while the removal is still running
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert fake.commands()[-1] == "rm"  # finished before the cancellation went on

    asyncio.run(scenario())


# --------------------------------------------------------------------------- detached + volumes


def test_start_detached_creates_and_starts(fake_docker: MakeFake) -> None:
    fake = fake_docker()
    name = runtime_for(fake).start_detached(SPEC)
    assert re.fullmatch(r"lf-lftest-[0-9a-f]{12}", name)
    assert fake.calls()[1] == ["start", name]
    assert fake.commands() == ["create", "start"]


def test_start_detached_removes_what_failed_to_start(fake_docker: MakeFake) -> None:
    fake = fake_docker({"start": {"exit": 1, "stderr": "cannot start"}})
    with pytest.raises(SandboxError, match="cannot start"):
        runtime_for(fake).start_detached(SPEC)
    assert fake.commands() == ["create", "start", "rm"]


def test_mounted_volumes_must_be_existing_factory_volumes(fake_docker: MakeFake) -> None:
    spec = ContainerSpec.from_config(
        CONFIG, argv=("true",), volumes=(VolumeMount("lf-a", "/a"), VolumeMount("lf-b", "/b"))
    )
    fake = fake_docker({"volume inspect": {"stdout": '{"local-factory":"1"}\n', "repeat": 2}})
    runtime_for(fake).start_detached(spec)
    assert fake.calls()[0] == ["volume", "inspect", "--format", "{{json .Labels}}", "lf-a", "lf-b"]
    assert fake.commands() == ["volume inspect", "create", "start"]


@pytest.mark.parametrize(
    ("rule", "match"),
    [
        ({"exit": 1, "stderr": "no such volume"}, "no such volume"),
        ({"stdout": "null\n"}, "not created by the factory"),
        ({"stdout": '{"local-factory":"0"}\n'}, "not created by the factory"),
        ({"stdout": "garbage\n"}, "not created by the factory"),
        ({"stdout": ""}, "unexpected docker volume inspect output"),
    ],
)
def test_unknown_volumes_are_refused_before_create(
    fake_docker: MakeFake, rule: dict[str, Any], match: str
) -> None:
    spec = ContainerSpec.from_config(CONFIG, argv=("true",), volumes=(VolumeMount("lf-a", "/a"),))
    fake = fake_docker({"volume inspect": rule})
    with pytest.raises(SandboxError, match=match):
        run(runtime_for(fake), spec)
    assert "create" not in fake.commands()  # docker would have created a disk-backed volume


def test_create_volume_is_tmpfs_backed_and_labelled(fake_docker: MakeFake) -> None:
    fake = fake_docker()
    name = runtime_for(fake).create_volume(size_mb=64, uid=10001, gid=10002)
    assert re.fullmatch(r"lf-lftest-[0-9a-f]{12}", name)
    assert fake.calls() == [
        [
            "volume", "create", "--driver", "local",
            "--opt", "type=tmpfs", "--opt", "device=tmpfs",
            "--opt", "o=size=64m,uid=10001,gid=10002,mode=0700",
            "--label", f"{LABEL}=1", "--label", f"{INSTANCE_LABEL}=lftest",
            name,
        ]
    ]  # fmt: skip


@pytest.mark.parametrize(
    "kwargs",
    [
        {"size_mb": 0, "uid": 1, "gid": 1},
        {"size_mb": 8, "uid": 0, "gid": 1},
        {"size_mb": 8, "uid": 1, "gid": True},
        {"size_mb": 8.5, "uid": 1, "gid": 1},
    ],
)
def test_create_volume_refuses_bad_values(fake_docker: MakeFake, kwargs: dict[str, Any]) -> None:
    fake = fake_docker()
    with pytest.raises(ValueError, match="positive integer"):
        runtime_for(fake).create_volume(**kwargs)
    assert fake.calls() == []


# --------------------------------------------------------------------------- ownership


@pytest.mark.parametrize(
    "name", ["lf-other-0123456789ab", "lf-lftest-0123", "lf-lftest-0123456789ab; rm", "-f", 3]
)
def test_only_own_objects_can_be_removed(fake_docker: MakeFake, name: object) -> None:
    fake = fake_docker()
    runtime = runtime_for(fake)
    with pytest.raises(ValueError, match="not an object of the runtime instance lftest"):
        runtime.remove_container(name)  # type: ignore[arg-type]  # the point: bad names
    with pytest.raises(ValueError, match="not an object of the runtime instance lftest"):
        runtime.remove_volume(name)  # type: ignore[arg-type]  # the point: bad names
    assert fake.calls() == []


def test_removing_something_already_gone_is_fine(fake_docker: MakeFake) -> None:
    fake = fake_docker({"rm": {"exit": 1, "stderr": "No such container"}, "volume rm": {"exit": 1}})
    runtime = runtime_for(fake)
    runtime.remove_container(NAME)
    runtime.remove_volume(NAME)
    assert fake.commands() == ["rm", "ps", "volume rm", "volume ls"]
    assert fake.calls()[0] == ["rm", "--force", "--volumes", NAME]
    assert fake.calls()[2] == ["volume", "rm", "--force", NAME]


def test_a_removal_that_leaves_the_object_is_an_error(fake_docker: MakeFake) -> None:
    fake = fake_docker(
        {
            "rm": {"exit": 1, "stderr": "in use"},
            "ps": {"stdout": NAME + "\n"},
            "volume rm": {"exit": 1, "stderr": "volume in use"},
            "volume ls": {"stdout": NAME + "\n"},
        }
    )
    runtime = runtime_for(fake)
    with pytest.raises(SandboxError, match="cannot remove the container"):
        runtime.remove_container(NAME)
    with pytest.raises(SandboxError, match=r"cannot remove the volume.*volume in use"):
        runtime.remove_volume(NAME)


def test_owned_lists_by_both_labels(fake_docker: MakeFake) -> None:
    fake = fake_docker({"ps": {"stdout": "lf-b\nlf-a\n"}, "volume ls": {"stdout": "lf-v\n"}})
    assert runtime_for(fake).owned() == (["lf-a", "lf-b"], ["lf-v"])
    filters = ["--filter", f"label={LABEL}=1", "--filter", f"label={INSTANCE_LABEL}=lftest"]
    assert fake.calls() == [
        ["ps", "--all", "--format", "{{.Names}}", *filters],
        ["volume", "ls", "--quiet", *filters],
    ]


def test_cleanup_removes_containers_before_volumes(fake_docker: MakeFake) -> None:
    fake = fake_docker({"ps": {"stdout": "lf-c1\nlf-c2\n"}, "volume ls": {"stdout": "lf-v1\n"}})
    runtime_for(fake).cleanup()
    removals = [c for c in fake.calls() if c[0] == "rm" or c[:2] == ["volume", "rm"]]
    assert [c[-1] for c in removals] == ["lf-c1", "lf-c2", "lf-v1"]
    assert removals[-1][:2] == ["volume", "rm"]


def test_cleanup_tries_everything_and_reports_what_stayed(fake_docker: MakeFake) -> None:
    fake = fake_docker(
        {
            "rm": {"exit": 1, "stderr": "busy"},
            "ps": {"stdout": "lf-c1\nlf-c2\n"},
            "volume rm": {"exit": 1, "stderr": "in use"},
            "volume ls": {"stdout": "lf-v1\n"},
        }
    )
    with pytest.raises(SandboxError, match="cleanup left objects behind") as caught:
        runtime_for(fake).cleanup()
    for name in ("lf-c1", "lf-c2", "lf-v1"):
        assert name in str(caught.value)


# --------------------------------------------------------------------------- helpers


def test_stdout_head_and_tail_buffers() -> None:
    head = _Stdout("head", 5, None)
    assert head.feed(b"abc") and not head.feed(b"defg")
    assert head.value() == b"abcde"
    tail = _Stdout("tail", 5, None)
    assert tail.feed(b"abc") and tail.feed(b"defg")
    assert tail.value() == b"cdefg"
    nothing = _Stdout("head", 0, io.BytesIO())
    assert nothing.feed(b"") and not nothing.feed(b"x")


def test_write_all_finishes_short_writes() -> None:
    class Trickle(io.BytesIO):
        def write(self, data: Any) -> int:
            return super().write(bytes(data)[:2])

    sink = Trickle()
    _write_all(sink, b"abcdefg")
    assert sink.getvalue() == b"abcdefg"


@pytest.mark.parametrize("written", [0, None, -1])
def test_write_all_refuses_a_stuck_sink(written: int | None) -> None:
    class Stuck(io.BytesIO):
        def write(self, data: Any) -> int:
            return written  # type: ignore[return-value]  # the point: what a broken sink returns

    with pytest.raises(SandboxError, match="short write"):
        _write_all(Stuck(), b"abc")


def test_write_all_reports_a_closed_sink() -> None:
    sink = io.BytesIO()
    sink.close()
    with pytest.raises(SandboxError, match="closed file"):
        _write_all(sink, b"abc")


def test_tail_drops_a_cut_character_only_when_truncated() -> None:
    tail = _Tail(4)
    tail.feed("éé".encode())
    assert tail.text() == "éé"
    tail.feed(b"x")
    assert tail.text() == "éx"
    junk = _Tail(3)
    junk.feed(b"\x80\x80\x80\x80")
    assert junk.text() == ""


def test_state_parses_what_the_runtime_uses() -> None:
    parsed = _State.parse(state(Status="exited", ExitCode=2, OOMKilled=True), "c")
    assert parsed == _State(status="exited", exit_code=2, oom_killed=True, error="")
    without_error = json.dumps({"Status": "exited", "ExitCode": 0, "OOMKilled": False})
    assert _State.parse(without_error, "c").error == ""


def test_a_missing_pipe_is_a_factory_error() -> None:
    with pytest.raises(SandboxError, match="pipe"):
        _stream(None)


def test_background_hands_over_results_and_errors() -> None:
    def boom() -> None:
        raise SandboxError("from the thread")

    async def scenario() -> None:
        ok = _Background(lambda: 42)
        assert await ok.result() == 42
        failing = _Background(boom)
        failing.wait()
        with pytest.raises(SandboxError, match="from the thread"):
            await failing.result()

    asyncio.run(scenario())
