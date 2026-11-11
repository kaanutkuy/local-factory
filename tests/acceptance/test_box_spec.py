"""Acceptance: container specs and their unconditional hardening (docs/specs/phase-1b.md § runtime).

No Docker needed: these check the exact `docker create` arguments. Protected.
"""

from __future__ import annotations

import dataclasses
import itertools
from pathlib import Path
from typing import Any

import pytest

from local_factory.config import SandboxConfig
from local_factory.sandbox.spec import (
    BindMount,
    ContainerSpec,
    SpecError,
    TmpfsMount,
    VolumeMount,
    create_args,
)

CONFIG = SandboxConfig(memory_mb=512, cpus=1.5, pids_limit=100, tmp_size_mb=32)


def spec(**overrides: Any) -> ContainerSpec:
    base = ContainerSpec.from_config(CONFIG, argv=("true",))
    return dataclasses.replace(base, **overrides)


def args_of(s: ContainerSpec) -> list[str]:
    return create_args(s, name="lf-test-0123456789ab", labels={"local-factory.instance": "lftest"})


def pairs(args: list[str]) -> set[tuple[str, str]]:
    return set(itertools.pairwise(args))


def test_from_config_takes_the_limits() -> None:
    s = ContainerSpec.from_config(CONFIG, argv=("echo", "hi"), workdir="/work")
    assert (s.image, s.argv, s.uid, s.gid) == (CONFIG.image, ("echo", "hi"), 10001, 10001)
    assert (s.memory_mb, s.cpus, s.pids_limit, s.tmp_size_mb) == (512, 1.5, 100, 32)
    assert s.workdir == "/work"


def test_hardening_is_always_present() -> None:
    args = args_of(spec())
    assert args[0] == "create"
    expected = {
        ("--name", "lf-test-0123456789ab"),
        ("--label", "local-factory=1"),
        ("--label", "local-factory.instance=lftest"),
        ("--user", "10001:10001"),
        ("--cap-drop", "ALL"),
        ("--security-opt", "no-new-privileges"),
        ("--network", "none"),
        ("--ipc", "private"),
        ("--pids-limit", "100"),
        ("--memory", "512m"),
        ("--memory-swap", "512m"),
        ("--cpus", "1.5"),
        ("--log-driver", "none"),
        ("--pull", "never"),
        ("--hostname", "sandbox"),
        ("--tmpfs", "/tmp:rw,nosuid,nodev,exec,size=32m"),
        ("--tmpfs", "/home/sandbox:rw,nosuid,nodev,exec,size=32m,uid=10001,gid=10001,mode=0700"),
        ("--env", "HOME=/home/sandbox"),
        ("--env", "TMPDIR=/tmp"),
        ("--env", "LANG=C.UTF-8"),
        ("--workdir", "/"),
    }
    assert expected <= pairs(args)
    assert "--read-only" in args
    assert "--init" in args
    assert args[-2:] == [CONFIG.image, "true"]


def test_nothing_dangerous_is_ever_emitted() -> None:
    s = spec(
        env={"FOO": "bar"},
        binds=(BindMount(Path("/srv/input.bundle"), "/in/base.bundle"),),
        volumes=(VolumeMount("lf-vol-1", "/work"),),
        tmpfs=(TmpfsMount("/scratch", 16, executable=True),),
        interactive=True,
    )
    args = args_of(s)
    for flag in (
        "-v",
        "--volume",
        "--privileged",
        "--cap-add",
        "--device",
        "--pid",
        "--userns",
        "--net",
        "-e",
        "--env-file",
        "-p",
        "--publish",
        "--add-host",
        "--rm",
    ):
        assert flag not in args, flag
    assert [a for a in args if a.startswith("--network")] == ["--network"]
    networks = [args[i + 1] for i, a in enumerate(args) if a == "--network"]
    assert networks == ["none"]
    envs = [args[i + 1] for i, a in enumerate(args) if a == "--env"]
    assert all("=" in e for e in envs)  # a bare --env KEY would copy the host's value
    assert "FOO=bar" in envs
    assert (
        "--mount",
        "type=bind,source=/srv/input.bundle,target=/in/base.bundle,readonly",
    ) in pairs(args)
    assert ("--mount", "type=volume,source=lf-vol-1,target=/work,volume-nocopy") in pairs(args)
    assert (
        "--tmpfs",
        "/scratch:rw,nosuid,nodev,exec,size=16m,uid=10001,gid=10001,mode=0700",
    ) in pairs(args)
    assert "--interactive" in args


def test_spec_env_and_labels_are_sorted() -> None:
    args = create_args(spec(env={"B": "2", "A": "1"}), name="n", labels={"z.k": "1", "a.k": "2"})
    envs = [args[i + 1] for i, a in enumerate(args) if a == "--env"]
    assert envs[-2:] == ["A=1", "B=2"]
    labels = [args[i + 1] for i, a in enumerate(args) if a == "--label"]
    assert labels == ["local-factory=1", "a.k=2", "z.k=1"]


def test_read_only_and_writable_mounts() -> None:
    args = args_of(
        spec(
            binds=(BindMount(Path("/srv/rw"), "/data", read_only=False),),
            volumes=(VolumeMount("lf-vol-2", "/work", read_only=True),),
            tmpfs=(TmpfsMount("/scratch", 8),),
        )
    )
    assert ("--mount", "type=bind,source=/srv/rw,target=/data") in pairs(args)
    assert ("--mount", "type=volume,source=lf-vol-2,target=/work,volume-nocopy,readonly") in pairs(
        args
    )
    assert ("--tmpfs", "/scratch:rw,nosuid,nodev,noexec,size=8m,uid=10001,gid=10001,mode=0700") in (
        pairs(args)
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"uid": 0},
        {"gid": 0},
        {"argv": ()},
        {"argv": ("echo", "a\x00b")},
        {"image": ""},
        {"image": "bad image"},
        {"memory_mb": 8},
        {"cpus": 0.0},
        {"pids_limit": 0},
        {"tmp_size_mb": 0},
        {"env": {"HOME": "/root"}},
        {"env": {"1BAD": "x"}},
        {"env": {"OK": "a\x00b"}},
        {"workdir": "relative"},
        {"workdir": "/a/../b"},
        {"tmpfs": (TmpfsMount("/scratch", 0),)},
        {"tmpfs": (TmpfsMount("/tmp", 8),)},
        {"tmpfs": (TmpfsMount("/home/sandbox", 8),)},
        {"tmpfs": (TmpfsMount("/", 8),)},
        {"volumes": (VolumeMount("bad name", "/work"),)},
        {"volumes": (VolumeMount("lf-v", "work"),)},
        {"volumes": (VolumeMount("lf-v", "/work,x"),)},
        {"volumes": (VolumeMount("lf-v", "/w"), VolumeMount("lf-u", "/w"))},
        {"binds": (BindMount(Path("relative/file"), "/in/x"),)},
        {"binds": (BindMount(Path("/srv/a=b"), "/in/x"),)},
        {"binds": (BindMount(Path("/"), "/host"),)},
        {"binds": (BindMount(Path.home(), "/host-home"),)},
        {"binds": (BindMount(Path.home().parent, "/homes"),)},
        {"binds": (BindMount(Path.home() / ".docker", "/d"),)},
        {"binds": (BindMount(Path.home() / ".docker" / "config.json", "/d.json"),)},
        {"binds": (BindMount(Path("/var/run/docker.sock"), "/var/run/docker.sock"),)},
        {"binds": (BindMount(Path("/run/docker.sock"), "/s"),)},
        {"binds": (BindMount(Path("/var/lib/docker"), "/d"),)},
        {"binds": (BindMount(Path("/run/containerd"), "/c"),)},
        {"binds": (BindMount(Path("/proc"), "/p"),)},
        {"binds": (BindMount(Path("/sys/fs"), "/s"),)},
        {"binds": (BindMount(Path("/dev"), "/d"),)},
    ],
)
def test_unsafe_specs_are_refused(overrides: dict[str, object]) -> None:
    with pytest.raises(SpecError):
        spec(**overrides)


def test_a_symlink_to_a_forbidden_source_is_refused(tmp_path: Path) -> None:
    sneaky = tmp_path / "innocent"
    sneaky.symlink_to("/var/run/docker.sock")
    with pytest.raises(SpecError):
        spec(binds=(BindMount(sneaky, "/in/x"),))


def test_a_directory_inside_home_is_allowed(tmp_path: Path) -> None:
    inside = Path.home() / "some-project" / "src"
    s = spec(binds=(BindMount(inside, "/opt/local-factory/local_factory"),))
    assert s.binds[0].source == inside
