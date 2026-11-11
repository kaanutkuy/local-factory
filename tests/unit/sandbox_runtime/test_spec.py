"""Unit tests for container specs: validation edge cases and the exact `docker create` argv."""

from __future__ import annotations

import dataclasses
import os
import pwd
import socket
from pathlib import Path
from typing import Any

import pytest

from local_factory.config import SandboxConfig
from local_factory.sandbox.spec import (
    BOX_HOME,
    BOX_TMP,
    BOX_WORK,
    INSTANCE_LABEL,
    LABEL,
    BindMount,
    ContainerSpec,
    SpecError,
    TmpfsMount,
    VolumeMount,
    check_image,
    create_args,
)

CONFIG = SandboxConfig(memory_mb=512, cpus=1.5, pids_limit=100, tmp_size_mb=32)


def make(**overrides: Any) -> ContainerSpec:
    return dataclasses.replace(ContainerSpec.from_config(CONFIG, argv=("true",)), **overrides)


def args_of(spec: ContainerSpec) -> list[str]:
    return create_args(spec, name="lf-x-0123456789ab", labels={INSTANCE_LABEL: "lfx"})


# --------------------------------------------------------------------------- construction


def test_from_config_passes_everything_through(tmp_path: Path) -> None:
    bind = BindMount(tmp_path / "in.bundle", "/in/base.bundle")
    volume = VolumeMount("lf-x-vol", BOX_WORK)
    scratch = TmpfsMount("/scratch", 8, executable=True)
    spec = ContainerSpec.from_config(
        CONFIG,
        argv=["git", "status"],
        env={"A": "1"},
        binds=[bind],
        volumes=[volume],
        tmpfs=[scratch],
        workdir=BOX_WORK,
        interactive=True,
    )
    assert spec.argv == ("git", "status")
    assert dict(spec.env) == {"A": "1"}
    assert (spec.binds, spec.volumes, spec.tmpfs) == ((bind,), (volume,), (scratch,))
    assert (spec.workdir, spec.interactive) == (BOX_WORK, True)


def test_a_single_string_argv_is_refused() -> None:
    with pytest.raises(SpecError, match="single string"):
        ContainerSpec.from_config(CONFIG, argv="true")
    with pytest.raises(SpecError, match="argv must be a sequence"):
        make(argv="true")


def test_containers_are_copied_so_the_caller_cannot_change_them() -> None:
    argv = ["echo", "hi"]
    env = {"A": "1"}
    spec = make(argv=argv, env=env)
    argv.append("--privileged")
    env["HOME"] = "/root"
    assert spec.argv == ("echo", "hi")
    assert dict(spec.env) == {"A": "1"}


def test_non_sequence_and_non_mapping_fields_are_refused() -> None:
    with pytest.raises(SpecError, match="binds must be a sequence"):
        make(binds=None)
    with pytest.raises(SpecError, match="env must be a mapping"):
        make(env=[("A", "1")])


@pytest.mark.parametrize(
    "overrides",
    [
        {"uid": True},
        {"gid": 1.0},
        {"memory_mb": 512.5},
        {"pids_limit": "100"},
        {"cpus": True},
        {"cpus": "1"},
        {"cpus": float("nan")},
        {"cpus": float("inf")},
        {"cpus": -1.0},
        {"cpus": 0.001},
        {"image": None},
        {"image": "img\nname"},
        {"image": "--privileged"},
        {"argv": ("echo", 1)},
        {"env": {1: "x"}},
        {"env": {"A-B": "x"}},
        {"env": {"é": "x"}},
        {"env": {"A": 1}},
        {"workdir": "//work"},
        {"workdir": "/work\n"},
        {"workdir": None},
        {"interactive": 1},
        {"binds": ("not a mount",)},
        {"volumes": (BindMount(Path("/srv/x"), "/x"),)},
        {"tmpfs": (VolumeMount("lf-v", "/x"),)},
        {"binds": (BindMount(Path("/srv/x"), "/x", read_only=1),)},  # type: ignore[arg-type]  # the point: a non-bool
        {"volumes": (VolumeMount("lf-v", "/x", read_only=None),)},  # type: ignore[arg-type]  # the point: a non-bool
        {"tmpfs": (TmpfsMount("/x", 8, executable="yes"),)},  # type: ignore[arg-type]  # the point: a non-bool
        {"tmpfs": (TmpfsMount("/x", 8.0),)},  # type: ignore[arg-type]  # the point: a non-int
        {"volumes": (VolumeMount("-lf-v", "/x"),)},
        {"volumes": (VolumeMount("lf-v", "/a:b"),)},
        {"volumes": (VolumeMount("lf-v", '/a"b'),)},
        {"volumes": (VolumeMount("lf-v", "/a\nreadonly"),)},
        {"volumes": (VolumeMount("lf-v", "//work"),)},
        {"volumes": (VolumeMount("lf-v", None),)},  # type: ignore[arg-type]  # the point: a non-str
        {"binds": (BindMount(Path("/srv/x"), "/same"),), "tmpfs": (TmpfsMount("/same", 8),)},
        {"binds": (BindMount("/srv/x", "/x"),)},  # type: ignore[arg-type]  # the point: a str source
        {"binds": (BindMount(Path("/srv/a,b"), "/x"),)},
        {"binds": (BindMount(Path('/srv/a"b'), "/x"),)},
        {"binds": (BindMount(Path("/srv/a\nb"), "/x"),)},
        {"binds": (BindMount(Path("/var"), "/x"),)},
        {"binds": (BindMount(Path("/var/lib"), "/x"),)},
        {"binds": (BindMount(Path("/var/lib/containerd"), "/x"),)},
        {"binds": (BindMount(Path("/run/user/1000"), "/x"),)},
        {"binds": (BindMount(Path("/var/run"), "/x"),)},
        {"binds": (BindMount(Path("/srv/../proc/self"), "/x"),)},
        {"binds": (BindMount(Path("/home"), "/x"),)},
        # The rules the acceptance suite also checks, so these unit tests stand on their own.
        {"uid": 0},
        {"memory_mb": 15},
        {"argv": ()},
        {"argv": ("a\x00b",)},
        {"tmpfs": (TmpfsMount(BOX_TMP, 8),)},
        {"volumes": (VolumeMount("lf-v", BOX_HOME),)},
        {"binds": (BindMount(Path("srv/x"), "/x"),)},
        {"binds": (BindMount(Path("/"), "/x"),)},
        {"binds": (BindMount(Path("/srv/.."), "/x"),)},
    ],
)
def test_unsafe_or_malformed_specs_are_refused(overrides: dict[str, object]) -> None:
    with pytest.raises(SpecError):
        make(**overrides)


def test_check_image_accepts_references_and_digests() -> None:
    check_image("local-factory/sandbox:v1")
    check_image("registry.local:5000/img@sha256:" + "0" * 64)
    with pytest.raises(SpecError):
        check_image("")


# --------------------------------------------------------------------------- bind sources


def test_a_socket_anywhere_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "agent.sock"
    with socket.socket(socket.AF_UNIX) as server:
        server.bind(str(path))
        with pytest.raises(SpecError, match="socket"):
            make(binds=(BindMount(path, "/in/x"),))


def test_a_symlink_to_the_home_directory_is_refused(tmp_path: Path) -> None:
    link = tmp_path / "home-link"
    link.symlink_to(Path.home())
    with pytest.raises(SpecError, match="home"):
        make(binds=(BindMount(link, "/in/x"),))


def test_a_symlink_into_docker_config_is_refused(tmp_path: Path) -> None:
    link = tmp_path / "cfg"
    link.symlink_to(Path.home() / ".docker" / "config.json")
    with pytest.raises(SpecError, match=r"\.docker"):
        make(binds=(BindMount(link, "/in/x"),))


def test_both_the_home_variable_and_the_passwd_home_are_protected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_home = tmp_path / "fake-home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    real_home = Path(pwd.getpwuid(os.getuid()).pw_dir)
    for source in (fake_home, real_home, fake_home / ".docker"):
        with pytest.raises(SpecError):
            make(binds=(BindMount(source, "/in/x"),))
    make(binds=(BindMount(fake_home / "project", "/in/x"),))


def test_a_missing_passwd_entry_falls_back_to_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_entry(uid: int) -> pwd.struct_passwd:
        raise KeyError(uid)

    monkeypatch.setattr(pwd, "getpwuid", no_entry)
    with pytest.raises(SpecError, match="home"):
        make(binds=(BindMount(Path.home(), "/in/x"),))
    make(binds=(BindMount(tmp_path / "file", "/in/x"),))


def test_a_missing_source_is_left_for_docker_to_report(tmp_path: Path) -> None:
    spec = make(binds=(BindMount(tmp_path / "missing", "/in/x"),))
    assert spec.binds[0].source == tmp_path / "missing"


# --------------------------------------------------------------------------- create_args


def test_the_complete_argv_in_order(tmp_path: Path) -> None:
    source = tmp_path / "in.bundle"
    spec = make(
        argv=("lf-entry", "exec", "--", "make"),
        env={"Z": "26", "A": "multi\nline"},
        binds=(BindMount(source, "/in/base.bundle"),),
        volumes=(VolumeMount("lf-x-vol", BOX_WORK),),
        tmpfs=(TmpfsMount("/scratch", 8),),
        workdir=BOX_WORK,
        interactive=True,
    )
    assert create_args(spec, name="lf-x-0123456789ab", labels={"b.k": "2", "a.k": "1"}) == [
        "create", "--name", "lf-x-0123456789ab",
        "--label", f"{LABEL}=1", "--label", "a.k=1", "--label", "b.k=2",
        "--user", "10001:10001",
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--read-only",
        "--network", "none",
        "--ipc", "private",
        "--pids-limit", "100",
        "--memory", "512m",
        "--memory-swap", "512m",
        "--cpus", "1.5",
        "--init",
        "--log-driver", "none",
        "--pull", "never",
        "--hostname", "sandbox",
        "--tmpfs", f"{BOX_TMP}:rw,nosuid,nodev,exec,size=32m",
        "--tmpfs", f"{BOX_HOME}:rw,nosuid,nodev,exec,size=32m,uid=10001,gid=10001,mode=0700",
        "--env", f"HOME={BOX_HOME}",
        "--env", "TMPDIR=/tmp",
        "--env", "LANG=C.UTF-8",
        "--env", "A=multi\nline",
        "--env", "Z=26",
        "--mount", f"type=bind,source={source},target=/in/base.bundle,readonly",
        "--mount", "type=volume,source=lf-x-vol,target=/work,volume-nocopy",
        "--tmpfs", "/scratch:rw,nosuid,nodev,noexec,size=8m,uid=10001,gid=10001,mode=0700",
        "--workdir", "/work",
        "--interactive",
        CONFIG.image, "lf-entry", "exec", "--", "make",
    ]  # fmt: skip


def test_no_interactive_flag_unless_asked() -> None:
    assert "--interactive" not in args_of(make())


@pytest.mark.parametrize(
    ("cpus", "text"),
    [(2.0, "2"), (2, "2"), (0.01, "0.01"), (1.123456789, "1.123456789"), (0.5, "0.5")],
)
def test_cpus_are_never_scientific_or_zero(cpus: float, text: str) -> None:
    args = args_of(make(cpus=cpus))
    assert args[args.index("--cpus") + 1] == text


@pytest.mark.parametrize("name", ["", "-lf", "a b", "a/b", None])
def test_invalid_container_names_are_refused(name: object) -> None:
    with pytest.raises(SpecError, match="container name"):
        create_args(make(), name=name, labels={})  # type: ignore[arg-type]  # the point: bad names


@pytest.mark.parametrize(
    "labels",
    [{LABEL: "0"}, {"bad key": "1"}, {"-k": "1"}, {"k": "a\nb"}, {"k": 1}, {1: "x"}],
)
def test_invalid_labels_are_refused(labels: dict[object, object]) -> None:
    with pytest.raises(SpecError, match="label"):
        create_args(make(), name="lf-x", labels=labels)  # type: ignore[arg-type]  # the point: bad labels


def test_create_args_needs_a_spec() -> None:
    with pytest.raises(SpecError, match="ContainerSpec"):
        create_args("true", name="lf-x", labels={})  # type: ignore[arg-type]  # the point: not a spec


def test_create_args_validates_again(tmp_path: Path) -> None:
    target = tmp_path / "harmless"
    target.write_text("x")
    link = tmp_path / "input"
    link.symlink_to(target)
    spec = make(binds=(BindMount(link, "/in/x"),))
    assert "--mount" in args_of(spec)
    link.unlink()
    link.symlink_to("/var/run/docker.sock")  # swapped after the spec was validated
    with pytest.raises(SpecError):
        args_of(spec)
    tampered = make(env={"A": "1"})
    tampered.env["HOME"] = "/root"  # type: ignore[index]  # the point: mutating the private copy
    with pytest.raises(SpecError, match="HOME"):
        args_of(tampered)
