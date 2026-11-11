"""Container specs for the box and their ``docker create`` arguments (docs/specs/phase-1b.md).

A :class:`ContainerSpec` describes one container: image, command, the box user, resource limits,
environment and mounts. It validates itself on construction and refuses anything that could
weaken the box or expose the host. :func:`create_args` turns it into ``docker create`` arguments.
The hardening is unconditional: no field or argument can remove or weaken it (PLAN §5.2).

Mount sources and targets end up inside comma-separated ``--mount`` / ``--tmpfs`` values, which
docker parses as CSV (``--mount``) or splits on ``:`` (``--tmpfs``). Characters that could start a
new field or record there (``,`` ``=`` ``"`` ``:`` or a control character such as a newline, which
would silently drop a trailing ``readonly``) are refused rather than escaped.
"""

from __future__ import annotations

import contextlib
import math
import os
import posixpath
import pwd
import re
import stat
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from local_factory.config import SandboxConfig

__all__ = [
    "BOX_FACTORY",
    "BOX_HOME",
    "BOX_IN",
    "BOX_TMP",
    "BOX_WORK",
    "INSTANCE_LABEL",
    "LABEL",
    "BindMount",
    "ContainerSpec",
    "SpecError",
    "TmpfsMount",
    "VolumeMount",
    "check_image",
    "create_args",
]

BOX_WORK = "/work"
BOX_HOME = "/home/sandbox"
BOX_FACTORY = "/opt/local-factory"
BOX_IN = "/in"
BOX_TMP = "/tmp"  # noqa: S108 - a path inside the container, not a host temp file
BOX_LANG = "C.UTF-8"
BOX_HOSTNAME = "sandbox"
LABEL = "local-factory"  # label value "1" on every object the factory creates
INSTANCE_LABEL = "local-factory.instance"  # value: the owning DockerRuntime's instance id

MIN_MEMORY_MB = 16
MIN_CPUS = 0.01  # docker's own lower bound; below it `docker create` fails anyway

_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")  # container and volume names
_LABEL_KEY_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_ENV_KEY_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_TARGET_FORBIDDEN = frozenset(',=":')
_SOURCE_FORBIDDEN = frozenset(',="')
_RESERVED_TARGETS = frozenset({BOX_TMP, BOX_HOME})  # always the box's own tmpfs mounts
_FORBIDDEN_ENV = frozenset({"HOME"})

# Host paths no bind mount may expose: the source may not be one of them, lie inside one, or be
# an ancestor of one (binding /var would expose /var/lib/docker).
_SENSITIVE_HOST_PATHS: tuple[str, ...] = (
    "/proc",
    "/sys",
    "/dev",
    "/var/lib/docker",
    "/var/lib/containerd",
    "/run",  # docker.sock, containerd, the users' runtime dirs (rootless docker sockets)
    "/var/run",
    "/run/containerd",
    "/run/docker.sock",
    "/var/run/docker.sock",
)


class SpecError(ValueError):
    """A container spec that the factory refuses to run."""


@dataclass(frozen=True)
class BindMount:
    """A host path mounted into the box (read-only unless stated otherwise)."""

    source: Path
    target: str
    read_only: bool = True


@dataclass(frozen=True)
class VolumeMount:
    """A docker volume (normally a task workspace) mounted into the box."""

    volume: str
    target: str
    read_only: bool = False


@dataclass(frozen=True)
class TmpfsMount:
    """A fresh, size-limited, memory-backed directory owned by the box user."""

    target: str
    size_mb: int
    executable: bool = False


@dataclass(frozen=True)
class ContainerSpec:
    """One box container. Validated on construction; see :func:`create_args` for the result."""

    image: str
    argv: tuple[str, ...]
    uid: int
    gid: int
    memory_mb: int
    cpus: float
    pids_limit: int
    tmp_size_mb: int
    env: Mapping[str, str] = field(default_factory=dict)
    binds: tuple[BindMount, ...] = ()
    volumes: tuple[VolumeMount, ...] = ()
    tmpfs: tuple[TmpfsMount, ...] = ()
    workdir: str = "/"
    interactive: bool = False  # attach stdin

    def __post_init__(self) -> None:
        # Private copies, so later changes to the caller's objects cannot reach the spec.
        # (create_args validates again anyway.)
        for name in ("argv", "binds", "volumes", "tmpfs"):
            value = getattr(self, name)
            if isinstance(value, str | bytes) or not isinstance(value, Sequence):
                raise SpecError(f"{name} must be a sequence, got {type(value).__name__}")
            object.__setattr__(self, name, tuple(value))
        if not isinstance(self.env, Mapping):
            raise SpecError(f"env must be a mapping, got {type(self.env).__name__}")
        object.__setattr__(self, "env", dict(self.env))
        _validate(self)

    @classmethod
    def from_config(
        cls,
        config: SandboxConfig,
        *,
        argv: Sequence[str],
        env: Mapping[str, str] | None = None,
        binds: Sequence[BindMount] = (),
        volumes: Sequence[VolumeMount] = (),
        tmpfs: Sequence[TmpfsMount] = (),
        workdir: str = "/",
        interactive: bool = False,
    ) -> ContainerSpec:
        """A spec whose image, user and limits come from ``config``."""
        if isinstance(argv, str):
            raise SpecError("argv must be a sequence of arguments, not a single string")
        return cls(
            image=config.image,
            argv=tuple(argv),
            uid=config.uid,
            gid=config.gid,
            memory_mb=config.memory_mb,
            cpus=config.cpus,
            pids_limit=config.pids_limit,
            tmp_size_mb=config.tmp_size_mb,
            env=dict(env or {}),
            binds=tuple(binds),
            volumes=tuple(volumes),
            tmpfs=tuple(tmpfs),
            workdir=workdir,
            interactive=interactive,
        )


def create_args(spec: ContainerSpec, *, name: str, labels: Mapping[str, str]) -> list[str]:
    """The ``docker create`` argv (without the docker binary) for ``spec``.

    Always: the box user, no capabilities, no-new-privileges, a read-only root filesystem, no
    network, a private IPC namespace, pids/memory (no swap)/CPU limits, an init process, no logs,
    no image pulls, and private tmpfs ``/tmp`` and ``$HOME``. Nothing else is ever emitted: no
    ``-v``, ``--privileged``, ``--cap-add``, ``--device``, ``--pid``, ``--userns``, and every
    ``--env`` carries ``=`` (a bare ``--env KEY`` would copy the variable from the host).
    """
    if not isinstance(spec, ContainerSpec):
        raise SpecError(f"not a ContainerSpec: {type(spec).__name__}")
    _validate(spec)  # again: bind sources may have changed on disk since construction
    _check_name(name, "container name")
    owner = f"uid={spec.uid},gid={spec.gid},mode=0700"
    tmp_size = f"size={spec.tmp_size_mb}m"
    args = ["create", "--name", name, "--label", f"{LABEL}=1"]
    for key, value in _sorted_labels(labels):
        args += ["--label", f"{key}={value}"]
    args += [
        "--user", f"{spec.uid}:{spec.gid}",
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--read-only",
        "--network", "none",
        "--ipc", "private",
        "--pids-limit", str(spec.pids_limit),
        "--memory", f"{spec.memory_mb}m",
        "--memory-swap", f"{spec.memory_mb}m",
        "--cpus", _format_cpus(spec.cpus),
        "--init",
        "--log-driver", "none",
        "--pull", "never",
        "--hostname", BOX_HOSTNAME,
        "--tmpfs", f"{BOX_TMP}:rw,nosuid,nodev,exec,{tmp_size}",
        "--tmpfs", f"{BOX_HOME}:rw,nosuid,nodev,exec,{tmp_size},{owner}",
        "--env", f"HOME={BOX_HOME}",
        "--env", f"TMPDIR={BOX_TMP}",
        "--env", f"LANG={BOX_LANG}",
    ]  # fmt: skip
    for key in sorted(spec.env):
        args += ["--env", f"{key}={spec.env[key]}"]
    for bind in spec.binds:
        readonly = ",readonly" if bind.read_only else ""
        args += ["--mount", f"type=bind,source={bind.source},target={bind.target}{readonly}"]
    for volume in spec.volumes:
        # volume-nocopy: otherwise docker copies the image's directory at the target (its
        # contents, owner and mode) into an empty volume, e.g. a fresh workspace's root.
        readonly = ",readonly" if volume.read_only else ""
        source = f"type=volume,source={volume.volume},target={volume.target}"
        args += ["--mount", f"{source},volume-nocopy{readonly}"]
    for tmpfs in spec.tmpfs:
        exec_flag = "exec" if tmpfs.executable else "noexec"
        options = f"rw,nosuid,nodev,{exec_flag},size={tmpfs.size_mb}m,{owner}"
        args += ["--tmpfs", f"{tmpfs.target}:{options}"]
    args += ["--workdir", spec.workdir]
    if spec.interactive:
        args.append("--interactive")
    args += [spec.image, *spec.argv]
    return args


# --------------------------------------------------------------------------- validation


def _validate(spec: ContainerSpec) -> None:
    check_image(spec.image)
    _check_argv(spec.argv)
    for what in ("uid", "gid"):
        _check_int(getattr(spec, what), what, minimum=1)
    _check_int(spec.memory_mb, "memory_mb", minimum=MIN_MEMORY_MB)
    _check_int(spec.pids_limit, "pids_limit", minimum=1)
    _check_int(spec.tmp_size_mb, "tmp_size_mb", minimum=1)
    _check_cpus(spec.cpus)
    _check_env(spec.env)
    _check_workdir(spec.workdir)
    _check_flag(spec.interactive, "interactive")
    targets: list[str] = []
    for bind in spec.binds:
        if not isinstance(bind, BindMount):
            raise SpecError(f"binds must hold BindMount values, got {bind!r}")
        _check_bind_source(bind.source)
        _check_flag(bind.read_only, f"bind {bind.target!r} read_only")
        targets.append(bind.target)
    for volume in spec.volumes:
        if not isinstance(volume, VolumeMount):
            raise SpecError(f"volumes must hold VolumeMount values, got {volume!r}")
        _check_name(volume.volume, "volume name")
        _check_flag(volume.read_only, f"volume {volume.target!r} read_only")
        targets.append(volume.target)
    for tmpfs in spec.tmpfs:
        if not isinstance(tmpfs, TmpfsMount):
            raise SpecError(f"tmpfs must hold TmpfsMount values, got {tmpfs!r}")
        _check_int(tmpfs.size_mb, f"tmpfs {tmpfs.target!r} size_mb", minimum=1)
        _check_flag(tmpfs.executable, f"tmpfs {tmpfs.target!r} executable")
        targets.append(tmpfs.target)
    seen: set[str] = set()
    for target in targets:
        _check_target(target)
        if target in seen:
            raise SpecError(f"two mounts share the target {target!r}")
        seen.add(target)


def _has_control(text: str) -> bool:
    return any(ord(char) < 32 or ord(char) == 127 for char in text)


def _check_int(value: object, what: str, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SpecError(f"{what} must be an integer, got {value!r}")
    if value < minimum:
        raise SpecError(f"{what} must be at least {minimum}, got {value}")


def _check_flag(value: object, what: str) -> None:
    if not isinstance(value, bool):
        raise SpecError(f"{what} must be a bool, got {value!r}")


def _check_cpus(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise SpecError(f"cpus must be a number, got {value!r}")
    if not math.isfinite(value) or value <= 0:
        raise SpecError(f"cpus must be a positive number, got {value!r}")
    if value < MIN_CPUS:
        raise SpecError(f"cpus must be at least {MIN_CPUS}, got {value!r}")


def _format_cpus(value: float) -> str:
    # Never scientific notation, never rounded down to "0" (which docker reads as "no limit").
    return format(value, ".9f").rstrip("0").rstrip(".")


def check_image(image: object) -> None:
    """Refuse an image reference that is empty, has whitespace or could be read as a flag."""
    if not isinstance(image, str) or not image:
        raise SpecError(f"image must be a non-empty string, got {image!r}")
    if any(char.isspace() for char in image) or _has_control(image):
        raise SpecError(f"image must not contain whitespace or control characters: {image!r}")
    if image.startswith("-"):
        raise SpecError(f"image must not start with '-': {image!r}")


def _check_argv(argv: tuple[object, ...]) -> None:
    if not argv:
        raise SpecError("argv must not be empty")
    for arg in argv:
        if not isinstance(arg, str):
            raise SpecError(f"argv must hold strings, got {arg!r}")
        if "\x00" in arg:
            raise SpecError(f"argv must not contain NUL: {arg!r}")


def _check_env(env: Mapping[str, str]) -> None:
    for key, value in env.items():
        if not isinstance(key, str) or not _ENV_KEY_RE.fullmatch(key):
            raise SpecError(f"env key must be an identifier: {key!r}")
        if key in _FORBIDDEN_ENV:
            raise SpecError(f"env must not set {key}: the box sets it")
        if not isinstance(value, str) or "\x00" in value:
            raise SpecError(f"env {key}: the value must be a string without NUL")


def _normalized_absolute(value: object) -> bool:
    return (
        isinstance(value, str)
        and value.startswith("/")
        and not value.startswith("//")
        and posixpath.normpath(value) == value
    )


def _check_workdir(workdir: object) -> None:
    if not _normalized_absolute(workdir) or _has_control(str(workdir)):
        raise SpecError(f"workdir must be a normalized absolute path: {workdir!r}")


def _check_target(target: object) -> None:
    if not _normalized_absolute(target) or target == "/":
        raise SpecError(f"mount target must be a normalized absolute path other than /: {target!r}")
    text = str(target)
    if _has_control(text) or any(char in _TARGET_FORBIDDEN for char in text):
        raise SpecError(f'mount target must not contain , = " : or control characters: {text!r}')
    if text in _RESERVED_TARGETS:
        raise SpecError(f"mount target {text!r} is reserved for the box's own tmpfs")


def _check_name(name: object, what: str) -> None:
    if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
        raise SpecError(f"invalid {what}: {name!r}")


def _sorted_labels(labels: Mapping[str, str]) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    for key, value in labels.items():
        if not isinstance(key, str) or not _LABEL_KEY_RE.fullmatch(key) or key == LABEL:
            raise SpecError(f"invalid label key: {key!r}")
        if not isinstance(value, str) or _has_control(value):
            raise SpecError(f"invalid value for label {key!r}: {value!r}")
        items.append((key, value))
    return sorted(items)


def _check_bind_source(source: object) -> None:
    if not isinstance(source, Path):
        raise SpecError(f"bind source must be a Path, got {source!r}")
    text = str(source)
    if not source.is_absolute():
        raise SpecError(f"bind source must be absolute: {text!r}")
    if _has_control(text) or any(char in _SOURCE_FORBIDDEN for char in text):
        raise SpecError(f'bind source must not contain , = " or control characters: {text!r}')
    resolved = os.path.realpath(text)
    for candidate in dict.fromkeys((os.path.normpath(text), resolved)):
        reason = _exposure(candidate)
        if reason:
            raise SpecError(f"bind source {text!r} ({candidate}) {reason}")
    if _is_socket(resolved):
        raise SpecError(f"bind source {text!r} is a socket")


def _exposure(path: str) -> str:
    """Why binding host ``path`` would expose too much ("" if it would not)."""
    if path == "/":
        return "is the root directory"
    for home in _host_homes():
        if _within(home, path):  # home itself, or one of its ancestors
            return "is the host user's home directory or one of its ancestors"
    for sensitive in _sensitive_paths():
        if _within(path, sensitive) or _within(sensitive, path):
            return f"overlaps {sensitive}"
    return ""


def _within(path: str, root: str) -> bool:
    """``path`` is ``root`` or lies inside it (both normalized absolute paths)."""
    return path == root or path.startswith(root.rstrip("/") + "/")


def _host_homes() -> Iterator[str]:
    candidates = [str(Path.home())]
    with contextlib.suppress(KeyError):  # no passwd entry (e.g. an arbitrary uid in a container)
        candidates.append(pwd.getpwuid(os.getuid()).pw_dir)
    for home in candidates:
        yield from _both(home)


def _sensitive_paths() -> Iterator[str]:
    for home in _host_homes():
        yield from _both(str(Path(home) / ".docker"))
    for path in _SENSITIVE_HOST_PATHS:
        yield from _both(path)


def _both(path: str) -> tuple[str, ...]:
    """``path`` lexically normalized, and resolved through symlinks."""
    return tuple(dict.fromkeys((os.path.normpath(path), os.path.realpath(path))))


def _is_socket(path: str) -> bool:
    try:
        return stat.S_ISSOCK(Path(path).stat().st_mode)
    except OSError:
        return False
