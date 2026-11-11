"""The Docker runtime: runs box containers and owns every docker object it creates.

Every docker CLI process runs with list argv, no shell, ``cwd="/"`` and :func:`docker_env` as
its whole environment, so no host variable can reach the CLI, let alone a container. Every
container and volume is named ``lf-<instance>-<12 hex>`` and labelled ``local-factory=1`` and
``local-factory.instance=<instance>``, so :meth:`DockerRuntime.cleanup` finds exactly this
runtime's objects and nothing else.

:meth:`DockerRuntime.run` is one container from create to removal: ``docker create`` ->
``docker start --attach`` (stdout read per the output mode, stderr kept as a tail, both read
concurrently so neither pipe can block) -> ``docker inspect`` for the exit code and the OOM flag.
The exit code is taken from ``inspect``, not from the attach process. The container is removed
on every path, including cancellation of the awaiting task: the blocking docker calls of the
cleanup run on their own threads, which a cancellation cannot interrupt, and a cancellation that
arrives *during* the cleanup waits for it to finish before it propagates.

:class:`SandboxError` means the factory's side failed (docker missing, daemon down, image
missing, create/start/inspect/remove failed). Whatever a command does inside the box is reported
in :class:`ContainerResult`, never as an exception.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import json
import math
import os
import re
import secrets
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Literal

from local_factory.sandbox.spec import (
    INSTANCE_LABEL,
    LABEL,
    ContainerSpec,
    SpecError,
    check_image,
    create_args,
)

__all__ = [
    "DOCKER_ENV_KEYS",
    "ContainerResult",
    "DockerRuntime",
    "OutputMode",
    "SandboxError",
    "docker_env",
]

OutputMode = Literal["head", "tail"]

# The docker CLI's own environment: nothing else from the host, ever.
DOCKER_ENV_KEYS: tuple[str, ...] = (
    "PATH",
    "HOME",
    "DOCKER_HOST",
    "DOCKER_CONTEXT",
    "DOCKER_CONFIG",
    "DOCKER_CERT_PATH",
    "DOCKER_TLS_VERIFY",
)

CLI_TIMEOUT_S = 120.0  # one docker CLI call (create, inspect, rm, ...), not the container's run
MESSAGE_CHARS = 2_000  # docker output quoted in a SandboxError
_CHUNK_BYTES = 65_536
_REAP_GRACE_S = 5.0  # waiting for the killed attach process (only slow if something held its pipes)
_INSTANCE_RE = re.compile(r"[a-z0-9][a-z0-9-]*")
_IMAGE_ID_RE = re.compile(r"sha256:[0-9a-f]{64}")


class SandboxError(Exception):
    """The factory's side of the box failed: docker missing, daemon down, image missing,
    create/start/inspect/remove failed. Never caused by what runs inside the box."""


@dataclass(frozen=True)
class ContainerResult:
    """How one container run ended."""

    exit_code: int | None  # State.ExitCode; None when timed_out or output_exceeded
    timed_out: bool
    oom_killed: bool  # State.OOMKilled
    output_exceeded: bool  # "head"/sink mode only: stdout passed max_stdout_bytes
    stdout: bytes  # head: first max bytes; tail: last max bytes; sink: b""
    stderr: str  # last stderr_tail_bytes, decoded with replacement
    duration_s: float


def docker_env(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """The docker CLI's environment: PATH and HOME, plus DOCKER_HOST, DOCKER_CONTEXT,
    DOCKER_CONFIG, DOCKER_CERT_PATH and DOCKER_TLS_VERIFY when set. Nothing else, ever."""
    source = os.environ if environ is None else environ
    return {key: source[key] for key in DOCKER_ENV_KEYS if key in source}


class DockerRuntime:
    """Runs box containers through the docker CLI and owns the objects it creates."""

    docker: str
    instance: str

    def __init__(self, *, docker: str = "docker", instance: str | None = None) -> None:
        if not isinstance(docker, str) or not docker or "\x00" in docker:
            raise ValueError(f"docker must name the docker CLI, got {docker!r}")
        if instance is None:
            instance = "lf" + secrets.token_hex(6)
        if not isinstance(instance, str) or not _INSTANCE_RE.fullmatch(instance):
            raise ValueError(f"instance must match [a-z0-9][a-z0-9-]*, got {instance!r}")
        # The CLI runs with cwd="/", so a relative path is anchored here, once.
        self.docker = str(Path(docker).absolute()) if "/" in docker else docker
        self.instance = instance
        self._labels = {INSTANCE_LABEL: instance}
        self._own_name = re.compile(rf"lf-{re.escape(instance)}-[0-9a-f]{{12}}")

    def __repr__(self) -> str:
        return f"DockerRuntime(docker={self.docker!r}, instance={self.instance!r})"

    # ----------------------------------------------------------------------------- images

    def check(self, image: str) -> str:
        """Check that the daemon is reachable and ``image`` is present; its id."""
        self._cli(["version", "--format", "{{.Server.Version}}"], what="reach the Docker daemon")
        return self.image_id(image)

    def image_id(self, image: str) -> str:
        """The local image's id (``sha256:...``). :class:`SandboxError` if it is missing."""
        try:
            check_image(image)
        except SpecError as exc:
            raise SandboxError(f"invalid sandbox image reference: {exc}") from exc
        try:
            out = self._cli(
                ["image", "inspect", "--format", "{{.Id}}", image], what=f"inspect {image}"
            )
        except SandboxError as exc:
            raise SandboxError(
                f"the sandbox image {image!r} is not available; build it with "
                f"scripts/build-images ({exc})"
            ) from exc
        image_id = out.strip()
        if not _IMAGE_ID_RE.fullmatch(image_id):
            raise SandboxError(f"unexpected id for the image {image!r}: {_bounded(image_id)!r}")
        return image_id

    # ------------------------------------------------------------------------- containers

    async def run(
        self,
        spec: ContainerSpec,
        *,
        timeout_s: float,
        stdin: bytes | None = None,
        max_stdout_bytes: int = 1_000_000,
        output: OutputMode = "head",
        stdout_sink: BinaryIO | None = None,
        stderr_tail_bytes: int = 16_000,
    ) -> ContainerResult:
        """Run one container to completion and remove it (on every path).

        ``output="head"`` keeps the first ``max_stdout_bytes`` of stdout and kills the container
        once it writes more (``output_exceeded``); ``"tail"`` keeps the last ``max_stdout_bytes``
        and never kills for size. With ``stdout_sink`` (head mode only) every byte up to the cap
        goes to the sink instead, and the container is killed once it writes more. After
        ``timeout_s`` seconds of wall clock from its start the container is killed
        (``timed_out``). ``stdin`` needs an interactive spec; it is written, then closed.
        """
        _check_run_options(
            spec,
            timeout_s=timeout_s,
            stdin=stdin,
            max_stdout_bytes=max_stdout_bytes,
            output=output,
            stdout_sink=stdout_sink,
            stderr_tail_bytes=stderr_tail_bytes,
        )
        name = self._new_name()
        args = create_args(spec, name=name, labels=self._labels)
        created = _Background(self._create, spec, args)
        try:
            await created.result()
            result = await self._run_created(
                name,
                spec,
                timeout_s=timeout_s,
                stdin=stdin,
                stdout=_Stdout(output, max_stdout_bytes, stdout_sink),
                stderr=_Tail(stderr_tail_bytes),
            )
        except BaseException:
            await self._dispose(name, created, quiet=True)
            raise
        await self._dispose(name, created, quiet=False)
        return result

    def start_detached(self, spec: ContainerSpec) -> str:
        """Create and start a container that keeps running; its name. Remove it yourself
        (:meth:`remove_container`) or with :meth:`cleanup`."""
        name = self._new_name()
        args = create_args(spec, name=name, labels=self._labels)
        try:
            self._create(spec, args)
            self._cli(["start", name], what=f"start the container {name}")
        except BaseException:
            with contextlib.suppress(SandboxError):
                self._remove_container(name)
            raise
        return name

    def remove_container(self, name: str) -> None:
        """Force-remove one of this runtime's containers; a missing one is fine."""
        self._require_own(name)
        self._remove_container(name)

    # ---------------------------------------------------------------------------- volumes

    def create_volume(self, *, size_mb: int, uid: int, gid: int) -> str:
        """A memory-backed (tmpfs) volume of at most ``size_mb``, owned by uid:gid, mode 0700.

        A tmpfs volume keeps its contents only while some container has it mounted.
        """
        for what, value in (("size_mb", size_mb), ("uid", uid), ("gid", gid)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{what} must be a positive integer, got {value!r}")
        name = self._new_name()
        args = [
            "volume", "create",
            "--driver", "local",
            "--opt", "type=tmpfs",
            "--opt", "device=tmpfs",
            "--opt", f"o=size={size_mb}m,uid={uid},gid={gid},mode=0700",
            "--label", f"{LABEL}=1",
            "--label", f"{INSTANCE_LABEL}={self.instance}",
            name,
        ]  # fmt: skip
        self._cli(args, what="create a workspace volume")
        return name

    def remove_volume(self, name: str) -> None:
        """Remove one of this runtime's volumes; a missing one is fine."""
        self._require_own(name)
        self._remove_volume(name)

    # -------------------------------------------------------------------------- ownership

    def owned(self) -> tuple[list[str], list[str]]:
        """The (containers, volumes) this runtime's instance owns, by label."""
        return self._list_containers(), self._list_volumes()

    def cleanup(self) -> None:
        """Remove everything this instance owns: containers first, then volumes."""
        containers, volumes = self.owned()
        failures: list[str] = []
        for name in containers:
            try:
                self._remove_container(name)
            except SandboxError as exc:
                failures.append(str(exc))
        for name in volumes:
            try:
                self._remove_volume(name)
            except SandboxError as exc:
                failures.append(str(exc))
        if failures:
            raise SandboxError("cleanup left objects behind: " + "; ".join(failures))

    # -------------------------------------------------------------------------- internals

    def _new_name(self) -> str:
        return f"lf-{self.instance}-{secrets.token_hex(6)}"

    def _require_own(self, name: str) -> None:
        if not isinstance(name, str) or not self._own_name.fullmatch(name):
            raise ValueError(f"{name!r} is not an object of the runtime instance {self.instance}")

    def _filters(self) -> list[str]:
        return [
            "--filter", f"label={LABEL}=1",
            "--filter", f"label={INSTANCE_LABEL}={self.instance}",
        ]  # fmt: skip

    def _list_containers(self) -> list[str]:
        out = self._cli(
            ["ps", "--all", "--format", "{{.Names}}", *self._filters()],
            what="list the factory's containers",
        )
        return sorted(out.split())

    def _list_volumes(self) -> list[str]:
        out = self._cli(
            ["volume", "ls", "--quiet", *self._filters()], what="list the factory's volumes"
        )
        return sorted(out.split())

    def _remove_container(self, name: str) -> None:
        result = self._cli_run(["rm", "--force", "--volumes", name])
        if result.returncode != 0 and name in self._list_containers():
            raise SandboxError(f"cannot remove the container {name}: {_bounded(result.stderr)}")

    def _remove_volume(self, name: str) -> None:
        result = self._cli_run(["volume", "rm", "--force", name])
        if result.returncode != 0 and name in self._list_volumes():
            raise SandboxError(f"cannot remove the volume {name}: {_bounded(result.stderr)}")

    def _create(self, spec: ContainerSpec, args: list[str]) -> None:
        self._check_volumes(spec)
        self._cli(args, what="create a container")

    def _check_volumes(self, spec: ContainerSpec) -> None:
        """Every volume a spec mounts must already exist and be the factory's: `docker create`
        would otherwise silently create an unlabelled, disk-backed volume without a quota."""
        names = [mount.volume for mount in spec.volumes]
        if not names:
            return
        out = self._cli(
            ["volume", "inspect", "--format", "{{json .Labels}}", *names],
            what=f"find the volume(s) {', '.join(names)}",
        )
        lines = out.splitlines()
        if len(lines) != len(names):
            raise SandboxError(f"unexpected docker volume inspect output: {_bounded(out)!r}")
        for volume, line in zip(names, lines, strict=True):
            labels = _json(line)
            if not isinstance(labels, dict) or labels.get(LABEL) != "1":
                raise SandboxError(f"the volume {volume} was not created by the factory")

    async def _run_created(
        self,
        name: str,
        spec: ContainerSpec,
        *,
        timeout_s: float,
        stdin: bytes | None,
        stdout: _Stdout,
        stderr: _Tail,
    ) -> ContainerResult:
        attached = await self._attach(
            name, spec, timeout_s=timeout_s, stdin=stdin, stdout=stdout, stderr=stderr
        )
        killed = attached.timed_out or attached.exceeded
        if killed:
            await asyncio.to_thread(self._kill, name)
        state = await asyncio.to_thread(self._inspect, name)
        if state.error:
            raise SandboxError(f"the container {name} could not be started: {state.error}")
        # Killed before it even started (a tiny timeout): "created" is a legitimate outcome then.
        if state.status != "exited" and not (killed and state.status == "created"):
            raise SandboxError(
                f"the container {name} did not run to completion (status {state.status!r}); "
                f"docker: {_bounded(stderr.text())}"
            )
        return ContainerResult(
            exit_code=None if killed else state.exit_code,
            timed_out=attached.timed_out,
            oom_killed=state.oom_killed,
            output_exceeded=attached.exceeded,
            stdout=stdout.value(),
            stderr=stderr.text(),
            duration_s=attached.duration_s,
        )

    async def _attach(
        self,
        name: str,
        spec: ContainerSpec,
        *,
        timeout_s: float,
        stdin: bytes | None,
        stdout: _Stdout,
        stderr: _Tail,
    ) -> _Attached:
        argv = [self.docker, "start", "--attach"]
        if spec.interactive:
            argv.append("--interactive")
        argv.append(name)
        started = time.monotonic()
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.PIPE if spec.interactive else asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=docker_env(),
                cwd="/",
            )
        except OSError as exc:
            raise SandboxError(f"cannot run the docker CLI {self.docker!r}: {exc}") from exc
        pumps = [
            asyncio.create_task(_pump(_stream(proc.stdout), stdout.feed)),
            asyncio.create_task(_pump(_stream(proc.stderr), stderr.feed)),
        ]
        feeder = (
            asyncio.create_task(_feed(proc.stdin, stdin or b"")) if proc.stdin is not None else None
        )
        timed_out = exceeded = False
        try:
            # Shielded: a timeout (or a cancellation) must not stop the pumps, which keep what
            # the box wrote until the attach process is gone (see _end).
            async with asyncio.timeout(timeout_s):
                exceeded = not await asyncio.shield(pumps[0])
                if not exceeded:
                    await asyncio.shield(pumps[1])
                    await proc.wait()
        except TimeoutError:
            timed_out = True
        finally:
            await _end(proc, pumps, feeder)
        return _Attached(timed_out=timed_out, exceeded=exceeded, duration_s=_since(started))

    def _kill(self, name: str) -> None:
        """Kill the container and wait until it has stopped (a no-op if it already has)."""
        self._cli(
            ["stop", "--signal", "KILL", "--timeout", "0", name],
            what=f"kill the container {name}",
        )

    def _inspect(self, name: str) -> _State:
        out = self._cli(
            ["inspect", "--type", "container", "--format", "{{json .State}}", name],
            what=f"inspect the container {name}",
        )
        return _State.parse(out, name)

    async def _dispose(self, name: str, created: _Background[None], *, quiet: bool) -> None:
        """Remove the container, after its `docker create` has finished either way."""
        removal = _Background(self._remove_when_created, name, created)
        try:
            await removal.result()
        except asyncio.CancelledError:
            removal.wait()  # cancelled again while cleaning up: finish the cleanup first
            raise
        except SandboxError:
            if not quiet:
                raise

    def _remove_when_created(self, name: str, created: _Background[None]) -> None:
        created.wait()
        self._remove_container(name)

    def _cli(self, args: Sequence[str], *, what: str) -> str:
        result = self._cli_run(args)
        if result.returncode != 0:
            raise SandboxError(
                f"cannot {what}: docker exited with {result.returncode}: {_bounded(result.stderr)}"
            )
        return result.stdout

    def _cli_run(self, args: Sequence[str]) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                [self.docker, *args],
                env=docker_env(),
                cwd="/",
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=CLI_TIMEOUT_S,
                check=False,
            )
        except OSError as exc:
            raise SandboxError(f"cannot run the docker CLI {self.docker!r}: {exc}") from exc
        except subprocess.TimeoutExpired as exc:
            raise SandboxError(
                f"docker {' '.join(args[:2])} did not finish within {CLI_TIMEOUT_S:g}s"
            ) from exc


# ------------------------------------------------------------------------------- helpers


def _check_run_options(
    spec: ContainerSpec,
    *,
    timeout_s: float,
    stdin: bytes | None,
    max_stdout_bytes: int,
    output: str,
    stdout_sink: BinaryIO | None,
    stderr_tail_bytes: int,
) -> None:
    if output not in ("head", "tail"):
        raise ValueError(f"output must be 'head' or 'tail', got {output!r}")
    if stdin is not None and not spec.interactive:
        raise ValueError("stdin needs an interactive spec (ContainerSpec.interactive=True)")
    if stdin is not None and not isinstance(stdin, bytes):
        raise TypeError(f"stdin must be bytes, got {type(stdin).__name__}")
    if stdout_sink is not None and output == "tail":
        raise ValueError("stdout_sink works in head mode only, not with output='tail'")
    if (
        isinstance(timeout_s, bool)
        or not isinstance(timeout_s, int | float)
        or not math.isfinite(timeout_s)
        or timeout_s <= 0
    ):
        raise ValueError(f"timeout_s must be a positive number, got {timeout_s!r}")
    for what, value in (
        ("max_stdout_bytes", max_stdout_bytes),
        ("stderr_tail_bytes", stderr_tail_bytes),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{what} must be a non-negative integer, got {value!r}")


def _bounded(text: str) -> str:
    text = text.strip()
    return text if len(text) <= MESSAGE_CHARS else "..." + text[-MESSAGE_CHARS:]


def _json(text: str) -> Any:
    try:
        return json.loads(text)
    except ValueError:
        return None


def _since(started: float) -> float:
    return time.monotonic() - started


def _stream(reader: asyncio.StreamReader | None) -> asyncio.StreamReader:
    if reader is None:  # cannot happen: the pipes are requested above
        raise SandboxError("the docker CLI's output pipe is missing")
    return reader


@dataclass(frozen=True)
class _Attached:
    timed_out: bool
    exceeded: bool
    duration_s: float


@dataclass(frozen=True)
class _State:
    """The fields of a container's ``State`` the runtime uses."""

    status: str
    exit_code: int
    oom_killed: bool
    error: str

    @classmethod
    def parse(cls, text: str, name: str) -> _State:
        raw = _json(text)
        if isinstance(raw, dict):
            status, code = raw.get("Status"), raw.get("ExitCode")
            oom, error = raw.get("OOMKilled"), raw.get("Error", "")
            if (
                isinstance(status, str)
                and isinstance(code, int)
                and not isinstance(code, bool)
                and isinstance(oom, bool)
                and isinstance(error, str)
            ):
                return cls(status=status, exit_code=code, oom_killed=oom, error=_bounded(error))
        raise SandboxError(f"unexpected docker inspect output for {name}: {_bounded(text)!r}")


class _Stdout:
    """Stdout per output mode: the head (or a sink) up to a cap, or a rolling tail."""

    def __init__(self, mode: OutputMode, limit: int, sink: BinaryIO | None) -> None:
        self._mode = mode
        self._limit = limit
        self._sink = sink
        self._buf = bytearray()
        self._total = 0

    def feed(self, chunk: bytes) -> bool:
        """Take a chunk; False once the head/sink cap has been passed (stop reading)."""
        if self._mode == "tail":
            self._buf += chunk
            excess = len(self._buf) - self._limit
            if excess > 0:
                del self._buf[:excess]
            return True
        kept = chunk[: self._limit - self._total]  # never negative: reading stops at the cap
        self._total += len(chunk)
        if self._sink is None:
            self._buf += kept
        elif kept:
            _write_all(self._sink, kept)
        return self._total <= self._limit

    def value(self) -> bytes:
        return bytes(self._buf)


def _write_all(sink: BinaryIO, data: bytes) -> None:
    view = memoryview(data)
    try:
        while view:
            written = sink.write(view)
            if not written or written < 0:
                raise OSError(f"short write to the output sink ({written!r} bytes)")
            view = view[written:]
    except (OSError, ValueError) as exc:
        raise SandboxError(f"cannot write the container's output: {exc}") from exc


class _Tail:
    """The last ``limit`` bytes of a stream, decoded with replacement characters."""

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._buf = bytearray()
        self._truncated = False

    def feed(self, chunk: bytes) -> bool:
        self._buf += chunk
        excess = len(self._buf) - self._limit
        if excess > 0:
            del self._buf[:excess]
            self._truncated = True
        return True

    def text(self) -> str:
        data = bytes(self._buf)
        if self._truncated:  # drop the rest of a UTF-8 character the cut went through
            skip = 0
            while skip < min(3, len(data)) and 0x80 <= data[skip] <= 0xBF:
                skip += 1
            data = data[skip:]
        return data.decode("utf-8", errors="replace")


async def _pump(stream: asyncio.StreamReader, feed: Callable[[bytes], bool]) -> bool:
    """Read ``stream`` into ``feed``; True at EOF, False when ``feed`` wanted no more."""
    while chunk := await stream.read(_CHUNK_BYTES):
        if not feed(chunk):
            return False
    return True


async def _feed(stdin: asyncio.StreamWriter, data: bytes) -> None:
    """Write ``data`` and close; a reader that went away early is not an error."""
    with contextlib.suppress(BrokenPipeError, ConnectionResetError):
        if data:
            stdin.write(data)
            await stdin.drain()
        stdin.close()


async def _end(
    proc: asyncio.subprocess.Process,
    pumps: list[asyncio.Task[bool]],
    feeder: asyncio.Task[None] | None,
) -> None:
    """Kill the attach process if it still runs, let its pipes run dry, and reap it."""
    if proc.returncode is None:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
    tasks: list[asyncio.Task[Any]] = [*pumps]
    if feeder is not None:
        feeder.cancel()
        tasks.append(feeder)
    if proc.stdin is not None:
        proc.stdin.close()
    try:
        # Bounded: the pipes only stay open past the kill if something else holds them.
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(_reap(proc, pumps), _REAP_GRACE_S)
    finally:  # even when cancelled (again) while reaping
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def _reap(proc: asyncio.subprocess.Process, pumps: list[asyncio.Task[bool]]) -> None:
    """Wait for the pumps to reach EOF, discard what a pump that stopped early (over the head
    cap, or a failing sink) left unread, and wait for the process.

    ``Process.wait()`` alone is not enough: it returns as soon as the exit has been seen, and a
    pipe nobody reads to EOF stays open, to be torn down by the garbage collector after the
    event loop has gone (ResourceWarnings on Python 3.12)."""
    await asyncio.wait(pumps)
    for stream in (_stream(proc.stdout), _stream(proc.stderr)):
        while await stream.read(_CHUNK_BYTES):
            pass
    await proc.wait()


class _Background[T]:
    """``fn(*args)`` on its own thread. Awaiting it can be cancelled, the thread cannot: it
    runs to the end, and cleanup code can still wait for it synchronously (:meth:`wait`)."""

    def __init__(self, fn: Callable[..., T], *args: object) -> None:
        self._future: concurrent.futures.Future[T] = concurrent.futures.Future()
        self._future.set_running_or_notify_cancel()  # so cancelling an awaiter cannot cancel it
        thread = threading.Thread(target=self._run, args=(fn, args), name="local-factory-docker")
        thread.start()

    def _run(self, fn: Callable[..., T], args: tuple[object, ...]) -> None:
        try:
            result = fn(*args)
        except BaseException as exc:  # handed over to whoever awaits the result
            self._future.set_exception(exc)
        else:
            self._future.set_result(result)

    async def result(self) -> T:
        return await asyncio.wrap_future(self._future)

    def wait(self) -> None:
        """Block until ``fn`` has finished, whatever its outcome."""
        concurrent.futures.wait([self._future])
