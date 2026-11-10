"""Deterministic gate runner (docs/specs/phase-1a.md § verification, PLAN §4.2, §6.1).

A gate is a configured shell command (``[commands]`` in ``factory.toml``). Its exit status alone
decides the result; its output is kept only as a bounded tail for feedback. Each gate runs in its
own process group so a timeout - or a gate that leaves background processes behind - can be
cleaned up completely.

Gates execute implementer-written code, so they never see the host environment (invariant 6): the
environment is PATH, ``LANG=C.UTF-8`` and a fresh per-gate HOME/TMPDIR, plus explicit overrides.
"""

from __future__ import annotations

import contextlib
import math
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO, cast

from local_factory.config import ProjectConfig
from local_factory.models import GateResult, GateStatus

DEFAULT_GATE_ORDER: tuple[str, ...] = ("lint", "typecheck", "unit", "integration", "e2e")
DEFAULT_TIMEOUT_S = 900.0
DEFAULT_TAIL_BYTES = 16_000

_CHUNK_BYTES = 65_536
# After the gate's process group is gone, how long to wait for its output pipe to close. It only
# stays open if a process escaped the group (e.g. via setsid) while holding the gate's stdout.
_DRAIN_GRACE_S = 2.0
_NOTE_PREFIX = "[local-factory] "
_HOME_PREFIX = "local-factory-gate-"
GATE_LANG = "C.UTF-8"


@dataclass(frozen=True)
class GateSpec:
    """One gate: a name, a bash command line and a wall-clock timeout."""

    name: str
    command: str
    timeout_s: float = DEFAULT_TIMEOUT_S

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("gate name must not be empty")
        if not self.command.strip():
            raise ValueError(f"gate {self.name!r}: command must not be blank")
        if not (math.isfinite(self.timeout_s) and self.timeout_s > 0):
            raise ValueError(
                f"gate {self.name!r}: timeout_s must be a positive number, got {self.timeout_s!r}"
            )


def gates_from_project(
    config: ProjectConfig, names: Sequence[str] = DEFAULT_GATE_ORDER
) -> list[GateSpec]:
    """Gate specs for ``names``, in that order; names without a (non-blank) command are omitted."""
    if isinstance(names, str):
        raise TypeError("names must be a sequence of gate names, not a single string")
    seen: set[str] = set()
    specs: list[GateSpec] = []
    for name in names:
        if name in seen:
            raise ValueError(f"duplicate gate name: {name!r}")
        seen.add(name)
        command = config.commands.get(name, "")
        if command.strip():
            specs.append(GateSpec(name=name, command=command))
    return specs


def run_gate(
    spec: GateSpec,
    *,
    cwd: Path,
    env: Mapping[str, str] | None = None,
    tail_bytes: int = DEFAULT_TAIL_BYTES,
) -> GateResult:
    """Run ``bash -c <command>`` in ``cwd`` and classify it.

    exit 0 -> PASS; non-zero (including death by signal, as a negative code) -> FAIL with
    ``exit_code``; timeout -> ERROR after killing the whole process group; a command that cannot be
    started at all (e.g. ``cwd`` missing) -> ERROR.

    The environment is minimal, never ``os.environ``: the host's PATH, ``LANG=C.UTF-8``, and
    HOME = TMPDIR = a fresh temp dir that is removed afterwards (also on timeout); ``env``
    overrides go on top. bash itself is resolved on the host PATH, so ``env`` may replace PATH.

    ``output_tail`` is the last ``tail_bytes`` bytes of interleaved stdout+stderr, decoded as
    UTF-8 with replacement characters, so it never has more than ``tail_bytes`` characters.
    """
    if tail_bytes < 0:
        raise ValueError(f"tail_bytes must be >= 0, got {tail_bytes}")
    # ignore_cleanup_errors: a process that escaped the group may still be writing into HOME;
    # TemporaryDirectory already resets permissions to delete read-only trees.
    with tempfile.TemporaryDirectory(prefix=_HOME_PREFIX, ignore_cleanup_errors=True) as home:
        return _run(spec, cwd=cwd, env=gate_env(home, env), tail=_Tail(tail_bytes))


def gate_env(home: str, overrides: Mapping[str, str] | None = None) -> dict[str, str]:
    """The minimal environment a gate runs with (see :func:`run_gate`)."""
    base = {
        "PATH": os.environ.get("PATH", os.defpath),
        "LANG": GATE_LANG,
        "HOME": home,
        "TMPDIR": home,
    }
    return {**base, **(overrides or {})}


def _run(spec: GateSpec, *, cwd: Path, env: dict[str, str], tail: _Tail) -> GateResult:
    started = time.monotonic()
    try:
        # Executing a configured shell command line is the gate runner's whole purpose (CLAUDE.md:
        # the one sanctioned shell). argv is a list; the command string is the project's own config.
        proc = subprocess.Popen(
            [shutil.which("bash") or "bash", "-c", spec.command],
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
            start_new_session=True,
        )
    except OSError as exc:
        tail.note(f"gate {spec.name!r} could not be started: {exc}")
        return GateResult(
            name=spec.name,
            status=GateStatus.ERROR,
            duration_s=time.monotonic() - started,
            output_tail=tail.text(),
        )

    stdout = cast("IO[bytes]", proc.stdout)  # always set: stdout=PIPE
    reader = threading.Thread(
        target=_drain, args=(stdout, tail), name=f"gate-output-{spec.name}", daemon=True
    )
    reader.start()
    try:
        timed_out = _wait(proc, spec.timeout_s)
    finally:
        # The group id is the leader's pid (start_new_session). Kill whatever is left in it: the
        # whole tree on timeout, or background stragglers after a normal exit.
        _kill_group(proc.pid)
        returncode = proc.wait()
    reader.join(_DRAIN_GRACE_S)
    duration = time.monotonic() - started

    if reader.is_alive():
        tail.note(
            f"gate {spec.name!r} exited but its output stayed open: a process escaped the "
            "gate's process group and was not killed"
        )
    if timed_out:
        tail.note(
            f"gate {spec.name!r} timed out after {spec.timeout_s:g}s; its process group was killed"
        )
        status, exit_code = GateStatus.ERROR, None
    elif returncode == 0:
        status, exit_code = GateStatus.PASS, 0
    else:
        status, exit_code = GateStatus.FAIL, returncode
    return GateResult(
        name=spec.name,
        status=status,
        exit_code=exit_code,
        duration_s=duration,
        output_tail=tail.text(),
    )


def run_gates(
    specs: Sequence[GateSpec],
    *,
    cwd: Path,
    env: Mapping[str, str] | None = None,
    fail_fast: bool = False,
) -> list[GateResult]:
    """Run ``specs`` in order. With ``fail_fast``, gates after the first non-PASS are SKIPPED
    without being executed."""
    results: list[GateResult] = []
    stopper: GateResult | None = None
    for spec in specs:
        if stopper is not None:
            results.append(
                GateResult(
                    name=spec.name,
                    status=GateStatus.SKIPPED,
                    output_tail=(
                        f"{_NOTE_PREFIX}not run: gate {stopper.name!r} finished with status "
                        f"{stopper.status.value!r} and fail_fast is set"
                    ),
                )
            )
            continue
        result = run_gate(spec, cwd=cwd, env=env)
        results.append(result)
        if fail_fast and result.status is not GateStatus.PASS:
            stopper = result
    return results


# --------------------------------------------------------------------------- internals


class _Tail:
    """A thread-safe byte sink that keeps only the last ``limit`` bytes written to it."""

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._buf = bytearray()
        self._truncated = False
        self._lock = threading.Lock()

    def feed(self, data: bytes) -> None:
        with self._lock:
            self._buf += data
            excess = len(self._buf) - self._limit
            if excess > 0:
                del self._buf[:excess]
                self._truncated = True

    def note(self, message: str) -> None:
        """Append a factory note on its own line (subject to the same bound as the output)."""
        self.feed(f"\n{_NOTE_PREFIX}{message}\n".encode())

    def text(self) -> str:
        with self._lock:
            data, truncated = bytes(self._buf), self._truncated
        if truncated:
            data = _drop_partial_utf8_char(data)
        return data.decode("utf-8", errors="replace")


def _drop_partial_utf8_char(data: bytes) -> bytes:
    """Drop the continuation bytes of a UTF-8 character that the truncation cut in half."""
    skip = 0
    while skip < min(3, len(data)) and 0x80 <= data[skip] <= 0xBF:
        skip += 1
    return data[skip:]


def _drain(stream: IO[bytes], tail: _Tail) -> None:
    with stream:
        while chunk := stream.read(_CHUNK_BYTES):
            tail.feed(chunk)


def _wait(proc: subprocess.Popen[bytes], timeout_s: float) -> bool:
    """Wait for the gate's leader process; True if it timed out."""
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return True
    return False


def _kill_group(pgid: int) -> None:
    # ProcessLookupError: the group is already empty. PermissionError: macOS reports EPERM for a
    # group that only has zombies left. Either way there is nothing left to kill.
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pgid, signal.SIGKILL)
