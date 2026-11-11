"""One command for :class:`~local_factory.agents.local.LocalExecutor` (imported on first use).

The command runs in its own process group (``start_new_session``), so a timeout, a cancelled
caller, or background processes it leaves behind are all cleaned up by killing the group. Its
stdout and stderr share one pipe that this module owns (not an asyncio subprocess pipe):
``Process.wait()`` then returns when the command exits even if an escaped process still holds
the pipe, and the pipe is always closed here.

The environment is minimal, as for gates (invariant 6): ``PATH``, ``LANG=C.UTF-8`` and
``HOME`` = ``TMPDIR`` = a fresh temporary directory, removed afterwards.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import tempfile
from collections.abc import Sequence
from io import FileIO
from pathlib import Path
from subprocess import DEVNULL

from local_factory.agents.local import clip
from local_factory.agents.tools import CommandResult, ToolError

COMMAND_LANG = "C.UTF-8"
NOTE_PREFIX = "[local-factory] "
# After the command and its process group are gone, how long to wait for the output pipe to
# close. It only stays open if a process escaped the group (setsid) while holding the pipe.
DRAIN_GRACE_S = 2.0
_CHUNK_BYTES = 65_536
_HOME_PREFIX = "local-factory-cmd-"


async def run_process(
    argv: Sequence[str], *, cwd: Path, timeout_s: float, max_output_bytes: int
) -> CommandResult:
    """Run ``argv`` in ``cwd``; ``exit_code`` is None iff it was killed for its timeout."""
    tail = _Tail(max_output_bytes)
    with tempfile.TemporaryDirectory(prefix=_HOME_PREFIX, ignore_cleanup_errors=True) as home:
        exit_code = await _run(argv, cwd=cwd, env=command_env(home), timeout_s=timeout_s, tail=tail)
    return CommandResult(exit_code=exit_code, output=tail.text(), timed_out=exit_code is None)


def command_env(home: str) -> dict[str, str]:
    """The whole environment a command sees: never the factory's own."""
    return {
        "PATH": os.environ.get("PATH", os.defpath),
        "LANG": COMMAND_LANG,
        "HOME": home,
        "TMPDIR": home,
    }


async def _run(
    argv: Sequence[str], *, cwd: Path, env: dict[str, str], timeout_s: float, tail: _Tail
) -> int | None:
    read_fd, write_fd = os.pipe()
    with FileIO(read_fd, "rb") as pipe:
        try:
            process = await _spawn(argv, cwd=cwd, env=env, output=write_fd)
        finally:
            os.close(write_fd)  # only the command's processes hold the write end now
        try:
            return await _supervise(process, pipe, timeout_s=timeout_s, tail=tail)
        finally:
            # Every path, including cancellation of the caller: nothing is left running.
            _kill_group(process.pid)
            if process.returncode is None:
                await process.wait()


async def _spawn(
    argv: Sequence[str], *, cwd: Path, env: dict[str, str], output: int
) -> asyncio.subprocess.Process:
    try:
        return await asyncio.create_subprocess_exec(
            *argv,
            cwd=cwd,
            env=env,
            stdin=DEVNULL,
            stdout=output,
            stderr=output,
            start_new_session=True,
        )
    except (OSError, ValueError) as exc:  # missing binary, not executable, NUL in an argument
        raise ToolError(clip(f"{type(exc).__name__}: {exc}")) from exc


async def _supervise(
    process: asyncio.subprocess.Process, pipe: FileIO, *, timeout_s: float, tail: _Tail
) -> int | None:
    reader = asyncio.StreamReader()
    transport, _ = await asyncio.get_running_loop().connect_read_pipe(
        lambda: asyncio.StreamReaderProtocol(reader), pipe
    )
    try:
        drain = asyncio.ensure_future(_drain(reader, tail))
        try:
            exit_code = await _wait(process, timeout_s)
            _kill_group(process.pid)  # background processes the command left behind
            if not await _drained(drain):
                tail.note(
                    "the command exited but its output stayed open: a process escaped its "
                    "process group and was not killed"
                )
            return exit_code
        finally:
            drain.cancel()
    finally:
        transport.close()


async def _wait(process: asyncio.subprocess.Process, timeout_s: float) -> int | None:
    """The command's exit code, or None after killing its group for the timeout."""
    try:
        return await asyncio.wait_for(process.wait(), timeout_s)
    except TimeoutError:
        _kill_group(process.pid)
        await process.wait()
        return None


async def _drained(drain: asyncio.Future[None]) -> bool:
    try:
        await asyncio.wait_for(drain, DRAIN_GRACE_S)
    except TimeoutError:
        return False
    return True


async def _drain(reader: asyncio.StreamReader, tail: _Tail) -> None:
    while chunk := await reader.read(_CHUNK_BYTES):
        tail.feed(chunk)


def _kill_group(pgid: int) -> None:
    # ProcessLookupError: the group is already empty. PermissionError: macOS reports EPERM for a
    # group that only has zombies left. Either way there is nothing left to kill.
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pgid, signal.SIGKILL)


class _Tail:
    """Keeps the last ``limit`` bytes written to it; decodes them without cutting a character."""

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._buf = bytearray()
        self._truncated = False

    def feed(self, data: bytes) -> None:
        self._buf += data
        excess = len(self._buf) - self._limit
        if excess > 0:
            del self._buf[:excess]
            self._truncated = True

    def note(self, message: str) -> None:
        """Append a factory note on its own line (subject to the same bound as the output)."""
        self.feed(f"\n{NOTE_PREFIX}{message}\n".encode())

    def text(self) -> str:
        data = bytes(self._buf)
        if self._truncated:
            data = _drop_partial_utf8_char(data)
        return data.decode("utf-8", errors="replace")


def _drop_partial_utf8_char(data: bytes) -> bytes:
    """Drop the continuation bytes of a UTF-8 character that the truncation cut in half."""
    skip = 0
    while skip < min(3, len(data)) and 0x80 <= data[skip] <= 0xBF:
        skip += 1
    return data[skip:]
