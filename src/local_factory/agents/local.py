"""Worker-side file tools on a plain directory, and :class:`LocalExecutor`.

The sync helpers :func:`write_file`, :func:`read_file` and :func:`delete_file` are the one
implementation of path confinement: worker-relative paths only, nothing that resolves outside the
root (``..``, symlinks, symlink loops), files only. They run on the host (through
:class:`LocalExecutor`) and inside the box, where ``local_factory.sandbox.entry`` imports them
under ``python3 -I`` on Python 3.12. This module therefore imports only the standard library and
:mod:`local_factory.agents.tools` at import time; :meth:`LocalExecutor.run_command` loads the
asyncio machinery (:mod:`local_factory.agents._command`) on first use.

Every failure raises :class:`ToolError` with a bounded message that is worded the same on every
supported Python version (messages are persisted in the ledger).

:class:`LocalExecutor` is a :class:`~local_factory.agents.tools.ToolExecutor` for in-process tests
and the Phase 1a host slice. It is **not** a security boundary: commands run as the host user.
"""

from __future__ import annotations

import errno
import math
import os
import stat
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

from local_factory.agents.tools import CommandResult, ToolError

__all__ = [
    "DEFAULT_MAX_OUTPUT_BYTES",
    "ESCAPES_ROOT",
    "MAX_TOOL_ERROR_CHARS",
    "SYMLINK_LOOP",
    "LocalExecutor",
    "clip",
    "delete_file",
    "read_file",
    "write_file",
]

DEFAULT_MAX_OUTPUT_BYTES = 1_000_000  # kept per command, and the largest file read_file returns
MAX_TOOL_ERROR_CHARS = 2_000  # hard cap on every ToolError message raised here
SYMLINK_LOOP = "path goes through a symlink loop"
ESCAPES_ROOT = "path escapes the worker directory"
_ELISION = "..."
_READ_CHUNK = 1024 * 1024
# O_NONBLOCK: opening a FIFO planted in the worker fails or returns at once instead of hanging.
_READ_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
_WRITE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC


# --------------------------------------------------------------------------- sync helpers


def write_file(root: Path, path: str, content: str) -> None:
    """Create or overwrite ``path`` under ``root`` with UTF-8 ``content``; parents are created.

    Writing through a symlink inside the root updates (or creates) its target, as long as the
    target is inside the root too.
    """
    with _tool_errors():
        _, target = _confine(_root(root), path)
        data = content.encode("utf-8")
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with _open_regular(target, _WRITE_FLAGS, "written") as fd:
                _write_all(fd, data)
        except IsADirectoryError as exc:
            raise ToolError("is a directory; only files can be written") from exc


def read_file(root: Path, path: str, *, max_bytes: int) -> bytes:
    """The bytes of the file at ``path`` under ``root``; more than ``max_bytes`` is an error."""
    if max_bytes < 0:
        raise ValueError(f"max_bytes must be >= 0, got {max_bytes}")
    with _tool_errors():
        _, target = _confine(_root(root), path)
        try:
            with _open_regular(target, _READ_FLAGS, "read") as fd:
                data = _read_at_most(fd, max_bytes + 1)  # never more than the limit + 1
        except FileNotFoundError as exc:
            raise ToolError("no such file") from exc
        if len(data) > max_bytes:
            raise ToolError(f"file is larger than the limit of {max_bytes} bytes")
        return data


def delete_file(root: Path, path: str) -> None:
    """Remove the file or symlink at ``path`` under ``root`` (a symlink itself, never its target).

    A missing entry or a directory is an error.
    """
    with _tool_errors():
        entry, _ = _confine(_root(root), path)
        if not entry.is_symlink():
            if not entry.exists():
                raise ToolError("no such file")
            if entry.is_dir():
                raise ToolError("is a directory; only files and symlinks can be deleted")
        entry.unlink()


def clip(text: str, limit: int = MAX_TOOL_ERROR_CHARS) -> str:
    """``text`` if it fits in ``limit`` characters, else its head and tail around ``...``."""
    if len(text) <= limit:
        return text
    keep = max(limit - len(_ELISION), 0)
    head = keep // 2
    return text[:head] + _ELISION + text[len(text) - (keep - head) :]


# --------------------------------------------------------------------------- executor


class LocalExecutor:
    """A :class:`~local_factory.agents.tools.ToolExecutor` acting directly on ``root``.

    Not a security boundary (see the module docstring). Commands run with ``cwd`` = the resolved
    root, stdin from ``/dev/null``, in their own process group, with a minimal environment like a
    gate's (``PATH``, ``LANG=C.UTF-8``, ``HOME`` = ``TMPDIR`` = a fresh temporary directory),
    never the factory's. Their stdout and stderr are interleaved; the last ``max_output_bytes``
    are kept. On timeout the whole process group is killed.
    """

    def __init__(self, root: Path, *, max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES) -> None:
        if max_output_bytes < 1:
            raise ValueError(f"max_output_bytes must be >= 1, got {max_output_bytes}")
        self.root = root
        self.max_output_bytes = max_output_bytes

    async def write_file(self, path: str, content: str) -> None:
        write_file(self.root, path, content)

    async def read_file(self, path: str) -> str:
        data = read_file(self.root, path, max_bytes=self.max_output_bytes)
        return data.decode("utf-8", errors="replace")

    async def delete_file(self, path: str) -> None:
        delete_file(self.root, path)

    async def run_command(self, argv: Sequence[str], *, timeout_s: float) -> CommandResult:
        if isinstance(argv, str):
            raise TypeError("argv must be a sequence of arguments, not a single string")
        if not argv:
            raise ToolError("empty argv")
        if not (math.isfinite(timeout_s) and timeout_s > 0):
            raise ValueError(f"timeout_s must be a positive number, got {timeout_s!r}")
        cwd = _root(self.root)
        # Deferred: the box imports this module for the sync helpers only (see module docstring).
        from local_factory.agents._command import run_process

        return await run_process(
            list(argv), cwd=cwd, timeout_s=timeout_s, max_output_bytes=self.max_output_bytes
        )


# --------------------------------------------------------------------------- confinement


def _root(root: Path) -> Path:
    resolved = Path(os.path.realpath(root))
    if not resolved.is_dir():
        raise ToolError(clip(f"worker directory {root} does not exist or is not a directory"))
    return resolved


def _confine(root: Path, raw: str) -> tuple[Path, Path]:
    """Map a worker-relative path to ``(entry, target)``, both inside the resolved ``root``.

    ``entry`` is the directory entry ``raw`` names, with its parent's symlinks resolved;
    ``target`` is what it finally points to with every symlink resolved. Absolute paths, paths
    that do not name an entry, and paths whose parent or target resolve outside ``root`` (via
    ``..`` or symlinks) are rejected before anything is touched.
    """
    if "\x00" in raw:
        raise ToolError("path contains a NUL byte")
    relative = Path(raw)
    if relative.is_absolute():
        raise ToolError("absolute paths are not allowed; use a path relative to the worker")
    if relative.name in ("", ".."):
        raise ToolError("path does not name a file")
    parent = _resolve((root / relative).parent)
    if not parent.is_relative_to(root):
        raise ToolError(ESCAPES_ROOT)
    entry = parent / relative.name
    target = _resolve(entry)
    if target == root or not target.is_relative_to(root):
        raise ToolError(ESCAPES_ROOT)
    return entry, target


def _resolve(path: Path) -> Path:
    """Resolve every symlink in ``path``, identically on all supported Python versions.

    ``Path.resolve()`` raises ``RuntimeError`` on a symlink loop before 3.13 and quietly returns
    the looping path from 3.13 on. ``os.path.realpath`` never raises for a loop, so loops are
    detected here, explicitly, and reported with one message. A missing ``path`` is fine (it may
    be about to be created).
    """
    resolved = Path(os.path.realpath(path))
    try:
        resolved.stat()
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise ToolError(SYMLINK_LOOP) from exc
    return resolved


@contextmanager
def _open_regular(path: Path, flags: int, verb: str) -> Iterator[int]:
    """An fd for ``path``, which must be a regular file (checked on the open fd)."""
    fd = os.open(path, flags, 0o666)
    try:
        mode = os.fstat(fd).st_mode
        if stat.S_ISDIR(mode):
            raise ToolError(f"is a directory; only files can be {verb}")
        if not stat.S_ISREG(mode):
            raise ToolError(f"not a regular file; only files can be {verb}")
        yield fd
    finally:
        os.close(fd)


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        view = view[os.write(fd, view) :]


def _read_at_most(fd: int, limit: int) -> bytes:
    chunks: list[bytes] = []
    remaining = limit
    while remaining > 0 and (chunk := os.read(fd, min(remaining, _READ_CHUNK))):
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


@contextmanager
def _tool_errors() -> Iterator[None]:
    """Turn OS and encoding failures into a :class:`ToolError` with a bounded message."""
    try:
        yield
    except (OSError, ValueError) as exc:  # UnicodeEncodeError is a ValueError
        raise ToolError(clip(f"{type(exc).__name__}: {exc}")) from exc
