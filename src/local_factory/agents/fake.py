"""A scripted backend: executes predetermined actions in the worker directory, no LLM involved.

`FakeBackend` stands in for an agent. Like a real agent it acts inside the worker directory (which
in production lives in the sandbox), so its file actions are confined to `work_dir` and its
commands run with `cwd=work_dir`. It drives the vertical slice and the red-team suite without
spending tokens.

Every scripted failure becomes an `AgentResult` with `status="error"`; `run` never raises for
them. Only a malformed script (an item that is not an action) is rejected, at construction.
Error messages are persisted in the ledger, so each is bounded (`MAX_ERROR_CHARS`) and worded the
same on every supported Python version.
"""

from __future__ import annotations

import asyncio
import errno
import os
import shlex
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from subprocess import DEVNULL, PIPE

from pydantic import JsonValue, TypeAdapter, ValidationError

from local_factory.agents.backend import AgentResult, AgentTask, Capability, RoleSpec, WorkerHandle
from local_factory.models import Halt, HaltReason, Usage

DEFAULT_USAGE = Usage(input_tokens=100, output_tokens=50)
ERROR_TAIL_CHARS = 2_000  # command output kept in an error: the last N characters
DESCRIPTION_CHARS = 160  # an action's description (path or argv) in an error
MAX_ERROR_CHARS = 4_000  # hard cap on any error message
SYMLINK_LOOP = "path goes through a symlink loop"
_ELISION = "..."

_JSON_OBJECT: TypeAdapter[dict[str, JsonValue]] = TypeAdapter(dict[str, JsonValue])


@dataclass(frozen=True)
class WriteFile:
    """Create or overwrite `path` (relative to the worker) with UTF-8 `content`."""

    path: str
    content: str


@dataclass(frozen=True)
class DeleteFile:
    """Remove the file (or symlink) at `path`. A missing file or a directory is an error."""

    path: str


@dataclass(frozen=True)
class RunCommand:
    """Run `argv` with `cwd=work_dir`, no shell. A non-zero exit is an error."""

    argv: tuple[str, ...]


@dataclass(frozen=True)
class EmitOutput:
    """Record structured output. If several are emitted, the last one wins."""

    data: dict[str, JsonValue]


@dataclass(frozen=True)
class RaiseHalt:
    """Stop and report a halt (rate limit, billing, budget, max turns)."""

    reason: HaltReason
    resume_at: datetime | None = None


@dataclass(frozen=True)
class Fail:
    """Stop and report an error with `message`."""

    message: str


Action = WriteFile | DeleteFile | RunCommand | EmitOutput | RaiseHalt | Fail
_ACTION_TYPES = (WriteFile, DeleteFile, RunCommand, EmitOutput, RaiseHalt, Fail)
_SideEffect = WriteFile | DeleteFile | RunCommand


class _ActionError(Exception):
    """A scripted action could not be carried out."""


class FakeBackend:
    """Runs the script registered for `task.task_id`, else for `role.name`."""

    name: str = "fake"
    capabilities: frozenset[Capability] = frozenset({Capability.STRUCTURED_OUTPUT})

    def __init__(
        self, scripts: Mapping[str, Sequence[Action]], *, usage: Usage = DEFAULT_USAGE
    ) -> None:
        self._scripts = {key: _checked_script(key, actions) for key, actions in scripts.items()}
        self._usage = usage
        self.calls: list[tuple[RoleSpec, AgentTask]] = []

    async def run(self, role: RoleSpec, task: AgentTask, worker: WorkerHandle) -> AgentResult:
        self.calls.append((role, task))
        script = self._script_for(role, task)
        if script is None:
            return self._error(f"no script for task {task.task_id!r} or role {role.name!r}")
        root = Path(os.path.realpath(worker.work_dir))
        if not root.is_dir():
            return self._error(
                f"worker directory {worker.work_dir} does not exist or is not a directory"
            )
        emitted: dict[str, JsonValue] | None = None
        for index, action in enumerate(script, start=1):
            if isinstance(action, EmitOutput):
                emitted = action.data
            elif isinstance(action, RaiseHalt):
                halt = Halt(reason=action.reason, resume_at=action.resume_at)
                return AgentResult(status="halted", halt=halt, usage=self._usage)
            elif isinstance(action, Fail):
                return self._error(action.message or "scripted failure (no message)")
            else:
                problem = await _perform(action, root)
                if problem is not None:
                    return self._error(f"action {index} ({_describe(action)}): {problem}")
        return self._finish(role, emitted)

    def _script_for(self, role: RoleSpec, task: AgentTask) -> tuple[Action, ...] | None:
        if task.task_id is not None and task.task_id in self._scripts:
            return self._scripts[task.task_id]
        return self._scripts.get(role.name)

    def _finish(self, role: RoleSpec, emitted: dict[str, JsonValue] | None) -> AgentResult:
        schema = role.output_schema
        payload: object = emitted
        if schema is not None:
            if emitted is None:
                return self._error(
                    f"role {role.name!r} requires structured output ({schema.__name__}) "
                    "but the script emitted none"
                )
            try:
                payload = schema.model_validate(emitted).model_dump(mode="json", by_alias=True)
            except ValidationError as exc:
                return self._error(f"structured output does not match {schema.__name__}: {exc}")
        if payload is None:
            return AgentResult(status="ok", usage=self._usage)
        try:
            output = _JSON_OBJECT.validate_python(payload)
        except ValidationError as exc:
            return self._error(f"structured output is not a JSON object: {exc}")
        return AgentResult(status="ok", structured_output=output, usage=self._usage)

    def _error(self, message: str) -> AgentResult:
        return AgentResult(status="error", error=_clip(message, MAX_ERROR_CHARS), usage=self._usage)


def _checked_script(key: str, actions: Sequence[Action]) -> tuple[Action, ...]:
    script = tuple(actions)
    for action in script:
        if not isinstance(action, _ACTION_TYPES):
            raise TypeError(f"script {key!r}: {action!r} is not a FakeBackend action")
    return script


async def _perform(action: _SideEffect, root: Path) -> str | None:
    """Carry out a side-effecting action; return a problem description, or None on success."""
    try:
        if isinstance(action, WriteFile):
            _write_file(root, action.path, action.content)
        elif isinstance(action, DeleteFile):
            _delete_file(root, action.path)
        else:
            await _run_command(root, action.argv)
    except _ActionError as exc:
        return str(exc)
    except (OSError, ValueError) as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def _confine(root: Path, raw: str) -> tuple[Path, Path]:
    """Map a worker-relative path to `(entry, target)`, both inside `root`.

    `entry` is the directory entry `raw` names, with its parent's symlinks resolved; `target` is
    what it finally points to with every symlink resolved. Absolute paths, paths that do not name
    an entry, and paths whose parent or target resolve outside `root` (via `..` or symlinks) are
    rejected before anything is touched.
    """
    if "\x00" in raw:
        raise _ActionError("path contains a NUL byte")
    relative = Path(raw)
    if relative.is_absolute():
        raise _ActionError("absolute paths are not allowed; use a path relative to the worker")
    if relative.name in ("", ".."):
        raise _ActionError("path does not name a file")
    parent = _resolve((root / relative).parent)
    if not parent.is_relative_to(root):
        raise _ActionError("path escapes the worker directory")
    entry = parent / relative.name
    target = _resolve(entry)
    if target == root or not target.is_relative_to(root):
        raise _ActionError("path escapes the worker directory")
    return entry, target


def _resolve(path: Path) -> Path:
    """Resolve every symlink in `path`, identically on all supported Python versions.

    `Path.resolve()` raises `RuntimeError` on a symlink loop before 3.13 and quietly returns the
    looping path from 3.13 on. `os.path.realpath` never raises for a loop, so loops are detected
    here, explicitly, and reported with one message. A missing `path` is fine (it may be created).
    """
    resolved = Path(os.path.realpath(path))
    try:
        resolved.stat()
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise _ActionError(SYMLINK_LOOP) from exc
    return resolved


def _write_file(root: Path, raw: str, content: str) -> None:
    _, target = _confine(root, raw)
    data = content.encode("utf-8")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)


def _delete_file(root: Path, raw: str) -> None:
    entry, _ = _confine(root, raw)
    if not entry.is_symlink():
        if not entry.exists():
            raise _ActionError("no such file")
        if entry.is_dir():
            raise _ActionError("is a directory; DeleteFile removes files only")
    entry.unlink()


async def _run_command(root: Path, argv: tuple[str, ...]) -> None:
    if not argv:
        raise _ActionError("empty argv")
    process = await asyncio.create_subprocess_exec(
        *argv, cwd=root, stdin=DEVNULL, stdout=PIPE, stderr=PIPE
    )
    stdout, stderr = await process.communicate()
    if process.returncode != 0:
        tail = (stderr or stdout).decode("utf-8", errors="replace").strip()[-ERROR_TAIL_CHARS:]
        status = f"exit status {process.returncode}"
        raise _ActionError(f"{status}: {tail}" if tail else status)


def _describe(action: _SideEffect) -> str:
    if isinstance(action, RunCommand):
        return _clip(f"RunCommand {shlex.join(action.argv)}", DESCRIPTION_CHARS)
    return _clip(f"{type(action).__name__} {action.path!r}", DESCRIPTION_CHARS)


def _clip(text: str, limit: int) -> str:
    """`text` if it fits in `limit` characters, else its head and tail around an elision mark."""
    if len(text) <= limit:
        return text
    keep = limit - len(_ELISION)
    head = keep // 2
    return text[:head] + _ELISION + text[len(text) - (keep - head) :]
