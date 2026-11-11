"""A scripted backend: executes predetermined actions on its worker, no LLM involved.

`FakeBackend` stands in for an agent. Like a real agent its "brain" (the script) runs in the
factory process and every side effect - writing and deleting files, running commands - goes
through the worker's tools (`WorkerHandle.tools()`), never directly: inside the box in
production, on a plain directory (`LocalExecutor`) in in-process tests. With an executor set it
does not touch `work_dir` on the host at all. It drives the vertical slice and the red-team suite
without spending tokens.

Every scripted failure becomes an `AgentResult` with `status="error"`; `run` never raises for
them. A malformed script (an item that is not an action) is rejected at construction. Failures of
the factory itself (e.g. the sandbox runtime) are not the worker's and propagate. Error messages
are persisted in the ledger, so each is bounded (`MAX_ERROR_CHARS`) and worded the same on every
supported Python version.
"""

from __future__ import annotations

import math
import os
import shlex
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from pydantic import JsonValue, TypeAdapter, ValidationError

from local_factory.agents.backend import AgentResult, AgentTask, Capability, RoleSpec, WorkerHandle
from local_factory.agents.local import clip
from local_factory.agents.tools import CommandResult, ToolError, ToolExecutor
from local_factory.models import Halt, HaltReason, Usage

DEFAULT_USAGE = Usage(input_tokens=100, output_tokens=50)
DEFAULT_COMMAND_TIMEOUT_S = 300.0  # a RunCommand without its own timeout_s
ERROR_TAIL_CHARS = 2_000  # command output kept in an error: the last N characters
DESCRIPTION_CHARS = 160  # an action's description (path or argv) in an error
MAX_ERROR_CHARS = 4_000  # hard cap on any error message

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
    """Run `argv` in the worker root, no shell; killed after `timeout_s` seconds (default: the
    backend's `command_timeout_s`). A non-zero exit or a timeout is an error."""

    argv: tuple[str, ...]
    timeout_s: float | None = None

    def __post_init__(self) -> None:
        if self.timeout_s is not None:
            _check_timeout("RunCommand.timeout_s", self.timeout_s)


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


class FakeBackend:
    """Runs the script registered for `task.task_id`, else for `role.name`."""

    name: str = "fake"
    capabilities: frozenset[Capability] = frozenset({Capability.STRUCTURED_OUTPUT})

    def __init__(
        self,
        scripts: Mapping[str, Sequence[Action]],
        *,
        usage: Usage = DEFAULT_USAGE,
        command_timeout_s: float = DEFAULT_COMMAND_TIMEOUT_S,
    ) -> None:
        _check_timeout("command_timeout_s", command_timeout_s)
        self._scripts = {key: _checked_script(key, actions) for key, actions in scripts.items()}
        self._usage = usage
        self._command_timeout_s = command_timeout_s
        self.calls: list[tuple[RoleSpec, AgentTask]] = []

    async def run(self, role: RoleSpec, task: AgentTask, worker: WorkerHandle) -> AgentResult:
        self.calls.append((role, task))
        script = self._script_for(role, task)
        if script is None:
            return self._error(f"no script for task {task.task_id!r} or role {role.name!r}")
        if worker.executor is None and not _is_directory(worker.work_dir):
            # A local worker lives on the host; a sandboxed one is never looked at from here.
            return self._error(
                f"worker directory {worker.work_dir} does not exist or is not a directory"
            )
        tools = worker.tools()
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
                problem = await self._perform(action, tools)
                if problem is not None:
                    return self._error(f"action {index} ({_describe(action)}): {problem}")
        return self._finish(role, emitted)

    async def _perform(self, action: _SideEffect, tools: ToolExecutor) -> str | None:
        """Carry out a side-effecting action; return a problem description, or None on success."""
        try:
            if isinstance(action, WriteFile):
                await tools.write_file(action.path, action.content)
            elif isinstance(action, DeleteFile):
                await tools.delete_file(action.path)
            else:
                timeout_s = action.timeout_s or self._command_timeout_s  # timeout_s is never 0
                result = await tools.run_command(action.argv, timeout_s=timeout_s)
                return _command_problem(result, timeout_s)
        except ToolError as exc:
            return str(exc)
        return None

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
        return AgentResult(status="error", error=clip(message, MAX_ERROR_CHARS), usage=self._usage)


def _checked_script(key: str, actions: Sequence[Action]) -> tuple[Action, ...]:
    script = tuple(actions)
    for action in script:
        if not isinstance(action, _ACTION_TYPES):
            raise TypeError(f"script {key!r}: {action!r} is not a FakeBackend action")
    return script


def _check_timeout(name: str, value: float) -> None:
    if not (math.isfinite(value) and value > 0):
        raise ValueError(f"{name} must be a positive number of seconds, got {value!r}")


def _is_directory(path: Path) -> bool:
    return Path(os.path.realpath(path)).is_dir()


def _command_problem(result: CommandResult, timeout_s: float) -> str | None:
    """None if the command succeeded, else its status and the tail of its output."""
    if result.ok:
        return None
    if result.timed_out:
        status = f"timed out after {timeout_s:g}s and was killed"
    else:
        status = f"exit status {result.exit_code}"
    tail = result.output.strip()[-ERROR_TAIL_CHARS:]
    return f"{status}: {tail}" if tail else status


def _describe(action: _SideEffect) -> str:
    if isinstance(action, RunCommand):
        return clip(f"RunCommand {shlex.join(action.argv)}", DESCRIPTION_CHARS)
    return clip(f"{type(action).__name__} {action.path!r}", DESCRIPTION_CHARS)
