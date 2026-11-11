"""The box's hands: a :class:`~local_factory.agents.tools.ToolExecutor` whose every call is one
fresh container on the task workspace (docs/specs/phase-1b.md § box).

Host side. Each operation starts ``lf-entry`` (:mod:`.entry`) in a container built from
``ContainerSpec.from_config(config, ...)``: the workspace volume at ``/work`` (read-write, also
the working directory), the factory's own package read-only at ``/opt/local-factory`` and nothing
else from the host. What comes back is data only: an exit code, a bounded output tail, and the
bounded ``error:`` reason of a refused operation.

A :class:`~local_factory.sandbox.runtime.SandboxError` (the factory's side failed: docker missing,
image missing, create failed) propagates unchanged; it is never reported as the worker's fault.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

from local_factory.agents.tools import CommandResult, ToolError
from local_factory.config import SandboxConfig
from local_factory.sandbox.protocol import LF_ENTRY, bounded, error_reason
from local_factory.sandbox.runtime import ContainerResult, DockerRuntime
from local_factory.sandbox.spec import BOX_FACTORY, BOX_WORK, BindMount, ContainerSpec
from local_factory.sandbox.workspace import TaskWorkspace
from local_factory.verification.gates import OutputTail

__all__ = ["SandboxExecutor", "factory_mount", "output_text"]

OUT_OF_MEMORY = "killed: out of memory"


def factory_mount(factory_src: Path) -> BindMount:
    """The factory's ``local_factory`` package, read-only where ``lf-entry`` imports it from."""
    return BindMount(factory_src, f"{BOX_FACTORY}/local_factory", read_only=True)


def output_text(result: ContainerResult, limit: int, *notes: str) -> str:
    """The tail of ``result.stdout`` plus factory ``notes``, at most ``limit`` characters.

    Pass ``max_stdout_bytes=limit + 1`` to the runtime: then a tail it cut is recognised as cut,
    and a UTF-8 character it split at the front is dropped instead of decoded into garbage.
    """
    tail = OutputTail(limit)
    tail.feed(result.stdout)
    for note in notes:
        tail.note(note)
    return tail.text()


def _check_text(value: str, what: str) -> None:
    if "\x00" in value:
        raise ToolError(f"{what} contains a NUL byte")


class SandboxExecutor:
    """File and command operations on one task workspace, each in its own container."""

    def __init__(
        self,
        runtime: DockerRuntime,
        config: SandboxConfig,
        workspace: TaskWorkspace,
        *,
        factory_src: Path,
    ) -> None:
        self.runtime = runtime
        self.config = config
        self.workspace = workspace
        self.factory_src = factory_src

    # ----------------------------------------------------------------------- ToolExecutor

    async def write_file(self, path: str, content: str) -> None:
        try:
            data = content.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ToolError(bounded(f"the content cannot be encoded as UTF-8: {exc}")) from exc
        await self._file_op("write", path, stdin=data)

    async def read_file(self, path: str) -> str:
        limit = self.config.max_output_bytes
        result = await self._file_op("read", path, "--max-bytes", str(limit))
        return result.stdout.decode("utf-8", errors="replace")

    async def delete_file(self, path: str) -> None:
        await self._file_op("delete", path)

    async def run_command(self, argv: Sequence[str], *, timeout_s: float) -> CommandResult:
        if isinstance(argv, str):
            raise TypeError("argv must be a sequence of arguments, not a single string")
        command = list(argv)
        if not command:
            raise ToolError("empty argv")
        for arg in command:
            _check_text(arg, "an argument")
        if not (math.isfinite(timeout_s) and timeout_s > 0):
            raise ValueError(f"timeout_s must be a positive number, got {timeout_s!r}")
        limit = self.config.max_output_bytes
        result = await self.runtime.run(
            self._spec("exec", "--", *command),
            timeout_s=timeout_s,
            max_stdout_bytes=limit + 1,
            output="tail",
        )
        notes = [OUT_OF_MEMORY] if result.oom_killed else []
        if result.timed_out:
            notes.append(f"killed: timed out after {timeout_s:g}s")
            return CommandResult(None, output_text(result, limit, *notes), timed_out=True)
        return CommandResult(result.exit_code, output_text(result, limit, *notes))

    # ----------------------------------------------------------------------- internals

    def _spec(self, *args: str, interactive: bool = False) -> ContainerSpec:
        return ContainerSpec.from_config(
            self.config,
            argv=(LF_ENTRY, *args),
            binds=(factory_mount(self.factory_src),),
            volumes=(self.workspace.mount(),),
            workdir=BOX_WORK,
            interactive=interactive,
        )

    async def _file_op(
        self, command: str, path: str, *extra: str, stdin: bytes | None = None
    ) -> ContainerResult:
        _check_text(path, "the path")
        spec = self._spec(
            command, "--root", BOX_WORK, "--path", path, *extra, interactive=stdin is not None
        )
        timeout_s = self.config.command_timeout_s
        limit = self.config.max_output_bytes
        result = await self.runtime.run(
            spec, timeout_s=timeout_s, stdin=stdin, max_stdout_bytes=limit
        )
        if result.timed_out:
            raise ToolError(f"{command} {path!r}: timed out after {timeout_s:g}s")
        if result.output_exceeded:
            raise ToolError(f"{command} {path!r}: more than {limit} bytes of output")
        if result.exit_code != 0:
            raise ToolError(bounded(error_reason(result.stderr) or _failure(command, result)))
        return result


def _failure(command: str, result: ContainerResult) -> str:
    """What to say about a box that failed without an ``error:`` reason."""
    if result.oom_killed:
        return f"{command}: the box ran out of memory"
    stderr = bounded(result.stderr, 1_000)
    detail = f": {stderr}" if stderr else ""
    return f"{command}: the box exited with status {result.exit_code}{detail}"
