"""The hands of an agent: the tool interface a backend acts through. Orchestrator-owned.

A backend's "brain" (FakeBackend's script, later a model's tool loop) runs in the factory process;
everything it *does* to a worker goes through a :class:`ToolExecutor`. In production that is
``local_factory.sandbox.SandboxExecutor``: every call runs inside the box (no network, not the host
user, only the task workspace writable). ``local_factory.agents.local.LocalExecutor`` does the
same on a plain directory, for in-process tests and the Phase 1a host slice only; it is not a
security boundary.

Paths are always relative to the worker root and use ``/`` separators. Every failure to carry out
an operation raises :class:`ToolError` with a bounded, human-readable message; a command that runs
and exits non-zero is *not* an error, it is a :class:`CommandResult`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

__all__ = ["CommandResult", "ToolError", "ToolExecutor"]


class ToolError(Exception):
    """A tool operation could not be carried out (refused path, missing file, box failure)."""


@dataclass(frozen=True)
class CommandResult:
    """How one command ended. ``exit_code`` is None when it was killed for its timeout."""

    exit_code: int | None
    output: str  # the tail of interleaved stdout + stderr, bounded by the executor
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


@runtime_checkable
class ToolExecutor(Protocol):
    """File and command operations on one worker, confined to its root."""

    async def write_file(self, path: str, content: str) -> None:
        """Create or overwrite ``path`` with UTF-8 ``content`` (parent dirs are created)."""
        ...

    async def read_file(self, path: str) -> str:
        """The UTF-8 text of ``path`` (invalid bytes replaced). Too large or missing: ToolError."""
        ...

    async def delete_file(self, path: str) -> None:
        """Remove the file or symlink at ``path``. Missing, or a directory: ToolError."""
        ...

    async def run_command(self, argv: Sequence[str], *, timeout_s: float) -> CommandResult:
        """Run ``argv`` (no shell) with the worker root as cwd; kill it after ``timeout_s``."""
        ...
