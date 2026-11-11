"""An in-process stand-in for ``local_factory.sandbox.Sandbox``: no Docker.

It has the box's whole interface (check, workspaces, materialize, executor, export, gates) and the
box's semantics, but every workspace is a temporary directory outside the work root and the "box"
side runs in this process (through ``gitops.worker_side``, the code that runs inside the real box).
That makes the sandboxed slice's flow testable with real git and no containers: what the real box
adds (isolation) is covered by the container tests and the red-team suite.

It records what the slice asked of it, so tests can check the box was used as specified.
"""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import cast

from local_factory.agents import LocalExecutor
from local_factory.config import SandboxConfig
from local_factory.gitops import (
    GitError,
    commit_all,
    export_task_bundle,
    has_task_commits,
    init_from_bundle,
)
from local_factory.models import GateResult
from local_factory.sandbox import ExportOutcome, Sandbox
from local_factory.sandbox.runtime import SandboxError
from local_factory.verification import GateSpec, run_gates

IMAGE_ID = "sha256:" + "d" * 64


class FakeWorkspace:
    """A workspace that is a directory (created by materialize) until it is closed."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.opened = False
        self.closed = False

    @property
    def volume(self) -> str:
        return f"fake-volume-{self.root.name}"

    def __enter__(self) -> FakeWorkspace:
        self.opened = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.closed = True
        shutil.rmtree(self.root, ignore_errors=True)


@dataclass
class FakeBox:
    """The recording, in-process ``Sandbox``. ``sandbox()`` is the typed view the slice takes."""

    scratch: Path
    config: SandboxConfig = field(default_factory=SandboxConfig)
    check_error: str | None = None  # make check() fail with this SandboxError message
    export_outcome: ExportOutcome | None = None  # answer every export with this, run nothing
    materialized: list[tuple[Path, str]] = field(default_factory=list)
    gated: list[tuple[Path, str]] = field(default_factory=list)
    exports: list[tuple[str, str, Path]] = field(default_factory=list)  # base, message, out
    checks: int = 0
    workspaces: list[FakeWorkspace] = field(default_factory=list)

    def sandbox(self) -> Sandbox:
        return cast("Sandbox", self)

    def check(self) -> str:
        self.checks += 1
        if self.check_error is not None:
            raise SandboxError(self.check_error)
        return IMAGE_ID

    def image_id(self) -> str:
        return IMAGE_ID

    def workspace(self) -> FakeWorkspace:
        ws = FakeWorkspace(self.scratch / f"workspace-{len(self.workspaces)}")
        self.workspaces.append(ws)
        return ws

    def executor(self, ws: FakeWorkspace) -> LocalExecutor:
        return LocalExecutor(ws.root)

    async def materialize(self, ws: FakeWorkspace, *, bundle: Path, ref: str) -> None:
        self.materialized.append((bundle, ref))
        init_from_bundle(bundle, ref, ws.root)

    async def export(
        self, ws: FakeWorkspace, *, base_sha: str, message: str, out: Path
    ) -> ExportOutcome:
        self.exports.append((base_sha, message, out))
        if self.export_outcome is not None:
            return self.export_outcome
        try:
            commit_all(ws.root, message)
            if not has_task_commits(ws.root, base_sha=base_sha):
                return ExportOutcome("no_changes", "the task made no changes")
            export_task_bundle(ws.root, out, base_sha=base_sha)
        except GitError as broken:
            return ExportOutcome("worker_git_error", str(broken))
        return ExportOutcome("exported")

    async def run_gates(
        self, specs: Sequence[GateSpec], *, bundle: Path, ref: str
    ) -> list[GateResult]:
        self.gated.append((bundle, ref))
        tree = self.scratch / f"gates-{len(self.gated)}"
        init_from_bundle(bundle, ref, tree)
        try:
            return run_gates(specs, cwd=tree)
        finally:
            shutil.rmtree(tree, ignore_errors=True)

    def open_workspaces(self) -> list[FakeWorkspace]:
        """Workspaces that were opened and never closed: there must be none after a slice."""
        return [ws for ws in self.workspaces if ws.opened and not ws.closed]
