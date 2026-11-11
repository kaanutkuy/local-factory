"""The box facade: workspaces, tools, task export and gates, each in hardened containers
(docs/specs/phase-1b.md § box, PLAN §5.1-5.2).

Host side. Nothing here runs git or anything else on a worker tree: the worker's repository lives
only in its task workspace (a tmpfs volume) and every operation on it runs ``lf-entry`` inside a
fresh container. Results leave the box as data only (invariant 7): the exported task bundle,
streamed to a factory-owned file under a hard size cap, exit codes and bounded text.

Two kinds of failure are kept apart. A failure on the factory's side (docker, the image, the
factory's own bundles) raises :class:`~local_factory.sandbox.runtime.SandboxError`. Whatever a
worker does to its own workspace can make an export fail or produce garbage, but never raise:
it becomes an :class:`ExportOutcome`, and the garbage is caught by the bundle import.
"""

from __future__ import annotations

import os
import stat
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import local_factory
from local_factory.config import SandboxConfig
from local_factory.gitops.worker_side import check_sha
from local_factory.models import GateResult, GateStatus
from local_factory.sandbox.executor import (
    OUT_OF_MEMORY,
    SandboxExecutor,
    factory_mount,
    output_text,
)
from local_factory.sandbox.protocol import (
    EXIT_MATERIALIZE_FAILED,
    EXIT_NO_CHANGES,
    LF_ENTRY,
    MAX_DETAIL,
    bounded,
    error_reason,
)
from local_factory.sandbox.runtime import ContainerResult, DockerRuntime, SandboxError
from local_factory.sandbox.spec import BOX_IN, BOX_WORK, BindMount, ContainerSpec, TmpfsMount
from local_factory.sandbox.workspace import TaskWorkspace
from local_factory.verification.gates import DEFAULT_TAIL_BYTES, GateSpec, skipped_result

__all__ = ["ExportOutcome", "ExportStatus", "Sandbox"]

ExportStatus = Literal["exported", "no_changes", "worker_git_error", "bundle_too_large"]
BASE_BUNDLE = f"{BOX_IN}/base.bundle"
TREE_BUNDLE = f"{BOX_IN}/tree.bundle"
_BUNDLE_MODE = 0o644  # the box user is another uid: it can read only world-readable files
_OUT_MODE = 0o600
_PYCACHE = "__pycache__"


@dataclass(frozen=True)
class ExportOutcome:
    """How a task export ended. ``detail`` explains a rejection (at most 2 000 characters)."""

    status: ExportStatus
    detail: str = ""


class Sandbox:
    """Hardened, network-less containers for every worker-side action of the pipeline."""

    def __init__(
        self, runtime: DockerRuntime, config: SandboxConfig, *, factory_src: Path | None = None
    ) -> None:
        if config.uid == os.getuid() or config.gid == os.getgid():
            raise SandboxError(
                f"the box must not run as the host user: configured uid:gid "
                f"{config.uid}:{config.gid}, host {os.getuid()}:{os.getgid()}"
            )
        src = Path(local_factory.__file__).parent if factory_src is None else factory_src
        self.runtime = runtime
        self.config = config
        self.factory_src = src.resolve()
        _check_world_readable(self.factory_src)

    def check(self) -> str:
        """Docker is reachable and the image is present; returns the image id."""
        return self.runtime.check(self.config.image)

    def image_id(self) -> str:
        return self.runtime.image_id(self.config.image)

    def workspace(self) -> TaskWorkspace:
        """A new, unopened workspace: use ``with sandbox.workspace() as ws``."""
        return TaskWorkspace(self.runtime, self.config)

    def executor(self, ws: TaskWorkspace) -> SandboxExecutor:
        return SandboxExecutor(self.runtime, self.config, ws, factory_src=self.factory_src)

    # ----------------------------------------------------------------------- materialize

    async def materialize(self, ws: TaskWorkspace, *, bundle: Path, ref: str) -> None:
        """Check ``ref`` out of the factory's ``bundle`` into the (empty) workspace as ``task``.

        Any failure is the factory's (``SandboxError``): the bundle is its own.
        """
        source = _box_readable(bundle)
        spec = ContainerSpec.from_config(
            self.config,
            argv=_materialize_argv(BASE_BUNDLE, ref),
            binds=(BindMount(source, BASE_BUNDLE), factory_mount(self.factory_src)),
            volumes=(ws.mount(),),
        )
        timeout_s = self.config.command_timeout_s
        result = await self.runtime.run(
            spec, timeout_s=timeout_s, max_stdout_bytes=self.config.max_output_bytes
        )
        if result.timed_out or result.exit_code != 0:
            why = _failure_reason(result, timeout_s)
            raise SandboxError(bounded(f"materializing {ref} from {bundle} failed: {why}"))

    # ----------------------------------------------------------------------- export

    async def export(
        self, ws: TaskWorkspace, *, base_sha: str, message: str, out: Path
    ) -> ExportOutcome:
        """Commit everything in the workspace and export ``refs/heads/task ^base_sha`` to ``out``.

        ``out`` must not exist (``FileExistsError``); it is created ``0600`` and receives the
        bundle as it streams out of the box, capped at ``config.max_bundle_bytes``. It is deleted
        again unless the outcome is ``exported``. Worker-caused failures never raise.
        """
        check_sha(base_sha)
        if not message.strip():
            raise ValueError("the commit message must not be empty")
        spec = ContainerSpec.from_config(
            self.config,
            argv=(LF_ENTRY, "export", "--work", BOX_WORK, "--base", base_sha, "--message", message),
            binds=(factory_mount(self.factory_src),),
            volumes=(ws.mount(),),
            workdir=BOX_WORK,
        )
        fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, _OUT_MODE)
        outcome: ExportOutcome | None = None
        try:
            with os.fdopen(fd, "wb") as sink:
                os.fchmod(sink.fileno(), _OUT_MODE)  # exactly 0600, whatever the umask
                result = await self.runtime.run(
                    spec,
                    timeout_s=self.config.command_timeout_s,
                    stdout_sink=sink,
                    max_stdout_bytes=self.config.max_bundle_bytes,
                )
                sink.flush()
                size = os.fstat(sink.fileno()).st_size
            outcome = self._export_outcome(result, size)
            return outcome
        finally:
            if outcome is None or outcome.status != "exported":
                Path(out).unlink(missing_ok=True)

    def _export_outcome(self, result: ContainerResult, size: int) -> ExportOutcome:
        if result.output_exceeded:
            cap = self.config.max_bundle_bytes
            return ExportOutcome("bundle_too_large", f"the task bundle exceeded {cap} bytes")
        if not result.timed_out and not result.oom_killed:
            if result.exit_code == 0 and size > 0:
                return ExportOutcome("exported")
            if result.exit_code == EXIT_NO_CHANGES:
                return ExportOutcome("no_changes", "the task made no changes")
            if result.exit_code == 0:
                return ExportOutcome("worker_git_error", "the export produced an empty bundle")
        why = _failure_reason(result, self.config.command_timeout_s)
        return ExportOutcome("worker_git_error", bounded(f"export failed: {why}"))

    # ----------------------------------------------------------------------- gates

    async def run_gates(
        self,
        specs: Sequence[GateSpec],
        *,
        bundle: Path,
        ref: str,
        fail_fast: bool = False,
        tail_bytes: int = DEFAULT_TAIL_BYTES,
    ) -> list[GateResult]:
        """Run ``specs`` in order, each in a fresh container on a fresh checkout of ``ref``.

        Semantics as ``verification.run_gates``: exit 0 PASS, non-zero FAIL with the exit code,
        timeout ERROR; with ``fail_fast`` the gates after the first non-PASS are SKIPPED without
        running. A tree that cannot be materialized is ERROR, like a gate that cannot be started.
        """
        if tail_bytes < 0:
            raise ValueError(f"tail_bytes must be >= 0, got {tail_bytes}")
        source = _box_readable(bundle)
        results: list[GateResult] = []
        stopper: GateResult | None = None
        for spec in specs:
            if stopper is not None:
                results.append(skipped_result(spec, stopper))
                continue
            result = await self._run_gate(spec, source, ref, tail_bytes)
            results.append(result)
            if fail_fast and result.status is not GateStatus.PASS:
                stopper = result
        return results

    async def _run_gate(
        self, gate: GateSpec, source: Path, ref: str, tail_bytes: int
    ) -> GateResult:
        caches = tuple(BindMount(m.source, m.target) for m in self.config.cache_mounts)
        spec = ContainerSpec.from_config(
            self.config,
            argv=(*_materialize_argv(TREE_BUNDLE, ref), "--", "bash", "-c", gate.command),
            env=self.config.gate_env,
            binds=(BindMount(source, TREE_BUNDLE), factory_mount(self.factory_src), *caches),
            tmpfs=(TmpfsMount(BOX_WORK, self.config.work_size_mb, executable=True),),
        )
        result = await self.runtime.run(
            spec, timeout_s=gate.timeout_s, max_stdout_bytes=tail_bytes + 1, output="tail"
        )
        notes = [f"gate {gate.name!r} {OUT_OF_MEMORY}"] if result.oom_killed else []
        status: GateStatus = GateStatus.ERROR  # unless the exit status says otherwise
        exit_code: int | None = None
        reason = error_reason(result.stderr)
        if result.timed_out:
            notes.append(
                f"gate {gate.name!r} timed out after {gate.timeout_s:g}s; its container was killed"
            )
        elif result.exit_code == EXIT_MATERIALIZE_FAILED and reason:
            notes.append(f"gate {gate.name!r} could not be started: {reason}")
        elif result.exit_code is None:  # defensive: tail mode never stops for output size
            notes.append(f"gate {gate.name!r} ended without an exit status")
        elif result.exit_code == 0:
            status, exit_code = GateStatus.PASS, 0
        else:
            status, exit_code = GateStatus.FAIL, result.exit_code
        return GateResult(
            name=gate.name,
            status=status,
            exit_code=exit_code,
            duration_s=max(result.duration_s, 0.0),
            output_tail=output_text(result, tail_bytes, *notes),
        )


# --------------------------------------------------------------------------- helpers


def _materialize_argv(bundle: str, ref: str) -> tuple[str, ...]:
    """``lf-entry materialize``: ``ref`` from the bundle mounted at ``bundle``, into /work."""
    return (LF_ENTRY, "materialize", "--bundle", bundle, "--ref", ref, "--dest", BOX_WORK)


def _failure_reason(result: ContainerResult, timeout_s: float) -> str:
    """Why a box run failed: its ``error:`` reason, or what happened to it."""
    if result.timed_out:
        return f"timed out after {timeout_s:g}s"
    if result.oom_killed:
        return "the box ran out of memory"
    reason = error_reason(result.stderr)
    if reason:
        return reason
    stderr = bounded(result.stderr, MAX_DETAIL // 2)
    return f"exit status {result.exit_code}" + (f": {stderr}" if stderr else "")


def _box_readable(bundle: Path) -> Path:
    """The factory's own ``bundle``, made world-readable (0644) so the box user can read it."""
    path = Path(bundle).resolve()
    try:
        info = path.stat()
        if not stat.S_ISREG(info.st_mode):
            raise SandboxError(f"bundle {bundle} is not a regular file")
        path.chmod(_BUNDLE_MODE)
    except OSError as exc:
        raise SandboxError(f"bundle {bundle} cannot be handed to the box: {exc}") from exc
    return path


def _check_world_readable(root: Path) -> None:
    """``SandboxError`` unless the box user (another uid) can read all of ``root``
    (``__pycache__`` directories excepted, see :func:`_walk`)."""
    try:
        if not root.is_dir():
            raise SandboxError(f"factory source {root} is not a directory")
        unreadable = [str(p) for p in _walk(root) if not _world_readable(p)]
    except OSError as exc:
        raise SandboxError(f"factory source {root} cannot be checked: {exc}") from exc
    if unreadable:
        shown = ", ".join(unreadable[:5]) + (" ..." if len(unreadable) > 5 else "")
        raise SandboxError(
            f"factory source {root} is mounted into the box as is, but the box user cannot read "
            f"{len(unreadable)} entries ({shown}); fix with: chmod -R a+rX {root}"
        )


def _walk(root: Path) -> list[Path]:
    """``root`` and everything under it, except ``__pycache__`` directories: the box compiles
    from source when it cannot read a cache, and the host interpreter writes caches with the
    user's umask."""

    def fail(error: OSError) -> None:
        raise error

    entries = [root]
    for dirpath, dirnames, filenames in os.walk(root, onerror=fail):
        dirnames[:] = [name for name in dirnames if name != _PYCACHE]  # not descended into either
        entries += [Path(dirpath, name) for name in (*dirnames, *filenames)]
    return entries


def _world_readable(path: Path) -> bool:
    mode = path.lstat().st_mode
    if stat.S_ISLNK(mode):
        return True  # resolved inside the box, never on the host; its target is checked if inside
    needed = stat.S_IROTH | stat.S_IXOTH if stat.S_ISDIR(mode) else stat.S_IROTH
    return mode & needed == needed
