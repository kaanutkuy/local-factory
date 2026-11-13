"""The sandboxed slice: the host slice's flow with every worker-side step in the box
(docs/specs/phase-1b.md § integrator).

*Brain on the host, hands in the box.* The backend decides in the factory process; everything it
does to a worker (write, read and delete files, run commands), the worker's git, the task export
and every gate run in hardened, network-less containers as a user that is neither root nor the host
user, with only the task workspace writable. Host-side code never runs git in, and never writes,
a worker tree: the worker's repository exists only in a memory-backed workspace volume, and its
work leaves the box only as a bundle file (written by the factory, size-capped) that is validated
on import. After a run the work root holds only directories and the factory's own ``.bundle``
files.

The flow is :mod:`.flow`'s; this module is the *box site* of it.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from local_factory.agents import AgentBackend, WorkerHandle
from local_factory.config import FactoryConfig, ProjectConfig
from local_factory.gitops import BareRepo, BundleLimits, export_base_bundle
from local_factory.hashing import utc_now
from local_factory.ledger import Ledger
from local_factory.models import GateResult
from local_factory.pipeline.flow import (
    Refused,
    Site,
    SliceOutcome,
    Verification,
    Worker,
    run_slice,
)
from local_factory.sandbox import Sandbox
from local_factory.sandbox.spec import BOX_WORK
from local_factory.sandbox.workspace import TaskWorkspace
from local_factory.verification import GateSpec

__all__ = ["box_site", "run_sandboxed_slice"]

IMAGE_KEY = "sandbox"  # the key of the sandbox image in the evidence's image digests


async def run_sandboxed_slice(
    *,
    ledger: Ledger,
    repo: BareRepo,
    project: ProjectConfig,
    backend: AgentBackend,
    sandbox: Sandbox,
    ticket_id: str,
    work_root: Path,
    key: bytes,
    main_ref: str = "refs/heads/main",
    clock: Callable[[], datetime] = utc_now,
    factory_config: FactoryConfig | None = None,
) -> SliceOutcome:
    """Run one task of ``ticket_id`` from worker to signed candidate evidence, in the box.

    Same preconditions, flow, ledger records and outcome as ``run_host_slice``, with these
    differences. ``sandbox.check()`` (Docker reachable, image present) is a precondition like the
    rest: it runs before the first ledger change. Each worker lives in a task workspace that is
    closed when the worker's work is exported, and on every other path, exceptions and
    cancellation included. A backend run is limited to ``sandbox.config.run_timeout_s``: one that
    outlives it is an agent error (``timeout: the run exceeded <n>s``). A task bundle is capped at
    ``sandbox.config.max_bundle_bytes``. Gates run in the box on the verified commit's bundle, and
    the verifier works in a fresh workspace of its own, discarded afterwards. The evidence records
    the sandbox image id.

    A failure of the factory's side (Docker, the image) raises
    :class:`~local_factory.sandbox.runtime.SandboxError`; nothing a worker does can cause one.
    """
    return await run_slice(
        site=_BoxSite(sandbox),
        ledger=ledger,
        repo=repo,
        project=project,
        backend=backend,
        ticket_id=ticket_id,
        work_root=work_root,
        key=key,
        main_ref=main_ref,
        clock=clock,
        factory_config=factory_config,
    )


# --------------------------------------------------------------------------- the box site


def box_site(sandbox: Sandbox) -> Site:
    """The box site: every worker-side step and every gate runs in ``sandbox``'s containers."""
    return _BoxSite(sandbox)


class _BoxSite:
    """Workers and verification trees live in task workspaces; gates run in their own containers."""

    def __init__(self, sandbox: Sandbox) -> None:
        self._sandbox = sandbox

    @property
    def run_limit_s(self) -> float:
        return self._sandbox.config.run_timeout_s

    @property
    def limits(self) -> BundleLimits:
        return BundleLimits(max_bundle_bytes=self._sandbox.config.max_bundle_bytes)

    def check(self) -> None:
        self._sandbox.check()

    def image_digests(self) -> dict[str, str]:
        return {IMAGE_KEY: self._sandbox.image_id()}

    @asynccontextmanager
    async def worker(self, bundle: Path, ref: str) -> AsyncIterator[Worker]:
        with self._sandbox.workspace() as ws:
            await self._sandbox.materialize(ws, bundle=bundle, ref=ref)
            yield _BoxWorker(self._sandbox, ws)

    @asynccontextmanager
    async def verification(
        self, repo: BareRepo, sha: str, ticket_dir: Path, *, round_no: int
    ) -> AsyncIterator[Verification]:
        ticket_dir.mkdir(parents=True, exist_ok=True)
        bundle = _unused(ticket_dir / f"verify-{sha[:12]}-{round_no}.bundle")
        ref = export_base_bundle(repo, sha=sha, out=bundle)
        with self._sandbox.workspace() as ws:  # the verifier's own, never the implementer's
            await self._sandbox.materialize(ws, bundle=bundle, ref=ref)
            yield _BoxVerification(self._sandbox, ws, bundle, ref)


def _unused(path: Path) -> Path:
    """``path``, or a variant of it if something (a concurrent run's bundle) is already there."""
    if not path.exists():
        return path
    return path.with_name(f"{path.stem}-{uuid.uuid4().hex[:8]}{path.suffix}")


def _handle(sandbox: Sandbox, ws: TaskWorkspace) -> WorkerHandle:
    """The backend's view of a workspace: ``/work`` inside the box, acted on through the box."""
    return WorkerHandle(work_dir=Path(BOX_WORK), executor=sandbox.executor(ws))


class _BoxWorker:
    def __init__(self, sandbox: Sandbox, ws: TaskWorkspace) -> None:
        self._sandbox = sandbox
        self._ws = ws
        self.handle = _handle(sandbox, ws)

    async def submit(self, *, base: str, message: str, out: Path) -> Path | Refused:
        outcome = await self._sandbox.export(self._ws, base_sha=base, message=message, out=out)
        if outcome.status == "exported":
            return out
        return Refused(outcome.status, outcome.detail)  # the status is the rejection code


class _BoxVerification:
    def __init__(self, sandbox: Sandbox, ws: TaskWorkspace, bundle: Path, ref: str) -> None:
        self._sandbox = sandbox
        self._bundle = bundle
        self._ref = ref
        self.handle = _handle(sandbox, ws)

    async def gates(self, specs: Sequence[GateSpec]) -> list[GateResult]:
        return await self._sandbox.run_gates(specs, bundle=self._bundle, ref=self._ref)
