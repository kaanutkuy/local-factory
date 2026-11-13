"""The Phase 1a host vertical slice (docs/specs/phase-1a-slice.md).

The flow itself lives in :mod:`.flow` and is shared with the sandboxed slice; this module is the
*host site* of it: the worker is a plain directory on the host, driven in-process through
:class:`~local_factory.agents.local.LocalExecutor`, and gates run as host subprocesses. **It is
not a security boundary**: a hostile worker has the host user's privileges here. It exists for
fast tests and for development; real work goes through
:func:`local_factory.pipeline.sandboxed_slice.run_sandboxed_slice`.
"""

from __future__ import annotations

import tempfile
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from local_factory.agents import AgentBackend, WorkerHandle
from local_factory.config import FactoryConfig, ProjectConfig
from local_factory.gitops import (
    BareRepo,
    BundleLimits,
    GitError,
    commit_all,
    export_task_bundle,
    has_task_commits,
    init_from_bundle,
)
from local_factory.hashing import utc_now
from local_factory.ledger import Ledger
from local_factory.models import GateResult
from local_factory.pipeline.flow import (
    ACTOR,
    FAKE_MODEL,
    IMPLEMENTER_ROLE,
    VERIFIER_ROLE,
    Refused,
    Site,
    SliceError,
    SliceOutcome,
    SliceStatus,
    Verification,
    VerifierReport,
    Worker,
    reconcile_criteria,
    run_slice,
)
from local_factory.verification import GateSpec, run_gates

__all__ = [
    "ACTOR",
    "FAKE_MODEL",
    "IMPLEMENTER_ROLE",
    "VERIFIER_ROLE",
    "SliceError",
    "SliceOutcome",
    "SliceStatus",
    "VerifierReport",
    "host_site",
    "reconcile_criteria",
    "run_host_slice",
]


async def run_host_slice(
    *,
    ledger: Ledger,
    repo: BareRepo,
    project: ProjectConfig,
    backend: AgentBackend,
    ticket_id: str,
    work_root: Path,
    key: bytes,
    main_ref: str = "refs/heads/main",
    clock: Callable[[], datetime] = utc_now,
    factory_config: FactoryConfig | None = None,
) -> SliceOutcome:
    """Run one task of ``ticket_id`` from worker to signed candidate evidence, on the host.

    **Not a security boundary.** The worker is a plain directory under ``work_root`` that the
    backend reaches through ``LocalExecutor`` (``WorkerHandle.tools()``), and the harness runs git
    in it: a hostile agent has the privileges of the host user. Use it for in-process tests and
    development only; ``run_sandboxed_slice`` runs the same flow with every worker-side step in the
    box.

    The ticket must be APPROVED (planned and moved to BUILDING here) or already BUILDING (a further
    round: a new task based on the ticket head). Everything is checked before the first change:
    the ticket's status and frozen contract, the key, the models (``factory_config`` resolves
    them for any backend but the fake one) and ``main_ref``.

    Raises :class:`SliceError` for unmet preconditions and for evidence that fails its binding
    check (a factory bug). Rejections, failed verification, halts and agent errors are outcomes.
    """
    return await run_slice(
        site=_HostSite(),
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


# --------------------------------------------------------------------------- the host site


def host_site() -> Site:
    """The host site: workers and verification trees are plain directories under the work root.

    **Not a security boundary** (see :func:`run_host_slice`): for in-process tests and development.
    """
    return _HostSite()


class _HostSite:
    """Workers and verification trees are plain directories under the work root."""

    run_limit_s = None  # the host slice puts no wall-clock limit on a backend run
    limits = BundleLimits()

    def check(self) -> None:
        """Nothing beyond the ledger and the repository: the host is always there."""

    def image_digests(self) -> dict[str, str]:
        return {}

    @asynccontextmanager
    async def worker(self, bundle: Path, ref: str) -> AsyncIterator[Worker]:
        work = bundle.parent / "work"  # <ticket>/task-<n>/work, next to the base bundle
        init_from_bundle(bundle, ref, work)  # a failure here is the factory's: it propagates
        yield _HostWorker(work)

    @asynccontextmanager
    async def verification(
        self,
        repo: BareRepo,
        sha: str,
        ticket_dir: Path,
        *,
        round_no: int,
    ) -> AsyncIterator[Verification]:
        ticket_dir.mkdir(parents=True, exist_ok=True)
        tree = Path(tempfile.mkdtemp(prefix=f"verify-{sha[:12]}-{round_no}-", dir=ticket_dir))
        repo.export_tree(sha, tree)
        yield _HostVerification(tree)


class _HostWorker:
    def __init__(self, work: Path) -> None:
        self._work = work

    @property
    def handle(self) -> WorkerHandle:
        return WorkerHandle(work_dir=self._work)

    async def submit(self, *, base: str, message: str, out: Path) -> Path | Refused:
        try:
            commit_all(self._work, message)
            has_commits = has_task_commits(self._work, base_sha=base)
        except GitError as broken:  # the agent had the repository: it broke it
            return Refused("worker_git_error", str(broken))
        if not has_commits:
            return Refused("no_changes", f"no commits beyond {base}")
        export_task_bundle(self._work, out, base_sha=base)  # failing now is a factory bug: raise
        return out


class _HostVerification:
    def __init__(self, tree: Path) -> None:
        self._tree = tree

    @property
    def handle(self) -> WorkerHandle:
        return WorkerHandle(work_dir=self._tree)

    async def gates(self, specs: Sequence[GateSpec]) -> list[GateResult]:
        return run_gates(specs, cwd=self._tree)
