"""The slice flow shared by the host slice and the sandboxed slice (docs/specs/phase-1a-slice.md,
docs/specs/phase-1b.md § integrator).

One deterministic pass over an approved ticket: one task, one worker, one harness commit, one
bundle import, the diff policy, integration into the ticket branch, verification of the ticket
head, a candidate squashed onto ``main``, verification of the candidate, and evidence bound to the
candidate SHA.

The flow is written once. *Where* the worker-side steps run is the only thing the two slices
disagree on, so they plug a :class:`Site` into it: the worker's workspace and what happens in it
(start, commit and export), and the fresh tree a verification runs gates and the verifier in. The
host slice's site works in plain directories (:mod:`.host_slice`); the sandboxed slice's site runs
every one of those steps in the box (:mod:`.sandboxed_slice`).

- Every state change goes through the ledger, the source of truth (invariant 1); this module
  keeps no workflow state of its own. Every rejected or halted task records its reason and detail
  (violations, conflict paths, error text) there, so the ledger alone explains each outcome.
- Host-side code never runs git inside a worker's directory (invariant 7): that is the site's
  business, and only :mod:`local_factory.gitops.worker_side` does it (inside the sandbox from
  Phase 1b on). The worker's output comes back only as a bundle file, validated on import.
- The slice never moves ``main`` and never pushes anywhere.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, JsonValue, ValidationError

from local_factory import __version__
from local_factory.agents import (
    AgentBackend,
    AgentResult,
    AgentTask,
    FakeBackend,
    RoleSpec,
    WorkerHandle,
)
from local_factory.config import FactoryConfig, ProjectConfig
from local_factory.evidence import build_evidence, check_binding, evidence_hash, sign
from local_factory.gitops import (
    BareRepo,
    BundleLimits,
    BundleRejected,
    export_base_bundle,
    import_task_bundle,
    merge_into,
    squash_onto,
)
from local_factory.hashing import content_hash
from local_factory.ledger import Ledger
from local_factory.models import (
    Contract,
    CriterionResult,
    CriterionStatus,
    Evidence,
    GateResult,
    Halt,
    RunRecord,
    Task,
    TaskStatus,
    Ticket,
    TicketStatus,
    Tier,
    Verdict,
    Violation,
)
from local_factory.policy import DiffPolicy, check_changes
from local_factory.verification import GateSpec, gates_from_project, make_verdict

__all__ = [
    "ACTOR",
    "FAKE_MODEL",
    "IMPLEMENTER_ROLE",
    "VERIFIER_ROLE",
    "Refused",
    "Site",
    "SliceError",
    "SliceOutcome",
    "SliceStatus",
    "Verification",
    "VerifierReport",
    "Worker",
    "reconcile_criteria",
    "run_limited",
    "run_slice",
]

ACTOR = "orchestrator"  # the actor of every ledger change the slice makes
FAKE_MODEL = "fake"  # the model recorded for runs of the scripted FakeBackend
UNREPORTED = "the verifier did not report on this criterion"
_STARTABLE = frozenset({TicketStatus.APPROVED, TicketStatus.BUILDING})


class VerifierReport(BaseModel):
    """The verifier role's structured output: one result per acceptance criterion."""

    criteria: list[CriterionResult]


IMPLEMENTER_ROLE = RoleSpec(
    name="implementer",
    tier=Tier.STANDARD,
    prompt=(
        "You are the implementer. Make the change the task describes in your working directory: "
        "the code plus its unit and integration tests. Never modify protected files (such as "
        "factory.toml or the acceptance tests) and never skip or suppress a check; if one seems "
        "wrong, say so in your task report. Everything you leave in the working tree is committed "
        "for you."
    ),
)
VERIFIER_ROLE = RoleSpec(
    name="verifier",
    tier=Tier.FRONTIER,
    output_schema=VerifierReport,
    prompt=(
        "You are the verifier. Your working directory holds the exact tree under verification. "
        "For every acceptance criterion in the task, decide whether this tree satisfies it and "
        "report it exactly once as pass, fail or inconclusive, with evidence (and feedback for "
        "anything that does not pass). Change nothing."
    ),
)

SliceStatus = Literal[
    "candidate_verified", "rejected", "failed_verification", "halted", "agent_error"
]


class SliceError(Exception):
    """The slice cannot start (preconditions), or it caught a factory bug (unbound evidence)."""


@dataclass(frozen=True)
class SliceOutcome:
    status: SliceStatus
    ticket: Ticket  # fresh snapshot at the end
    task: Task
    # a bundle rejection code | "policy" | "conflict" | "candidate_conflict" | "no_changes" |
    # "worker_git_error" | "bundle_too_large"
    rejection: str | None = None
    violations: tuple[Violation, ...] = ()
    verdict: Verdict | None = None  # the last verdict this run recorded
    candidate_sha: str | None = None
    evidence: Evidence | None = None
    signature: str | None = None


# --------------------------------------------------------------------------- the site


@dataclass(frozen=True)
class Refused:
    """A worker's submission that never became a bundle: the rejection code and its error text."""

    code: str
    error: str


class Worker(Protocol):
    """One task's worker: where the implementer acts, and how its work leaves as a bundle."""

    @property
    def handle(self) -> WorkerHandle:
        """What the backend gets to act on the worker with."""
        ...

    async def submit(self, *, base: str, message: str, out: Path) -> Path | Refused:
        """Commit whatever the worker left and write the task bundle to ``out``.

        Returns ``out``, or why there is no bundle: ``no_changes``, ``worker_git_error`` for a
        worker that broke its own repository, or ``bundle_too_large``. Anything else is a factory
        failure and raises.
        """
        ...


class Verification(Protocol):
    """A fresh tree of one commit under verification, for the gates and the verifier."""

    @property
    def handle(self) -> WorkerHandle:
        """Where the verifier works: a tree of its own, discarded after the verification."""
        ...

    async def gates(self, specs: Sequence[GateSpec]) -> list[GateResult]:
        """Run the project's gates on this commit's tree."""
        ...


class Site(Protocol):
    """Where the worker-side steps of the slice run."""

    @property
    def run_limit_s(self) -> float | None:
        """Wall-clock limit of one backend run, or None for no limit."""
        ...

    @property
    def limits(self) -> BundleLimits:
        """What a task bundle from a worker here may be."""
        ...

    def check(self) -> None:
        """Whatever must hold before the slice changes anything (raises if it does not)."""
        ...

    def image_digests(self) -> dict[str, str]:
        """The images the work ran in, for the evidence."""
        ...

    def worker(self, bundle: Path, ref: str) -> AbstractAsyncContextManager[Worker]:
        """A worker on ``ref`` of the factory's base ``bundle``; released when the block ends,
        however it ends. The bundle sits in the task's directory under the work root, which is
        where a site that needs a place of its own for the worker puts it."""
        ...

    def verification(
        self, repo: BareRepo, sha: str, ticket_dir: Path, *, round_no: int
    ) -> AbstractAsyncContextManager[Verification]:
        """A fresh tree of ``sha`` from ``repo``; discarded when the block ends."""
        ...


# --------------------------------------------------------------------------- entry point


async def run_slice(
    *,
    site: Site,
    ledger: Ledger,
    repo: BareRepo,
    project: ProjectConfig,
    backend: AgentBackend,
    ticket_id: str,
    work_root: Path,
    key: bytes,
    main_ref: str,
    clock: Callable[[], datetime],
    factory_config: FactoryConfig | None,
) -> SliceOutcome:
    """Run one task of ``ticket_id`` from worker to signed candidate evidence, on ``site``.

    The ticket must be APPROVED (planned and moved to BUILDING here) or already BUILDING (a further
    round: a new task based on the ticket head). Everything is checked before the first change:
    the ticket's status and frozen contract, the key, the models (``factory_config`` resolves
    them for any backend but the fake one), ``main_ref`` and the site itself.

    Raises :class:`SliceError` for unmet preconditions and for evidence that fails its binding
    check (a factory bug). Rejections, failed verification, halts and agent errors are outcomes.
    """
    ticket = ledger.get_ticket(ticket_id)
    if ticket.status not in _STARTABLE:
        raise SliceError(f"{ticket.id} is {ticket.status}; the slice needs approved or building")
    if not key:
        raise ValueError("the evidence signing key must not be empty")
    run = _Slice(
        site=site,
        ledger=ledger,
        repo=repo,
        project=project,
        backend=backend,
        ticket_id=ticket.id,
        frozen=_frozen(ticket),
        work_root=work_root,
        key=key,
        main_ref=main_ref,
        clock=clock,
        models=_models(backend.name, factory_config),
    )
    main = repo.rev_parse(main_ref)
    site.check()
    return await run.execute(main, plan=ticket.status is TicketStatus.APPROVED)


def reconcile_criteria(
    reported: Sequence[CriterionResult], contract: Contract
) -> tuple[CriterionResult, ...]:
    """The verifier's results, one per contract criterion, in contract order (fail closed).

    A criterion the verifier did not report is INCONCLUSIVE; a criterion reported more than once
    takes its first non-passing result; results for ids outside the contract are dropped.
    """
    chosen: dict[str, CriterionResult] = {}
    for result in reported:
        current = chosen.get(result.id)
        if current is None or (
            current.status is CriterionStatus.PASS and result.status is not CriterionStatus.PASS
        ):
            chosen[result.id] = result
    return tuple(
        chosen.get(c.id)
        or CriterionResult(id=c.id, status=CriterionStatus.INCONCLUSIVE, feedback=UNREPORTED)
        for c in contract.criteria
    )


async def run_limited(run: Awaitable[AgentResult], limit_s: float | None) -> AgentResult:
    """``run`` under a wall-clock limit (None: unlimited). One that outlives it is cancelled and
    becomes a backend error, recorded like any other (``timeout: the run exceeded <n>s``)."""
    if limit_s is None:
        return await run
    try:
        return await asyncio.wait_for(run, limit_s)
    except TimeoutError:
        return AgentResult(status="error", error=f"timeout: the run exceeded {limit_s:g}s")


# --------------------------------------------------------------------------- internals


@dataclass(frozen=True)
class _Frozen:
    """What approval froze: the contract, its hash and the oracle version."""

    contract: Contract
    contract_hash: str
    oracle_version: str


def _frozen(ticket: Ticket) -> _Frozen:
    if ticket.contract is None or ticket.contract_hash is None or ticket.oracle_version is None:
        raise SliceError(f"{ticket.id} has no frozen contract and oracle version")
    return _Frozen(ticket.contract, ticket.contract_hash, ticket.oracle_version)


def _models(backend_name: str, config: FactoryConfig | None) -> dict[str, str]:
    """role -> model id: ``fake`` for the fake backend, else the configured model for the tier."""
    roles = (IMPLEMENTER_ROLE, VERIFIER_ROLE)
    if backend_name == FakeBackend.name:
        return {role.name: FAKE_MODEL for role in roles}
    if config is None:
        raise SliceError(f"backend {backend_name!r} needs a factory config to resolve its models")
    return {role.name: config.model_for(backend_name, role.tier) for role in roles}


def _criteria_context(contract: Contract) -> list[JsonValue]:
    return [{"id": c.id, "ears": c.ears} for c in contract.criteria]


def _violations_detail(violations: Sequence[Violation]) -> list[JsonValue]:
    return [v.model_dump(mode="json") for v in violations]


def _criteria_text(contract: Contract) -> str:
    return "\n".join(f"- {c.id}: {c.ears}" for c in contract.criteria)


@dataclass(frozen=True)
class _Slice:
    """The collaborators of one slice run. Holds no workflow state: that lives in the ledger."""

    site: Site
    ledger: Ledger
    repo: BareRepo
    project: ProjectConfig
    backend: AgentBackend
    ticket_id: str
    frozen: _Frozen
    work_root: Path
    key: bytes
    main_ref: str
    clock: Callable[[], datetime]
    models: dict[str, str]

    @property
    def ticket_ref(self) -> str:
        return f"refs/heads/factory/{self.ticket_id}"

    async def execute(self, main: str, *, plan: bool) -> SliceOutcome:
        task = self._plan(plan=plan)  # 1
        base = self._task_base(main)  # 2
        done = await self._work(task, base)  # 3 - 6
        if isinstance(done, AgentResult):
            return self._ended(task.id, done)
        if isinstance(done, Refused):
            return self._reject(task.id, done.code, {"error": done.error})
        self._set_task(task.id, TaskStatus.SUBMITTED)
        integrated = self._integrate(task.id, done, base)  # 7, 8, 9
        if isinstance(integrated, SliceOutcome):
            return integrated
        return await self._converge(task.id, integrated)  # 10 - 13

    # ------------------------------------------------------------------ steps 1 - 9

    def _plan(self, *, plan: bool) -> Task:
        contract = self.frozen.contract
        if plan:
            self._transition(TicketStatus.PLANNED)
        task = self.ledger.create_task(
            self.ticket_id,
            goal=contract.summary,
            role=IMPLEMENTER_ROLE.name,
            tier=IMPLEMENTER_ROLE.tier,
            criteria_served=[c.id for c in contract.criteria],
            actor=ACTOR,
        )
        if plan:
            self._transition(TicketStatus.BUILDING)
        return task

    def _task_base(self, main: str) -> str:
        """The ticket head, after creating the ticket branch at ``main`` if it is missing."""
        if not self.repo.ref_exists(self.ticket_ref):
            absent = "0" * len(main)  # git's null object id: the ref must not exist yet
            self.repo.update_ref(self.ticket_ref, main, old_sha=absent)
        return self.repo.rev_parse(self.ticket_ref)

    async def _work(self, task: Task, base: str) -> AgentResult | Refused | Path:
        """Steps 3 - 6: the worker runs and submits. Its workspace is released before this returns.

        The result: the agent's halt or error, why there is no bundle, or the task bundle.
        """
        task_dir = self.work_root.joinpath(*task.id.split("/"))  # <ticket>/task-<n>
        task_dir.mkdir(parents=True)
        bundle = task_dir / "base.bundle"
        ref = export_base_bundle(self.repo, sha=base, out=bundle)
        async with self.site.worker(bundle, ref) as worker:
            self.ledger.update_task(task.id, actor=ACTOR, status=TaskStatus.RUNNING, base_sha=base)
            result = await self._implement(task, base, worker.handle)
            if result.halt is not None or result.status == "error":
                return result
            return await worker.submit(
                base=base, message=f"{task.id}: {task.goal}", out=task_dir / "task.bundle"
            )

    async def _implement(self, task: Task, base: str, handle: WorkerHandle) -> AgentResult:
        contract = self.frozen.contract
        agent_task = AgentTask(
            ticket_id=self.ticket_id,
            task_id=task.id,
            instructions=f"{task.goal}\n\nAcceptance criteria:\n{_criteria_text(contract)}",
            context={"base_sha": base, "criteria": _criteria_context(contract)},
        )
        return await self._run_role(IMPLEMENTER_ROLE, agent_task, handle)

    def _ended(self, task_id: str, result: AgentResult) -> SliceOutcome:
        """The implementer halted or failed: its task ends here."""
        if result.halt is not None:
            halt = result.halt
            self.ledger.halt(self.ticket_id, halt, actor=ACTOR)
            detail = {"halt": halt.model_dump(mode="json")}
            self._end_task(task_id, TaskStatus.HALTED, halt.reason.value, detail)
            return self._outcome("halted", task_id)
        self._end_task(task_id, TaskStatus.REJECTED, "agent_error", {"error": result.error})
        return self._outcome("agent_error", task_id)

    def _integrate(self, task_id: str, bundle: Path, base: str) -> str | SliceOutcome:
        """Import, policy and merge into the ticket branch: the new ticket head, or a rejection."""
        try:
            imported = import_task_bundle(
                self.repo,
                bundle,
                base_sha=base,
                dest_ref=f"refs/factory/tasks/{task_id}",
                limits=self.site.limits,
            )
        except BundleRejected as rejected:
            return self._reject(task_id, rejected.code, {"error": rejected.detail})
        violations = check_changes(imported.changes, DiffPolicy.from_project(self.project))
        if violations:
            detail = {"violations": _violations_detail(violations)}
            return self._reject(task_id, "policy", detail, violations)
        merged = merge_into(
            self.repo,
            target_ref=self.ticket_ref,
            source=imported.head_sha,
            message=f"Merge {task_id} into factory/{self.ticket_id}",
        )
        if merged.sha is None:
            return self._reject(task_id, "conflict", {"conflicts": list(merged.conflicts)})
        self.ledger.update_task(
            task_id, actor=ACTOR, status=TaskStatus.ACCEPTED, head_sha=imported.head_sha
        )
        return merged.sha

    # ------------------------------------------------------------------ steps 10 - 13

    async def _converge(self, task_id: str, ticket_head: str) -> SliceOutcome:
        self._transition(TicketStatus.VERIFYING)
        verdict = await self._verify(ticket_head)
        if isinstance(verdict, Halt):
            return self._halted(task_id, verdict)
        if not verdict.passed:
            return self._failed(task_id, verdict)
        self._transition(TicketStatus.CONVERGED)
        return await self._candidate(task_id, ticket_head, verdict)

    async def _candidate(self, task_id: str, ticket_head: str, head: Verdict) -> SliceOutcome:
        main = self.repo.rev_parse(self.main_ref)
        title = self.ledger.get_ticket(self.ticket_id).title
        squashed = squash_onto(
            self.repo, base=main, head=ticket_head, message=f"{self.ticket_id}: {title}"
        )
        if squashed.sha is None:
            self._transition(
                TicketStatus.BUILDING,
                reason="candidate_conflict",
                data={"conflicts": list(squashed.conflicts)},
            )
            return self._outcome("rejected", task_id, rejection="candidate_conflict", verdict=head)
        candidate = squashed.sha
        verdict = await self._verify(candidate)
        if isinstance(verdict, Halt):
            return self._halted(task_id, verdict, verdict=head, candidate_sha=candidate)
        if not verdict.passed:
            return self._failed(task_id, verdict, candidate_sha=candidate)
        return self._certify(task_id, verdict, candidate=candidate, base=main)

    def _certify(
        self, task_id: str, verdict: Verdict, *, candidate: str, base: str
    ) -> SliceOutcome:
        evidence = build_evidence(
            ticket_id=self.ticket_id,
            verdict=verdict,
            tree_hash=self.repo.tree_hash(candidate),
            base_sha=base,
            contract_hash=self.frozen.contract_hash,
            oracle_version=self.frozen.oracle_version,
            factory_version=__version__,
            image_digests=self.site.image_digests(),
            models=self.models,
            clock=self.clock,
        )
        self._check_bound(evidence, candidate)
        signature = sign(evidence, self.key)
        self.ledger.record_evidence(
            evidence, evidence_hash=evidence_hash(evidence), signature=signature
        )
        self._transition(TicketStatus.CANDIDATE_VERIFIED)
        return self._outcome(
            "candidate_verified",
            task_id,
            verdict=verdict,
            candidate_sha=candidate,
            evidence=evidence,
            signature=signature,
        )

    def _check_bound(self, evidence: Evidence, candidate: str) -> None:
        """Re-derive every anchor independently (repo, ledger); any mismatch is a factory bug."""
        current = _frozen(self.ledger.get_ticket(self.ticket_id))
        problems = check_binding(
            evidence,
            sha=candidate,
            tree_hash=self.repo.tree_hash(candidate),
            contract_hash=content_hash(current.contract),
            oracle_version=current.oracle_version,
        )
        if problems:
            raise SliceError(
                f"{self.ticket_id}: evidence for {candidate} is not bound (factory bug): "
                + "; ".join(problems)
            )

    # ------------------------------------------------------------------ verification

    async def _verify(self, sha: str) -> Verdict | Halt:
        """Gates and the verifier on a fresh tree of ``sha``; the recorded verdict, or a halt."""
        ticket_dir = self.work_root / self.ticket_id
        async with self.site.verification(
            self.repo, sha, ticket_dir, round_no=self._next_round()
        ) as tree:
            gates = await tree.gates(gates_from_project(self.project))
            result = await self._run_role(VERIFIER_ROLE, self._verifier_task(sha), tree.handle)
        if result.halt is not None:
            return result.halt
        verdict = make_verdict(
            ticket_id=self.ticket_id,
            round=self._next_round(),  # numbered when recorded: no await in between
            sha=sha,
            gates=gates,
            criteria=self._criteria(result),
        )
        self.ledger.record_verdict(verdict)
        return verdict

    def _verifier_task(self, sha: str) -> AgentTask:
        contract = self.frozen.contract
        return AgentTask(
            ticket_id=self.ticket_id,
            instructions=f"Verify {sha} against these criteria:\n{_criteria_text(contract)}",
            context={"sha": sha, "criteria": _criteria_context(contract)},
        )

    def _criteria(self, result: AgentResult) -> tuple[CriterionResult, ...]:
        """The verifier's criterion results; none at all if it failed (so the verdict fails)."""
        if result.status != "ok" or result.structured_output is None:
            return ()
        try:
            report = VerifierReport.model_validate(result.structured_output)
        except ValidationError:
            return ()
        return reconcile_criteria(report.criteria, self.frozen.contract)

    def _next_round(self) -> int:
        return max((v.round for v in self.ledger.verdicts(self.ticket_id)), default=0) + 1

    # ------------------------------------------------------------------ backend and ledger

    async def _run_role(self, role: RoleSpec, task: AgentTask, handle: WorkerHandle) -> AgentResult:
        started = self.clock()
        result = await run_limited(self.backend.run(role, task, handle), self.site.run_limit_s)
        self.ledger.record_run(
            RunRecord(
                id=f"run-{uuid.uuid4().hex}",
                ticket_id=self.ticket_id,
                task_id=task.task_id,
                role=role.name,
                backend=self.backend.name,
                model=self.models[role.name],
                tier=role.tier,
                started_at=started,
                finished_at=self.clock(),
                status=result.status,
                usage=result.usage,
                session_id=result.session_id,
                transcript_path=result.transcript_path,
                error=result.error,
            )
        )
        return result

    def _transition(
        self, to: TicketStatus, *, reason: str = "", data: Mapping[str, JsonValue] | None = None
    ) -> None:
        self.ledger.transition(self.ticket_id, to, actor=ACTOR, reason=reason, data=data)

    def _set_task(self, task_id: str, status: TaskStatus) -> None:
        self.ledger.update_task(task_id, actor=ACTOR, status=status)

    def _end_task(
        self, task_id: str, status: TaskStatus, reason: str, detail: Mapping[str, JsonValue]
    ) -> None:
        """A task's final update (REJECTED or HALTED), with the reason and detail persisted."""
        self.ledger.update_task(task_id, actor=ACTOR, status=status, reason=reason, data=detail)

    def _reject(
        self,
        task_id: str,
        code: str,
        detail: Mapping[str, JsonValue],
        violations: Sequence[Violation] = (),
    ) -> SliceOutcome:
        self._end_task(task_id, TaskStatus.REJECTED, code, detail)
        return self._outcome("rejected", task_id, rejection=code, violations=violations)

    def _failed(
        self, task_id: str, verdict: Verdict, *, candidate_sha: str | None = None
    ) -> SliceOutcome:
        self._transition(
            TicketStatus.BUILDING, reason="failed_verification: " + ", ".join(verdict.failing)
        )
        return self._outcome(
            "failed_verification", task_id, verdict=verdict, candidate_sha=candidate_sha
        )

    def _halted(
        self,
        task_id: str,
        halt: Halt,
        *,
        verdict: Verdict | None = None,
        candidate_sha: str | None = None,
    ) -> SliceOutcome:
        """A halt during verification: the ticket halts where it is; the task stays accepted."""
        self.ledger.halt(self.ticket_id, halt, actor=ACTOR)
        return self._outcome("halted", task_id, verdict=verdict, candidate_sha=candidate_sha)

    def _outcome(
        self,
        status: SliceStatus,
        task_id: str,
        *,
        rejection: str | None = None,
        violations: Sequence[Violation] = (),
        verdict: Verdict | None = None,
        candidate_sha: str | None = None,
        evidence: Evidence | None = None,
        signature: str | None = None,
    ) -> SliceOutcome:
        return SliceOutcome(
            status=status,
            ticket=self.ledger.get_ticket(self.ticket_id),
            task=self.ledger.get_task(task_id),
            rejection=rejection,
            violations=tuple(violations),
            verdict=verdict,
            candidate_sha=candidate_sha,
            evidence=evidence,
            signature=signature,
        )
