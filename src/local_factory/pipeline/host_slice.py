"""The Phase 1a host vertical slice (docs/specs/phase-1a-slice.md).

One deterministic pass over an approved ticket: one task, one worker, one harness commit, one
bundle import, the diff policy, integration into the ticket branch, verification of the ticket
head, a candidate squashed onto ``main``, verification of the candidate, and evidence bound to the
candidate SHA.

- Every state change goes through the ledger, the source of truth (invariant 1); this module
  keeps no workflow state of its own. Every rejected or halted task records its reason and detail
  (violations, conflict paths, error text) there, so the ledger alone explains each outcome.
- Host-side code never runs git inside the worker directory (invariant 7): only
  :mod:`local_factory.gitops.worker_side` touches it (inside the sandbox from Phase 1b on), and
  the worker's output comes back only as a bundle file, validated on import.
- The slice never moves ``main`` and never pushes anywhere.
"""

from __future__ import annotations

import tempfile
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

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
    BundleRejected,
    GitError,
    commit_all,
    export_base_bundle,
    export_task_bundle,
    has_task_commits,
    import_task_bundle,
    init_from_bundle,
    merge_into,
    squash_onto,
)
from local_factory.hashing import content_hash, utc_now
from local_factory.ledger import Ledger
from local_factory.models import (
    Contract,
    CriterionResult,
    CriterionStatus,
    Evidence,
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
from local_factory.verification import gates_from_project, make_verdict, run_gates

__all__ = [
    "ACTOR",
    "FAKE_MODEL",
    "IMPLEMENTER_ROLE",
    "VERIFIER_ROLE",
    "SliceError",
    "SliceOutcome",
    "SliceStatus",
    "VerifierReport",
    "reconcile_criteria",
    "run_host_slice",
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
    # a bundle rejection code | "policy" | "conflict" | "candidate_conflict"
    rejection: str | None = None
    violations: tuple[Violation, ...] = ()
    verdict: Verdict | None = None  # the last verdict this run recorded
    candidate_sha: str | None = None
    evidence: Evidence | None = None
    signature: str | None = None


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
    """Run one task of ``ticket_id`` from worker to signed candidate evidence.

    The ticket must be APPROVED (planned and moved to BUILDING here) or already BUILDING (a further
    round: a new task based on the ticket head). Everything is checked before the first change:
    the ticket's status and frozen contract, the key, the models (``factory_config`` resolves
    them for any backend but the fake one) and ``main_ref``.

    Raises :class:`SliceError` for unmet preconditions and for evidence that fails its binding
    check (a factory bug). Rejections, failed verification, halts and agent errors are outcomes.
    """
    ticket = ledger.get_ticket(ticket_id)
    if ticket.status not in _STARTABLE:
        raise SliceError(f"{ticket.id} is {ticket.status}; the slice needs approved or building")
    if not key:
        raise ValueError("the evidence signing key must not be empty")
    run = _Slice(
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
        work = self._start_worker(task, base)  # 3
        result = await self._implement(task, base, work)  # 4
        if result.halt is not None:
            halt = result.halt
            self.ledger.halt(self.ticket_id, halt, actor=ACTOR)
            detail = {"halt": halt.model_dump(mode="json")}
            self._end_task(task.id, TaskStatus.HALTED, halt.reason.value, detail)
            return self._outcome("halted", task.id)
        if result.status == "error":
            self._end_task(task.id, TaskStatus.REJECTED, "agent_error", {"error": result.error})
            return self._outcome("agent_error", task.id)
        bundle = self._submit(task, work, base)  # 5, 6
        if isinstance(bundle, SliceOutcome):
            return bundle
        integrated = self._integrate(task.id, bundle, base)  # 7, 8, 9
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

    def _start_worker(self, task: Task, base: str) -> Path:
        task_dir = self.work_root.joinpath(*task.id.split("/"))  # <ticket>/task-<n>
        task_dir.mkdir(parents=True)
        bundle = task_dir / "base.bundle"
        ref = export_base_bundle(self.repo, sha=base, out=bundle)
        work = task_dir / "work"
        init_from_bundle(bundle, ref, work)
        self.ledger.update_task(task.id, actor=ACTOR, status=TaskStatus.RUNNING, base_sha=base)
        return work

    async def _implement(self, task: Task, base: str, work: Path) -> AgentResult:
        contract = self.frozen.contract
        agent_task = AgentTask(
            ticket_id=self.ticket_id,
            task_id=task.id,
            instructions=f"{task.goal}\n\nAcceptance criteria:\n{_criteria_text(contract)}",
            context={"base_sha": base, "criteria": _criteria_context(contract)},
        )
        return await self._run_role(IMPLEMENTER_ROLE, agent_task, work)

    def _submit(self, task: Task, work: Path, base: str) -> Path | SliceOutcome:
        """Harness commit and task bundle, or the rejection: no commits, or a broken worker repo."""
        try:
            commit_all(work, f"{task.id}: {task.goal}")
            has_commits = has_task_commits(work, base_sha=base)
        except GitError as broken:  # the agent had the repository: it broke it
            return self._reject(task.id, "worker_git_error", {"error": str(broken)})
        if not has_commits:
            return self._reject(task.id, "no_changes", {"error": f"no commits beyond {base}"})
        bundle = work.parent / "task.bundle"
        export_task_bundle(work, bundle, base_sha=base)  # failing now is a factory bug: raise
        self._set_task(task.id, TaskStatus.SUBMITTED)
        return bundle

    def _integrate(self, task_id: str, bundle: Path, base: str) -> str | SliceOutcome:
        """Import, policy and merge into the ticket branch: the new ticket head, or a rejection."""
        try:
            imported = import_task_bundle(
                self.repo, bundle, base_sha=base, dest_ref=f"refs/factory/tasks/{task_id}"
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
            conflicts = ", ".join(squashed.conflicts)
            self._transition(TicketStatus.BUILDING, reason=f"candidate_conflict: {conflicts}")
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
        """Gates and the verifier on a fresh export of ``sha``; the recorded verdict, or a halt."""
        tree = self._fresh_dir(f"verify-{sha[:12]}-")
        self.repo.export_tree(sha, tree)
        gates = run_gates(gates_from_project(self.project), cwd=tree)
        result = await self._run_role(VERIFIER_ROLE, self._verifier_task(sha), tree)
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

    def _fresh_dir(self, prefix: str) -> Path:
        parent = self.work_root / self.ticket_id
        parent.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix=prefix, dir=parent))

    # ------------------------------------------------------------------ ledger helpers

    async def _run_role(self, role: RoleSpec, task: AgentTask, work_dir: Path) -> AgentResult:
        started = self.clock()
        result = await self.backend.run(role, task, WorkerHandle(work_dir=work_dir))
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

    def _transition(self, to: TicketStatus, *, reason: str = "") -> None:
        self.ledger.transition(self.ticket_id, to, actor=ACTOR, reason=reason)

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
