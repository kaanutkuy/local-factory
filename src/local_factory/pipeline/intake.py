"""Gate 1: a request becomes an approved, frozen contract (docs/specs/phase-2b.md § intake).

One ticket at a time, in the foreground. The four intake roles (triage, spec writer, ambiguity
critic, test architect) run on a :class:`~local_factory.pipeline.flow.Site`, and **every model run
happens in a fresh tree of main that is discarded afterwards**; only the test architect's work
leaves its box, as a bundle, and only test files under the ticket's acceptance directory can pass
the oracle policy. Nothing here moves ``main`` or touches the user's source repository.

- The ledger is the source of truth (invariant 1): this module keeps no workflow state. Questions,
  drafts, critiques and every red-first result are events; the ticket's status says where intake
  continues.
- The factory, not the model, decides what a ticket's contract says: a role's structured output
  is validated again here, criteria are numbered by the factory, and the visible tests only count
  once red-first has seen them fail on the base (a keep criterion's checks: pass on it). Probe
  expectations live only in the holdout store (and in the model's own output): never in a
  repository, a box or the ledger.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ValidationError

from local_factory.agents import (
    AgentBackend,
    AgentResult,
    AgentTask,
    FakeBackend,
    RoleSpec,
    WorkerHandle,
)
from local_factory.agents.roles import (
    CRITIC_ROLE,
    INTAKE_ROLES,
    SPEC_WRITER_ROLE,
    TEST_ARCHITECT_ROLE,
    TRIAGE_ROLE,
)
from local_factory.config import ACCEPTANCE_COMMAND, FactoryConfig, ProjectConfig
from local_factory.gitops import BareRepo, BundleRejected, export_base_bundle, import_task_bundle
from local_factory.gitops.bundles import ImportResult
from local_factory.hashing import utc_now
from local_factory.ledger import Ledger, TicketNotFound
from local_factory.ledger.state_machine import TERMINAL
from local_factory.models import (
    ChangedFile,
    ChangeKind,
    Contract,
    Criterion,
    CriterionKind,
    Critique,
    Halt,
    Holdout,
    HoldoutProbe,
    OracleManifest,
    Question,
    QuestionDraft,
    RedCheck,
    RiskTier,
    RunRecord,
    SpecDraft,
    Ticket,
    TicketSize,
    TicketStatus,
    TicketType,
    TriageResult,
)
from local_factory.oracle import (
    HoldoutError,
    HoldoutStore,
    acceptance_dir,
    check_oracle_changes,
    oracle_ref,
    oracle_version,
    red_first,
    split_test_id,
)
from local_factory.pipeline.flow import FAKE_MODEL, Refused, Site, run_limited

__all__ = [
    "ACTOR",
    "MAX_CLARIFY_ROUNDS",
    "MAX_OPEN_TICKETS_SHOWN",
    "MAX_ORACLE_ATTEMPTS",
    "MAX_REQUEST_CHARS",
    "Intake",
    "IntakeError",
    "IntakeOutcome",
    "IntakeStatus",
    "answer",
    "approve",
    "open_questions",
    "questions",
    "receive",
    "run_intake",
]

ACTOR = "factory"  # the actor of every ledger change intake makes by itself
MAX_ORACLE_ATTEMPTS = 3
MAX_CLARIFY_ROUNDS = 3
MAX_REQUEST_CHARS = 20_000
MAX_OPEN_TICKETS_SHOWN = 50

TITLE_CHARS = 80  # a ticket's first title: the request's first line, clipped
OPEN_REQUEST_CHARS = 500  # an open ticket's request, as triage sees it
ERROR_CHARS = 2_000  # a role's error text as recorded in an intake_error event
FACTORY_ASKER = "factory"  # `asked_by` of a question the factory asks itself (round 0)
HUMAN = "human"

IntakeStatus = Literal["contract_proposed", "needs_info", "oracle_failed", "agent_error", "halted"]

_QUESTION_ASKED = "question_asked"
_QUESTION_ANSWERED = "question_answered"
_SPEC_DRAFTED = "spec_drafted"
_CRITIQUE = "critique"
_INTAKE_ERROR = "intake_error"
_RED_FIRST = "red_first"
_ORACLE_REJECTED = "oracle_rejected"
_ORACLE_FAILED = "oracle_failed"
_TRIAGED = "triaged"
_RESUMABLE = frozenset({TicketStatus.RECEIVED, TicketStatus.TRIAGED, TicketStatus.NEEDS_INFO})


class IntakeError(Exception):
    """Misuse of intake: a ticket in the wrong status, an unknown question, a bad request text."""


@dataclass(frozen=True)
class IntakeOutcome:
    status: IntakeStatus
    ticket: Ticket  # fresh snapshot at the end
    questions: tuple[Question, ...] = ()  # the open ones (needs_info)
    red_checks: tuple[RedCheck, ...] = ()  # the last red-first run, if any
    error: str | None = None  # agent_error / oracle_failed / halted: why


@dataclass(frozen=True)
class Intake:
    """The collaborators of intake. Holds no workflow state: that lives in the ledger."""

    site: Site
    ledger: Ledger
    repo: BareRepo
    project: ProjectConfig
    backend: AgentBackend
    holdout: HoldoutStore
    work_root: Path
    main_ref: str = "refs/heads/main"
    clock: Callable[[], datetime] = utc_now
    factory_config: FactoryConfig | None = None  # required unless the backend is the fake one


# --------------------------------------------------------------------------- receive


def receive(ledger: Ledger, text: str, *, author: str = HUMAN) -> Ticket:
    """Record a request as a RECEIVED ticket. Until triage says otherwise it is a feature of the
    highest risk (an unknown risk is treated as the highest)."""
    request = text.strip()
    if not request:
        raise IntakeError("the request is empty")
    if len(request) > MAX_REQUEST_CHARS:
        raise IntakeError(
            f"the request is {len(request)} characters; the limit is {MAX_REQUEST_CHARS}"
        )
    title = request.splitlines()[0][:TITLE_CHARS]
    return ledger.create_ticket(
        title=title, type=TicketType.FEATURE, risk=RiskTier.R2, request=request, actor=author
    )


# --------------------------------------------------------------------------- questions


def questions(ledger: Ledger, ticket_id: str) -> list[Question]:
    """Every question of the ticket in the order asked, answers filled in (folded from events)."""
    folded: dict[str, Question] = {}
    for event in ledger.events(ticket_id):
        if event.kind == _QUESTION_ASKED:
            asked = Question.model_validate(event.data["question"])
            folded[asked.id] = asked
        elif event.kind == _QUESTION_ANSWERED:
            question = folded.get(str(event.data["id"]))
            if question is not None:
                folded[question.id] = question.model_copy(
                    update={
                        "answer": str(event.data["answer"]),
                        "answered_by": str(event.data["answered_by"]),
                    }
                )
    return list(folded.values())


def open_questions(ledger: Ledger, ticket_id: str) -> list[Question]:
    return [q for q in questions(ledger, ticket_id) if q.answer is None]


def answer(
    ledger: Ledger, ticket_id: str, question_id: str, text: str, *, author: str = HUMAN
) -> Question:
    """Answer one open question of a ticket that is waiting for answers."""
    ticket = ledger.get_ticket(ticket_id)
    if ticket.status is not TicketStatus.NEEDS_INFO:
        raise IntakeError(f"{ticket_id} is {ticket.status}, not waiting for answers")
    found = {q.id: q for q in questions(ledger, ticket_id)}.get(question_id)
    if found is None:
        raise IntakeError(f"{ticket_id} has no question {question_id}")
    if found.answer is not None:
        raise IntakeError(f"{question_id} is already answered")
    reply = text.strip()
    if not reply:
        raise IntakeError("the answer is empty")
    ledger.record_event(
        ticket_id,
        _QUESTION_ANSWERED,
        actor=author,
        data={"id": question_id, "answer": reply, "answered_by": author},
    )
    return found.model_copy(update={"answer": reply, "answered_by": author})


# --------------------------------------------------------------------------- approval


def approve(
    ledger: Ledger, repo: BareRepo, holdout: HoldoutStore, ticket_id: str, *, author: str = HUMAN
) -> Ticket:
    """Freeze the proposed contract and the oracle behind it, and approve the ticket.

    Refused unless the oracle is exactly what was proposed: the sealed holdout record must hash
    to the ticket's oracle version and name the commit the oracle ref points at now.
    """
    ticket = ledger.get_ticket(ticket_id)
    if ticket.status is not TicketStatus.CONTRACT_PROPOSED or ticket.contract is None:
        raise IntakeError(
            f"{ticket_id} is {ticket.status}; only a proposed contract can be approved"
        )
    try:
        record = holdout.load(ticket_id)
    except HoldoutError as exc:
        raise IntakeError(f"{ticket_id}: the holdout record is unusable: {exc}") from exc
    if record is None:
        raise IntakeError(f"{ticket_id}: there is no holdout record")
    if oracle_version(record) != ticket.oracle_version:
        raise IntakeError(
            f"{ticket_id}: the holdout record changed after the contract was proposed"
        )
    ref = oracle_ref(ticket_id)
    if not repo.ref_exists(ref) or repo.rev_parse(ref) != record.oracle_sha:
        raise IntakeError(f"{ticket_id}: the visible tests changed after the contract was proposed")
    return ledger.freeze_contract(ticket_id, actor=author)


# --------------------------------------------------------------------------- running intake


async def run_intake(intake: Intake, ticket_id: str) -> IntakeOutcome:
    """Continue ``ticket_id`` from its status as far as it can go without a human, never moving
    ``main``. Raises :class:`IntakeError` for a ticket that intake does not continue (it is past
    intake, or halted: resume it first)."""
    ledger = intake.ledger
    status = ledger.get_ticket(ticket_id).status
    if status is TicketStatus.CONTRACT_PROPOSED:
        return _outcome(intake, ticket_id, "contract_proposed")
    if status is TicketStatus.NEEDS_INFO:
        waiting = open_questions(ledger, ticket_id)
        if waiting:
            return _outcome(intake, ticket_id, "needs_info", questions=tuple(waiting))
    elif status not in _RESUMABLE:
        raise IntakeError(f"{ticket_id} is {status}; intake continues received or triaged tickets")
    run = _prepare(intake, ticket_id)
    if status is TicketStatus.NEEDS_INFO:
        ledger.transition(ticket_id, TicketStatus.TRIAGED, actor=ACTOR, reason="questions answered")
    if status is TicketStatus.RECEIVED:
        stopped = await run.triage()
        if stopped is not None:
            return stopped
    return await run.specify_and_build()


def _outcome(
    intake: Intake,
    ticket_id: str,
    status: IntakeStatus,
    *,
    questions: tuple[Question, ...] = (),
    red_checks: tuple[RedCheck, ...] = (),
    error: str | None = None,
) -> IntakeOutcome:
    return IntakeOutcome(
        status=status,
        ticket=intake.ledger.get_ticket(ticket_id),
        questions=questions,
        red_checks=red_checks,
        error=error,
    )


def _prepare(intake: Intake, ticket_id: str) -> _Run:
    """Everything a run needs, checked before the first model or ledger change."""
    template = intake.project.commands.get(ACCEPTANCE_COMMAND)
    if template is None or "{path}" not in template:
        raise IntakeError(
            f"factory.toml needs [commands] {ACCEPTANCE_COMMAND} with a {{path}} placeholder"
        )
    models = _models(intake.backend.name, intake.factory_config)
    main = intake.repo.rev_parse(intake.main_ref)
    intake.site.check()
    return _Run(intake=intake, ticket_id=ticket_id, main=main, models=models, acceptance=template)


def _models(backend_name: str, config: FactoryConfig | None) -> dict[str, str]:
    """role -> model id: ``fake`` for the fake backend, else the configured model for the tier."""
    if backend_name == FakeBackend.name:
        return {role.name: FAKE_MODEL for role in INTAKE_ROLES}
    if config is None:
        raise IntakeError(f"backend {backend_name!r} needs a factory config to resolve its models")
    return {role.name: config.model_for(backend_name, role.tier) for role in INTAKE_ROLES}


# --------------------------------------------------------------------------- what a run knows


@dataclass(frozen=True)
class _Spec:
    version: int
    draft: SpecDraft


@dataclass(frozen=True)
class _Brief:
    """What the test architect works from: the specification plus every assumption it rests on."""

    spec: _Spec
    assumptions: tuple[str, ...]


@dataclass(frozen=True)
class _Accepted:
    """An oracle that passed every check: what the contract and the holdout record are made of."""

    manifest: OracleManifest
    probes: tuple[HoldoutProbe, ...]
    oracle_sha: str


@dataclass(frozen=True)
class _Attempt:
    """One go at the oracle: the problems found (none: accepted) and what red-first saw."""

    problems: tuple[str, ...]
    red: tuple[RedCheck, ...] = ()
    accepted: _Accepted | None = None


class _SpecEvent(BaseModel):
    version: int
    draft: SpecDraft


class _TriagedEvent(BaseModel):
    triage: TriageResult


def _draft_event(ledger: Ledger, ticket_id: str) -> _Spec | None:
    """The latest ``spec_drafted`` event's draft."""
    latest: _Spec | None = None
    for event in ledger.events(ticket_id):
        if event.kind == _SPEC_DRAFTED:
            parsed = _SpecEvent.model_validate(event.data)
            latest = _Spec(version=parsed.version, draft=parsed.draft)
    return latest


def _triage_event(ledger: Ledger, ticket_id: str) -> TriageResult | None:
    """The latest triage result recorded with the ticket's ``triaged`` event."""
    latest: TriageResult | None = None
    for event in ledger.events(ticket_id):
        if event.kind == _TRIAGED and "triage" in event.data:
            latest = _TriagedEvent.model_validate(event.data).triage
    return latest


# --------------------------------------------------------------------------- the run


@dataclass(frozen=True)
class _Run:
    intake: Intake
    ticket_id: str
    main: str
    models: dict[str, str]
    acceptance: str

    @property
    def ledger(self) -> Ledger:
        return self.intake.ledger

    @property
    def ticket_dir(self) -> Path:
        return self.intake.work_root / self.ticket_id

    # ------------------------------------------------------------------ 1. triage

    async def triage(self) -> IntakeOutcome | None:
        """Classify the request. An outcome when intake must stop (an error, a halt, a question
        for the human); None to carry on."""
        ticket = self.ledger.get_ticket(self.ticket_id)
        result = await self._in_fresh_tree(
            TRIAGE_ROLE, _triage_instructions(ticket, self._open_tickets()), TriageResult
        )
        if isinstance(result, IntakeOutcome):
            return result
        self.ledger.triage(
            self.ticket_id,
            title=result.title,
            type=result.type,
            risk=result.risk,
            actor=ACTOR,
            data={"triage": result.model_dump(mode="json")},
        )
        asked = self._factory_questions(result)
        if not asked:
            return None
        return self._ask(asked)

    def _open_tickets(self) -> list[Ticket]:
        """The open tickets triage compares the request with: the latest ones, but this."""
        others = [
            t
            for t in self.ledger.list_tickets()
            if t.status not in TERMINAL and t.id != self.ticket_id
        ]
        return others[-MAX_OPEN_TICKETS_SHOWN:]

    def _factory_questions(self, result: TriageResult) -> list[Question]:
        """What the factory asks itself about a triage result: a suspected duplicate, an epic."""
        texts: list[str] = []
        duplicate = self._duplicate(result)
        if duplicate is not None:
            texts.append(
                f"Triage thinks this duplicates {duplicate.id} ({duplicate.title}): "
                f"{result.rationale} Answer to continue with this ticket anyway."
            )
        if result.size is TicketSize.XL:
            texts.append(
                f"Triage thinks this is too large for one ticket: {result.rationale} "
                "Answer with the part to build first, or to continue as one ticket."
            )
        return self._numbered([(text, "", FACTORY_ASKER, 0) for text in texts])

    def _duplicate(self, result: TriageResult) -> Ticket | None:
        """The existing ticket triage names as the same change; None when it names this ticket or
        none that exists."""
        if result.duplicate_of is None or result.duplicate_of == self.ticket_id:
            return None
        try:
            return self.ledger.get_ticket(result.duplicate_of)
        except TicketNotFound:
            return None

    def _numbered(self, drafts: Sequence[tuple[str, str, str, int]]) -> list[Question]:
        """Questions ``(text, why_blocking, asked_by, round)`` with ids continuing the ticket's."""
        first = len(questions(self.ledger, self.ticket_id)) + 1
        return [
            Question(
                id=f"Q-{number}", text=text, why_blocking=why, asked_by=asked_by, round=round_no
            )
            for number, (text, why, asked_by, round_no) in enumerate(drafts, start=first)
        ]

    def _ask(self, asked: Sequence[Question]) -> IntakeOutcome:
        """Record the questions and wait for the human."""
        for question in asked:
            self.ledger.record_event(
                self.ticket_id,
                _QUESTION_ASKED,
                actor=ACTOR,
                data={"question": question.model_dump(mode="json")},
            )
        self.ledger.transition(
            self.ticket_id, TicketStatus.NEEDS_INFO, actor=ACTOR, reason="questions asked"
        )
        return _outcome(
            self.intake,
            self.ticket_id,
            "needs_info",
            questions=tuple(open_questions(self.ledger, self.ticket_id)),
        )

    # ------------------------------------------------------------------ 2. specify, 3. clarify

    async def specify_and_build(self) -> IntakeOutcome:
        spec = await self._specify()
        if isinstance(spec, IntakeOutcome):
            return spec
        extra = await self._clarify(spec)
        if isinstance(extra, IntakeOutcome):
            return extra
        assumptions = tuple(dict.fromkeys([*spec.draft.assumptions, *extra]))
        return await self._oracle(_Brief(spec=spec, assumptions=assumptions))

    async def _specify(self) -> _Spec | IntakeOutcome:
        ticket = self.ledger.get_ticket(self.ticket_id)
        previous = _draft_event(self.ledger, self.ticket_id)
        instructions = _specify_instructions(
            ticket,
            _triage_event(self.ledger, self.ticket_id),
            questions(self.ledger, self.ticket_id),
            previous,
        )
        draft = await self._in_fresh_tree(SPEC_WRITER_ROLE, instructions, SpecDraft)
        if isinstance(draft, IntakeOutcome):
            return draft
        version = 1 if previous is None else previous.version + 1
        self.ledger.record_event(
            self.ticket_id,
            _SPEC_DRAFTED,
            actor=ACTOR,
            data={"version": version, "draft": draft.model_dump(mode="json")},
        )
        return _Spec(version=version, draft=draft)

    async def _clarify(self, spec: _Spec) -> tuple[str, ...] | IntakeOutcome:
        """A fresh critic reads the draft. Its blocking questions go to the human (an outcome)
        while rounds remain; its assumptions, and the questions that no longer block, are what
        the contract keeps (the tuple)."""
        ticket = self.ledger.get_ticket(self.ticket_id)
        asked = questions(self.ledger, self.ticket_id)
        instructions = _critic_instructions(
            ticket, _triage_event(self.ledger, self.ticket_id), spec, asked
        )
        critique = await self._in_fresh_tree(CRITIC_ROLE, instructions, Critique)
        if isinstance(critique, IntakeOutcome):
            return critique
        self.ledger.record_event(
            self.ticket_id,
            _CRITIQUE,
            actor=ACTOR,
            data=critique.model_dump(mode="json"),
        )
        rounds = len({q.round for q in asked if q.asked_by == CRITIC_ROLE.name})
        if critique.blocking and rounds < MAX_CLARIFY_ROUNDS:
            return self._ask(self._numbered(_critic_drafts(critique.blocking, rounds + 1)))
        unresolved = tuple(
            f"Unresolved after {MAX_CLARIFY_ROUNDS} rounds: {q.text}" for q in critique.blocking
        )
        return (*critique.assumptions, *unresolved)

    # ------------------------------------------------------------------ 4. the oracle

    async def _oracle(self, brief: _Brief) -> IntakeOutcome:
        problems: tuple[str, ...] = ()
        last_red: tuple[RedCheck, ...] = ()
        for attempt in range(1, MAX_ORACLE_ATTEMPTS + 1):
            tried = await self._attempt(attempt, brief, problems)
            if isinstance(tried, IntakeOutcome):
                return tried
            last_red = tried.red or last_red
            if tried.accepted is not None:
                return self._accept(brief, tried.accepted, tried.red)
            problems = tried.problems
            self.ledger.record_event(
                self.ticket_id,
                _ORACLE_REJECTED,
                actor=ACTOR,
                data={"attempt": attempt, "problems": list(problems)},
            )
        self.ledger.record_event(
            self.ticket_id, _ORACLE_FAILED, actor=ACTOR, data={"attempts": MAX_ORACLE_ATTEMPTS}
        )
        error = (
            f"the tests were rejected {MAX_ORACLE_ATTEMPTS} times; the last problems:\n"
            + "\n".join(problems)
        )
        return _outcome(
            self.intake, self.ticket_id, "oracle_failed", red_checks=last_red, error=error
        )

    async def _attempt(
        self, attempt: int, brief: _Brief, previous: Sequence[str]
    ) -> _Attempt | IntakeOutcome:
        """One test architect run from main, and everything the factory then checks."""
        intake = self.intake
        directory = _fresh_dir(self.ticket_dir, f"oracle-{attempt}")
        base = directory / "base.bundle"
        ref = export_base_bundle(intake.repo, sha=self.main, out=base)
        instructions = _architect_instructions(
            self.ticket_id, self.acceptance, brief, previous, attempt
        )
        async with intake.site.worker(base, ref) as worker:
            manifest = await self._run(
                TEST_ARCHITECT_ROLE, instructions, OracleManifest, worker.handle
            )
            if isinstance(manifest, IntakeOutcome):
                return manifest
            submitted = await worker.submit(
                base=self.main,
                message=f"{self.ticket_id}: acceptance tests",
                out=directory / "task.bundle",
            )
        if isinstance(submitted, Refused):
            return _Attempt(problems=(f"{submitted.code}: {submitted.error}",))
        candidate = f"refs/factory/oracle-candidate/{self.ticket_id}/{attempt}"
        try:
            imported = import_task_bundle(
                intake.repo,
                submitted,
                base_sha=self.main,
                dest_ref=candidate,
                limits=intake.site.limits,
            )
        except BundleRejected as rejected:
            return _Attempt(problems=(str(rejected),))
        try:
            judged = await self._judge(attempt, brief, manifest, imported)
            if judged.accepted is not None:
                # claim the commit under the oracle ref before the candidate ref goes
                intake.repo.update_ref(oracle_ref(self.ticket_id), imported.head_sha)
            return judged
        finally:
            intake.repo.delete_ref(candidate)

    async def _judge(
        self, attempt: int, brief: _Brief, manifest: OracleManifest, imported: ImportResult
    ) -> _Attempt:
        """The policy and manifest checks, then (only without problems) red-first."""
        problems = [
            f"{v.rule} {v.path}: {v.message}"
            for v in check_oracle_changes(imported.changes, self.ticket_id)
        ]
        kinds = _criterion_kinds(brief.spec.draft)
        problems += _manifest_problems(manifest, imported.changes, kinds)
        if problems:
            return _Attempt(problems=tuple(problems))
        probes = tuple(
            HoldoutProbe(
                id=f"HP-{number}",
                criteria=draft.criteria,
                argv=draft.argv,
                stdin=draft.stdin,
                expect=draft.expect,
            )
            for number, draft in enumerate(manifest.holdout, start=1)
        )
        checks = await red_first(
            site=self.intake.site,
            repo=self.intake.repo,
            oracle_sha=imported.head_sha,
            base_sha=self.main,
            visible=[visible.test for visible in manifest.visible],
            probes=probes,
            acceptance=self.acceptance,
            ticket_dir=self.ticket_dir,
            keep=_keep_ids(manifest, probes, kinds),
        )
        self.ledger.record_event(
            self.ticket_id,
            _RED_FIRST,
            actor=ACTOR,
            data={
                "attempt": attempt,
                "oracle_sha": imported.head_sha,
                "checks": [check.model_dump(mode="json") for check in checks],
            },
        )
        failed = tuple(_check_problem(check) for check in checks if not check.ok)
        accepted = None if failed else _Accepted(manifest, probes, imported.head_sha)
        return _Attempt(problems=failed, red=tuple(checks), accepted=accepted)

    def _accept(
        self, brief: _Brief, accepted: _Accepted, red: tuple[RedCheck, ...]
    ) -> IntakeOutcome:
        """Seal the holdout and propose the contract; the visible tests are already under the
        oracle ref."""
        record = Holdout(
            ticket_id=self.ticket_id, oracle_sha=accepted.oracle_sha, probes=accepted.probes
        )
        self.intake.holdout.save(record)
        contract = _contract(brief, accepted.manifest)
        self.ledger.propose(self.ticket_id, contract, oracle_version(record), actor=ACTOR)
        return _outcome(self.intake, self.ticket_id, "contract_proposed", red_checks=red)

    # ------------------------------------------------------------------ one model run

    async def _in_fresh_tree[M: BaseModel](
        self, role: RoleSpec, instructions: str, schema: type[M]
    ) -> M | IntakeOutcome:
        """``role`` in a fresh tree of main that is discarded afterwards."""
        intake = self.intake
        round_no = len(self.ledger.runs(self.ticket_id)) + 1
        async with intake.site.verification(
            intake.repo, self.main, self.ticket_dir, round_no=round_no
        ) as tree:
            return await self._run(role, instructions, schema, tree.handle)

    async def _run[M: BaseModel](
        self, role: RoleSpec, instructions: str, schema: type[M], handle: WorkerHandle
    ) -> M | IntakeOutcome:
        """Run ``role`` on ``handle``, record the run, and validate what it submitted."""
        intake = self.intake
        task = AgentTask(ticket_id=self.ticket_id, instructions=instructions)
        started = intake.clock()
        result = await run_limited(intake.backend.run(role, task, handle), intake.site.run_limit_s)
        self.ledger.record_run(
            RunRecord(
                id=f"run-{uuid.uuid4().hex}",
                ticket_id=self.ticket_id,
                role=role.name,
                backend=intake.backend.name,
                model=self.models[role.name],
                tier=role.tier,
                started_at=started,
                finished_at=intake.clock(),
                status=result.status,
                usage=result.usage,
                session_id=result.session_id,
                transcript_path=result.transcript_path,
                error=result.error,
            )
        )
        if result.halt is not None:
            return self._halted(result.halt)
        parsed = _parsed(result, schema)
        if isinstance(parsed, str):
            return self._failed(role, parsed)
        return parsed

    def _halted(self, halt: Halt) -> IntakeOutcome:
        self.ledger.halt(self.ticket_id, halt, actor=ACTOR)
        reason = halt.reason.value + (f": {halt.detail}" if halt.detail else "")
        return _outcome(self.intake, self.ticket_id, "halted", error=reason)

    def _failed(self, role: RoleSpec, error: str) -> IntakeOutcome:
        """The ticket stays where it is: running intake again retries."""
        text = _clip(error, ERROR_CHARS)
        self.ledger.record_event(
            self.ticket_id, _INTAKE_ERROR, actor=ACTOR, data={"role": role.name, "error": text}
        )
        return _outcome(self.intake, self.ticket_id, "agent_error", error=text)


def _parsed[M: BaseModel](result: AgentResult, schema: type[M]) -> M | str:
    """What a finished run submitted, validated again here; or why it is unusable."""
    if result.status == "error":
        return result.error or "the run failed"
    try:
        return schema.model_validate(result.structured_output)
    except ValidationError as exc:
        found = "; ".join(
            f"{'.'.join(str(part) for part in err['loc']) or 'result'}: {err['msg']}"
            for err in exc.errors(include_url=False, include_input=False, include_context=False)
        )
        return f"the submitted result does not match {schema.__name__}: {found}"


# --------------------------------------------------------------------------- what the roles read


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _oneline(text: str, limit: int) -> str:
    return _clip(" ".join(text.split()), limit)


def _list(title: str, items: Sequence[str]) -> list[str]:
    if not items:
        return [f"{title}: none"]
    return [f"{title}:", *(f"- {item}" for item in items)]


def _request_block(ticket: Ticket) -> str:
    return f"Request:\n<request>\n{ticket.request}\n</request>"


def _triage_block(result: TriageResult | None) -> str:
    if result is None:
        return "Triage result: none recorded."
    lines = [
        "Triage result:",
        f"- title: {result.title}",
        f"- type: {result.type.value}, risk: {result.risk.value}, size: {result.size.value}",
        f"- areas: {', '.join(result.areas) or 'none'}",
        f"- rationale: {result.rationale}",
    ]
    return "\n".join(lines)


def _questions_block(asked: Sequence[Question]) -> str:
    if not asked:
        return "Questions and answers: none."
    lines = ["Questions and answers:"]
    for question in asked:
        lines.append(f"- {question.id} ({question.asked_by}): {question.text}")
        reply = "(not answered yet)" if question.answer is None else question.answer
        lines.append(f"  Answer: {reply}")
    return "\n".join(lines)


def _criteria_lines(draft: SpecDraft) -> list[str]:
    lines: list[str] = []
    for number, criterion in enumerate(draft.criteria, start=1):
        lines.append(
            f"AC-{number} [{criterion.layer.value}, {criterion.kind.value}]: {criterion.ears}"
        )
        lines.append(f"  Test plan: {criterion.test_plan}")
    return lines


def _draft_block(spec: _Spec) -> str:
    draft = spec.draft
    lines = [f"Specification draft, version {spec.version}:", f"Summary: {draft.summary}"]
    lines += ["Acceptance criteria:", *_criteria_lines(draft)]
    lines += _list("Non-functional requirements", draft.nfr)
    lines += _list("Out of scope", draft.out_of_scope)
    lines += _list("Assumptions", draft.assumptions)
    lines.append(f"Migration impact: {draft.migration_impact.value}")
    return "\n".join(lines)


def _triage_instructions(ticket: Ticket, others: Sequence[Ticket]) -> str:
    lines = [f"Triage ticket {ticket.id}.", "", _request_block(ticket), ""]
    if others:
        lines.append(
            "Open tickets (id, title, start of the request). Put one in duplicate_of only if it "
            "asks for the same change:"
        )
        for other in others:
            lines.append(f"- {other.id}: {other.title}")
            lines.append(f"  {_oneline(other.request, OPEN_REQUEST_CHARS)}")
    else:
        lines.append("There are no other open tickets.")
    return "\n".join(lines)


def _specify_instructions(
    ticket: Ticket,
    triage: TriageResult | None,
    asked: Sequence[Question],
    previous: _Spec | None,
) -> str:
    lines = [
        f"Write the specification for ticket {ticket.id}.",
        "",
        _request_block(ticket),
        "",
        _triage_block(triage),
        "",
        _questions_block(asked),
    ]
    if previous is not None:
        lines += [
            "",
            _draft_block(previous),
            "Improve this draft, taking the answers into account.",
        ]
    return "\n".join(lines)


def _critic_instructions(
    ticket: Ticket, triage: TriageResult | None, spec: _Spec, asked: Sequence[Question]
) -> str:
    return "\n".join(
        [
            f"Review the specification draft of ticket {ticket.id} for ambiguity.",
            "",
            _request_block(ticket),
            "",
            _triage_block(triage),
            "",
            _draft_block(spec),
            "",
            _questions_block(asked),
            "Do not ask a question that was already asked.",
        ]
    )


def _architect_instructions(
    ticket_id: str, acceptance: str, brief: _Brief, previous: Sequence[str], attempt: int
) -> str:
    draft = brief.spec.draft
    lines = [
        f"Write the acceptance tests for ticket {ticket_id} (attempt {attempt}).",
        "",
        f"Acceptance directory (write your test files only here): {acceptance_dir(ticket_id)}",
        "Acceptance command (it runs one test from the repository root; {path} and {name} are "
        f"replaced by the two halves of the test id, shell-quoted): {acceptance}",
        "",
        f"Summary: {draft.summary}",
        "Acceptance criteria:",
        *_criteria_lines(draft),
        "A change criterion's tests and probes must fail on the current code; a keep criterion's "
        "must pass on it. A test or probe checks criteria of one kind only.",
        *_list("Out of scope", draft.out_of_scope),
        *_list("Assumptions", brief.assumptions),
    ]
    if previous:
        lines += [
            "",
            "Your previous attempt was rejected. Fix every one of these problems:",
            *previous,
        ]
    return "\n".join(lines)


def _critic_drafts(
    blocking: Sequence[QuestionDraft], round_no: int
) -> list[tuple[str, str, str, int]]:
    return [(q.text, q.why_blocking, CRITIC_ROLE.name, round_no) for q in blocking]


# --------------------------------------------------------------------------- the oracle's checks


def _fresh_dir(parent: Path, name: str) -> Path:
    """A new directory ``parent/name`` (``name-2``, ... when an earlier run left one)."""
    path = parent / name
    suffix = 1
    while path.exists():
        suffix += 1
        path = parent / f"{name}-{suffix}"
    path.mkdir(parents=True)
    return path


def _criterion_kinds(draft: SpecDraft) -> dict[str, CriterionKind]:
    """``AC-n`` -> its kind, numbered the way the factory numbers the draft's criteria."""
    return {f"AC-{n}": criterion.kind for n, criterion in enumerate(draft.criteria, start=1)}


def _mixes_kinds(criteria: Sequence[str], kinds: Mapping[str, CriterionKind]) -> bool:
    """Whether the known ones among ``criteria`` include both a change and a keep criterion."""
    return len({kinds[ac] for ac in criteria if ac in kinds}) > 1


def _keep_ids(
    manifest: OracleManifest, probes: Sequence[HoldoutProbe], kinds: Mapping[str, CriterionKind]
) -> frozenset[str]:
    """The ids of the tests and probes that guard keep criteria (none mixes kinds by now)."""
    guards = [(visible.test, visible.criteria) for visible in manifest.visible]
    guards += [(probe.id, probe.criteria) for probe in probes]
    return frozenset(
        check_id
        for check_id, criteria in guards
        if all(kinds[ac] is CriterionKind.KEEP for ac in criteria)
    )


def _check_problem(check: RedCheck) -> str:
    """Why a red-first check is not what its criteria's kind needs."""
    text = f"{check.id}: {check.reason}"
    return f"{text}; a keep check must pass on the base" if check.keep else text


def _manifest_problems(
    manifest: OracleManifest, changes: Sequence[ChangedFile], kinds: Mapping[str, CriterionKind]
) -> list[str]:
    """What is wrong with the manifest, each naming the test, probe or criterion it is about."""
    added = {change.path for change in changes if change.kind is ChangeKind.ADDED}
    problems: list[str] = []
    seen: set[str] = set()
    for visible in manifest.visible:
        if visible.test in seen:
            problems.append(f"{visible.test}: listed more than once")
        seen.add(visible.test)
        path = split_test_id(visible.test)[0]
        if path not in added:
            problems.append(f"{visible.test}: {path} is not a file the test architect added")
        problems += [
            f"{visible.test}: unknown criterion {ac}" for ac in visible.criteria if ac not in kinds
        ]
        if _mixes_kinds(visible.criteria, kinds):
            problems.append(f"{visible.test}: mixes change and keep criteria")
    for number, probe in enumerate(manifest.holdout, start=1):
        problems += [
            f"HP-{number}: unknown criterion {ac}" for ac in probe.criteria if ac not in kinds
        ]
        if _mixes_kinds(probe.criteria, kinds):
            problems.append(f"HP-{number}: mixes change and keep criteria")
    tested = {ac for visible in manifest.visible for ac in visible.criteria}
    probed = {ac for probe in manifest.holdout for ac in probe.criteria}
    for ac in kinds:
        if ac not in tested:
            problems.append(f"{ac}: no visible test checks it")
        if ac not in probed:
            problems.append(f"{ac}: no holdout probe checks it")
    return problems


def _contract(brief: _Brief, manifest: OracleManifest) -> Contract:
    """The proposed contract: the latest draft, each criterion with its tests and probe count."""
    draft = brief.spec.draft
    criteria = []
    for number, criterion in enumerate(draft.criteria, start=1):
        ac = f"AC-{number}"
        criteria.append(
            Criterion(
                id=ac,
                ears=criterion.ears,
                layer=criterion.layer,
                kind=criterion.kind,
                tests=tuple(v.test for v in manifest.visible if ac in v.criteria),
                holdout_probes=sum(ac in probe.criteria for probe in manifest.holdout),
            )
        )
    return Contract(
        version=brief.spec.version,
        summary=draft.summary,
        criteria=tuple(criteria),
        nfr=draft.nfr,
        out_of_scope=draft.out_of_scope,
        assumptions=brief.assumptions,
        migration_impact=draft.migration_impact,
    )
