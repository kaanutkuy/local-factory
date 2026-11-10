"""Integration: the host slice end to end with real git, real gates and the scripted backend."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import JsonValue

from local_factory import __version__
from local_factory.agents import (
    Action,
    AgentResult,
    AgentTask,
    Fail,
    FakeBackend,
    RaiseHalt,
    RoleSpec,
    WorkerHandle,
    WriteFile,
)
from local_factory.config import FactoryConfig
from local_factory.evidence import evidence_hash, verify_signature
from local_factory.models import (
    CriterionStatus,
    HaltReason,
    TaskStatus,
    TicketStatus,
    Tier,
    Usage,
)
from local_factory.pipeline import SliceOutcome
from local_factory.pipeline.host_slice import UNREPORTED

from .conftest import (
    IMPLEMENT_SHOUT,
    MAIN,
    TICKET_REF,
    VERIFIER_FAIL,
    VERIFIER_PASS,
    ClaudeLikeBackend,
    PerCallBackend,
    RecordingBackend,
    World,
    report,
)

pytestmark = pytest.mark.integration

S = TaskStatus
T = TicketStatus


def _verified(outcome: SliceOutcome) -> str:
    assert outcome.status == "candidate_verified", outcome
    assert outcome.candidate_sha is not None
    return outcome.candidate_sha


# --------------------------------------------------------------------------- happy path


def test_runs_carry_backend_model_tier_and_usage(world: World) -> None:
    usage = Usage(input_tokens=10, output_tokens=5, cost_usd=0.25)
    backend = FakeBackend({"implementer": IMPLEMENT_SHOUT, "verifier": VERIFIER_PASS}, usage=usage)
    _verified(world.run(backend))

    runs = world.ledger.runs(world.ticket_id)
    assert [(r.role, r.task_id, r.tier) for r in runs] == [
        ("implementer", "T-0001/task-1", Tier.STANDARD),
        ("verifier", None, Tier.FRONTIER),
        ("verifier", None, Tier.FRONTIER),
    ]
    assert {(r.backend, r.model, r.status) for r in runs} == {("fake", "fake", "ok")}
    assert all(r.usage == usage and r.started_at == r.finished_at for r in runs)
    assert len({r.id for r in runs}) == 3
    assert world.ledger.ticket_cost_usd(world.ticket_id) == pytest.approx(0.75)


def test_evidence_is_bound_signed_and_recorded(world: World) -> None:
    outcome = world.run_scripts(IMPLEMENT_SHOUT)
    candidate = _verified(outcome)

    evidence = outcome.evidence
    assert evidence is not None
    assert outcome.signature is not None
    assert outcome.verdict is not None
    assert (outcome.verdict.round, outcome.verdict.sha) == (2, candidate)
    assert evidence.base_sha == world.main
    assert evidence.tree_hash == world.repo.tree_hash(candidate)
    assert evidence.factory_version == __version__
    assert evidence.models == {"implementer": "fake", "verifier": "fake"}
    assert evidence.gates == outcome.verdict.gates
    assert verify_signature(evidence, outcome.signature, world.key)
    record = world.ledger.evidence_record(world.ticket_id, candidate)
    assert record is not None
    assert (record.evidence, record.signature) == (evidence, outcome.signature)
    assert record.evidence_hash == evidence_hash(evidence)


def test_git_topology_after_a_verified_candidate(world: World) -> None:
    outcome = world.run_scripts(IMPLEMENT_SHOUT)
    candidate = _verified(outcome)
    task = outcome.task

    assert (task.id, task.base_sha) == ("T-0001/task-1", world.main)
    assert task.head_sha is not None
    assert task.criteria_served == ("AC-1", "AC-2")
    assert (task.goal, task.role, task.tier) == (
        "greet supports --shout",
        "implementer",
        Tier.STANDARD,
    )
    assert world.repo.rev_parse("refs/factory/tasks/T-0001/task-1") == task.head_sha
    assert world.repo.rev_parse(TICKET_REF) == task.head_sha
    assert world.message(task.head_sha) == "T-0001/task-1: greet supports --shout"
    assert world.parents(candidate) == [world.main]
    assert world.message(candidate) == "T-0001: shout"
    assert world.repo.tree_hash(candidate) == world.repo.tree_hash(task.head_sha)
    world.assert_boundaries()


def test_every_verification_runs_in_a_fresh_export(world: World) -> None:
    backend = RecordingBackend({"implementer": IMPLEMENT_SHOUT, "verifier": VERIFIER_PASS})
    _verified(world.run(backend))

    (implementer, verifiers) = backend.workers[0], backend.workers[1:]
    assert implementer[:2] == ("implementer", "T-0001/task-1")
    assert [(role, task) for role, task, _ in verifiers] == [("verifier", None)] * 2
    dirs = [path for _, _, path in verifiers]
    assert len(set(dirs)) == 2
    assert implementer[2] not in dirs
    for path in dirs:
        assert path.parent == world.work_root / world.ticket_id
        assert b"--shout" in (path / "greet.py").read_bytes()
        assert not (path / ".git").exists()  # an export, never a checkout


def test_agent_tasks_carry_the_contract(world: World) -> None:
    backend = FakeBackend({"implementer": IMPLEMENT_SHOUT, "verifier": VERIFIER_PASS})
    candidate = _verified(world.run(backend))

    (_, implement), (_, verify_head), (_, verify_candidate) = backend.calls
    assert implement.task_id == "T-0001/task-1"
    assert "AC-1: When --shout is given, greet shall shout." in implement.instructions
    assert implement.context["base_sha"] == world.main
    assert verify_head.task_id is None
    assert verify_candidate.context["sha"] == candidate
    assert implement.context["criteria"] == verify_head.context["criteria"]


def test_an_agent_commit_is_kept_as_is(world: World) -> None:
    """The harness commits only what is left; the agent's own commits are exported unchanged."""
    outcome = world.run_scripts(
        [
            *IMPLEMENT_SHOUT,
            world.git.as_agent("add", "-A"),
            world.git.as_agent("commit", "-q", "-m", "agent: shout"),
        ]
    )
    _verified(outcome)
    head = outcome.task.head_sha
    assert head is not None
    assert world.repo.list_commits(world.main, head) == [head]
    assert world.message(head) == "agent: shout"


def test_a_configured_backend_records_its_configured_models(world: World, tmp_path: Path) -> None:
    config = FactoryConfig(home=tmp_path / "factory-home")
    backend = ClaudeLikeBackend({"implementer": IMPLEMENT_SHOUT, "verifier": VERIFIER_PASS})
    outcome = world.run(backend, factory_config=config)
    _verified(outcome)

    expected = {
        "implementer": config.model_for("claude", Tier.STANDARD),
        "verifier": config.model_for("claude", Tier.FRONTIER),
    }
    assert outcome.evidence is not None
    assert outcome.evidence.models == expected
    runs = world.ledger.runs(world.ticket_id)
    assert [(r.backend, r.model) for r in runs] == [
        ("claude", expected["implementer"]),
        ("claude", expected["verifier"]),
        ("claude", expected["verifier"]),
    ]


# --------------------------------------------------------------------------- agent errors


@pytest.mark.parametrize(
    "script",
    [[Fail("boom")], [*IMPLEMENT_SHOUT, Fail("gave up")], [WriteFile("/etc/evil", "x")]],
    ids=["fail", "fail-after-writing", "escape"],
)
def test_an_agent_error_rejects_the_task(world: World, script: list[Action]) -> None:
    outcome = world.run_scripts(script)

    assert (outcome.status, outcome.rejection) == ("agent_error", None)
    assert outcome.task.status is S.REJECTED
    assert outcome.task.head_sha is None
    assert outcome.ticket.status is T.BUILDING
    assert (outcome.verdict, outcome.evidence, outcome.candidate_sha) == (None, None, None)
    (run,) = world.ledger.runs(world.ticket_id)
    assert run.status == "error"
    assert run.error
    assert world.task_ends()[outcome.task.id].detail == {"error": run.error}
    assert world.refs("refs/factory/") == []
    assert world.repo.rev_parse(TICKET_REF) == world.main
    world.assert_boundaries()


# --------------------------------------------------------------------------- further rounds


def test_a_building_ticket_gets_a_new_task_and_rounds_continue(world: World) -> None:
    first = world.run_scripts(IMPLEMENT_SHOUT, verifier=VERIFIER_FAIL)
    assert first.status == "failed_verification"
    assert first.task.status is S.ACCEPTED
    first_head = first.task.head_sha

    second = world.run_scripts([WriteFile("NOTES.md", "shout notes\n")])
    candidate = _verified(second)

    assert second.task.id == "T-0001/task-2"
    assert second.task.base_sha == first_head
    assert [t.status for t in world.ledger.list_tasks(world.ticket_id)] == [S.ACCEPTED] * 2
    assert [v.round for v in world.ledger.verdicts(world.ticket_id)] == [1, 2, 3]
    assert world.transitions()[3:] == [
        T.PLANNED,
        T.BUILDING,
        T.VERIFYING,
        T.BUILDING,  # round 1 failed: back to building, the same ticket branch
        T.VERIFYING,
        T.CONVERGED,
        T.CANDIDATE_VERIFIED,
    ]
    assert world.parents(candidate) == [world.main]
    assert b"--shout" in world.repo.read_file(candidate, "greet.py")
    assert world.repo.read_file(candidate, "NOTES.md") == b"shout notes\n"
    world.assert_boundaries()


def test_a_failed_verification_says_why_in_the_ledger(world: World) -> None:
    outcome = world.run_scripts(IMPLEMENT_SHOUT, verifier=VERIFIER_FAIL)

    assert outcome.verdict is not None
    assert outcome.verdict.failing == ("criterion:AC-1",)
    last = world.ledger.events(world.ticket_id)[-1]
    assert (last.kind, last.to_status) == ("transition", outcome.ticket.status)
    assert last.data.get("reason") == "failed_verification: criterion:AC-1"


def test_a_halted_task_is_replaced_after_resume(world: World) -> None:
    backend = FakeBackend(
        {
            "T-0001/task-1": [RaiseHalt(HaltReason.BILLING)],
            "T-0001/task-2": IMPLEMENT_SHOUT,
            "verifier": VERIFIER_PASS,
        }
    )
    halted = world.run(backend)
    assert halted.status == "halted"
    assert [r.status for r in world.ledger.runs(world.ticket_id)] == ["halted"]
    world.ledger.resume(world.ticket_id, actor="scheduler")

    outcome = world.run(backend)
    _verified(outcome)
    assert outcome.task.id == "T-0001/task-2"
    assert world.ledger.get_task("T-0001/task-1").status is S.HALTED


# --------------------------------------------------------------------------- the verifier


def test_an_unreported_criterion_fails_closed(world: World) -> None:
    outcome = world.run_scripts(IMPLEMENT_SHOUT, verifier=report(("AC-1", "pass")))

    assert outcome.status == "failed_verification"
    assert outcome.verdict is not None
    assert outcome.verdict.failing == ("criterion:AC-2",)
    unreported = outcome.verdict.criteria[1]
    assert (unreported.id, unreported.status) == ("AC-2", CriterionStatus.INCONCLUSIVE)
    assert unreported.feedback == UNREPORTED


def test_a_verifier_error_leaves_no_criteria(world: World) -> None:
    outcome = world.run_scripts(IMPLEMENT_SHOUT, verifier=[Fail("verifier crashed")])

    assert outcome.verdict is not None
    assert outcome.verdict.criteria == ()
    assert outcome.verdict.failing == ("criteria:none",)
    assert [r.status for r in world.ledger.runs(world.ticket_id)] == ["ok", "error"]


class SchemalessBackend(FakeBackend):
    """A backend without native structured output: its verifier "succeeds" with raw output."""

    def __init__(self, output: dict[str, JsonValue] | None) -> None:
        super().__init__({"implementer": IMPLEMENT_SHOUT})
        self.output = output

    async def run(self, role: RoleSpec, task: AgentTask, worker: WorkerHandle) -> AgentResult:
        if role.name != "verifier":
            return await super().run(role, task, worker)
        return AgentResult(status="ok", structured_output=self.output)


@pytest.mark.parametrize(
    "output",
    [None, {"criteria": "all good"}, {"verdict": "pass"}],
    ids=["no-output", "malformed", "wrong-shape"],
)
def test_unvalidated_verifier_output_cannot_pass(
    world: World, output: dict[str, JsonValue] | None
) -> None:
    outcome = world.run(SchemalessBackend(output))

    assert outcome.status == "failed_verification"
    assert outcome.verdict is not None
    assert outcome.verdict.failing == ("criteria:none",)


def test_a_verifier_halt_on_the_ticket_head_halts_in_verifying(world: World) -> None:
    backend = PerCallBackend(
        {"implementer": IMPLEMENT_SHOUT},
        {"verifier": [[RaiseHalt(HaltReason.RATE_LIMIT)]]},
    )
    outcome = world.run(backend)

    assert outcome.status == "halted"
    assert outcome.ticket.status is T.HALTED
    assert outcome.task.status is S.ACCEPTED
    assert (outcome.verdict, outcome.candidate_sha) == (None, None)
    assert world.ledger.verdicts(world.ticket_id) == []
    assert world.ledger.resume(world.ticket_id, actor="scheduler").status is T.VERIFYING


def test_a_verifier_halt_on_the_candidate_halts_in_converged(world: World) -> None:
    backend = PerCallBackend(
        {"implementer": IMPLEMENT_SHOUT},
        {"verifier": [VERIFIER_PASS, [RaiseHalt(HaltReason.RATE_LIMIT)]]},
    )
    outcome = world.run(backend)

    assert outcome.status == "halted"
    assert outcome.verdict is not None
    assert (outcome.verdict.round, outcome.verdict.passed) == (1, True)
    assert outcome.candidate_sha is not None
    assert outcome.evidence is None
    assert [v.round for v in world.ledger.verdicts(world.ticket_id)] == [1]
    halt = world.ledger.current_halt(world.ticket_id)
    assert halt is not None
    assert halt.reason is HaltReason.RATE_LIMIT
    assert world.ledger.resume(world.ticket_id, actor="scheduler").status is T.CONVERGED
    assert world.repo.rev_parse(MAIN) == world.main
