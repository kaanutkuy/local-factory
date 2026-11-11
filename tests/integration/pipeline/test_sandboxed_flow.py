"""Integration: the sandboxed slice's flow, with real git and an in-process stand-in for the box.

No Docker here (``fake_box.FakeBox``): these tests pin what the slice asks of the box and how it
maps the box's answers to ledger records. The real containers are covered by
``test_sandboxed_box.py``, the acceptance test and the red-team suite.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from local_factory.agents import (
    Action,
    AgentResult,
    AgentTask,
    FakeBackend,
    LocalExecutor,
    RoleSpec,
    RunCommand,
    WorkerHandle,
    WriteFile,
)
from local_factory.config import SandboxConfig
from local_factory.gitops import GitError
from local_factory.models import RunStatus, TaskStatus, TicketStatus
from local_factory.sandbox import ExportOutcome
from local_factory.sandbox.runtime import SandboxError

from .conftest import (
    IMPLEMENT_SHOUT,
    TICKET_REF,
    VERIFIER_PASS,
    GatedBackend,
    TaskEnd,
    World,
    greet_with,
)
from .fake_box import IMAGE_ID, FakeBox

pytestmark = pytest.mark.integration

T = TicketStatus
GREET_CONCAT = greet_with('print("Hello, " + args.name + "!")')
GREET_FORMAT = greet_with('print("Hello, {}!".format(args.name))')


@pytest.fixture
def box(tmp_path: Path) -> FakeBox:
    scratch = tmp_path / "box"
    scratch.mkdir()
    return FakeBox(scratch)


class HandleBackend(FakeBackend):
    """A FakeBackend that also keeps every (role, WorkerHandle) it was given."""

    def __init__(self, scripts: dict[str, list[Action]]) -> None:
        super().__init__(scripts)
        self.handles: list[tuple[str, WorkerHandle]] = []

    async def run(self, role: RoleSpec, task: AgentTask, worker: WorkerHandle) -> AgentResult:
        self.handles.append((role.name, worker))
        return await super().run(role, task, worker)


class HangingBackend(FakeBackend):
    """Runs the scripts of every role but `hangs`, which never returns."""

    def __init__(self, scripts: dict[str, list[Action]], *, hangs: str) -> None:
        super().__init__(scripts)
        self.hangs = hangs

    async def run(self, role: RoleSpec, task: AgentTask, worker: WorkerHandle) -> AgentResult:
        if role.name == self.hangs:
            await asyncio.Event().wait()
        return await super().run(role, task, worker)


class CrashingBackend(FakeBackend):
    async def run(self, role: RoleSpec, task: AgentTask, worker: WorkerHandle) -> AgentResult:
        await worker.tools().write_file("scratch.txt", "half done")
        raise RuntimeError("the backend crashed")


# --------------------------------------------------------------------------- the flow in the box


def test_every_worker_side_step_goes_through_the_box(world: World, box: FakeBox) -> None:
    backend = HandleBackend({"implementer": IMPLEMENT_SHOUT, "verifier": VERIFIER_PASS})
    outcome = world.run(backend, sandbox=box.sandbox())

    assert outcome.status == "candidate_verified", outcome
    assert outcome.task.status is TaskStatus.ACCEPTED
    candidate, head = outcome.candidate_sha, outcome.task.head_sha
    assert candidate is not None and head is not None
    assert world.repo.rev_parse(TICKET_REF) == head
    assert world.parents(candidate) == [world.main]
    assert box.checks == 1

    task_dir = world.work_root / "T-0001" / "task-1"
    ticket_dir = world.work_root / "T-0001"
    verify = [
        ticket_dir / f"verify-{sha[:12]}-{n}.bundle" for n, sha in enumerate((head, candidate), 1)
    ]
    # the box materialized the base, then each verified commit, each in a workspace of its own
    assert [bundle for bundle, _ in box.materialized] == [task_dir / "base.bundle", *verify]
    assert len({ws.root for ws in box.workspaces}) == 3
    assert all(ws.closed for ws in box.workspaces) and box.open_workspaces() == []
    # the gates ran on the very bundle (and ref) the verifier's workspace was materialized from
    assert box.gated == box.materialized[1:]
    assert all(bundle.is_file() for bundle in verify)
    assert box.exports == [
        (world.main, "T-0001/task-1: greet supports --shout", task_dir / "task.bundle")
    ]

    # the backend acts through the box's executor on /work, never on a host directory
    assert [role for role, _ in backend.handles] == ["implementer", "verifier", "verifier"]
    for (_, handle), ws in zip(backend.handles, box.workspaces, strict=True):
        assert handle.work_dir == Path("/work")
        assert isinstance(handle.executor, LocalExecutor) and handle.executor.root == ws.root

    assert world.work_root_oddities() == []
    world.assert_boundaries()


def test_the_evidence_records_the_sandbox_image(world: World, box: FakeBox) -> None:
    outcome = world.run_scripts(IMPLEMENT_SHOUT, sandbox=box.sandbox())
    assert outcome.evidence is not None
    assert outcome.evidence.image_digests == {"sandbox": IMAGE_ID}
    assert (
        world.ledger.evidence_for(world.ticket_id, outcome.candidate_sha or "") == outcome.evidence
    )


def test_the_host_slice_records_no_image(world: World) -> None:
    outcome = world.run_scripts(IMPLEMENT_SHOUT)
    assert outcome.evidence is not None and outcome.evidence.image_digests == {}


def test_a_verifier_never_sees_another_workspace(world: World, box: FakeBox) -> None:
    """Both verifications run `touch seen` after checking it is absent: each starts clean."""
    fresh = RunCommand(("sh", "-c", "test ! -e seen && touch seen"))
    outcome = world.run_scripts(IMPLEMENT_SHOUT, [fresh, *VERIFIER_PASS], sandbox=box.sandbox())
    assert outcome.status == "candidate_verified", outcome
    assert outcome.candidate_sha is not None
    with pytest.raises(GitError):
        world.repo.read_file(outcome.candidate_sha, "seen")


def test_gates_run_in_the_box_and_a_failure_fails_verification(world: World, box: FakeBox) -> None:
    outcome = world.run_scripts([WriteFile("greet.py", "def broken(:\n")], sandbox=box.sandbox())
    assert outcome.status == "failed_verification"
    assert outcome.verdict is not None and "gate:lint" in outcome.verdict.failing
    assert outcome.ticket.status is T.BUILDING
    assert len(box.gated) == 1  # the ticket head failed: no candidate, no second verification


# --------------------------------------------------------------------------- export outcomes


EXPORT_ANSWERS = [
    ExportOutcome("no_changes", "the task made no changes"),
    ExportOutcome("worker_git_error", "export failed: exit status 4"),
    ExportOutcome("bundle_too_large", "the task bundle exceeded 5000000 bytes"),
]


@pytest.mark.parametrize("answer", EXPORT_ANSWERS, ids=lambda answer: answer.status)
def test_every_export_outcome_maps_to_its_rejection(
    world: World, box: FakeBox, answer: ExportOutcome
) -> None:
    box.export_outcome = answer
    outcome = world.run_scripts(IMPLEMENT_SHOUT, sandbox=box.sandbox())

    assert (outcome.status, outcome.rejection) == ("rejected", answer.status)
    assert outcome.task.status is TaskStatus.REJECTED
    assert outcome.ticket.status is T.BUILDING
    assert world.task_ends()[outcome.task.id] == TaskEnd(
        TaskStatus.REJECTED, answer.status, {"error": answer.detail}
    )
    assert world.repo.rev_parse(TICKET_REF) == world.main
    assert len(box.workspaces) == 1 and box.open_workspaces() == []  # closed, nothing verified
    assert world.work_root_oddities() == []
    world.assert_boundaries()


def test_a_worker_that_does_nothing_is_rejected_in_the_box(world: World, box: FakeBox) -> None:
    outcome = world.run_scripts([], sandbox=box.sandbox())
    assert (outcome.status, outcome.rejection) == ("rejected", "no_changes")
    assert box.open_workspaces() == []


def test_a_worker_that_breaks_its_repository_is_rejected_in_the_box(
    world: World, box: FakeBox
) -> None:
    outcome = world.run_scripts([RunCommand(("rm", "-rf", ".git"))], sandbox=box.sandbox())
    assert (outcome.status, outcome.rejection) == ("rejected", "worker_git_error")
    end = world.task_ends()[outcome.task.id]
    assert end.reason == "worker_git_error" and end.detail.get("error")


def test_the_bundle_cap_is_the_sandbox_configs(world: World, tmp_path: Path) -> None:
    """The box caps the stream; the import enforces the same cap on the file it is handed."""
    scratch = tmp_path / "tight"
    scratch.mkdir()
    tight = FakeBox(scratch, config=SandboxConfig(max_bundle_bytes=200))
    outcome = world.run_scripts(IMPLEMENT_SHOUT, sandbox=tight.sandbox())
    assert (outcome.status, outcome.rejection) == ("rejected", "bundle_too_large")
    end = world.task_ends()[outcome.task.id]
    assert "limit 200" in str(end.detail["error"])


# --------------------------------------------------------------------------- limits and failures


def test_a_run_that_outlives_the_limit_is_an_agent_error(world: World, tmp_path: Path) -> None:
    scratch = tmp_path / "quick"
    scratch.mkdir()
    quick = FakeBox(scratch, config=SandboxConfig(run_timeout_s=0.05))
    backend = HangingBackend({}, hangs="implementer")

    outcome = world.run(backend, sandbox=quick.sandbox())

    message = "timeout: the run exceeded 0.05s"
    assert (outcome.status, outcome.rejection) == ("agent_error", None)
    assert outcome.task.status is TaskStatus.REJECTED
    assert outcome.ticket.status is T.BUILDING
    (run,) = world.ledger.runs(world.ticket_id)
    assert (run.status, run.error) == ("error", message)
    assert world.task_ends()[outcome.task.id] == TaskEnd(
        TaskStatus.REJECTED, "agent_error", {"error": message}
    )
    assert quick.exports == [] and quick.open_workspaces() == []


def test_a_verifier_that_outlives_the_limit_fails_the_verification(
    world: World, tmp_path: Path
) -> None:
    scratch = tmp_path / "quick"
    scratch.mkdir()
    quick = FakeBox(scratch, config=SandboxConfig(run_timeout_s=0.05))
    backend = HangingBackend({"implementer": IMPLEMENT_SHOUT}, hangs="verifier")

    outcome = world.run(backend, sandbox=quick.sandbox())

    assert outcome.status == "failed_verification"
    assert outcome.verdict is not None and "criteria:none" in outcome.verdict.failing
    runs = world.ledger.runs(world.ticket_id)
    verifier = runs[-1]
    assert (verifier.role, verifier.status) == ("verifier", "error")
    assert verifier.error is not None and verifier.error.startswith("timeout:")
    assert quick.open_workspaces() == []


def test_a_crashing_backend_propagates_and_its_workspace_is_closed(
    world: World, box: FakeBox
) -> None:
    with pytest.raises(RuntimeError, match="the backend crashed"):
        world.run(CrashingBackend({}), sandbox=box.sandbox())
    assert len(box.workspaces) == 1 and box.open_workspaces() == []


def test_a_cancelled_run_closes_its_workspace(world: World, box: FakeBox) -> None:
    backend = GatedBackend({"verifier": VERIFIER_PASS}, gated="T-0001/task-1")

    async def scenario() -> None:
        running = asyncio.create_task(world.start(backend, sandbox=box.sandbox()))
        await backend.waiting.wait()
        assert len(box.open_workspaces()) == 1  # the implementer's, held while it runs
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running

    asyncio.run(scenario())
    assert len(box.workspaces) == 1 and box.open_workspaces() == []


def test_a_verification_that_crashes_closes_the_verifiers_workspace(
    world: World, box: FakeBox
) -> None:
    class VerifierCrashes(FakeBackend):
        async def run(self, role: RoleSpec, task: AgentTask, worker: WorkerHandle) -> AgentResult:
            if role.name == "verifier":
                raise RuntimeError("the verifier crashed")
            return await super().run(role, task, worker)

    backend = VerifierCrashes({"implementer": IMPLEMENT_SHOUT})
    with pytest.raises(RuntimeError, match="the verifier crashed"):
        world.run(backend, sandbox=box.sandbox())
    assert len(box.workspaces) == 2 and box.open_workspaces() == []


# --------------------------------------------------------------------------- preconditions


def test_a_sandbox_that_is_not_ready_is_refused_before_any_change(
    world: World, box: FakeBox
) -> None:
    box.check_error = "image local-factory/sandbox:v1 is missing; run scripts/build-images"
    backend = FakeBackend({"implementer": IMPLEMENT_SHOUT, "verifier": VERIFIER_PASS})
    with pytest.raises(SandboxError, match="scripts/build-images"):
        world.run(backend, sandbox=box.sandbox())

    assert box.checks == 1
    assert len(world.ledger.events(world.ticket_id)) == world.initial_events
    assert backend.calls == []
    assert box.workspaces == []
    assert not world.work_root.exists()
    assert world.refs("refs/heads/factory/") == []
    assert world.ledger.get_ticket(world.ticket_id).status is T.APPROVED


def test_the_other_preconditions_are_checked_too(world: World, box: FakeBox) -> None:
    with pytest.raises(GitError):
        world.run(FakeBackend({}), sandbox=box.sandbox(), main_ref="refs/heads/nope")
    with pytest.raises(ValueError, match="key must not be empty"):
        world.run(FakeBackend({}), sandbox=box.sandbox(), key=b"")
    assert len(world.ledger.events(world.ticket_id)) == world.initial_events
    assert not world.work_root.exists()


# --------------------------------------------------------------------------- structured conflict


def test_a_candidate_conflict_is_recorded_with_its_paths(world: World, box: FakeBox) -> None:
    world.repo.update_ref(TICKET_REF, world.main)  # the ticket branch was cut at the old main
    world.advance_main({"greet.py": GREET_FORMAT})

    outcome = world.run_scripts([WriteFile("greet.py", GREET_CONCAT)], sandbox=box.sandbox())

    assert (outcome.status, outcome.rejection) == ("rejected", "candidate_conflict")
    assert outcome.ticket.status is T.BUILDING
    assert outcome.task.status is TaskStatus.ACCEPTED
    back = world.ledger.events(world.ticket_id)[-1]
    assert (back.kind, back.to_status) == ("transition", T.BUILDING)
    assert back.data == {"reason": "candidate_conflict", "detail": {"conflicts": ["greet.py"]}}
    assert box.open_workspaces() == []
    assert run_statuses(world) == ["ok", "ok"]  # the implementer and the head's verifier


def run_statuses(world: World) -> list[RunStatus]:
    return [run.status for run in world.ledger.runs(world.ticket_id)]
