"""Acceptance: the vertical slice with every worker-side step in the box
(docs/specs/phase-1b.md § integrator). This is the Phase 1b exit check: one ticket, one worker,
one isolated container boundary, one commit, bundle import, deterministic verification, one
integration, evidence bound to the candidate SHA, $0 LLM spend. Protected.
"""

from __future__ import annotations

import asyncio
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import JsonValue

from local_factory.agents.fake import Action, FakeBackend, RunCommand, WriteFile
from local_factory.config import ProjectConfig, load_project_config
from local_factory.evidence.binding import check_binding, load_or_create_key, verify_signature
from local_factory.gitops import GitError
from local_factory.gitops.bare import BareRepo
from local_factory.hashing import content_hash
from local_factory.ledger import Ledger
from local_factory.models import (
    Contract,
    Criterion,
    RiskTier,
    TaskStatus,
    TestLayer,
    TicketStatus,
    TicketType,
)
from local_factory.pipeline.host_slice import SliceOutcome
from local_factory.pipeline.sandboxed_slice import run_sandboxed_slice
from local_factory.sandbox import Sandbox
from local_factory.sandbox.runtime import DockerRuntime

from .conftest import T0, Git, docker_objects, small_sandbox_config
from .test_host_slice import IMPLEMENT_SHOUT, VERIFIER_PASS

pytestmark = [pytest.mark.container, pytest.mark.integration]

S = TicketStatus
FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "cli-target"
ORACLE = "sha256:" + "b" * 64
CONTRACT = Contract(
    version=1,
    summary="greet supports --shout",
    criteria=(
        Criterion(
            id="AC-1",
            ears="When --shout is given, greet shall print the greeting in upper case.",
            layer=TestLayer.E2E,
        ),
    ),
)


@dataclass
class World:
    ledger: Ledger
    repo: BareRepo
    project: ProjectConfig
    ticket_id: str
    main: str
    key: bytes
    work_root: Path
    target: Path
    runtime: DockerRuntime

    def sandbox(self, **overrides: object) -> Sandbox:
        return Sandbox(self.runtime, small_sandbox_config(**overrides))

    def run(
        self,
        scripts: Mapping[str, Sequence[Action]],
        *,
        sandbox: Sandbox | None = None,
    ) -> SliceOutcome:
        return asyncio.run(
            run_sandboxed_slice(
                ledger=self.ledger,
                repo=self.repo,
                project=self.project,
                backend=FakeBackend(scripts),
                sandbox=sandbox or self.sandbox(),
                ticket_id=self.ticket_id,
                work_root=self.work_root,
                key=self.key,
                clock=lambda: T0,
            )
        )

    def implement(
        self, implementer: Sequence[Action], *, sandbox: Sandbox | None = None
    ) -> SliceOutcome:
        return self.run({"implementer": implementer, "verifier": VERIFIER_PASS}, sandbox=sandbox)

    def task_events(self) -> list[dict[str, JsonValue]]:
        return [e.data for e in self.ledger.events(self.ticket_id) if e.kind == "task_updated"]


@pytest.fixture
def world(tmp_path: Path, git: Git, docker_runtime: DockerRuntime):
    target = tmp_path / "target"
    shutil.copytree(FIXTURE, target)
    git("init", "-q", cwd=target)
    git("add", "-A", cwd=target)
    git("commit", "-q", "-m", "target", cwd=target)
    state = tmp_path / "state"
    repo = BareRepo.init(state / "repo.git")
    repo.fetch_from(target, ["refs/heads/main:refs/heads/main"])
    with Ledger(state / "ledger.db") as ledger:
        ticket = ledger.create_ticket(
            title="shout", type=TicketType.FEATURE, risk=RiskTier.R1, request="--shout", actor="h"
        )
        ledger.transition(ticket.id, S.TRIAGED, actor="triage")
        ledger.transition(ticket.id, S.CONTRACT_PROPOSED, actor="spec")
        ledger.propose_contract(ticket.id, CONTRACT, actor="spec")
        ledger.set_oracle_version(ticket.id, ORACLE, actor="test-architect")
        ledger.freeze_contract(ticket.id, actor="human")
        yield World(
            ledger=ledger,
            repo=repo,
            project=load_project_config(target),
            ticket_id=ticket.id,
            main=repo.rev_parse("refs/heads/main"),
            key=load_or_create_key(state / "evidence.key"),
            work_root=tmp_path / "work",
            target=target,
            runtime=docker_runtime,
        )


def only_factory_bundles(root: Path) -> list[str]:
    """Everything under `root` that is not a directory or a regular `.bundle` file."""
    odd = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not (
            path.is_dir() or (path.is_file() and path.suffix == ".bundle")
        ):
            odd.append(str(path.relative_to(root)))
    return odd


# --------------------------------------------------------------------------- the exit check


def test_happy_path_in_the_box(world: World, git: Git) -> None:
    sandbox = world.sandbox()
    outcome = world.implement(IMPLEMENT_SHOUT, sandbox=sandbox)

    assert outcome.status == "candidate_verified", outcome
    assert outcome.ticket.status is S.CANDIDATE_VERIFIED
    assert outcome.task.status is TaskStatus.ACCEPTED
    candidate = outcome.candidate_sha
    assert candidate is not None
    parents = git("rev-list", "--parents", "-n", "1", candidate, cwd=world.repo.path).split()[1:]
    assert parents == [world.main]
    assert b"--shout" in world.repo.read_file(candidate, "greet.py")
    assert world.repo.rev_parse("refs/heads/main") == world.main

    evidence = outcome.evidence
    assert evidence is not None and evidence.passed
    assert evidence.image_digests == {"sandbox": sandbox.image_id()}
    assert (
        check_binding(
            evidence,
            sha=candidate,
            tree_hash=world.repo.tree_hash(candidate),
            contract_hash=content_hash(CONTRACT),
            oracle_version=ORACLE,
        )
        == []
    )
    assert outcome.signature is not None
    assert verify_signature(evidence, outcome.signature, world.key)
    assert world.ledger.evidence_for(world.ticket_id, candidate) == evidence

    verdicts = world.ledger.verdicts(world.ticket_id)
    assert [v.round for v in verdicts] == [1, 2]
    assert [v.sha for v in verdicts] == [outcome.task.head_sha, candidate]
    assert {g.name for g in verdicts[1].gates} == {"lint", "unit"}
    assert all(g.status.value == "pass" for g in verdicts[1].gates)
    runs = world.ledger.runs(world.ticket_id)
    assert [r.role for r in runs] == ["implementer", "verifier", "verifier"]
    assert world.ledger.ticket_cost_usd(world.ticket_id) == 0.0

    assert only_factory_bundles(world.work_root) == []  # no worker tree ever lands on the host
    assert docker_objects(world.runtime.instance) == ([], [])


def test_verifier_writes_never_reach_the_candidate(world: World) -> None:
    verifier: list[Action] = [WriteFile("verifier-was-here.txt", "x"), *VERIFIER_PASS]
    outcome = world.run({"implementer": IMPLEMENT_SHOUT, "verifier": verifier})
    assert outcome.status == "candidate_verified", outcome
    assert outcome.candidate_sha is not None
    with pytest.raises(GitError):
        world.repo.read_file(outcome.candidate_sha, "verifier-was-here.txt")


# --------------------------------------------------------------------------- rejections


def test_protected_edit_is_rejected(world: World) -> None:
    outcome = world.implement(
        [*IMPLEMENT_SHOUT, WriteFile("factory.toml", '[project]\nname="x"\n')]
    )
    assert (outcome.status, outcome.rejection) == ("rejected", "policy")
    assert [(v.rule, v.path) for v in outcome.violations] == [("protected_path", "factory.toml")]
    assert world.repo.rev_parse(f"refs/heads/factory/{world.ticket_id}") == world.main


def test_doing_nothing_is_rejected(world: World) -> None:
    outcome = world.implement([])
    assert (outcome.status, outcome.rejection) == ("rejected", "no_changes")
    assert outcome.task.status is TaskStatus.REJECTED


def test_failing_gate_fails_verification(world: World) -> None:
    outcome = world.implement([WriteFile("greet.py", "def broken(:\n")])
    assert outcome.status == "failed_verification"
    assert outcome.verdict is not None and "gate:lint" in outcome.verdict.failing
    assert outcome.evidence is None


def test_a_broken_worker_repo_is_rejected_not_raised(world: World) -> None:
    outcome = world.implement([*IMPLEMENT_SHOUT, RunCommand(("rm", "-rf", ".git"))])
    assert (outcome.status, outcome.rejection) == ("rejected", "worker_git_error")
    assert outcome.ticket.status is S.BUILDING
    last = world.task_events()[-1]
    assert last["reason"] == "worker_git_error"
    detail = last["detail"]
    assert isinstance(detail, dict) and detail.get("error")


def test_an_oversized_bundle_is_rejected(world: World) -> None:
    tight = world.sandbox(max_bundle_bytes=50_000)
    noise = RunCommand(("sh", "-c", "head -c 400000 /dev/urandom > noise.bin"))
    outcome = world.implement([*IMPLEMENT_SHOUT, noise], sandbox=tight)
    assert (outcome.status, outcome.rejection) == ("rejected", "bundle_too_large")


def test_a_run_that_outlives_its_limit_is_stopped(world: World) -> None:
    quick = world.sandbox(run_timeout_s=4.0, command_timeout_s=120.0)
    outcome = world.implement([RunCommand(("sleep", "120"))], sandbox=quick)
    assert outcome.status == "agent_error"
    assert outcome.task.status is TaskStatus.REJECTED
    (run,) = world.ledger.runs(world.ticket_id)
    assert run.status == "error"
    assert run.error is not None and "timeout" in run.error
    assert world.task_events()[-1]["reason"] == "agent_error"
    assert docker_objects(world.runtime.instance) == ([], [])


# --------------------------------------------------------------------------- structured conflict


def test_candidate_conflict_is_recorded_with_paths(world: World, git: Git) -> None:
    first = world.run({"T-0001/task-1": [], "verifier": VERIFIER_PASS})
    assert first.rejection == "no_changes"  # the ticket branch now exists at the old main
    (world.target / "greet.py").write_text("print('main moved on')\n")
    git("commit", "-q", "-am", "main moves", cwd=world.target)
    world.repo.fetch_from(world.target, ["+refs/heads/main:refs/heads/main"])

    second = world.run({"T-0001/task-2": IMPLEMENT_SHOUT, "verifier": VERIFIER_PASS})
    assert (second.status, second.rejection) == ("rejected", "candidate_conflict")
    back = [
        e
        for e in world.ledger.events(world.ticket_id)
        if e.kind == "transition" and e.to_status is S.BUILDING
    ][-1]
    assert back.data == {"reason": "candidate_conflict", "detail": {"conflicts": ["greet.py"]}}
