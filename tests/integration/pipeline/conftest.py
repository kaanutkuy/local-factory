"""Fixtures for the host-slice integration tests: a real target, bare repo, ledger and key.

Fixture repositories are built with an independent git harness (not local_factory.gitops).
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
from collections.abc import Coroutine, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import JsonValue

from local_factory.agents import (
    Action,
    AgentResult,
    AgentTask,
    EmitOutput,
    FakeBackend,
    RoleSpec,
    RunCommand,
    WorkerHandle,
    WriteFile,
)
from local_factory.config import FactoryConfig, ProjectConfig, load_project_config
from local_factory.evidence import load_or_create_key
from local_factory.gitops import BareRepo
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
from local_factory.pipeline import SliceOutcome, run_host_slice

T0 = datetime(2026, 1, 1, tzinfo=UTC)
FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "cli-target"
ORACLE = "sha256:" + "b" * 64
MAIN = "refs/heads/main"
TICKET_REF = "refs/heads/factory/T-0001"

CONTRACT = Contract(
    version=1,
    summary="greet supports --shout",
    criteria=(
        Criterion(id="AC-1", ears="When --shout is given, greet shall shout.", layer=TestLayer.E2E),
        Criterion(id="AC-2", ears="Without --shout, greet shall not shout.", layer=TestLayer.E2E),
    ),
)

GREET = (FIXTURE / "greet.py").read_text()
GREET_PRINT = '    print(f"Hello, {args.name}!")\n'


def greet_with(print_line: str) -> str:
    """The fixture greet.py with its print line replaced (same output, different text)."""
    assert GREET_PRINT in GREET
    return GREET.replace(GREET_PRINT, f"    {print_line}\n")


SHOUT_GREET = GREET.replace(
    "    args = parser.parse_args(argv)\n",
    '    parser.add_argument("--shout", action="store_true")\n    args = parser.parse_args(argv)\n',
).replace(
    GREET_PRINT,
    '    greeting = f"Hello, {args.name}!"\n'
    "    print(greeting.upper() if args.shout else greeting)\n",
)
SHOUT_TEST = """import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class ShoutTest(unittest.TestCase):
    def test_shout(self):
        result = subprocess.run(
            [sys.executable, "greet.py", "--name", "Ada", "--shout"],
            cwd=ROOT, capture_output=True, text=True,
        )
        self.assertEqual(result.stdout, "HELLO, ADA!\\n")
"""
IMPLEMENT_SHOUT: list[Action] = [
    WriteFile("greet.py", SHOUT_GREET),
    WriteFile("tests/test_shout.py", SHOUT_TEST),
]


def report(*statuses: tuple[str, str]) -> list[Action]:
    """A verifier script reporting (criterion id, status) pairs."""
    return [EmitOutput({"criteria": [{"id": cid, "status": status} for cid, status in statuses]})]


VERIFIER_PASS = report(("AC-1", "pass"), ("AC-2", "pass"))
VERIFIER_FAIL = report(("AC-1", "fail"), ("AC-2", "pass"))


class Git:
    """Minimal independent git harness for building fixtures."""

    def __init__(self, home: Path) -> None:
        self.home = home
        self.env = {
            "PATH": os.environ["PATH"],
            "LANG": "C",
            "HOME": str(home),
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_AUTHOR_NAME": "tester",
            "GIT_AUTHOR_EMAIL": "tester@localhost",
            "GIT_COMMITTER_NAME": "tester",
            "GIT_COMMITTER_EMAIL": "tester@localhost",
        }

    def __call__(self, *args: str, cwd: Path) -> str:
        result = subprocess.run(
            ["git", "-c", "init.defaultBranch=main", *args],
            cwd=cwd,
            env=self.env,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()

    def as_agent(self, *args: str) -> RunCommand:
        """A FakeBackend command running git in the worker with a clean env (no user config)."""
        env = [f"{name}={value}" for name, value in self.env.items()]
        return RunCommand(("env", "-i", *env, "git", *args))


# --------------------------------------------------------------------------- backends


class RecordingBackend(FakeBackend):
    """A FakeBackend that also records (role, task id, worker directory) for every call."""

    def __init__(self, scripts: Mapping[str, Sequence[Action]], **kwargs: Any) -> None:
        super().__init__(scripts, **kwargs)
        self.workers: list[tuple[str, str | None, Path]] = []

    async def run(self, role: RoleSpec, task: AgentTask, worker: WorkerHandle) -> AgentResult:
        self.workers.append((role.name, task.task_id, worker.work_dir))
        return await super().run(role, task, worker)


class PerCallBackend(FakeBackend):
    """Each call of a role in `rounds` runs that role's next script (in order)."""

    def __init__(
        self,
        scripts: Mapping[str, Sequence[Action]],
        rounds: Mapping[str, Sequence[Sequence[Action]]],
    ) -> None:
        super().__init__(scripts)
        self._rounds = {role: list(scripts) for role, scripts in rounds.items()}

    async def run(self, role: RoleSpec, task: AgentTask, worker: WorkerHandle) -> AgentResult:
        queue = self._rounds.get(role.name)
        if not queue:
            return await super().run(role, task, worker)
        return await FakeBackend({role.name: queue.pop(0)}).run(role, task, worker)


class GatedBackend(FakeBackend):
    """Holds the run of task `gated` until `release` is set; `waiting` is set once it arrives."""

    def __init__(self, scripts: Mapping[str, Sequence[Action]], *, gated: str) -> None:
        super().__init__(scripts)
        self.gated = gated
        self.waiting = asyncio.Event()
        self.release = asyncio.Event()

    async def run(self, role: RoleSpec, task: AgentTask, worker: WorkerHandle) -> AgentResult:
        if task.task_id == self.gated:
            self.waiting.set()
            await self.release.wait()
        return await super().run(role, task, worker)


class ClaudeLikeBackend(FakeBackend):
    """Scripted, but not named "fake": its models come from the factory config."""

    name = "claude"


# --------------------------------------------------------------------------- the world


@dataclass(frozen=True)
class TaskEnd:
    """How a task ended badly, as the ledger's task_updated event records it."""

    status: TaskStatus
    reason: str
    detail: dict[str, JsonValue]


@dataclass
class World:
    ledger: Ledger
    repo: BareRepo
    project: ProjectConfig
    ticket_id: str
    main: str  # the current main; updated by advance_main, never by the slice
    key: bytes
    work_root: Path
    target: Path
    git: Git
    initial_events: int = field(default=0)

    def start(
        self,
        backend: FakeBackend,
        *,
        main_ref: str = MAIN,
        factory_config: FactoryConfig | None = None,
        key: bytes | None = None,
    ) -> Coroutine[Any, Any, SliceOutcome]:
        return run_host_slice(
            ledger=self.ledger,
            repo=self.repo,
            project=self.project,
            backend=backend,
            ticket_id=self.ticket_id,
            work_root=self.work_root,
            key=self.key if key is None else key,
            main_ref=main_ref,
            clock=lambda: T0,
            factory_config=factory_config,
        )

    def run(self, backend: FakeBackend, **kwargs: Any) -> SliceOutcome:
        return asyncio.run(self.start(backend, **kwargs))

    def run_scripts(
        self, implementer: Sequence[Action], verifier: Sequence[Action] = VERIFIER_PASS
    ) -> SliceOutcome:
        return self.run(FakeBackend({"implementer": implementer, "verifier": verifier}))

    def advance_main(self, files: Mapping[str, str], message: str = "main moved") -> str:
        """Commit `files` on the target's main and fetch it into the bare repo (fast-forward)."""
        for path, content in files.items():
            (self.target / path).parent.mkdir(parents=True, exist_ok=True)
            (self.target / path).write_text(content)
        self.git("add", "-A", cwd=self.target)
        self.git("commit", "-q", "-m", message, cwd=self.target)
        self.repo.fetch_from(self.target, [f"{MAIN}:{MAIN}"])
        self.main = self.repo.rev_parse(MAIN)
        return self.main

    def refs(self, prefix: str = "refs/") -> list[str]:
        out = self.git("for-each-ref", "--format=%(refname)", prefix, cwd=self.repo.path)
        return out.splitlines()

    def parents(self, sha: str) -> list[str]:
        return self.git("rev-list", "--parents", "-n", "1", sha, cwd=self.repo.path).split()[1:]

    def message(self, sha: str) -> str:
        return self.git("log", "-1", "--format=%B", sha, cwd=self.repo.path)

    def transitions(self) -> list[TicketStatus | None]:
        events = self.ledger.events(self.ticket_id)
        return [e.to_status for e in events if e.kind == "transition"]

    def task_ends(self) -> dict[str, TaskEnd]:
        """Every task's REJECTED/HALTED update, read back from the ledger's events alone."""
        ends: dict[str, TaskEnd] = {}
        for event in self.ledger.events(self.ticket_id):
            if event.kind != "task_updated":
                continue
            task_id, reason = event.data["task_id"], event.data["reason"]
            changes, detail = event.data["changes"], event.data["detail"]
            assert isinstance(task_id, str) and isinstance(reason, str)
            assert isinstance(changes, dict) and isinstance(detail, dict)
            status = changes.get("status")
            if status in (TaskStatus.REJECTED, TaskStatus.HALTED):
                assert isinstance(status, str)
                ends[task_id] = TaskEnd(TaskStatus(status), reason, detail)
        return ends

    def assert_boundaries(self) -> None:
        """The slice never moves main and leaves no quarantine or temporary base refs behind."""
        assert self.repo.rev_parse(MAIN) == self.main
        assert self.refs("refs/factory/quarantine/") == []
        assert [ref for ref in self.refs("refs/heads/") if "factory-base-" in ref] == []


@pytest.fixture
def git(tmp_path: Path) -> Git:
    home = tmp_path / "home"
    home.mkdir()
    return Git(home)


@pytest.fixture
def world(tmp_path: Path, git: Git) -> Iterator[World]:
    target = tmp_path / "target"
    shutil.copytree(FIXTURE, target)
    git("init", "-q", cwd=target)
    git("add", "-A", cwd=target)
    git("commit", "-q", "-m", "target", cwd=target)

    state = tmp_path / "state"
    repo = BareRepo.init(state / "repo.git")
    repo.fetch_from(target, [f"{MAIN}:{MAIN}"])
    with Ledger(state / "ledger.db", clock=lambda: T0) as ledger:
        ticket = ledger.create_ticket(
            title="shout",
            type=TicketType.FEATURE,
            risk=RiskTier.R1,
            request="--shout",
            actor="human",
        )
        ledger.transition(ticket.id, TicketStatus.TRIAGED, actor="triage")
        ledger.transition(ticket.id, TicketStatus.CONTRACT_PROPOSED, actor="spec")
        ledger.propose_contract(ticket.id, CONTRACT, actor="spec")
        ledger.set_oracle_version(ticket.id, ORACLE, actor="test-architect")
        ledger.freeze_contract(ticket.id, actor="human")
        yield World(
            ledger=ledger,
            repo=repo,
            project=load_project_config(target),
            ticket_id=ticket.id,
            main=repo.rev_parse(MAIN),
            key=load_or_create_key(state / "evidence.key"),
            work_root=tmp_path / "work",
            target=target,
            git=git,
            initial_events=len(ledger.events(ticket.id)),
        )
