"""The Gate 1 test world (docs/specs/phase-2b.md). Protected.

A target repository made from `tests/fixtures/cli-target`, the factory's bare repository, ledger
and holdout store for it, and `ScriptedBackend`: per role, a queue of `FakeBackend` scripts, one
per run of that role, recording every task it was given.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import JsonValue

from local_factory.agents import AgentResult, AgentTask, FakeBackend, RoleSpec, WorkerHandle
from local_factory.agents.backend import AgentBackend, Capability
from local_factory.agents.fake import Action, EmitOutput, WriteFile
from local_factory.config import ProjectConfig, load_project_config
from local_factory.gitops.bare import BareRepo
from local_factory.ledger import Ledger
from local_factory.oracle import HoldoutStore
from local_factory.pipeline.flow import Site
from local_factory.pipeline.host_slice import host_site
from tests.acceptance.conftest import Git

if TYPE_CHECKING:
    from local_factory.pipeline.intake import Intake

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "cli-target"
MAIN = "refs/heads/main"
ACCEPTANCE = "python3 -m unittest {path} -k {name}"

REQUEST = (
    "greet.py should accept a --shout flag that prints the greeting in upper case.\n"
    "Without the flag nothing changes."
)
EARS = (
    "When greet.py is run with --name NAME --shout, "
    "the system shall print the greeting in upper case."
)
TEST_PLAN = "Run greet.py --name Ada --shout and compare its output."
KEEP_EARS = (
    "When greet.py is run with --name NAME and without --shout, "
    "the system shall print the greeting exactly as it does today."
)
KEEP_PLAN = "Run greet.py --name Ada and compare its output with today's."
SPEC_ASSUMPTION = "Names are plain ASCII."
CRITIC_ASSUMPTION = "Upper case means str.upper()."

# What a probe expects is the holdout's secret; tests look for these strings where they must not be.
SHOUT_PROBE_STDOUT = "HELLO, GRACE HOPPER!"
PLAIN_PROBE_STDOUT = "Hello, Grace!"


def acceptance_path(ticket: str = "T-0001", file: str = "test_shout.py") -> str:
    return f"tests/acceptance/{ticket}/{file}"


TEST_HEADER = """import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def greet(*args):
    return subprocess.run(
        [sys.executable, "greet.py", *args], cwd=ROOT, capture_output=True, text=True
    )

"""

SHOUT_TEST = (
    TEST_HEADER
    + """
class ShoutTest(unittest.TestCase):
    def test_shout(self):
        self.assertEqual(greet("--name", "Ada", "--shout").stdout, "HELLO, ADA!\\n")
"""
)

# Passes on the base: it tests nothing new.
VACUOUS_TEST = (
    TEST_HEADER
    + """
class PlainTest(unittest.TestCase):
    def test_plain(self):
        self.assertEqual(greet("--name", "Ada").stdout, "Hello, Ada!\\n")
"""
)


# A keep test that is wrong about today's behaviour: it fails on the base.
WRONG_KEEP_TEST = (
    TEST_HEADER
    + """
class PlainTest(unittest.TestCase):
    def test_plain(self):
        self.assertEqual(greet("--name", "Ada").stdout, "Hi, Ada!\\n")
"""
)


def shout_id(ticket: str = "T-0001") -> str:
    return f"{acceptance_path(ticket)}::ShoutTest.test_shout"


SHOUT_ID = shout_id()
VACUOUS_ID = f"{acceptance_path(file='test_plain.py')}::PlainTest.test_plain"
KEEP_ID = VACUOUS_ID  # vacuous for a change, exactly right for a keep criterion


def shout_probe(criteria: Sequence[str] = ("AC-1",)) -> dict[str, JsonValue]:
    return {
        "criteria": list(criteria),
        "argv": ["python3", "greet.py", "--name", "Grace Hopper", "--shout"],
        "expect": {"exit_code": 0, "stdout": SHOUT_PROBE_STDOUT},
    }


def plain_probe(criteria: Sequence[str] = ("AC-1",)) -> dict[str, JsonValue]:
    """Passes on the base."""
    return {
        "criteria": list(criteria),
        "argv": ["python3", "greet.py", "--name", "Grace"],
        "expect": {"exit_code": 0, "stdout": PLAIN_PROBE_STDOUT},
    }


def triage_out(**overrides: JsonValue) -> dict[str, JsonValue]:
    out: dict[str, JsonValue] = {
        "title": "Add --shout to greet.py",
        "type": "feature",
        "risk": "R1",
        "size": "S",
        "areas": ["greet.py"],
        "rationale": "A small CLI flag.",
    }
    out.update(overrides)
    return out


def criterion(
    ears: str = EARS, plan: str = TEST_PLAN, kind: str = "change"
) -> dict[str, JsonValue]:
    return {"ears": ears, "layer": "e2e", "test_plan": plan, "kind": kind}


def keep_criterion() -> dict[str, JsonValue]:
    return criterion(KEEP_EARS, KEEP_PLAN, "keep")


def spec_out(*criteria: dict[str, JsonValue], **overrides: JsonValue) -> dict[str, JsonValue]:
    out: dict[str, JsonValue] = {
        "summary": "greet.py --shout prints the greeting in upper case.",
        "criteria": list(criteria) or [criterion()],
        "nfr": [],
        "out_of_scope": ["Other scripts."],
        "assumptions": [SPEC_ASSUMPTION],
        "migration_impact": "none",
    }
    out.update(overrides)
    return out


def critique_out(*questions: str, assumptions: Sequence[str] = ()) -> dict[str, JsonValue]:
    return {
        "blocking": [{"text": q, "why_blocking": "A wrong guess is rejected."} for q in questions],
        "assumptions": list(assumptions),
    }


def manifest(
    visible: Sequence[tuple[str, Sequence[str]]] | None = None,
    holdout: Sequence[dict[str, JsonValue]] | None = None,
    *,
    ticket: str = "T-0001",
) -> dict[str, JsonValue]:
    if visible is None:
        visible = ((shout_id(ticket), ("AC-1",)),)
    return {
        "visible": [{"test": test, "criteria": list(criteria)} for test, criteria in visible],
        "holdout": list(holdout) if holdout is not None else [shout_probe()],
        "notes": "",
    }


def oracle_script(
    files: Mapping[str, str] | None = None,
    out: dict[str, JsonValue] | None = None,
    *,
    ticket: str = "T-0001",
) -> list[Action]:
    """The test architect: write `files`, submit `out` (default: the shout test and probe)."""
    files = {acceptance_path(ticket): SHOUT_TEST} if files is None else files
    actions: list[Action] = [WriteFile(path, content) for path, content in files.items()]
    actions.append(EmitOutput(manifest(ticket=ticket) if out is None else out))
    return actions


def happy_scripts(
    *, critic: dict[str, JsonValue] | None = None, ticket: str = "T-0001"
) -> dict[str, list[list[Action]]]:
    return {
        "triage": [[EmitOutput(triage_out())]],
        "spec-writer": [[EmitOutput(spec_out())]],
        "ambiguity-critic": [
            [
                EmitOutput(
                    critique_out(assumptions=[CRITIC_ASSUMPTION]) if critic is None else critic
                )
            ]
        ],
        "test-architect": [oracle_script(ticket=ticket)],
    }


def keep_oracle(
    plain_test: str = VACUOUS_TEST,
    visible: Sequence[tuple[str, Sequence[str]]] | None = None,
    holdout: Sequence[dict[str, JsonValue]] | None = None,
) -> list[Action]:
    """The test architect for a spec with AC-1 (change: --shout) and AC-2 (keep: no flag)."""
    return oracle_script(
        files={acceptance_path(): SHOUT_TEST, acceptance_path(file="test_plain.py"): plain_test},
        out=manifest(
            visible=((SHOUT_ID, ("AC-1",)), (KEEP_ID, ("AC-2",))) if visible is None else visible,
            holdout=[shout_probe(), plain_probe(("AC-2",))] if holdout is None else holdout,
        ),
    )


def keep_scripts() -> dict[str, list[list[Action]]]:
    """The happy path for a spec whose AC-2 keeps today's behaviour."""
    scripts = happy_scripts()
    scripts["spec-writer"] = [[EmitOutput(spec_out(criterion(), keep_criterion()))]]
    scripts["test-architect"] = [keep_oracle()]
    return scripts


@dataclass
class ScriptedBackend:
    """Each run of a role takes that role's next script; every call is recorded."""

    scripts: dict[str, list[list[Action]]]
    calls: list[tuple[RoleSpec, AgentTask]] = field(default_factory=list)
    name: str = FakeBackend.name
    capabilities: frozenset[Capability] = FakeBackend.capabilities

    def add(self, role: str, *scripts: list[Action]) -> None:
        self.scripts.setdefault(role, []).extend(scripts)

    async def run(self, role: RoleSpec, task: AgentTask, worker: WorkerHandle) -> AgentResult:
        self.calls.append((role, task))
        queue = self.scripts.get(role.name, [])
        if not queue:
            return AgentResult(status="error", error=f"no script left for {role.name}")
        return await FakeBackend({role.name: queue.pop(0)}).run(role, task, worker)

    def runs(self, role: str) -> list[AgentTask]:
        return [task for spec, task in self.calls if spec.name == role]

    def roles(self) -> list[str]:
        return [spec.name for spec, _ in self.calls]


@dataclass
class World:
    root: Path
    target: Path
    repo: BareRepo
    ledger: Ledger
    holdout: HoldoutStore
    project: ProjectConfig
    work_root: Path
    git: Git

    @property
    def main(self) -> str:
        return self.repo.rev_parse(MAIN)

    def intake(self, backend: AgentBackend, site: Site | None = None) -> Intake:
        # imported here so the oracle agent's tests import without the intake module
        from local_factory.pipeline.intake import Intake

        return Intake(
            site=host_site() if site is None else site,
            ledger=self.ledger,
            repo=self.repo,
            project=self.project,
            backend=backend,
            holdout=self.holdout,
            work_root=self.work_root,
        )

    def files_at(self, rev: str) -> dict[str, bytes]:
        paths = [entry.path.decode() for entry in self.repo.ls_tree(rev)]
        return {path: self.repo.read_file(rev, path) for path in paths}

    def all_objects(self) -> bytes:
        """Every object in the bare repository, concatenated (to search for leaks)."""
        return self.repo.git(["cat-file", "--batch-all-objects", "--batch"]).stdout

    def refs(self, prefix: str = "refs/") -> list[str]:
        out = self.repo.git(["for-each-ref", "--format=%(refname)", prefix]).stdout
        return [line for line in out.decode().splitlines() if line]


def make_target(root: Path, git: Git) -> Path:
    target = root / "target"
    shutil.copytree(FIXTURE, target)
    git("init", "-q", cwd=target)
    git("add", "-A", cwd=target)
    git("commit", "-q", "-m", "target", cwd=target)
    return target


@contextmanager
def make_world(root: Path) -> Iterator[World]:
    (root / "githome").mkdir(parents=True, exist_ok=True)
    git = Git(root / "githome")
    target = make_target(root, git)
    state = root / "state"
    repo = BareRepo.init(state / "repo.git")
    repo.fetch_from(target, [f"{MAIN}:{MAIN}"])
    with Ledger(state / "ledger.db") as ledger:
        yield World(
            root=root,
            target=target,
            repo=repo,
            ledger=ledger,
            holdout=HoldoutStore(state / "holdout"),
            project=load_project_config(target),
            work_root=state / "work",
            git=git,
        )
