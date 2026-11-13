"""Red team, Phase 2b: the holdout stays sealed and the test architect stays in its lane
(docs/specs/phase-2b.md § Red-team items). Protected; `container`.
"""

from __future__ import annotations

import asyncio
import stat
from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import JsonValue

from local_factory.agents.fake import Action, EmitOutput, RunCommand, WriteFile
from local_factory.oracle import oracle_ref
from local_factory.pipeline.intake import (
    MAX_ORACLE_ATTEMPTS,
    IntakeOutcome,
    receive,
    run_intake,
)
from local_factory.pipeline.sandboxed_slice import box_site
from local_factory.sandbox import Sandbox
from tests.acceptance.intake_world import (
    REQUEST,
    SHOUT_ID,
    SHOUT_TEST,
    TEST_HEADER,
    ScriptedBackend,
    World,
    acceptance_path,
    happy_scripts,
    make_world,
    manifest,
    oracle_script,
    shout_probe,
)

pytestmark = [pytest.mark.container, pytest.mark.integration]

CANARY = "LFHOLDOUT-7c1d-canary"
# Searches the probe's box for the canary without spelling it out in its own argv.
SEARCH = (
    'p="LFHOLD""OUT-7c1d"; '
    'grep -rsl --exclude-dir=proc --exclude-dir=sys --exclude-dir=dev -- "${p}-canary" /; '
    "exit 3"
)


@pytest.fixture
def gate1(tmp_path: Path) -> Iterator[World]:
    with make_world(tmp_path) as world:
        yield world


def intake(w: World, backend: ScriptedBackend, sandbox: Sandbox, ticket_id: str) -> IntakeOutcome:
    return asyncio.run(run_intake(w.intake(backend, box_site(sandbox)), ticket_id))


def files_containing(root: Path, needle: bytes) -> list[str]:
    return sorted(
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file() and needle in p.read_bytes()
    )


def test_a_probe_expectation_never_leaves_the_holdout_store(gate1: World, sandbox: Sandbox) -> None:
    secret_probe: dict[str, JsonValue] = {
        "criteria": ["AC-1"],
        "argv": ["python3", "greet.py", "--name", "Ada", "--shout"],
        "expect": {"exit_code": 0, "stdout": CANARY},
    }
    search_probe: dict[str, JsonValue] = {
        "criteria": ["AC-1"],
        "argv": ["sh", "-c", SEARCH],
        "expect": {"exit_code": 0},
    }
    scripts = happy_scripts()
    scripts["test-architect"] = [oracle_script(out=manifest(holdout=[secret_probe, search_probe]))]
    ticket = receive(gate1.ledger, REQUEST)
    outcome = intake(gate1, ScriptedBackend(scripts), sandbox, ticket.id)

    assert outcome.status == "contract_proposed", outcome
    checks = {c.id: c for c in outcome.red_checks}
    assert checks["HP-2"].reason == "failed on the base: exit code 3 differs"
    assert checks["HP-2"].output == ""  # no file in the probe's box holds the expectation

    # nowhere in the factory's state but the sealed record
    state = gate1.ledger.path.parent
    assert files_containing(state, CANARY.encode()) == [f"holdout/{ticket.id}.json"]
    assert CANARY.encode() not in gate1.all_objects()
    record = gate1.holdout.path(ticket.id)
    assert stat.S_IMODE(record.stat().st_mode) == 0o600
    assert stat.S_IMODE(record.parent.stat().st_mode) == 0o700


def test_the_test_architect_can_only_add_its_own_tests(gate1: World, sandbox: Sandbox) -> None:
    edits: list[Action] = [
        WriteFile(acceptance_path(), SHOUT_TEST),
        WriteFile("greet.py", "print('HELLO, ADA!')\n"),
        WriteFile("factory.toml", '[project]\nname = "cli-target"\n'),
        WriteFile(acceptance_path("T-0002"), SHOUT_TEST),
        WriteFile(".github/workflows/ci.yml", "on: push\n"),
        EmitOutput(manifest()),
    ]
    # A symlink out of the repository never even imports: the bundle check rejects it first.
    link: list[Action] = [
        WriteFile(acceptance_path(), SHOUT_TEST),
        RunCommand(("ln", "-s", "/etc/passwd", acceptance_path(file="passwd"))),
        EmitOutput(manifest()),
    ]
    scripts = happy_scripts()
    scripts["test-architect"] = [list(edits), list(link), list(edits)]
    assert len(scripts["test-architect"]) == MAX_ORACLE_ATTEMPTS
    ticket = receive(gate1.ledger, REQUEST)
    outcome = intake(gate1, ScriptedBackend(scripts), sandbox, ticket.id)

    assert outcome.status == "oracle_failed"
    rejected = [e.data for e in gate1.ledger.events(ticket.id) if e.kind == "oracle_rejected"]
    assert [r["attempt"] for r in rejected] == [1, 2, 3]
    problems = str(rejected[0]["problems"])
    for path in ("greet.py", "factory.toml", "T-0002", ".github/workflows/ci.yml"):
        assert path in problems, path
    assert "symlink_escape" in str(rejected[1]["problems"])
    assert "/etc/passwd" in str(rejected[1]["problems"])
    assert gate1.refs("refs/factory/") == []
    assert gate1.holdout.load(ticket.id) is None
    assert [e for e in gate1.ledger.events(ticket.id) if e.kind == "red_first"] == []


def test_test_code_runs_only_in_the_box(gate1: World, sandbox: Sandbox, tmp_path: Path) -> None:
    planted = tmp_path / "written-by-a-test"
    escape = (
        TEST_HEADER
        + f"""
from pathlib import Path as _P

try:
    _P({str(planted)!r}).write_text("escaped")
except OSError:
    pass


class ShoutTest(unittest.TestCase):
    def test_shout(self):
        self.assertEqual(greet("--name", "Ada", "--shout").stdout, "HELLO, ADA!\\n")
"""
    )
    probe = shout_probe()
    probe["argv"] = ["sh", "-c", f"echo escaped > {planted}; exit 1"]
    scripts = happy_scripts()
    scripts["test-architect"] = [
        oracle_script(
            files={acceptance_path(): escape},
            out=manifest(visible=((SHOUT_ID, ("AC-1",)),), holdout=[probe]),
        )
    ]
    ticket = receive(gate1.ledger, REQUEST)
    outcome = intake(gate1, ScriptedBackend(scripts), sandbox, ticket.id)
    assert outcome.status == "contract_proposed", outcome
    assert gate1.repo.rev_parse(oracle_ref(ticket.id))
    assert not planted.exists()
