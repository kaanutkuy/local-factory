"""Unit tests for the oracle loop of intake: what a rejected attempt leaves behind, what the
contract is made of, and the cleanup guarantees (docs/specs/phase-2b.md § intake, step 4).

The common rejections (vacuous tests, probes that pass, a manifest that does not add up) are in
tests/acceptance/test_intake.py; these cover the rest.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import JsonValue

from local_factory.agents.fake import Action, EmitOutput, RunCommand
from local_factory.models import RedCheck
from local_factory.oracle import oracle_ref
from local_factory.pipeline import IntakeOutcome, receive, run_intake
from local_factory.pipeline import intake as intake_module
from local_factory.pipeline.intake import MAX_ORACLE_ATTEMPTS
from tests.acceptance.intake_world import (
    CRITIC_ASSUMPTION,
    REQUEST,
    SHOUT_ID,
    SHOUT_TEST,
    SPEC_ASSUMPTION,
    VACUOUS_ID,
    VACUOUS_TEST,
    ScriptedBackend,
    World,
    acceptance_path,
    criterion,
    critique_out,
    happy_scripts,
    make_world,
    manifest,
    oracle_script,
    shout_probe,
    spec_out,
)


@pytest.fixture
def gate1(tmp_path: Path) -> Iterator[World]:
    with make_world(tmp_path) as world:
        yield world


def attempts(w: World, scripts: list[list[Action]]) -> tuple[ScriptedBackend, IntakeOutcome]:
    """Intake on a new ticket whose test architect plays ``scripts``, one per attempt."""
    all_scripts = happy_scripts()
    all_scripts["test-architect"] = scripts
    backend = ScriptedBackend(all_scripts)
    ticket = receive(w.ledger, REQUEST)
    return backend, asyncio.run(run_intake(w.intake(backend), ticket.id))


def rejections(w: World) -> list[list[str]]:
    found = [e.data["problems"] for e in w.ledger.events("T-0001") if e.kind == "oracle_rejected"]
    return [
        [str(problem) for problem in problems] for problems in found if isinstance(problems, list)
    ]


# --------------------------------------------------------------------------- what gets rejected


def test_every_violation_is_reported_in_one_go(gate1: World) -> None:
    attack = oracle_script(
        files={
            acceptance_path(): SHOUT_TEST,
            "greet.py": "print('rigged')\n",
            "factory.toml": "[project]\nname = 'x'\n",
            acceptance_path("T-0002"): SHOUT_TEST,
            ".github/workflows/ci.yml": "on: push\n",
        }
    )
    backend, outcome = attempts(gate1, [attack, attack, attack])
    assert outcome.status == "oracle_failed"
    [first, *_] = rejections(gate1)
    assert len(first) == 4 and all(problem.startswith("oracle_path ") for problem in first)
    for path in ("greet.py", "factory.toml", "T-0002", ".github/workflows/ci.yml"):
        assert sum(path in problem for problem in first) == 1, path
    assert gate1.refs("refs/factory/") == [] and gate1.holdout.load("T-0001") is None
    assert [e for e in gate1.ledger.events("T-0001") if e.kind == "red_first"] == []
    assert len(backend.runs("test-architect")) == MAX_ORACLE_ATTEMPTS


def test_a_symlink_the_import_refuses_is_the_attempts_problem(gate1: World) -> None:
    attack = oracle_script()
    attack.insert(1, RunCommand(("ln", "-s", "/etc/passwd", acceptance_path(file="passwd"))))
    _, outcome = attempts(gate1, [attack, attack, attack])
    assert outcome.status == "oracle_failed"
    [first, *_] = rejections(gate1)
    [problem] = first
    assert problem.startswith("symlink_escape: ") and "tests/acceptance/T-0001/passwd" in problem
    assert gate1.refs("refs/factory/") == []


def test_the_failure_says_why_and_keeps_the_last_red_first_run(gate1: World) -> None:
    vacuous = oracle_script(
        files={acceptance_path(): SHOUT_TEST, acceptance_path(file="test_plain.py"): VACUOUS_TEST},
        out=manifest(visible=((SHOUT_ID, ("AC-1",)), (VACUOUS_ID, ("AC-1",)))),
    )
    rigged = oracle_script(files={"greet.py": "x = 1\n"})
    backend, outcome = attempts(gate1, [vacuous, rigged, rigged])

    assert outcome.status == "oracle_failed"
    assert outcome.error is not None
    assert outcome.error.startswith(f"the tests were rejected {MAX_ORACLE_ATTEMPTS} times")
    assert "greet.py" in outcome.error and VACUOUS_ID not in outcome.error  # the last attempt's
    assert [(c.id, c.red) for c in outcome.red_checks] == [
        (SHOUT_ID, True),
        (VACUOUS_ID, False),
        ("HP-1", True),
    ]  # attempt 1's: the last run of red-first
    assert outcome.ticket.status.value == "triaged"
    retry = backend.runs("test-architect")[1].instructions
    assert "attempt 2" in retry and f"{VACUOUS_ID}: passed on the base" in retry
    third = backend.runs("test-architect")[2].instructions
    assert "greet.py" in third and VACUOUS_ID not in third  # only the previous attempt's problems


def test_the_first_attempt_is_told_nothing_about_problems(gate1: World) -> None:
    backend, outcome = attempts(gate1, [oracle_script()])
    assert outcome.status == "contract_proposed"
    [first] = backend.runs("test-architect")
    assert "attempt 1" in first.instructions and "rejected" not in first.instructions


def test_each_attempt_works_in_a_directory_of_its_own(gate1: World) -> None:
    rigged = oracle_script(files={"greet.py": "x = 1\n"})
    backend, outcome = attempts(gate1, [rigged] * MAX_ORACLE_ATTEMPTS)
    assert outcome.status == "oracle_failed"
    backend.add("spec-writer", [EmitOutput(spec_out())])
    backend.add("ambiguity-critic", [EmitOutput(critique_out())])
    backend.add("test-architect", oracle_script())
    assert asyncio.run(run_intake(gate1.intake(backend), "T-0001")).status == "contract_proposed"
    names = sorted(p.name for p in (gate1.work_root / "T-0001").glob("oracle-*"))
    assert names == ["oracle-1", "oracle-1-2", "oracle-2", "oracle-3"]


# --------------------------------------------------------------------------- cleanup


def test_the_candidate_ref_goes_even_when_red_first_breaks(
    gate1: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def broken(**_: object) -> tuple[RedCheck, ...]:
        raise RuntimeError("the box fell over")

    monkeypatch.setattr(intake_module, "red_first", broken)
    ticket = receive(gate1.ledger, REQUEST)
    backend = ScriptedBackend(happy_scripts())
    with pytest.raises(RuntimeError, match="fell over"):
        asyncio.run(run_intake(gate1.intake(backend), ticket.id))
    assert gate1.refs("refs/factory/") == []
    assert gate1.ledger.get_ticket(ticket.id).status.value == "triaged"


def test_an_accepted_oracle_leaves_only_the_oracle_ref(gate1: World) -> None:
    _, outcome = attempts(gate1, [oracle_script()])
    assert outcome.status == "contract_proposed"
    assert gate1.refs("refs/factory/") == [oracle_ref("T-0001")]


# --------------------------------------------------------------------------- the contract


def test_the_contract_collects_tests_probes_and_assumptions(gate1: World) -> None:
    second = criterion(
        "If --shout is given twice, the system shall print the greeting in upper case once.",
        "Run greet.py --name Ada --shout --shout.",
    )
    twice = SHOUT_TEST.replace("ShoutTest", "TwiceTest").replace(
        '"--shout")', '"--shout", "--shout")'
    )
    twice_id = f"{acceptance_path(file='test_twice.py')}::TwiceTest.test_shout"
    twice_probe: dict[str, JsonValue] = {
        "criteria": ["AC-2"],
        "argv": ["python3", "greet.py", "--name", "Ada", "--shout", "--shout"],
        "expect": {"exit_code": 0, "stdout": "HELLO, ADA!"},
    }
    architect = oracle_script(
        files={acceptance_path(): SHOUT_TEST, acceptance_path(file="test_twice.py"): twice},
        out=manifest(
            visible=((SHOUT_ID, ("AC-1",)), (twice_id, ("AC-1", "AC-2"))),
            holdout=[shout_probe(("AC-1",)), shout_probe(("AC-1", "AC-2")), twice_probe],
        ),
    )
    scripts = happy_scripts(critic=critique_out(assumptions=[SPEC_ASSUMPTION, "Extra."]))
    scripts["spec-writer"] = [[EmitOutput(spec_out(criterion(), second))]]
    scripts["test-architect"] = [architect]
    ticket = receive(gate1.ledger, REQUEST)
    outcome = asyncio.run(run_intake(gate1.intake(ScriptedBackend(scripts)), ticket.id))
    assert outcome.status == "contract_proposed", outcome.error

    contract = gate1.ledger.get_ticket(ticket.id).contract
    assert contract is not None
    first, other = contract.criteria
    assert (first.id, first.tests, first.holdout_probes) == ("AC-1", (SHOUT_ID, twice_id), 2)
    assert (other.id, other.tests, other.holdout_probes) == ("AC-2", (twice_id,), 2)
    # the draft's assumption, then the critic's new one (a repeat keeps its first place)
    assert contract.assumptions == (SPEC_ASSUMPTION, "Extra.")
    assert CRITIC_ASSUMPTION not in contract.assumptions

    record = gate1.holdout.load(ticket.id)
    assert record is not None
    assert [(p.id, p.criteria) for p in record.probes] == [
        ("HP-1", ("AC-1",)),
        ("HP-2", ("AC-1", "AC-2")),
        ("HP-3", ("AC-2",)),
    ]
