"""Gate 1, in process: intake roles, questions, the oracle, red-first, approval
(docs/specs/phase-2b.md § intake). Protected.

The fake backend plays the roles and the in-process host site stands in for the box (the box runs
the same flow in test_intake_box.py), so every rule here is checked without Docker or tokens.
"""

from __future__ import annotations

import asyncio
import stat
from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import JsonValue

from local_factory.agents import BOX_TOOLS
from local_factory.agents.fake import Action, EmitOutput, Fail, RaiseHalt, WriteFile
from local_factory.agents.roles import (
    CRITIC_ROLE,
    INTAKE_ROLES,
    SPEC_WRITER_ROLE,
    TEST_ARCHITECT_ROLE,
    TRIAGE_ROLE,
)
from local_factory.hashing import content_hash
from local_factory.models import (
    ChangeKind,
    CriterionKind,
    Critique,
    HaltReason,
    Holdout,
    OracleManifest,
    RiskTier,
    SpecDraft,
    TicketStatus,
    TicketType,
    Tier,
    TriageResult,
)
from local_factory.oracle import oracle_ref, oracle_version
from local_factory.pipeline.intake import (
    MAX_CLARIFY_ROUNDS,
    MAX_ORACLE_ATTEMPTS,
    IntakeError,
    IntakeOutcome,
    answer,
    approve,
    open_questions,
    questions,
    receive,
    run_intake,
)
from tests.acceptance.intake_world import (
    ACCEPTANCE,
    CRITIC_ASSUMPTION,
    EARS,
    KEEP_EARS,
    KEEP_ID,
    MAIN,
    REQUEST,
    SHOUT_ID,
    SHOUT_PROBE_STDOUT,
    SHOUT_TEST,
    SPEC_ASSUMPTION,
    TEST_PLAN,
    VACUOUS_ID,
    VACUOUS_TEST,
    WRONG_KEEP_TEST,
    ScriptedBackend,
    World,
    acceptance_path,
    criterion,
    critique_out,
    happy_scripts,
    keep_criterion,
    keep_oracle,
    keep_scripts,
    make_world,
    manifest,
    oracle_script,
    plain_probe,
    shout_probe,
    spec_out,
    triage_out,
)

pytestmark = pytest.mark.integration


@pytest.fixture
def gate1(tmp_path: Path) -> Iterator[World]:
    with make_world(tmp_path) as world:
        yield world


def intake(w: World, backend: ScriptedBackend, ticket_id: str) -> IntakeOutcome:
    return asyncio.run(run_intake(w.intake(backend), ticket_id))


def events(w: World, ticket_id: str, kind: str) -> list[dict[str, JsonValue]]:
    return [e.data for e in w.ledger.events(ticket_id) if e.kind == kind]


def red_flags(data: dict[str, JsonValue]) -> list[JsonValue]:
    checks = data["checks"]
    assert isinstance(checks, list)
    return [check["red"] for check in checks if isinstance(check, dict)]


def propose(w: World, backend: ScriptedBackend | None = None) -> IntakeOutcome:
    ticket = receive(w.ledger, REQUEST)
    return intake(w, backend or ScriptedBackend(happy_scripts()), ticket.id)


# --------------------------------------------------------------------------- roles


def test_the_intake_roles() -> None:
    expected = {
        TRIAGE_ROLE: ("triage", Tier.STANDARD, "medium", TriageResult),
        SPEC_WRITER_ROLE: ("spec-writer", Tier.FRONTIER, "high", SpecDraft),
        CRITIC_ROLE: ("ambiguity-critic", Tier.FRONTIER, "high", Critique),
        TEST_ARCHITECT_ROLE: ("test-architect", Tier.FRONTIER, "high", OracleManifest),
    }
    assert INTAKE_ROLES == (TRIAGE_ROLE, SPEC_WRITER_ROLE, CRITIC_ROLE, TEST_ARCHITECT_ROLE)
    for role, (name, tier, effort, schema) in expected.items():
        assert (role.name, role.tier, role.effort, role.output_schema) == (
            name,
            tier,
            effort,
            schema,
        )
        assert role.tools == BOX_TOOLS
        assert "data" in role.prompt and "instructions" in role.prompt  # repo text is data
    assert "tests/acceptance" in TEST_ARCHITECT_ROLE.prompt


# --------------------------------------------------------------------------- receive


def test_receive_records_a_conservative_ticket(gate1: World) -> None:
    ticket = receive(gate1.ledger, "\n\n  " + "x" * 100 + "\nmore detail\n  ")
    assert ticket.status is TicketStatus.RECEIVED
    assert ticket.risk is RiskTier.R2  # unknown risk is the highest until triage
    assert ticket.type is TicketType.FEATURE
    assert ticket.title == "x" * 80
    assert ticket.request == "x" * 100 + "\nmore detail"


@pytest.mark.parametrize("text", ["", "   \n ", "y" * 20_001])
def test_receive_refuses_an_empty_or_huge_request(gate1: World, text: str) -> None:
    with pytest.raises(IntakeError):
        receive(gate1.ledger, text)
    assert gate1.ledger.list_tickets() == []


# --------------------------------------------------------------------------- the happy path


def test_a_request_becomes_a_proposed_contract(gate1: World) -> None:
    main = gate1.main
    backend = ScriptedBackend(happy_scripts())
    outcome = propose(gate1, backend)

    assert outcome.status == "contract_proposed", outcome
    ticket = gate1.ledger.get_ticket("T-0001")
    assert outcome.ticket == ticket
    assert ticket.status is TicketStatus.CONTRACT_PROPOSED
    assert backend.roles() == ["triage", "spec-writer", "ambiguity-critic", "test-architect"]

    contract = ticket.contract
    assert contract is not None and contract.version == 1
    [ac] = contract.criteria
    assert (ac.id, ac.ears, ac.tests, ac.holdout_probes) == ("AC-1", EARS, (SHOUT_ID,), 1)
    assert contract.assumptions == (SPEC_ASSUMPTION, CRITIC_ASSUMPTION)
    assert contract.out_of_scope == ("Other scripts.",)

    # the visible tests: one commit on main that only adds the ticket's test files
    sha = gate1.repo.rev_parse(oracle_ref("T-0001"))
    assert gate1.repo.rev_parse(f"{sha}^") == main
    changed = gate1.repo.diff(main, sha)
    assert [(c.path, c.kind) for c in changed] == [(acceptance_path(), ChangeKind.ADDED)]
    assert gate1.repo.read_file(sha, acceptance_path()).decode() == SHOUT_TEST
    assert gate1.main == main  # main never moves
    assert gate1.refs("refs/factory/oracle-candidate/") == []

    # the sealed holdout, bound to the ticket by the oracle version
    record = gate1.holdout.load("T-0001")
    assert record is not None and record.oracle_sha == sha
    [probe] = record.probes
    assert probe.id == "HP-1" and probe.expect.stdout == SHOUT_PROBE_STDOUT
    assert ticket.oracle_version == oracle_version(record) == content_hash(record)
    path = gate1.holdout.path("T-0001")
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700

    # red-first: everything failed on the base
    assert [(c.kind, c.id, c.red) for c in outcome.red_checks] == [
        ("visible", SHOUT_ID, True),
        ("holdout", "HP-1", True),
    ]
    assert outcome.red_checks[0].reason == "failed on the base (exit 1)"
    assert outcome.red_checks[1].reason.startswith("failed on the base: ")
    assert all(SHOUT_PROBE_STDOUT not in c.reason for c in outcome.red_checks)


def test_the_ledger_explains_the_intake(gate1: World) -> None:
    propose(gate1)
    statuses = [(e.from_status, e.to_status) for e in gate1.ledger.events("T-0001") if e.to_status]
    assert statuses == [
        (None, TicketStatus.RECEIVED),
        (TicketStatus.RECEIVED, TicketStatus.TRIAGED),
        (TicketStatus.TRIAGED, TicketStatus.CONTRACT_PROPOSED),
    ]
    [triaged] = events(gate1, "T-0001", "triaged")
    assert triaged["risk"] == "R1" and triaged["title"] == "Add --shout to greet.py"
    [drafted] = events(gate1, "T-0001", "spec_drafted")
    assert drafted["version"] == 1
    [critique] = events(gate1, "T-0001", "critique")
    assert critique["assumptions"] == [CRITIC_ASSUMPTION]
    [red] = events(gate1, "T-0001", "red_first")
    assert red["attempt"] == 1
    assert red["oracle_sha"] == gate1.repo.rev_parse(oracle_ref("T-0001"))
    assert red_flags(red) == [True, True]

    runs = gate1.ledger.runs("T-0001")
    assert [(r.role, r.model, r.backend, r.status) for r in runs] == [
        ("triage", "fake", "fake", "ok"),
        ("spec-writer", "fake", "fake", "ok"),
        ("ambiguity-critic", "fake", "fake", "ok"),
        ("test-architect", "fake", "fake", "ok"),
    ]
    assert [r.tier for r in runs] == [Tier.STANDARD, Tier.FRONTIER, Tier.FRONTIER, Tier.FRONTIER]


def test_triage_sets_title_type_and_risk(gate1: World) -> None:
    scripts = happy_scripts()
    scripts["triage"] = [[EmitOutput(triage_out(title="Fix shouting", type="bug", risk="R0"))]]
    propose(gate1, ScriptedBackend(scripts))
    ticket = gate1.ledger.get_ticket("T-0001")
    assert (ticket.title, ticket.type, ticket.risk) == ("Fix shouting", TicketType.BUG, RiskTier.R0)


def test_each_role_is_told_what_it_needs(gate1: World) -> None:
    other = receive(gate1.ledger, "slugify.py should keep digits")
    gate1.ledger.transition(other.id, TicketStatus.ABANDONED, actor="human")
    open_one = receive(gate1.ledger, "Document the greet.py options in a README")
    backend = ScriptedBackend(happy_scripts(ticket="T-0003"))
    ticket = receive(gate1.ledger, REQUEST)
    assert ticket.id == "T-0003"
    intake(gate1, backend, ticket.id)

    [triage] = backend.runs("triage")
    assert REQUEST in triage.instructions
    assert open_one.id in triage.instructions and open_one.title in triage.instructions
    assert other.id not in triage.instructions  # closed tickets are not candidates
    assert triage.ticket_id == ticket.id

    [spec] = backend.runs("spec-writer")
    assert REQUEST in spec.instructions and "Add --shout to greet.py" in spec.instructions

    [critic] = backend.runs("ambiguity-critic")
    assert REQUEST in critic.instructions
    assert "AC-1" in critic.instructions and EARS in critic.instructions

    [architect] = backend.runs("test-architect")
    for needed in (f"tests/acceptance/{ticket.id}/", ACCEPTANCE, "AC-1", EARS, TEST_PLAN):
        assert needed in architect.instructions, needed


def test_what_the_other_roles_write_is_discarded(gate1: World) -> None:
    main = gate1.main
    original = gate1.repo.read_file(main, "greet.py")
    scripts = happy_scripts()
    for role in ("triage", "spec-writer", "ambiguity-critic"):
        scripts[role][0].insert(0, WriteFile("greet.py", "print('hijacked')\n"))
        scripts[role][0].insert(0, WriteFile(acceptance_path(file="test_planted.py"), "x = 1\n"))
    outcome = propose(gate1, ScriptedBackend(scripts))
    assert outcome.status == "contract_proposed"
    sha = gate1.repo.rev_parse(oracle_ref("T-0001"))
    files = gate1.files_at(sha)
    assert files["greet.py"] == original
    assert acceptance_path(file="test_planted.py") not in files
    assert gate1.main == main


# --------------------------------------------------------------------------- triage's questions


def test_a_suspected_duplicate_asks_the_human(gate1: World) -> None:
    first = receive(gate1.ledger, "Make greet.py shout")
    scripts = happy_scripts(ticket="T-0002")
    scripts["triage"] = [[EmitOutput(triage_out(duplicate_of=first.id))]]
    backend = ScriptedBackend(scripts)
    ticket = receive(gate1.ledger, REQUEST)

    outcome = intake(gate1, backend, ticket.id)
    assert outcome.status == "needs_info"
    [question] = outcome.questions
    assert (question.id, question.asked_by, question.round) == ("Q-1", "factory", 0)
    assert first.id in question.text and "A small CLI flag." in question.text
    assert gate1.ledger.get_ticket(ticket.id).status is TicketStatus.NEEDS_INFO

    answer(gate1.ledger, ticket.id, "Q-1", "It is a different change; continue.")
    outcome = intake(gate1, backend, ticket.id)
    assert outcome.status == "contract_proposed"
    [spec] = backend.runs("spec-writer")
    assert "It is a different change; continue." in spec.instructions


def test_an_epic_asks_the_human(gate1: World) -> None:
    scripts = happy_scripts()
    scripts["triage"] = [[EmitOutput(triage_out(size="XL", rationale="Three features."))]]
    ticket = receive(gate1.ledger, REQUEST)
    outcome = intake(gate1, ScriptedBackend(scripts), ticket.id)
    assert outcome.status == "needs_info"
    [question] = outcome.questions
    assert "too large" in question.text and "Three features." in question.text


@pytest.mark.parametrize("duplicate", ["T-0001", "T-0099"])
def test_a_duplicate_of_itself_or_of_nothing_is_ignored(gate1: World, duplicate: str) -> None:
    scripts = happy_scripts()
    scripts["triage"] = [[EmitOutput(triage_out(duplicate_of=duplicate))]]
    assert propose(gate1, ScriptedBackend(scripts)).status == "contract_proposed"


# --------------------------------------------------------------------------- clarify


def test_blocking_questions_wait_for_the_answers(gate1: World) -> None:
    scripts = happy_scripts(critic=critique_out("Upper case for non-ASCII too?", "Flag name?"))
    scripts["spec-writer"].append([EmitOutput(spec_out(summary="Shout, revised."))])
    scripts["ambiguity-critic"].append([EmitOutput(critique_out())])
    backend = ScriptedBackend(scripts)
    ticket = receive(gate1.ledger, REQUEST)

    outcome = intake(gate1, backend, ticket.id)
    assert outcome.status == "needs_info"
    assert [(q.id, q.asked_by, q.round) for q in outcome.questions] == [
        ("Q-1", "ambiguity-critic", 1),
        ("Q-2", "ambiguity-critic", 1),
    ]
    assert outcome.questions[0].why_blocking == "A wrong guess is rejected."

    # nothing runs while a question is open
    runs = len(backend.calls)
    assert intake(gate1, backend, ticket.id).status == "needs_info"
    assert len(backend.calls) == runs

    answered = answer(gate1.ledger, ticket.id, "Q-1", "Yes, str.upper().")
    assert (answered.answer, answered.answered_by) == ("Yes, str.upper().", "human")
    assert [q.id for q in open_questions(gate1.ledger, ticket.id)] == ["Q-2"]
    assert intake(gate1, backend, ticket.id).status == "needs_info"
    answer(gate1.ledger, ticket.id, "Q-2", "--shout", author="kaan")

    outcome = intake(gate1, backend, ticket.id)
    assert outcome.status == "contract_proposed", outcome
    assert open_questions(gate1.ledger, ticket.id) == []
    assert [q.answer for q in questions(gate1.ledger, ticket.id)] == [
        "Yes, str.upper().",
        "--shout",
    ]
    contract = gate1.ledger.get_ticket(ticket.id).contract
    assert contract is not None and (contract.version, contract.summary) == (2, "Shout, revised.")

    second_spec = backend.runs("spec-writer")[1].instructions
    second_critic = backend.runs("ambiguity-critic")[1].instructions
    for text in ("Upper case for non-ASCII too?", "Yes, str.upper().", "Flag name?", "--shout"):
        assert text in second_spec and text in second_critic
    statuses = [e.to_status for e in gate1.ledger.events(ticket.id) if e.to_status]
    assert statuses[-3:] == [
        TicketStatus.NEEDS_INFO,
        TicketStatus.TRIAGED,
        TicketStatus.CONTRACT_PROPOSED,
    ]


def test_answer_refuses_misuse(gate1: World) -> None:
    ticket = receive(gate1.ledger, REQUEST)
    with pytest.raises(IntakeError):  # not waiting for answers
        answer(gate1.ledger, ticket.id, "Q-1", "anything")
    scripts = happy_scripts(critic=critique_out("Which flag?"))
    intake(gate1, ScriptedBackend(scripts), ticket.id)
    for qid, text in (("Q-9", "no such question"), ("Q-1", "   ")):
        with pytest.raises(IntakeError):
            answer(gate1.ledger, ticket.id, qid, text)
    answer(gate1.ledger, ticket.id, "Q-1", "--shout")
    with pytest.raises(IntakeError):  # already answered
        answer(gate1.ledger, ticket.id, "Q-1", "--loud")
    assert [q.answer for q in questions(gate1.ledger, ticket.id)] == ["--shout"]


def test_questions_stop_blocking_after_the_last_round(gate1: World) -> None:
    rounds = MAX_CLARIFY_ROUNDS + 1
    scripts = happy_scripts()
    scripts["spec-writer"] = [[EmitOutput(spec_out())] for _ in range(rounds)]
    scripts["ambiguity-critic"] = [
        [EmitOutput(critique_out(f"Question {n}?"))] for n in range(1, rounds + 1)
    ]
    backend = ScriptedBackend(scripts)
    ticket = receive(gate1.ledger, REQUEST)
    for n in range(1, MAX_CLARIFY_ROUNDS + 1):
        outcome = intake(gate1, backend, ticket.id)
        assert outcome.status == "needs_info"
        [question] = outcome.questions
        assert (question.id, question.round) == (f"Q-{n}", n)
        answer(gate1.ledger, ticket.id, question.id, f"Answer {n}")

    outcome = intake(gate1, backend, ticket.id)
    assert outcome.status == "contract_proposed"
    contract = gate1.ledger.get_ticket(ticket.id).contract
    assert contract is not None
    assert f"Unresolved after 3 rounds: Question {rounds}?" in contract.assumptions
    assert len(questions(gate1.ledger, ticket.id)) == MAX_CLARIFY_ROUNDS


# --------------------------------------------------------------------------- the oracle


def retry_scripts(first: list[Action]) -> dict[str, list[list[Action]]]:
    scripts = happy_scripts()
    scripts["test-architect"] = [first, oracle_script()]
    return scripts


def test_a_vacuous_visible_test_goes_back_to_the_test_architect(gate1: World) -> None:
    first = oracle_script(
        files={acceptance_path(): SHOUT_TEST, acceptance_path(file="test_plain.py"): VACUOUS_TEST},
        out=manifest(visible=((SHOUT_ID, ("AC-1",)), (VACUOUS_ID, ("AC-1",)))),
    )
    backend = ScriptedBackend(retry_scripts(first))
    outcome = propose(gate1, backend)

    assert outcome.status == "contract_proposed"
    [rejected] = events(gate1, "T-0001", "oracle_rejected")
    assert rejected["attempt"] == 1
    assert rejected["problems"] == [f"{VACUOUS_ID}: passed on the base"]
    retry = backend.runs("test-architect")[1].instructions
    assert f"{VACUOUS_ID}: passed on the base" in retry
    red = events(gate1, "T-0001", "red_first")
    assert [r["attempt"] for r in red] == [1, 2]
    # the accepted oracle is the second attempt's, built from main again
    sha = gate1.repo.rev_parse(oracle_ref("T-0001"))
    assert acceptance_path(file="test_plain.py") not in gate1.files_at(sha)
    assert gate1.refs("refs/factory/oracle-candidate/") == []


def test_a_probe_that_passes_on_the_base_goes_back(gate1: World) -> None:
    first = oracle_script(out=manifest(holdout=[shout_probe(), plain_probe()]))
    backend = ScriptedBackend(retry_scripts(first))
    assert propose(gate1, backend).status == "contract_proposed"
    [rejected] = events(gate1, "T-0001", "oracle_rejected")
    assert rejected["problems"] == ["HP-2: passed on the base"]
    record = gate1.holdout.load("T-0001")
    assert record is not None and [p.id for p in record.probes] == ["HP-1"]


def test_a_test_id_that_runs_nothing_goes_back(gate1: World) -> None:
    missing = f"{acceptance_path()}::ShoutTest.test_no_such_test"
    first = oracle_script(out=manifest(visible=((missing, ("AC-1",)),)))
    assert propose(gate1, ScriptedBackend(retry_scripts(first))).status == "contract_proposed"
    [rejected] = events(gate1, "T-0001", "oracle_rejected")
    assert rejected["problems"] == [f"{missing}: no test ran (exit 5)"]


@pytest.mark.parametrize(
    ("files", "needle"),
    [
        ({"greet.py": "print('rigged')\n"}, "greet.py"),
        ({"factory.toml": "[project]\nname = 'x'\n"}, "factory.toml"),
        ({acceptance_path("T-0002"): SHOUT_TEST}, "T-0002"),
        ({"tests/test_cli.py": "x = 1\n"}, "tests/test_cli.py"),
    ],
)
def test_the_test_architect_may_only_add_its_own_tests(
    gate1: World, files: dict[str, str], needle: str
) -> None:
    first = oracle_script(files={acceptance_path(): SHOUT_TEST, **files})
    backend = ScriptedBackend(retry_scripts(first))
    assert propose(gate1, backend).status == "contract_proposed"
    [rejected] = events(gate1, "T-0001", "oracle_rejected")
    problems = rejected["problems"]
    assert isinstance(problems, list) and len(problems) == 1
    problem = str(problems[0])
    assert "oracle_path" in problem and needle in problem
    assert [r["attempt"] for r in events(gate1, "T-0001", "red_first")] == [2]  # never ran


def test_writing_no_tests_is_a_problem(gate1: World) -> None:
    first = oracle_script(files={})
    backend = ScriptedBackend(retry_scripts(first))
    assert propose(gate1, backend).status == "contract_proposed"
    [rejected] = events(gate1, "T-0001", "oracle_rejected")
    assert "no_changes" in str(rejected["problems"])


TWO_CRITERIA = spec_out(
    criterion(),
    criterion(
        "When greet.py is run without --shout, the system shall print the greeting unchanged.",
        "Run greet.py --name Ada.",
    ),
)


@pytest.mark.parametrize(
    ("out", "needle"),
    [
        (manifest(), "AC-2"),  # AC-2 has neither a visible test nor a probe
        (manifest(visible=((SHOUT_ID, ("AC-1", "AC-2")),)), "AC-2"),  # no probe for AC-2
        (manifest(holdout=[shout_probe(("AC-1", "AC-2"))]), "AC-2"),  # no visible test for AC-2
        (
            manifest(
                visible=((SHOUT_ID, ("AC-1", "AC-2")),),
                holdout=[shout_probe(("AC-1", "AC-2", "AC-7"))],
            ),
            "AC-7",
        ),
        (
            manifest(
                visible=(
                    (SHOUT_ID, ("AC-1", "AC-2")),
                    (f"{acceptance_path(file='test_gone.py')}::T.test_x", ("AC-1",)),
                ),
                holdout=[shout_probe(("AC-1", "AC-2"))],
            ),
            "test_gone.py",
        ),
        (
            manifest(
                visible=((SHOUT_ID, ("AC-1", "AC-2")), (SHOUT_ID, ("AC-1",))),
                holdout=[shout_probe(("AC-1", "AC-2"))],
            ),
            SHOUT_ID,
        ),
    ],
    ids=["uncovered", "no-probe", "no-visible", "unknown-criterion", "missing-file", "duplicate"],
)
def test_manifest_problems_go_back(gate1: World, out: dict[str, JsonValue], needle: str) -> None:
    good = manifest(
        visible=((SHOUT_ID, ("AC-1", "AC-2")),), holdout=[shout_probe(("AC-1", "AC-2"))]
    )
    scripts = happy_scripts()
    scripts["spec-writer"] = [[EmitOutput(TWO_CRITERIA)]]
    scripts["test-architect"] = [oracle_script(out=out), oracle_script(out=good)]
    backend = ScriptedBackend(scripts)
    assert propose(gate1, backend).status == "contract_proposed"
    [rejected] = events(gate1, "T-0001", "oracle_rejected")
    assert needle in str(rejected["problems"])
    assert [r["attempt"] for r in events(gate1, "T-0001", "red_first")] == [2]
    contract = gate1.ledger.get_ticket("T-0001").contract
    assert contract is not None
    assert [(c.tests, c.holdout_probes) for c in contract.criteria] == [
        ((SHOUT_ID,), 1),
        ((SHOUT_ID,), 1),
    ]


def test_the_oracle_fails_after_the_last_attempt(gate1: World) -> None:
    vacuous = oracle_script(out=manifest(holdout=[plain_probe()]))
    scripts = happy_scripts()
    scripts["test-architect"] = [list(vacuous) for _ in range(MAX_ORACLE_ATTEMPTS)]
    backend = ScriptedBackend(scripts)
    outcome = propose(gate1, backend)

    assert outcome.status == "oracle_failed"
    assert outcome.error is not None and "HP-1: passed on the base" in outcome.error
    assert len(backend.runs("test-architect")) == MAX_ORACLE_ATTEMPTS
    ticket = gate1.ledger.get_ticket("T-0001")
    assert ticket.status is TicketStatus.TRIAGED and ticket.oracle_version is None
    assert events(gate1, "T-0001", "oracle_failed") == [{"attempts": MAX_ORACLE_ATTEMPTS}]
    assert gate1.holdout.load("T-0001") is None
    assert gate1.refs("refs/factory/") == []

    backend.add("spec-writer", [EmitOutput(spec_out())])  # a later run starts over from triaged
    backend.add("ambiguity-critic", [EmitOutput(critique_out())])
    backend.add("test-architect", oracle_script())
    assert intake(gate1, backend, "T-0001").status == "contract_proposed"


# --------------------------------------------------------------------------- keep criteria


def test_a_keep_criterion_guards_todays_behaviour(gate1: World) -> None:
    outcome = propose(gate1, ScriptedBackend(keep_scripts()))

    assert outcome.status == "contract_proposed", outcome
    # the change checks fail on the base, the keep checks pass on it
    assert [(c.id, c.keep, c.red, c.green, c.ok) for c in outcome.red_checks] == [
        (SHOUT_ID, False, True, False, True),
        (KEEP_ID, True, False, True, True),
        ("HP-1", False, True, False, True),
        ("HP-2", True, False, True, True),
    ]
    assert outcome.red_checks[1].reason == "passed on the base"
    contract = gate1.ledger.get_ticket("T-0001").contract
    assert contract is not None
    assert [(c.id, c.kind, c.ears, c.tests, c.holdout_probes) for c in contract.criteria] == [
        ("AC-1", CriterionKind.CHANGE, EARS, (SHOUT_ID,), 1),
        ("AC-2", CriterionKind.KEEP, KEEP_EARS, (KEEP_ID,), 1),
    ]
    [red] = events(gate1, "T-0001", "red_first")
    checks = red["checks"]
    assert isinstance(checks, list)
    assert [(c["keep"], c["green"]) for c in checks if isinstance(c, dict)] == [
        (False, False),
        (True, True),
        (False, False),
        (True, True),
    ]
    sha = gate1.repo.rev_parse(oracle_ref("T-0001"))
    assert sorted(c.path for c in gate1.repo.diff(gate1.main, sha)) == [
        acceptance_path(file="test_plain.py"),
        acceptance_path(),
    ]
    approved = approve(gate1.ledger, gate1.repo, gate1.holdout, "T-0001")
    assert approved.status is TicketStatus.APPROVED


def test_a_keep_check_that_fails_on_the_base_goes_back(gate1: World) -> None:
    wrong_probe = plain_probe(("AC-2",))
    wrong_probe["expect"] = {"exit_code": 0, "stdout": "Hi, Grace!"}
    scripts = keep_scripts()
    scripts["test-architect"].insert(
        0, keep_oracle(WRONG_KEEP_TEST, holdout=[shout_probe(), wrong_probe])
    )
    backend = ScriptedBackend(scripts)
    outcome = propose(gate1, backend)

    assert outcome.status == "contract_proposed", outcome
    [rejected] = events(gate1, "T-0001", "oracle_rejected")
    expected = [
        f"{KEEP_ID}: failed on the base (exit 1); a keep check must pass on the base",
        "HP-2: failed on the base: stdout differs; a keep check must pass on the base",
    ]
    assert rejected == {"attempt": 1, "problems": expected}
    retry = backend.runs("test-architect")[1].instructions
    assert all(problem in retry for problem in expected)
    assert "Hi, Grace!" not in retry  # a probe's expectation is still never echoed


@pytest.mark.parametrize("mixed", ["visible", "holdout"])
def test_a_test_or_probe_checks_one_kind_only(gate1: World, mixed: str) -> None:
    both = ("AC-1", "AC-2")
    first = keep_oracle(
        visible=(
            (SHOUT_ID, both if mixed == "visible" else ("AC-1",)),
            (KEEP_ID, ("AC-2",)),
        ),
        holdout=[shout_probe(both if mixed == "holdout" else ("AC-1",)), plain_probe(("AC-2",))],
    )
    scripts = keep_scripts()
    scripts["test-architect"].insert(0, first)
    assert propose(gate1, ScriptedBackend(scripts)).status == "contract_proposed"

    [rejected] = events(gate1, "T-0001", "oracle_rejected")
    culprit = SHOUT_ID if mixed == "visible" else "HP-1"
    assert rejected["problems"] == [f"{culprit}: mixes change and keep criteria"]
    # caught by the manifest check, before anything ran
    assert [r["attempt"] for r in events(gate1, "T-0001", "red_first")] == [2]


def test_a_spec_must_change_something(gate1: World) -> None:
    scripts = happy_scripts()
    scripts["spec-writer"] = [[EmitOutput(spec_out(keep_criterion()))]]
    ticket = receive(gate1.ledger, REQUEST)
    outcome = intake(gate1, ScriptedBackend(scripts), ticket.id)
    assert outcome.status == "agent_error"
    assert outcome.error is not None and "at least one change criterion" in outcome.error
    assert gate1.ledger.get_ticket(ticket.id).status is TicketStatus.TRIAGED


def test_the_roles_see_each_criterion_kind(gate1: World) -> None:
    backend = ScriptedBackend(keep_scripts())
    assert propose(gate1, backend).status == "contract_proposed"
    [critic] = backend.runs("ambiguity-critic")
    [architect] = backend.runs("test-architect")
    for instructions in (critic.instructions, architect.instructions):
        assert f"AC-1 [e2e, change]: {EARS}" in instructions
        assert f"AC-2 [e2e, keep]: {KEEP_EARS}" in instructions


def test_tool_caches_are_not_committed(gate1: World) -> None:
    """A test architect that ran its own tests leaves caches behind; only the tests are kept."""
    caches = {
        acceptance_path(file="__pycache__/test_shout.cpython-312.pyc"): "bytecode stand-in",
        "tests/__pycache__/test_cli.cpython-312.pyc": "bytecode stand-in",
        ".pytest_cache/v/cache/lastfailed": "{}",
        ".mypy_cache/3.12/greet.meta.json": "{}",
        ".ruff_cache/CACHEDIR.TAG": "Signature: 8a477f597d28d172789f06886806bc55",
    }
    scripts = happy_scripts()
    scripts["test-architect"] = [oracle_script(files={acceptance_path(): SHOUT_TEST, **caches})]
    outcome = propose(gate1, ScriptedBackend(scripts))

    assert outcome.status == "contract_proposed", outcome
    assert events(gate1, "T-0001", "oracle_rejected") == []
    sha = gate1.repo.rev_parse(oracle_ref("T-0001"))
    assert [c.path for c in gate1.repo.diff(gate1.main, sha)] == [acceptance_path()]


# --------------------------------------------------------------------------- failures and halts


def test_an_agent_error_leaves_the_ticket_where_it_was(gate1: World) -> None:
    scripts = happy_scripts()
    scripts["triage"] = [[Fail("the model wandered off")], [EmitOutput(triage_out())]]
    backend = ScriptedBackend(scripts)
    ticket = receive(gate1.ledger, REQUEST)

    outcome = intake(gate1, backend, ticket.id)
    assert outcome.status == "agent_error"
    assert outcome.error is not None and "the model wandered off" in outcome.error
    assert gate1.ledger.get_ticket(ticket.id).status is TicketStatus.RECEIVED
    [error] = events(gate1, ticket.id, "intake_error")
    assert error["role"] == "triage" and "the model wandered off" in str(error["error"])
    assert [r.status for r in gate1.ledger.runs(ticket.id)] == ["error"]

    assert intake(gate1, backend, ticket.id).status == "contract_proposed"


def test_output_that_does_not_fit_the_schema_is_an_agent_error(gate1: World) -> None:
    scripts = happy_scripts()
    scripts["spec-writer"] = [[EmitOutput({"summary": "no criteria", "criteria": []})]]
    ticket = receive(gate1.ledger, REQUEST)
    outcome = intake(gate1, ScriptedBackend(scripts), ticket.id)
    assert outcome.status == "agent_error"
    assert gate1.ledger.get_ticket(ticket.id).status is TicketStatus.TRIAGED


def test_a_halt_halts_the_ticket_until_it_is_resumed(gate1: World) -> None:
    scripts = happy_scripts()
    scripts["spec-writer"].insert(0, [RaiseHalt(HaltReason.RATE_LIMIT)])
    backend = ScriptedBackend(scripts)
    ticket = receive(gate1.ledger, REQUEST)

    outcome = intake(gate1, backend, ticket.id)
    assert outcome.status == "halted"
    assert outcome.error is not None and "rate_limit" in outcome.error
    assert gate1.ledger.get_ticket(ticket.id).status is TicketStatus.HALTED
    with pytest.raises(IntakeError):
        intake(gate1, backend, ticket.id)

    gate1.ledger.resume(ticket.id, actor="human")
    assert intake(gate1, backend, ticket.id).status == "contract_proposed"


def test_intake_does_nothing_more_once_proposed(gate1: World) -> None:
    backend = ScriptedBackend(happy_scripts())
    propose(gate1, backend)
    calls = len(backend.calls)
    outcome = intake(gate1, backend, "T-0001")
    assert outcome.status == "contract_proposed" and len(backend.calls) == calls


# --------------------------------------------------------------------------- approval


def test_approval_freezes_the_contract_and_the_oracle(gate1: World) -> None:
    propose(gate1)
    proposed = gate1.ledger.get_ticket("T-0001")
    ticket = approve(gate1.ledger, gate1.repo, gate1.holdout, "T-0001", author="kaan")
    assert ticket.status is TicketStatus.APPROVED
    assert ticket.contract == proposed.contract and ticket.contract is not None
    assert ticket.contract_hash == content_hash(ticket.contract)
    assert ticket.oracle_version == proposed.oracle_version
    [frozen] = [e for e in gate1.ledger.events("T-0001") if e.kind == "contract_frozen"]
    assert frozen.actor == "kaan"
    with pytest.raises(IntakeError):
        approve(gate1.ledger, gate1.repo, gate1.holdout, "T-0001")


def test_approval_needs_a_proposed_contract(gate1: World) -> None:
    ticket = receive(gate1.ledger, REQUEST)
    with pytest.raises(IntakeError):
        approve(gate1.ledger, gate1.repo, gate1.holdout, ticket.id)


def test_approval_refuses_a_changed_holdout(gate1: World) -> None:
    propose(gate1)
    record = gate1.holdout.load("T-0001")
    assert record is not None
    weaker = record.probes[0].model_copy(update={"stdin": "changed"})
    gate1.holdout.save(Holdout(ticket_id="T-0001", oracle_sha=record.oracle_sha, probes=(weaker,)))
    with pytest.raises(IntakeError):
        approve(gate1.ledger, gate1.repo, gate1.holdout, "T-0001")
    gate1.holdout.path("T-0001").unlink()
    with pytest.raises(IntakeError):
        approve(gate1.ledger, gate1.repo, gate1.holdout, "T-0001")
    assert gate1.ledger.get_ticket("T-0001").status is TicketStatus.CONTRACT_PROPOSED


def test_approval_refuses_moved_visible_tests(gate1: World) -> None:
    propose(gate1)
    gate1.repo.update_ref(oracle_ref("T-0001"), gate1.main)
    with pytest.raises(IntakeError):
        approve(gate1.ledger, gate1.repo, gate1.holdout, "T-0001")
    assert gate1.ledger.get_ticket("T-0001").status is TicketStatus.CONTRACT_PROPOSED


def test_the_source_repository_is_never_touched(gate1: World) -> None:
    before = gate1.git("rev-parse", "HEAD", cwd=gate1.target)
    status = gate1.git("status", "--porcelain", cwd=gate1.target)
    propose(gate1)
    approve(gate1.ledger, gate1.repo, gate1.holdout, "T-0001")
    assert gate1.git("rev-parse", "HEAD", cwd=gate1.target) == before
    assert gate1.git("status", "--porcelain", cwd=gate1.target) == status
    assert gate1.repo.rev_parse(MAIN) == before
