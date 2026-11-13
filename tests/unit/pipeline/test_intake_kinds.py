"""Unit tests for criterion kinds in intake (docs/specs/phase-2b.md, amendment C): what the roles
read, the manifest's one-kind rule, which checks are keep checks and what a failed one says.

The flow end to end (a keep criterion's checks must pass on the base, a mixed test goes back) is
in tests/acceptance/test_intake.py.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest

from local_factory.agents.fake import EmitOutput, Fail
from local_factory.models import (
    ChangedFile,
    ChangeKind,
    CriterionDraft,
    CriterionKind,
    HoldoutProbe,
    OracleManifest,
    ProbeDraft,
    ProbeExpectation,
    RedCheck,
    SpecDraft,
    TestLayer,
    VisibleTest,
)
from local_factory.oracle import red_first
from local_factory.pipeline import intake as intake_module
from local_factory.pipeline import receive, run_intake
from local_factory.pipeline.intake import (
    _check_problem,
    _criteria_lines,
    _criterion_kinds,
    _keep_ids,
    _manifest_problems,
)
from tests.acceptance.intake_world import (
    CRITIC_ASSUMPTION,
    EARS,
    KEEP_EARS,
    KEEP_ID,
    KEEP_PLAN,
    REQUEST,
    SHOUT_ID,
    TEST_PLAN,
    ScriptedBackend,
    World,
    acceptance_path,
    critique_out,
    happy_scripts,
    keep_oracle,
    keep_scripts,
    make_world,
    spec_out,
)

CHANGE, KEEP = CriterionKind.CHANGE, CriterionKind.KEEP
# AC-1 and AC-4 change something; AC-2 and AC-3 keep today's behaviour
KINDS = {"AC-1": CHANGE, "AC-2": KEEP, "AC-3": KEEP, "AC-4": CHANGE}
TESTS = ("tests/acceptance/T-0001/test_a.py", "tests/acceptance/T-0001/test_b.py")


@pytest.fixture
def gate1(tmp_path: Path) -> Iterator[World]:
    with make_world(tmp_path) as world:
        yield world


def draft(*kinds: CriterionKind) -> SpecDraft:
    return SpecDraft(
        summary="s",
        criteria=tuple(
            CriterionDraft(ears=f"criterion {n}", layer=TestLayer.E2E, test_plan="plan", kind=kind)
            for n, kind in enumerate(kinds, start=1)
        ),
    )


def probe(*criteria: str) -> ProbeDraft:
    return ProbeDraft(criteria=criteria, argv=("true",), expect=ProbeExpectation(exit_code=0))


def manifest(
    visible: Sequence[tuple[str, Sequence[str]]], holdout: Sequence[Sequence[str]]
) -> OracleManifest:
    return OracleManifest(
        visible=tuple(VisibleTest(test=test, criteria=tuple(ac)) for test, ac in visible),
        holdout=tuple(probe(*ac) for ac in holdout),
    )


def added(*paths: str) -> list[ChangedFile]:
    return [ChangedFile(path=path, kind=ChangeKind.ADDED) for path in paths]


def problems(
    visible: Sequence[tuple[str, Sequence[str]]], holdout: Sequence[Sequence[str]]
) -> list[str]:
    paths = {test.partition("::")[0] for test, _ in visible}
    return _manifest_problems(manifest(visible, holdout), added(*paths), KINDS)


# --------------------------------------------------------------------------- what the roles read


def test_a_criterion_line_names_layer_and_kind_then_the_test_plan() -> None:
    spec = SpecDraft(
        summary="s",
        criteria=(
            CriterionDraft(ears="Do x.", layer=TestLayer.UNIT, test_plan="Check x."),
            CriterionDraft(ears="Keep y.", layer=TestLayer.E2E, test_plan="Check y.", kind=KEEP),
        ),
    )
    assert _criteria_lines(spec) == [
        "AC-1 [unit, change]: Do x.",
        "  Test plan: Check x.",
        "AC-2 [e2e, keep]: Keep y.",
        "  Test plan: Check y.",
    ]


def test_the_kinds_are_numbered_the_way_the_criteria_are() -> None:
    assert _criterion_kinds(draft(CHANGE, KEEP, KEEP)) == {
        "AC-1": CHANGE,
        "AC-2": KEEP,
        "AC-3": KEEP,
    }


def test_the_spec_writer_sees_the_kinds_of_its_previous_draft(gate1: World) -> None:
    scripts = keep_scripts()
    scripts["ambiguity-critic"] = [[Fail("critic fell over")], *scripts["ambiguity-critic"]]
    spec_again = spec_out()  # a revised draft: one change criterion
    ticket = receive(gate1.ledger, REQUEST)
    backend = ScriptedBackend(scripts)
    assert asyncio.run(run_intake(gate1.intake(backend), ticket.id)).status == "agent_error"
    backend.add("spec-writer", [EmitOutput(spec_again)])
    backend.add("test-architect", *scripts["test-architect"])
    asyncio.run(run_intake(gate1.intake(backend), ticket.id))

    _, second = backend.runs("spec-writer")
    assert f"AC-1 [e2e, change]: {EARS}\n  Test plan: {TEST_PLAN}" in second.instructions
    assert f"AC-2 [e2e, keep]: {KEEP_EARS}\n  Test plan: {KEEP_PLAN}" in second.instructions


def test_the_test_architect_is_told_what_each_kind_must_do_on_the_base(gate1: World) -> None:
    backend = ScriptedBackend(keep_scripts())
    ticket = receive(gate1.ledger, REQUEST)
    asyncio.run(run_intake(gate1.intake(backend), ticket.id))
    [brief] = backend.runs("test-architect")
    sentence = (
        "A change criterion's tests and probes must fail on the current code; a keep "
        "criterion's must pass on it. A test or probe checks criteria of one kind only."
    )
    assert sentence in brief.instructions
    assert brief.instructions.index(f"AC-2 [e2e, keep]: {KEEP_EARS}") < brief.instructions.index(
        sentence
    )
    assert "Out of scope" in brief.instructions.split(sentence)[1]  # between criteria and the rest


# --------------------------------------------------------------------------- the manifest


def test_a_manifest_of_single_kind_tests_and_probes_has_no_problems() -> None:
    assert (
        problems(
            [(f"{TESTS[0]}::A.t", ("AC-1", "AC-4")), (f"{TESTS[1]}::B.t", ("AC-2", "AC-3"))],
            [("AC-1", "AC-4"), ("AC-2", "AC-3")],
        )
        == []
    )


def test_a_visible_test_that_mixes_kinds_is_named() -> None:
    mixed = f"{TESTS[0]}::A.t"
    found = problems(
        [(mixed, ("AC-1", "AC-2", "AC-3", "AC-4"))], [("AC-1", "AC-4"), ("AC-2", "AC-3")]
    )
    assert found == [f"{mixed}: mixes change and keep criteria"]


def test_a_probe_that_mixes_kinds_is_named_by_its_position() -> None:
    found = problems(
        [(f"{TESTS[0]}::A.t", ("AC-1", "AC-4")), (f"{TESTS[1]}::B.t", ("AC-2", "AC-3"))],
        [("AC-1", "AC-4"), ("AC-3", "AC-4"), ("AC-2", "AC-3")],
    )
    assert found == ["HP-2: mixes change and keep criteria"]


def test_every_mixed_test_and_probe_is_reported_tests_first() -> None:
    a, b = f"{TESTS[0]}::A.t", f"{TESTS[1]}::B.t"
    found = problems(
        [(a, ("AC-1", "AC-2")), (b, ("AC-3", "AC-4"))],
        [("AC-1", "AC-2"), ("AC-3",), ("AC-4", "AC-3")],
    )
    assert found == [
        f"{a}: mixes change and keep criteria",
        f"{b}: mixes change and keep criteria",
        "HP-1: mixes change and keep criteria",
        "HP-3: mixes change and keep criteria",
    ]


def test_an_unknown_criterion_is_not_a_kind_to_mix() -> None:
    # AC-9 does not exist: that is the problem, not a mix of kinds
    found = problems(
        [(f"{TESTS[0]}::A.t", ("AC-1", "AC-9")), (f"{TESTS[1]}::B.t", ("AC-2", "AC-3", "AC-4"))],
        [("AC-1", "AC-4", "AC-9"), ("AC-2", "AC-3")],
    )
    assert found == [
        f"{TESTS[0]}::A.t: unknown criterion AC-9",
        f"{TESTS[1]}::B.t: mixes change and keep criteria",
        "HP-1: unknown criterion AC-9",
    ]


def test_the_other_manifest_problems_are_as_before() -> None:
    gone = "tests/acceptance/T-0001/test_gone.py::G.t"
    manifest_ = manifest(
        [(f"{TESTS[0]}::A.t", ("AC-1",)), (f"{TESTS[0]}::A.t", ("AC-1",)), (gone, ("AC-1",))],
        [("AC-1",)],
    )
    found = _manifest_problems(manifest_, added(TESTS[0]), KINDS)
    assert found == [
        f"{TESTS[0]}::A.t: listed more than once",
        f"{gone}: tests/acceptance/T-0001/test_gone.py is not a file the test architect added",
        "AC-2: no visible test checks it",
        "AC-2: no holdout probe checks it",
        "AC-3: no visible test checks it",
        "AC-3: no holdout probe checks it",
        "AC-4: no visible test checks it",
        "AC-4: no holdout probe checks it",
    ]


# --------------------------------------------------------------------------- the keep checks


def hp(number: int, *criteria: str) -> HoldoutProbe:
    return HoldoutProbe(
        id=f"HP-{number}",
        criteria=criteria,
        argv=("true",),
        expect=ProbeExpectation(exit_code=0),
    )


def test_a_test_or_probe_whose_criteria_are_all_keep_guards_behaviour() -> None:
    visible = [(f"{TESTS[0]}::A.t", ("AC-1",)), (f"{TESTS[1]}::B.t", ("AC-2", "AC-3"))]
    probes = [hp(1, "AC-4"), hp(2, "AC-3"), hp(3, "AC-2", "AC-3")]
    keep = _keep_ids(manifest(visible, [("AC-4",)]), probes, KINDS)
    assert keep == {f"{TESTS[1]}::B.t", "HP-2", "HP-3"}


def test_nothing_is_a_keep_check_when_nothing_is_kept() -> None:
    only_change = {"AC-1": CHANGE}
    keep = _keep_ids(
        manifest([(f"{TESTS[0]}::A.t", ("AC-1",))], [("AC-1",)]), [hp(1, "AC-1")], only_change
    )
    assert keep == frozenset()


def check(kind: str, reason: str, *, keep: bool) -> RedCheck:
    return RedCheck.model_validate(
        {"kind": kind, "id": "x", "reason": reason, "keep": keep, "red": False}
    )


@pytest.mark.parametrize(
    "reason",
    [
        "passed on the base",
        "no test ran (exit 5)",
        "did not run cleanly: boom",
        "timed out after 30s",
    ],
)
def test_a_change_check_that_is_not_red_is_reported_as_it_always_was(reason: str) -> None:
    assert _check_problem(check("visible", reason, keep=False)) == f"x: {reason}"


@pytest.mark.parametrize(
    "reason",
    [
        "failed on the base (exit 1)",
        "no test ran (exit 5)",
        "did not run cleanly: boom",
        "failed on the base: stdout differs",
        "timed out after 30s",
    ],
)
def test_a_keep_check_that_is_not_green_says_it_must_pass_on_the_base(reason: str) -> None:
    for kind in ("visible", "holdout"):
        assert (
            _check_problem(check(kind, reason, keep=True))
            == f"x: {reason}; a keep check must pass on the base"
        )


# --------------------------------------------------------------------------- red-first's input


def spy_red_first(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Let red-first run for real, recording what intake handed it."""
    seen: list[dict[str, Any]] = []

    async def recording(**kwargs: Any) -> tuple[RedCheck, ...]:
        seen.append(kwargs)
        return await red_first(**kwargs)

    monkeypatch.setattr(intake_module, "red_first", recording)
    return seen


def test_red_first_is_told_which_checks_guard_keep_criteria(
    gate1: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = spy_red_first(monkeypatch)
    ticket = receive(gate1.ledger, REQUEST)
    outcome = asyncio.run(run_intake(gate1.intake(ScriptedBackend(keep_scripts())), ticket.id))
    assert outcome.status == "contract_proposed", outcome
    [call] = seen
    assert set(call["keep"]) == {KEEP_ID, "HP-2"}
    assert call["visible"] == [SHOUT_ID, KEEP_ID]


def test_a_spec_of_change_criteria_only_has_no_keep_checks(
    gate1: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = spy_red_first(monkeypatch)
    ticket = receive(gate1.ledger, REQUEST)
    backend = ScriptedBackend(happy_scripts(critic=critique_out(assumptions=[CRITIC_ASSUMPTION])))
    assert asyncio.run(run_intake(gate1.intake(backend), ticket.id)).status == "contract_proposed"
    [call] = seen
    assert set(call["keep"]) == set()


def test_a_keep_test_the_architect_never_wrote_fails_before_anything_ran(gate1: World) -> None:
    """A keep test that is not in the commit is a manifest problem; red-first never starts."""
    gone = f"{acceptance_path(file='test_gone.py')}::PlainTest.test_plain"
    scripts = keep_scripts()
    scripts["test-architect"].insert(
        0,
        keep_oracle(visible=((SHOUT_ID, ("AC-1",)), (gone, ("AC-2",)))),
    )
    ticket = receive(gate1.ledger, REQUEST)
    outcome = asyncio.run(run_intake(gate1.intake(ScriptedBackend(scripts)), ticket.id))
    assert outcome.status == "contract_proposed", outcome
    [rejected] = [e.data for e in gate1.ledger.events(ticket.id) if e.kind == "oracle_rejected"]
    assert rejected["problems"] == [
        f"{gone}: {acceptance_path(file='test_gone.py')} is not a file the test architect added"
    ]
