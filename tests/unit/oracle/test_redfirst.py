"""Unit tests for red-first with a scripted site: what runs where, in which order, and how the
results are judged. The real thing (git trees, gates, a real app) is the acceptance test."""

from __future__ import annotations

import asyncio
import shlex
from collections.abc import AsyncIterator, Collection, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import pytest

from local_factory.agents import WorkerHandle
from local_factory.agents.tools import CommandResult
from local_factory.gitops import BareRepo
from local_factory.models import (
    GateResult,
    GateStatus,
    HoldoutProbe,
    ProbeExpectation,
    RedCheck,
)
from local_factory.oracle import (
    NO_TESTS_RAN_EXIT,
    RED_TEST_TIMEOUT_S,
    judge_visible,
    red_first,
    render_acceptance,
    split_test_id,
)
from local_factory.pipeline.flow import Site
from local_factory.verification import GateSpec

ORACLE = "1" * 40
BASE = "2" * 40
TEMPLATE = "python3 -m unittest {path} -k {name}"


@dataclass
class Script:
    """What the scripted site was asked, in order, and what it answers."""

    log: list[tuple[object, ...]] = field(default_factory=list)
    gate_results: dict[str, GateResult] = field(default_factory=dict)
    commands: dict[str, CommandResult] = field(default_factory=dict)  # by the probe's first arg


class Executor:
    def __init__(self, script: Script) -> None:
        self.script = script

    async def write_file(self, path: str, content: str) -> None:
        raise AssertionError("unexpected")

    async def read_file(self, path: str) -> str:
        raise AssertionError("unexpected")

    async def delete_file(self, path: str) -> None:
        raise AssertionError("unexpected")

    async def run_command(self, argv: Sequence[str], *, timeout_s: float) -> CommandResult:
        key = argv[5]  # sh -c SCRIPT probe STDIN <first arg of the probe>
        self.script.log.append(("run", key, timeout_s))
        return self.script.commands[key]


class Verification:
    def __init__(self, script: Script, tmp: Path) -> None:
        self.script = script
        self.handle = WorkerHandle(work_dir=tmp, executor=Executor(script))

    async def gates(self, specs: Sequence[GateSpec]) -> list[GateResult]:
        self.script.log.append(("gates", tuple(spec.name for spec in specs)))
        [spec] = specs
        assert spec.timeout_s == RED_TEST_TIMEOUT_S
        return [self.script.gate_results[spec.name]]


class FakeSite:
    def __init__(self, script: Script, tmp: Path) -> None:
        self.script = script
        self.tmp = tmp

    @asynccontextmanager
    async def verification(
        self, repo: BareRepo, sha: str, ticket_dir: Path, *, round_no: int
    ) -> AsyncIterator[Verification]:
        self.script.log.append(("open", sha, ticket_dir, round_no))
        yield Verification(self.script, self.tmp)
        self.script.log.append(("close", sha, round_no))


def gate(test_id: str, status: GateStatus, exit_code: int | None, tail: str = "") -> GateResult:
    return GateResult(
        name=f"red-first {test_id}", status=status, exit_code=exit_code, output_tail=tail
    )


def probe(n: int, first: str, **expect: object) -> HoldoutProbe:
    return HoldoutProbe(
        id=f"HP-{n}",
        criteria=("AC-1",),
        argv=(first,),
        expect=ProbeExpectation.model_validate(expect or {"exit_code": 0}),
    )


def run(
    tmp_path: Path,
    script: Script,
    *,
    visible: Sequence[str] = (),
    probes: Sequence[HoldoutProbe] = (),
    acceptance: str = TEMPLATE,
    keep: Collection[str] = (),
) -> tuple[RedCheck, ...]:
    return asyncio.run(
        red_first(
            site=cast("Site", FakeSite(script, tmp_path)),
            repo=cast("BareRepo", object()),
            oracle_sha=ORACLE,
            base_sha=BASE,
            visible=visible,
            probes=probes,
            acceptance=acceptance,
            ticket_dir=tmp_path / "T-0001",
            keep=keep,
        )
    )


def test_visible_tests_share_one_tree_and_run_one_gate_at_a_time(tmp_path: Path) -> None:
    script = Script(
        gate_results={
            "red-first a.py::t1": gate("a.py::t1", GateStatus.FAIL, 1),
            "red-first a.py::t2": gate("a.py::t2", GateStatus.FAIL, 1),
        }
    )
    checks = run(tmp_path, script, visible=["a.py::t1", "a.py::t2"])
    assert [(c.kind, c.id, c.red) for c in checks] == [
        ("visible", "a.py::t1", True),
        ("visible", "a.py::t2", True),
    ]
    assert script.log == [
        ("open", ORACLE, tmp_path / "T-0001", 1),
        ("gates", ("red-first a.py::t1",)),
        ("gates", ("red-first a.py::t2",)),
        ("close", ORACLE, 1),
    ]


def test_every_probe_gets_its_own_fresh_tree_of_the_base(tmp_path: Path) -> None:
    script = Script(
        gate_results={"red-first a.py::t": gate("a.py::t", GateStatus.FAIL, 1)},
        commands={
            "first": CommandResult(exit_code=1, output="no"),
            "second": CommandResult(exit_code=0, output="yes"),
        },
    )
    checks = run(
        tmp_path,
        script,
        visible=["a.py::t"],
        probes=[probe(1, "first"), probe(2, "second")],
    )
    assert [(c.kind, c.id, c.red, c.reason) for c in checks] == [
        ("visible", "a.py::t", True, "failed on the base (exit 1)"),
        ("holdout", "HP-1", True, "failed on the base: exit code 1 differs"),
        ("holdout", "HP-2", False, "passed on the base"),
    ]
    opened = [entry for entry in script.log if entry[0] == "open"]
    assert [(sha, rounds) for _, sha, _, rounds in opened] == [(ORACLE, 1), (BASE, 2), (BASE, 3)]
    assert len({rounds for *_, rounds in opened}) == len(opened)  # distinct rounds
    kinds = [entry[0] for entry in script.log]
    assert kinds == ["open", "gates", "close", "open", "run", "close", "open", "run", "close"]


def test_probes_alone_open_no_oracle_tree(tmp_path: Path) -> None:
    script = Script(commands={"x": CommandResult(exit_code=0, output="")})
    checks = run(tmp_path, script, probes=[probe(1, "x")])
    assert [c.id for c in checks] == ["HP-1"]
    assert script.log[0] == ("open", BASE, tmp_path / "T-0001", 2)


def test_nothing_to_check_runs_nothing(tmp_path: Path) -> None:
    script = Script()
    assert run(tmp_path, script) == ()
    assert script.log == []


def test_a_probe_that_timed_out_is_not_red(tmp_path: Path) -> None:
    script = Script(commands={"x": CommandResult(exit_code=None, output="partial", timed_out=True)})
    [check] = run(tmp_path, script, probes=[probe(1, "x")])
    assert (check.red, check.reason, check.output) == (False, "timed out after 30s", "partial")


def test_a_probe_s_output_is_clipped_to_its_tail(tmp_path: Path) -> None:
    script = Script(commands={"x": CommandResult(exit_code=1, output="y" * 5_000 + "END")})
    [check] = run(tmp_path, script, probes=[probe(1, "x")])
    assert len(check.output) == 2_000 and check.output.endswith("END")


def test_a_bad_command_template_fails_before_anything_runs(tmp_path: Path) -> None:
    script = Script()
    with pytest.raises(ValueError, match=r"\{path\}"):
        run(tmp_path, script, visible=["a.py::t"], acceptance="python3 -m unittest")
    with pytest.raises(ValueError, match="path::name"):
        run(tmp_path, script, visible=["a.py::t", "no-name"])
    assert script.log == []


def test_each_gate_is_named_after_its_test_and_runs_the_rendered_command(tmp_path: Path) -> None:
    seen: list[GateSpec] = []

    class Recording(Verification):
        async def gates(self, specs: Sequence[GateSpec]) -> list[GateResult]:
            seen.extend(specs)
            return [gate("x", GateStatus.PASS, 0)]

    class RecordingSite(FakeSite):
        @asynccontextmanager
        async def verification(
            self, repo: BareRepo, sha: str, ticket_dir: Path, *, round_no: int
        ) -> AsyncIterator[Verification]:
            yield Recording(self.script, self.tmp)

    asyncio.run(
        red_first(
            site=cast("Site", RecordingSite(Script(), tmp_path)),
            repo=cast("BareRepo", object()),
            oracle_sha=ORACLE,
            base_sha=BASE,
            visible=["tests/a b.py::C.t"],
            probes=[],
            acceptance=TEMPLATE,
            ticket_dir=tmp_path,
        )
    )
    [spec] = seen
    assert spec.name == "red-first tests/a b.py::C.t"
    assert spec.command == "python3 -m unittest 'tests/a b.py' -k C.t"
    assert spec.timeout_s == RED_TEST_TIMEOUT_S == 120.0


# --------------------------------------------------------------------------- helpers


@pytest.mark.parametrize(
    ("test_id", "parts"),
    [
        ("a.py::t", ("a.py", "t")),
        ("a.py::C::t::u", ("a.py", "C::t::u")),
        ("a.py::", ("a.py", "")),
    ],
)
def test_split_test_id(test_id: str, parts: tuple[str, str]) -> None:
    assert split_test_id(test_id) == parts


def test_a_test_id_needs_a_separator() -> None:
    with pytest.raises(ValueError, match="path::name"):
        split_test_id("a.py")


def test_placeholders_in_a_name_or_path_are_never_substituted_again() -> None:
    # {name} inside the already-quoted path must not be replaced with the (quoted) name
    command = render_acceptance(TEMPLATE, "x{name}.py::$(id)")
    assert shlex.split(command) == ["python3", "-m", "unittest", "x{name}.py", "-k", "$(id)"]
    command = render_acceptance("run {path}", "{path}::{name}")
    assert shlex.split(command) == ["run", "{path}"]
    assert render_acceptance("{name} {path} {name}", "a::b") == "b a b"


@pytest.mark.parametrize("hostile", ["'; touch pwned; '", "$(touch pwned)", "`touch pwned`", "\n"])
def test_hostile_ids_stay_one_argument(hostile: str) -> None:
    words = shlex.split(render_acceptance(TEMPLATE, f"a.py::{hostile}"))
    assert words[-1] == hostile
    assert len(words) == 6


def test_judge_visible_clips_the_unclean_reason_and_the_output() -> None:
    long = "z" * 3_000 + "TAIL"
    check = judge_visible("a.py::t", gate("a.py::t", GateStatus.ERROR, None, long))
    assert check.reason == "did not run cleanly: " + long[-300:]
    assert check.output == long[-2_000:]
    assert NO_TESTS_RAN_EXIT == 5
    assert judge_visible("a.py::t", gate("a.py::t", GateStatus.ERROR, None, "")).reason == (
        "did not run cleanly: "
    )


def test_a_passing_test_is_not_red_even_with_a_failure_looking_exit_code() -> None:
    check = judge_visible("a.py::t", gate("a.py::t", GateStatus.PASS, 5))
    assert (check.red, check.reason) == (False, "passed on the base")


def test_a_timed_out_gate_is_not_red() -> None:
    check = judge_visible("a.py::t", gate("a.py::t", GateStatus.ERROR, None, "timed out"))
    assert check.red is False


# --------------------------------------------------------------------------- keep checks


@pytest.mark.parametrize(
    ("result", "red", "green"),
    [
        (gate("t", GateStatus.FAIL, 1), True, False),
        (gate("t", GateStatus.FAIL, NO_TESTS_RAN_EXIT), False, False),
        (gate("t", GateStatus.PASS, 0), False, True),
        (gate("t", GateStatus.ERROR, None, "boom"), False, False),
        (gate("t", GateStatus.SKIPPED, None, "slow"), False, False),
    ],
)
def test_only_a_pass_is_green_and_only_a_real_failure_is_red(
    result: GateResult, red: bool, green: bool
) -> None:
    plain = judge_visible("a.py::t", result)
    kept = judge_visible("a.py::t", result, keep=True)
    assert (plain.red, plain.green, plain.keep, plain.ok) == (red, green, False, red)
    assert (kept.red, kept.green, kept.keep, kept.ok) == (red, green, True, green)
    assert kept.model_copy(update={"keep": False}) == plain  # keep is the only difference


def test_keep_marks_the_named_checks_and_nothing_else_changes(tmp_path: Path) -> None:
    script = Script(
        gate_results={
            "red-first a.py::change": gate("a.py::change", GateStatus.FAIL, 1),
            "red-first a.py::keep": gate("a.py::keep", GateStatus.PASS, 0),
        },
        commands={
            "change": CommandResult(exit_code=1, output="no"),
            "keep": CommandResult(exit_code=0, output="yes"),
        },
    )
    visible = ["a.py::change", "a.py::keep"]
    probes = [probe(1, "change"), probe(2, "keep")]
    checks = run(
        tmp_path, script, visible=visible, probes=probes, keep={"a.py::keep", "HP-2", "HP-9"}
    )
    assert [(c.id, c.keep, c.red, c.green, c.ok) for c in checks] == [
        ("a.py::change", False, True, False, True),
        ("a.py::keep", True, False, True, True),
        ("HP-1", False, True, False, True),
        ("HP-2", True, False, True, True),
    ]
    # the same runs, in the same order, whether or not anything is a keep check
    plain_script = Script(gate_results=script.gate_results, commands=script.commands)
    unmarked = run(tmp_path, plain_script, visible=visible, probes=probes)
    assert script.log == plain_script.log
    assert [c.model_copy(update={"keep": False}) for c in checks] == list(unmarked)


def test_a_keep_check_that_fails_is_not_ok(tmp_path: Path) -> None:
    script = Script(
        gate_results={"red-first a.py::k": gate("a.py::k", GateStatus.FAIL, 1)},
        commands={"x": CommandResult(exit_code=1, output="wrong")},
    )
    checks = run(
        tmp_path, script, visible=["a.py::k"], probes=[probe(1, "x")], keep={"a.py::k", "HP-1"}
    )
    assert [(c.keep, c.red, c.green, c.ok) for c in checks] == [(True, True, False, False)] * 2


def test_a_probe_that_timed_out_is_neither_red_nor_green(tmp_path: Path) -> None:
    script = Script(commands={"x": CommandResult(exit_code=None, output="", timed_out=True)})
    [check] = run(tmp_path, script, probes=[probe(1, "x")], keep={"HP-1"})
    assert (check.red, check.green, check.keep, check.ok) == (False, False, True, False)
