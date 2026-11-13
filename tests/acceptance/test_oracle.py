"""The oracle's building blocks: holdout store, oracle policy, probes, red-first, and the ledger,
gitops, policy and config additions of Phase 2b (docs/specs/phase-2b.md § oracle). Protected.
"""

from __future__ import annotations

import asyncio
import shlex
import stat
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from local_factory.agents import WorkerHandle
from local_factory.agents.tools import CommandResult
from local_factory.config import (
    DEFAULT_TEST_GLOBS,
    ConfigError,
    parse_project_config,
)
from local_factory.gitops.bare import BareRepo
from local_factory.gitops.bundles import unsafe_symlinks
from local_factory.hashing import canonical_json, content_hash
from local_factory.ledger import IllegalTransition, Ledger, LedgerError, TicketNotFound
from local_factory.models import (
    MODE_EXECUTABLE,
    MODE_GITLINK,
    MODE_REGULAR,
    MODE_SYMLINK,
    ChangedFile,
    ChangeKind,
    Contract,
    Criterion,
    GateResult,
    GateStatus,
    Holdout,
    HoldoutProbe,
    ProbeDraft,
    ProbeExpectation,
    RiskTier,
    TestLayer,
    TicketStatus,
    TicketType,
)
from local_factory.oracle import (
    MAX_ORACLE_FILE_BYTES,
    MAX_ORACLE_FILES,
    ORACLE_REF_PREFIX,
    PROBE_SCRIPT,
    PROBE_TIMEOUT_S,
    HoldoutError,
    HoldoutStore,
    acceptance_dir,
    check_oracle_changes,
    judge_probe,
    judge_visible,
    oracle_ref,
    oracle_version,
    probe_argv,
    red_first,
    render_acceptance,
    run_probe,
    split_test_id,
)
from local_factory.pipeline.host_slice import host_site
from local_factory.policy.diff_policy import DiffPolicy
from tests.acceptance.intake_world import (
    ACCEPTANCE,
    MAIN,
    SHOUT_ID,
    SHOUT_TEST,
    VACUOUS_ID,
    VACUOUS_TEST,
    World,
    acceptance_path,
    make_world,
)

SHA = "a" * 40
NOW = datetime(2026, 11, 13, tzinfo=UTC)


def probe(n: int = 1, **expect: object) -> HoldoutProbe:
    return HoldoutProbe(
        id=f"HP-{n}",
        criteria=("AC-1",),
        argv=("python3", "greet.py", "--name", "Ada"),
        expect=ProbeExpectation.model_validate(expect or {"exit_code": 0}),
    )


def holdout(ticket: str = "T-0001", *probes: HoldoutProbe) -> Holdout:
    return Holdout(ticket_id=ticket, oracle_sha=SHA, probes=probes or (probe(),))


# --------------------------------------------------------------------------- holdout store


def test_the_holdout_store_seals_each_record(tmp_path: Path) -> None:
    store = HoldoutStore(tmp_path / "holdout")
    assert store.load("T-0001") is None
    record = holdout()
    path = store.save(record)
    assert path == store.path("T-0001") == tmp_path / "holdout" / "T-0001.json"
    assert path.read_bytes() == canonical_json(record)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert store.load("T-0001") == record

    replaced = holdout("T-0001", probe(1), probe(2))
    store.save(replaced)
    assert store.load("T-0001") == replaced
    assert sorted(p.name for p in path.parent.iterdir()) == ["T-0001.json"]  # no temp files left


def test_the_holdout_store_tightens_an_open_directory(tmp_path: Path) -> None:
    root = tmp_path / "holdout"
    root.mkdir(mode=0o755)
    root.chmod(0o755)
    HoldoutStore(root).save(holdout())
    assert stat.S_IMODE(root.stat().st_mode) == 0o700


@pytest.mark.parametrize("ticket", ["../T-0001", "T-1", "T-0001/x", "", "t-0001"])
def test_the_holdout_store_takes_only_ticket_ids(tmp_path: Path, ticket: str) -> None:
    store = HoldoutStore(tmp_path / "holdout")
    for call in (store.path, store.load):
        with pytest.raises(ValueError, match=r"."):
            call(ticket)


def test_the_holdout_store_refuses_symlinks(tmp_path: Path) -> None:
    secret = tmp_path / "elsewhere.json"
    secret.write_bytes(canonical_json(holdout()))
    store = HoldoutStore(tmp_path / "holdout")
    store.save(holdout("T-0002"))
    store.path("T-0001").symlink_to(secret)
    with pytest.raises(HoldoutError):
        store.load("T-0001")
    with pytest.raises(HoldoutError):
        store.save(holdout())
    assert secret.read_bytes() == canonical_json(holdout())

    (tmp_path / "real").mkdir()
    (tmp_path / "linked").symlink_to(tmp_path / "real")
    with pytest.raises(HoldoutError):
        HoldoutStore(tmp_path / "linked").save(holdout())
    assert list((tmp_path / "real").iterdir()) == []


def test_a_corrupt_or_misfiled_record_is_an_error(tmp_path: Path) -> None:
    store = HoldoutStore(tmp_path / "holdout")
    store.save(holdout())
    store.path("T-0001").write_text("{not json")
    with pytest.raises(HoldoutError):
        store.load("T-0001")
    store.path("T-0002").write_bytes(canonical_json(holdout("T-0003")))
    with pytest.raises(HoldoutError):
        store.load("T-0002")


def test_the_oracle_names() -> None:
    record = holdout()
    assert oracle_version(record) == content_hash(record)
    assert oracle_ref("T-0007") == f"{ORACLE_REF_PREFIX}T-0007" == "refs/factory/oracle/T-0007"
    assert acceptance_dir("T-0007") == "tests/acceptance/T-0007/"


# --------------------------------------------------------------------------- oracle policy


def added(
    path: str, *, mode: str = MODE_REGULAR, size: int = 10, binary: bool = False
) -> ChangedFile:
    return ChangedFile(path=path, kind=ChangeKind.ADDED, mode=mode, size=size, is_binary=binary)


def test_new_test_files_in_the_ticket_directory_pass() -> None:
    changes = [
        added("tests/acceptance/T-0001/test_a.py"),
        added("tests/acceptance/T-0001/sub/helper.sh", mode=MODE_EXECUTABLE),
        added("tests/acceptance/T-0001/data.txt", size=MAX_ORACLE_FILE_BYTES),
    ]
    assert check_oracle_changes(changes, "T-0001") == []
    assert check_oracle_changes([], "T-0001") == []


def test_everything_else_is_a_violation() -> None:
    changes = [
        added("greet.py"),
        ChangedFile(
            path="tests/acceptance/T-0001/test_b.py",
            kind=ChangeKind.MODIFIED,
            mode=MODE_REGULAR,
            size=5,
        ),
        ChangedFile(path="tests/acceptance/T-0001/test_c.py", kind=ChangeKind.DELETED),
        added("tests/acceptance/T-0002/test_a.py"),
        added("tests/acceptance/T-0001/link", mode=MODE_SYMLINK),
        added("tests/acceptance/T-0001/blob.bin", binary=True),
        added("tests/acceptance/T-0001/big.py", size=MAX_ORACLE_FILE_BYTES + 1),
        added("tests/acceptance/T-0001/sub", mode=MODE_GITLINK),
        added("tests/acceptance/T-00010/test_a.py"),
    ]
    violations = check_oracle_changes(changes, "T-0001")
    by_path = {v.path: v.rule for v in violations}
    assert by_path == {
        "greet.py": "oracle_path",
        "tests/acceptance/T-0001/test_b.py": "oracle_path",
        "tests/acceptance/T-0001/test_c.py": "oracle_path",
        "tests/acceptance/T-0002/test_a.py": "oracle_path",
        "tests/acceptance/T-0001/link": "oracle_symlink",
        "tests/acceptance/T-0001/blob.bin": "oracle_binary",
        "tests/acceptance/T-0001/big.py": "oracle_size",
        "tests/acceptance/T-0001/sub": "oracle_path",
        "tests/acceptance/T-00010/test_a.py": "oracle_path",
    }
    assert [v.path for v in violations] == sorted(v.path for v in violations)
    assert all(v.message for v in violations)


def test_too_many_files_is_one_violation() -> None:
    changes = [added(f"tests/acceptance/T-0001/test_{n}.py") for n in range(MAX_ORACLE_FILES + 1)]
    violations = check_oracle_changes(changes, "T-0001")
    assert [(v.rule, v.path) for v in violations] == [("oracle_files", "")]


# --------------------------------------------------------------------------- probes


def test_a_probe_runs_through_a_shell_free_wrapper() -> None:
    draft = ProbeDraft(
        criteria=("AC-1",),
        argv=("python3", "greet.py"),
        stdin="in",
        expect=ProbeExpectation(exit_code=0),
    )
    assert probe_argv(draft) == ["sh", "-c", PROBE_SCRIPT, "probe", "in", "python3", "greet.py"]
    assert PROBE_TIMEOUT_S == 30.0


def test_run_probe_feeds_stdin_and_keeps_only_stdout(tmp_path: Path) -> None:
    (tmp_path / "here.txt").write_text("cwd is the root\n")
    script = (
        "import sys; data = sys.stdin.read(); sys.stderr.write('noise'); "
        "print(open('here.txt').read().strip()); print(data.upper()); sys.exit(3)"
    )
    draft = ProbeDraft(
        criteria=("AC-1",),
        argv=("python3", "-c", script),
        stdin="héllo $HOME `x`",
        expect=ProbeExpectation(exit_code=3),
    )
    result = asyncio.run(run_probe(WorkerHandle(work_dir=tmp_path), draft))
    assert (result.exit_code, result.timed_out) == (3, False)
    assert result.output == "cwd is the root\nHÉLLO $HOME `X`\n"


def outcome(output: str, exit_code: int | None = 0, *, timed_out: bool = False) -> CommandResult:
    return CommandResult(exit_code=exit_code, output=output, timed_out=timed_out)


def expectation(**fields: object) -> ProbeDraft:
    return ProbeDraft(
        criteria=("AC-1",), argv=("x",), expect=ProbeExpectation.model_validate(fields)
    )


@pytest.mark.parametrize(
    ("expect", "result", "passed", "reason"),
    [
        ({"stdout": "HELLO"}, outcome("HELLO\n"), True, "passed on the base"),
        ({"stdout": "HELLO\n"}, outcome("HELLO"), True, "passed on the base"),
        ({"stdout": "HELLO"}, outcome("HELLO \n"), False, "failed on the base: stdout differs"),
        ({"exit_code": 0}, outcome("", 2), False, "failed on the base: exit code 2 differs"),
        ({"stdout_contains": ["LL", "HE"]}, outcome("HELLO"), True, "passed on the base"),
        (
            {"stdout_contains": ["LL", "XYZZY"]},
            outcome("HELLO"),
            False,
            "failed on the base: stdout lacks an expected fragment",
        ),
        (
            {"exit_code": 0, "stdout": "SECRET-ONE", "stdout_contains": ["SECRET-TWO"]},
            outcome("Hello", 1),
            False,
            "failed on the base: exit code 1 differs; stdout differs; "
            "stdout lacks an expected fragment",
        ),
        ({"exit_code": 0}, outcome("", None, timed_out=True), False, "timed out after 30s"),
    ],
)
def test_judge_probe(
    expect: dict[str, object], result: CommandResult, passed: bool, reason: str
) -> None:
    assert judge_probe(expectation(**expect), result) == (passed, reason)


def test_a_probe_reason_never_shows_the_expectation() -> None:
    draft = expectation(exit_code=0, stdout="SECRET-ONE", stdout_contains=["SECRET-TWO"])
    for result in (
        outcome("SECRET-ON", 0),
        outcome("nothing", 1),
        outcome("x", None, timed_out=True),
    ):
        _, reason = judge_probe(draft, result)
        assert "SECRET" not in reason


# --------------------------------------------------------------------------- red-first helpers


def test_split_and_render_a_test_id() -> None:
    assert split_test_id("tests/a.py::Case.test_x") == ("tests/a.py", "Case.test_x")
    assert split_test_id("tests/a.py::Case::test_x") == ("tests/a.py", "Case::test_x")
    template = "python3 -m unittest {path} -k {name}"
    assert render_acceptance(template, "tests/acceptance/T-0001/t x.py::C.test_{a}") == (
        f"python3 -m unittest {shlex.quote('tests/acceptance/T-0001/t x.py')} -k "
        f"{shlex.quote('C.test_{a}')}"
    )
    assert render_acceptance("pytest {path}::{name}", "tests/a.py::test_b") == (
        "pytest tests/a.py::test_b"
    )
    assert render_acceptance("run {path} {other}", "a.py::b") == "run a.py {other}"
    assert render_acceptance("x $(rm -rf ~) {path}", "a.py::b") == "x $(rm -rf ~) a.py"
    hostile = render_acceptance(template, "a.py::$(touch pwned); `id`")
    assert shlex.split(hostile)[-1] == "$(touch pwned); `id`"
    with pytest.raises(ValueError, match=r"\{path\}"):
        render_acceptance("python3 -m unittest", "a.py::b")


def gate(status: GateStatus, exit_code: int | None, tail: str = "") -> GateResult:
    return GateResult(name="red-first x", status=status, exit_code=exit_code, output_tail=tail)


@pytest.mark.parametrize(
    ("result", "red", "green", "reason"),
    [
        (gate(GateStatus.FAIL, 1), True, False, "failed on the base (exit 1)"),
        (gate(GateStatus.FAIL, 2), True, False, "failed on the base (exit 2)"),
        (gate(GateStatus.FAIL, 5), False, False, "no test ran (exit 5)"),
        (gate(GateStatus.PASS, 0), False, True, "passed on the base"),
        (gate(GateStatus.ERROR, None, "timed out"), False, False, "did not run cleanly: timed out"),
    ],
)
def test_judge_visible(result: GateResult, red: bool, green: bool, reason: str) -> None:
    check = judge_visible("tests/a.py::t", result)
    assert (check.kind, check.id, check.red, check.green, check.keep, check.reason) == (
        "visible",
        "tests/a.py::t",
        red,
        green,
        False,
        reason,
    )
    assert check.ok is red
    kept = judge_visible("tests/a.py::t", result, keep=True)
    assert (kept.red, kept.green, kept.keep, kept.reason) == (red, green, True, reason)
    assert kept.ok is green  # a keep check must pass on the base


def test_judge_visible_keeps_a_bounded_tail() -> None:
    check = judge_visible("tests/a.py::t", gate(GateStatus.FAIL, 1, "x" * 5_000 + "END"))
    assert check.output.endswith("END") and len(check.output) <= 2_000
    errored = judge_visible("tests/a.py::t", gate(GateStatus.ERROR, None, "y" * 5_000))
    assert len(errored.reason) <= len("did not run cleanly: ") + 300


# --------------------------------------------------------------------------- red-first


@pytest.fixture
def gate1(tmp_path: Path) -> Iterator[World]:
    with make_world(tmp_path) as world:
        yield world


def oracle_commit(w: World, files: dict[str, str]) -> str:
    """Commit `files` on top of main in a scratch clone and fetch it into the bare repo."""
    clone = w.root / "scratch"
    w.git("clone", "-q", str(w.target), str(clone), cwd=w.root)
    for path, content in files.items():
        (clone / path).parent.mkdir(parents=True, exist_ok=True)
        (clone / path).write_text(content)
    w.git("add", "-A", cwd=clone)
    w.git("commit", "-q", "-m", "oracle", cwd=clone)
    w.repo.fetch_from(clone, ["HEAD:refs/heads/oracle-under-test"])
    return w.repo.rev_parse("refs/heads/oracle-under-test")


def shell_probe(n: int, script: str, **expect: object) -> HoldoutProbe:
    return HoldoutProbe(
        id=f"HP-{n}",
        criteria=("AC-1",),
        argv=("sh", "-c", script),
        expect=ProbeExpectation.model_validate(expect),
    )


@pytest.mark.integration
def test_red_first_runs_every_test_and_probe_on_the_base(gate1: World) -> None:
    sha = oracle_commit(
        gate1,
        {acceptance_path(): SHOUT_TEST, acceptance_path(file="test_plain.py"): VACUOUS_TEST},
    )
    missing = f"{acceptance_path()}::ShoutTest.test_nothing_here"
    probes = (
        probe(1, exit_code=0, stdout="HELLO, ADA!"),
        probe(2, exit_code=0, stdout="Hello, Ada!"),
        HoldoutProbe(
            id="HP-3",
            criteria=("AC-1",),
            argv=("python3", "-c", "import sys; print(sys.stdin.read().upper())"),
            stdin="abc",
            expect=ProbeExpectation(stdout="ABC"),
        ),
        # probes see the app only: main, not the commit with the visible tests
        shell_probe(4, f"test -e {acceptance_path()}", exit_code=0),
        # each probe gets a fresh tree
        shell_probe(5, "touch left-behind", exit_code=0),
        shell_probe(6, "test -e left-behind", exit_code=0),
    )
    checks = asyncio.run(
        red_first(
            site=host_site(),
            repo=gate1.repo,
            oracle_sha=sha,
            base_sha=gate1.main,
            visible=(SHOUT_ID, VACUOUS_ID, missing),
            probes=probes,
            acceptance=ACCEPTANCE,
            ticket_dir=gate1.work_root / "T-0001",
        )
    )
    assert [(c.kind, c.id, c.red, c.reason) for c in checks] == [
        ("visible", SHOUT_ID, True, "failed on the base (exit 1)"),
        ("visible", VACUOUS_ID, False, "passed on the base"),
        ("visible", missing, False, "no test ran (exit 5)"),
        ("holdout", "HP-1", True, "failed on the base: stdout differs"),
        ("holdout", "HP-2", False, "passed on the base"),
        ("holdout", "HP-3", False, "passed on the base"),
        ("holdout", "HP-4", True, "failed on the base: exit code 1 differs"),
        ("holdout", "HP-5", False, "passed on the base"),
        ("holdout", "HP-6", True, "failed on the base: exit code 1 differs"),
    ]
    assert checks[3].output == "Hello, Ada!\n"  # what the app printed on the base (stdout)
    assert [c.green for c in checks] == [False, True, False, False, True, True, False, True, False]
    assert not any(c.keep for c in checks)
    assert gate1.main == gate1.repo.rev_parse(MAIN)


@pytest.mark.integration
def test_red_first_marks_the_keep_checks(gate1: World) -> None:
    sha = oracle_commit(
        gate1,
        {acceptance_path(): SHOUT_TEST, acceptance_path(file="test_plain.py"): VACUOUS_TEST},
    )
    probes = (
        probe(1, exit_code=0, stdout="HELLO, ADA!"),
        probe(2, exit_code=0, stdout="Hello, Ada!"),
    )
    checks = asyncio.run(
        red_first(
            site=host_site(),
            repo=gate1.repo,
            oracle_sha=sha,
            base_sha=gate1.main,
            visible=(SHOUT_ID, VACUOUS_ID),
            probes=probes,
            acceptance=ACCEPTANCE,
            ticket_dir=gate1.work_root / "T-0001",
            keep={VACUOUS_ID, "HP-2"},
        )
    )
    assert [(c.id, c.keep, c.red, c.green, c.ok) for c in checks] == [
        (SHOUT_ID, False, True, False, True),
        (VACUOUS_ID, True, False, True, True),
        ("HP-1", False, True, False, True),
        ("HP-2", True, False, True, True),
    ]


# --------------------------------------------------------------------------- ledger additions


@pytest.fixture
def ledger(tmp_path: Path) -> Iterator[Ledger]:
    with Ledger(tmp_path / "ledger.db", clock=lambda: NOW) as led:
        yield led


def new_ticket(ledger: Ledger) -> str:
    return ledger.create_ticket(
        title="t", type=TicketType.FEATURE, risk=RiskTier.R2, request="r", actor="human"
    ).id


RESERVED = [
    "created",
    "transition",
    "contract_proposed",
    "oracle_version_set",
    "contract_frozen",
    "halted",
    "resumed",
    "task_created",
    "task_updated",
    "run_recorded",
    "verdict_recorded",
    "evidence_recorded",
    "triaged",
]


def test_record_event_appends_but_never_impersonates_the_ledger(ledger: Ledger) -> None:
    ticket = new_ticket(ledger)
    event = ledger.record_event(ticket, "critique", actor="factory", data={"blocking": []})
    assert (event.kind, event.actor, event.data, event.ticket_id) == (
        "critique",
        "factory",
        {"blocking": []},
        ticket,
    )
    assert ledger.events(ticket)[-1] == event
    assert ledger.record_event(ticket, "note", actor="factory").data == {}
    for kind in RESERVED:
        with pytest.raises(ValueError, match=kind):
            ledger.record_event(ticket, kind, actor="factory")
    with pytest.raises(TicketNotFound):
        ledger.record_event("T-0099", "note", actor="factory")


def test_triage_sets_the_fields_and_moves_on(ledger: Ledger) -> None:
    ticket = new_ticket(ledger)
    triaged = ledger.triage(
        ticket,
        title="Better title",
        type=TicketType.BUG,
        risk=RiskTier.R1,
        actor="factory",
        data={"size": "S"},
    )
    assert (triaged.title, triaged.type, triaged.risk, triaged.status) == (
        "Better title",
        TicketType.BUG,
        RiskTier.R1,
        TicketStatus.TRIAGED,
    )
    [event] = [e for e in ledger.events(ticket) if e.kind == "triaged"]
    assert event.data == {"size": "S", "title": "Better title", "type": "bug", "risk": "R1"}
    assert [e.to_status for e in ledger.events(ticket) if e.to_status] == [
        TicketStatus.RECEIVED,
        TicketStatus.TRIAGED,
    ]
    with pytest.raises(IllegalTransition):
        ledger.triage(ticket, title="x", type=TicketType.BUG, risk=RiskTier.R0, actor="factory")


def contract() -> Contract:
    return Contract(
        version=1,
        summary="s",
        criteria=(Criterion(id="AC-1", ears="e", layer=TestLayer.E2E, tests=("a.py::t",)),),
    )


def test_propose_stores_contract_and_oracle_together(ledger: Ledger) -> None:
    ticket = new_ticket(ledger)
    version = "sha256:" + "b" * 64
    with pytest.raises(LedgerError):  # still received
        ledger.propose(ticket, contract(), version, actor="factory")
    assert ledger.get_ticket(ticket).contract is None
    ledger.triage(ticket, title="t", type=TicketType.FEATURE, risk=RiskTier.R1, actor="factory")
    proposed = ledger.propose(ticket, contract(), version, actor="factory")
    assert proposed.status is TicketStatus.CONTRACT_PROPOSED
    assert (proposed.contract, proposed.oracle_version) == (contract(), version)
    kinds = [e.kind for e in ledger.events(ticket)]
    assert {"contract_proposed", "oracle_version_set"} <= set(kinds)
    assert ledger.freeze_contract(ticket, actor="human").status is TicketStatus.APPROVED
    with pytest.raises(LedgerError):
        ledger.propose(ticket, contract(), version, actor="factory")


def test_propose_also_from_needs_info(ledger: Ledger) -> None:
    ticket = new_ticket(ledger)
    ledger.triage(ticket, title="t", type=TicketType.FEATURE, risk=RiskTier.R1, actor="factory")
    ledger.transition(ticket, TicketStatus.NEEDS_INFO, actor="factory")
    proposed = ledger.propose(ticket, contract(), "sha256:" + "c" * 64, actor="factory")
    assert proposed.status is TicketStatus.CONTRACT_PROPOSED


# --------------------------------------------------------------------------- unsafe symlinks


@pytest.mark.integration
def test_unsafe_symlinks_lists_every_escape(gate1: World) -> None:
    (gate1.target / "abs").symlink_to("/etc/passwd")
    (gate1.target / "up").symlink_to("../outside")
    (gate1.target / "fine").symlink_to("greet.py")
    (gate1.target / "docs").mkdir()
    (gate1.target / "docs" / "deep").symlink_to("../../x")
    gate1.git("add", "-A", cwd=gate1.target)
    gate1.git("commit", "-q", "-m", "links", cwd=gate1.target)
    repo = BareRepo.init(gate1.root / "links.git")
    repo.fetch_from(gate1.target, [f"{MAIN}:{MAIN}"])
    found = unsafe_symlinks(repo, MAIN)
    assert [path for path, _ in found] == ["abs", "docs/deep", "up"]
    assert all(reason for _, reason in found)
    assert unsafe_symlinks(gate1.repo, MAIN) == []


# --------------------------------------------------------------------------- test globs


def test_test_globs_are_configurable() -> None:
    default = parse_project_config({"project": {"name": "p"}})
    assert default.test_globs == DEFAULT_TEST_GLOBS
    assert DiffPolicy.from_project(default).test_globs == DEFAULT_TEST_GLOBS
    custom = parse_project_config(
        {"project": {"name": "p"}, "paths": {"test_globs": ["tests/**", "web/**/*.spec.ts"]}}
    )
    assert custom.test_globs == ("tests/**", "web/**/*.spec.ts")
    assert DiffPolicy.from_project(custom).test_globs == ("tests/**", "web/**/*.spec.ts")
    bad_values: tuple[object, ...] = ("tests/**", [])
    for bad in bad_values:
        with pytest.raises(ConfigError):
            parse_project_config({"project": {"name": "p"}, "paths": {"test_globs": bad}})
