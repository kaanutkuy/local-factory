"""The Gate 1 CLI (docs/specs/phase-2b.md § cli.py). Protected.

`main(argv, services=...)` runs with the fake backend and the in-process host site; the factory
home is a temporary `LOCAL_FACTORY_HOME`.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from local_factory.agents.fake import EmitOutput, Fail, RaiseHalt
from local_factory.cli import Services, main
from local_factory.config import FactoryConfig
from local_factory.hashing import content_hash
from local_factory.ledger import Ledger
from local_factory.models import HaltReason, TicketStatus
from local_factory.pipeline.flow import Site
from local_factory.pipeline.host_slice import host_site
from local_factory.projects import project_paths
from tests.acceptance.conftest import Git
from tests.acceptance.intake_world import (
    EARS,
    REQUEST,
    SHOUT_ID,
    SHOUT_PROBE_STDOUT,
    ScriptedBackend,
    critique_out,
    happy_scripts,
    make_target,
    spec_out,
)

pytestmark = pytest.mark.integration


@dataclass
class Harness:
    home: Path
    target: Path
    git: Git
    backend: ScriptedBackend
    capsys: pytest.CaptureFixture[str]
    sites: list[FactoryConfig] = field(default_factory=list)

    def services(self) -> Services:
        def site(config: FactoryConfig) -> AbstractContextManager[Site]:
            self.sites.append(config)
            return nullcontext(host_site())

        def backend(_config: FactoryConfig, _transcripts: Path) -> ScriptedBackend:
            return self.backend

        return Services(backend=backend, site=site)

    def run(self, *argv: str) -> tuple[int, str, str]:
        self.capsys.readouterr()
        try:
            code = main(list(argv), services=self.services())
        except SystemExit as exc:  # argparse
            code = exc.code if isinstance(exc.code, int) else 1
        out, err = self.capsys.readouterr()
        return code, out, err

    def ok(self, *argv: str) -> str:
        code, out, err = self.run(*argv)
        assert code == 0, (argv, out, err)
        return out

    def ledger(self, project: str = "cli-target") -> Ledger:
        config = FactoryConfig(home=self.home)
        return Ledger(project_paths(config, project).ledger)


@pytest.fixture
def cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> Harness:
    home = tmp_path / "factory-home"
    monkeypatch.setenv("LOCAL_FACTORY_HOME", str(home))
    (tmp_path / "githome").mkdir()
    git = Git(tmp_path / "githome")
    return Harness(
        home=home,
        target=make_target(tmp_path, git),
        git=git,
        backend=ScriptedBackend(happy_scripts()),
        capsys=capsys,
    )


def adopted(cli: Harness) -> Harness:
    assert "project cli-target added" in cli.ok("project", "add", str(cli.target))
    return cli


def first_line(out: str) -> str:
    return out.splitlines()[0]


def test_help_lists_the_commands(cli: Harness) -> None:
    code, out, _ = cli.run("--help")
    assert code == 0
    for command in ("project", "intake", "questions", "answer", "show", "approve"):
        assert command in out


def test_project_add(cli: Harness) -> None:
    adopted(cli)
    code, _, err = cli.run("project", "add", str(cli.target))
    assert code == 1 and err.startswith("error:") and "already" in err
    code, _, err = cli.run("project", "add", str(cli.home / "nowhere"))
    assert code == 1 and err.startswith("error:")
    assert cli.sites == []  # adopting a repository runs no model


def test_a_request_to_an_approved_contract(cli: Harness) -> None:
    adopted(cli)
    out = cli.ok("intake", REQUEST)
    assert first_line(out) == "T-0001: contract_proposed"
    assert "approve" in out
    assert len(cli.sites) == 1

    shown = cli.ok("show", "T-0001")
    for needed in ("T-0001", "contract_proposed", f"AC-1 [e2e, change] {EARS}", SHOUT_ID, "R1"):
        assert needed in shown, needed
    assert SHOUT_PROBE_STDOUT not in shown and "Grace Hopper" not in shown
    with_holdout = cli.ok("show", "T-0001", "--holdout")
    assert SHOUT_PROBE_STDOUT in with_holdout and "Grace Hopper" in with_holdout

    approved = cli.ok("approve", "T-0001")
    assert first_line(approved) == "T-0001: approved"
    with cli.ledger() as ledger:
        ticket = ledger.get_ticket("T-0001")
    assert ticket.status is TicketStatus.APPROVED and ticket.contract is not None
    assert ticket.contract_hash == content_hash(ticket.contract)
    assert ticket.contract_hash in approved
    assert len(cli.sites) == 1  # show and approve run no model


def test_questions_and_answers(cli: Harness) -> None:
    adopted(cli)
    cli.backend = ScriptedBackend(
        happy_scripts(critic=critique_out("Which flag name?", "Exit code?"))
    )
    cli.backend.add("spec-writer", [EmitOutput(spec_out(summary="s2"))])
    cli.backend.add("ambiguity-critic", [EmitOutput(critique_out())])

    out = cli.ok("intake", REQUEST)
    assert first_line(out) == "T-0001: needs_info"
    assert "Q-1: Which flag name?" in out and "Q-2: Exit code?" in out

    listed = cli.ok("questions", "T-0001")
    assert "Q-1 [open] Which flag name?" in listed and "Q-2 [open] Exit code?" in listed

    out = cli.ok("answer", "T-0001", "Q-1", "--shout")
    assert "Q-1 answered" in out and "contract_proposed" not in out
    sites = len(cli.sites)
    out = cli.ok("answer", "T-0001", "Q-2", "0")
    assert "Q-2 answered" in out
    assert "T-0001: contract_proposed" in out  # the last answer continues intake
    assert len(cli.sites) == sites + 1

    listed = cli.ok("questions", "T-0001")
    assert "Q-1 [answered] Which flag name?" in listed and "--shout" in listed
    shown = cli.ok("show", "T-0001")
    assert "Which flag name?" in shown and "--shout" in shown


@pytest.mark.parametrize(
    "argv",
    [
        ("show", "T-0042"),
        ("approve", "T-0001"),
        ("answer", "T-0001", "Q-7", "x"),
        ("answer", "T-0042", "Q-1", "x"),
        ("questions", "T-0042"),
        ("intake", "--resume", "T-0042"),
        ("intake", "   "),
    ],
)
def test_refusals_exit_1_with_a_message(cli: Harness, argv: Sequence[str]) -> None:
    adopted(cli)
    cli.backend = ScriptedBackend(happy_scripts(critic=critique_out("Which flag?")))
    cli.ok("intake", REQUEST)  # T-0001 waits for an answer
    code, _, err = cli.run(*argv)
    assert code == 1, argv
    assert err.startswith("error:")


def test_an_agent_error_exits_1(cli: Harness) -> None:
    adopted(cli)
    cli.backend.scripts["triage"] = [[Fail("triage fell over")]]
    code, out, err = cli.run("intake", REQUEST)
    assert code == 1
    assert first_line(out) == "T-0001: agent_error"
    assert "triage fell over" in out + err


def test_a_halted_intake_resumes(cli: Harness) -> None:
    adopted(cli)
    cli.backend.scripts["spec-writer"].insert(0, [RaiseHalt(HaltReason.BILLING)])
    code, out, err = cli.run("intake", REQUEST)
    assert code == 1 and first_line(out) == "T-0001: halted"
    assert "billing" in out + err
    out = cli.ok("intake", "--resume", "T-0001")
    assert first_line(out) == "T-0001: contract_proposed"


def test_without_a_project_intake_says_how_to_add_one(cli: Harness) -> None:
    code, _, err = cli.run("intake", REQUEST)
    assert code == 1 and "project add" in err


def test_two_projects_need_a_choice(cli: Harness) -> None:
    adopted(cli)
    cli.ok("project", "add", str(cli.target), "--name", "second")
    code, _, err = cli.run("intake", REQUEST)
    assert code == 1 and "--project" in err
    out = cli.ok("intake", REQUEST, "--project", "second")
    assert first_line(out) == "T-0001: contract_proposed"
    with cli.ledger("second") as ledger:
        assert [t.id for t in ledger.list_tickets()] == ["T-0001"]
    with cli.ledger() as ledger:
        assert ledger.list_tickets() == []


def test_usage_errors_exit_2(cli: Harness) -> None:
    adopted(cli)
    assert cli.run("intake")[0] == 2
    assert cli.run("intake", REQUEST, "--resume", "T-0001")[0] == 2
    assert cli.run("answer", "T-0001")[0] == 2
