"""Unit tests for the CLI: the parser, the services and the messages around failures.

The Gate 1 commands end to end are in tests/acceptance/test_cli.py.
"""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import pytest
from pydantic import JsonValue

from local_factory import __version__
from local_factory.agents import ClaudeAPIBackend
from local_factory.agents.fake import EmitOutput, Fail, RaiseHalt
from local_factory.cli import Services, build_parser, default_services, main
from local_factory.config import FactoryConfig, SandboxConfig
from local_factory.hashing import utc_now
from local_factory.ledger import Ledger
from local_factory.models import Contract, Criterion, HaltReason, TestLayer
from local_factory.pipeline.flow import Site
from local_factory.pipeline.host_slice import host_site
from local_factory.pipeline.intake import questions, receive
from local_factory.projects import project_paths
from local_factory.sandbox.runtime import SandboxError
from tests.acceptance.conftest import Git
from tests.acceptance.intake_world import (
    EARS,
    KEEP_EARS,
    KEEP_ID,
    REQUEST,
    SHOUT_ID,
    ScriptedBackend,
    critique_out,
    happy_scripts,
    keep_scripts,
    make_target,
    manifest,
    oracle_script,
    shout_probe,
    spec_out,
    triage_out,
)


def test_no_command_prints_help_and_succeeds(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    assert "usage: factory" in capsys.readouterr().out


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


# --------------------------------------------------------------------------- the Gate 1 commands
# The commands end to end are in tests/acceptance/test_cli.py; these cover the parser's corners,
# the services and the messages around failures.


@dataclass
class Cli:
    home: Path
    target: Path
    backend: ScriptedBackend
    capsys: pytest.CaptureFixture[str]
    site_error: Exception | None = None
    sites: list[FactoryConfig] = field(default_factory=list)

    def services(self) -> Services:
        def site(config: FactoryConfig) -> AbstractContextManager[Site]:
            self.sites.append(config)
            if self.site_error is not None:
                raise self.site_error
            return nullcontext(host_site())

        return Services(backend=lambda _config, _transcripts: self.backend, site=site)

    def run(self, *argv: str) -> tuple[int, str, str]:
        self.capsys.readouterr()
        try:
            code = main(list(argv), services=self.services())
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 1
        out, err = self.capsys.readouterr()
        return code, out, err

    def ok(self, *argv: str) -> str:
        code, out, err = self.run(*argv)
        assert code == 0, (argv, out, err)
        return out

    def ledger(self, project: str = "cli-target") -> Ledger:
        return Ledger(project_paths(FactoryConfig(home=self.home), project).ledger)


@pytest.fixture
def cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> Cli:
    home = tmp_path / "factory-home"
    monkeypatch.setenv("LOCAL_FACTORY_HOME", str(home))
    (tmp_path / "githome").mkdir()
    target = make_target(tmp_path, Git(tmp_path / "githome"))
    cli = Cli(home=home, target=target, backend=ScriptedBackend(happy_scripts()), capsys=capsys)
    assert "added" in cli.ok("project", "add", str(target))
    return cli


# ------------------------------------------------------------------ the parser


@pytest.mark.parametrize(
    "argv",
    [
        ("--bogus",),
        ("project",),
        ("project", "remove", "x"),
        ("intake",),
        ("intake", "text", "--resume", "T-0001"),
        ("intake", "--resume", "T-0001", "--shout"),
        ("answer", "T-0001"),
        ("answer", "T-0001", "Q-1"),
        ("answer", "T-0001", "Q-1", "one", "two"),
        ("show",),
        ("show", "T-0001", "--bogus"),
        ("approve", "T-0001", "extra"),
        ("questions",),
    ],
)
def test_bad_usage_exits_2_and_changes_nothing(cli: Cli, argv: tuple[str, ...]) -> None:
    code, out, err = cli.run(*argv)
    assert code == 2 and out == "" and "usage:" in err
    assert cli.sites == []
    with cli.ledger() as ledger:
        assert ledger.list_tickets() == []


def test_a_global_option_without_a_command_is_a_usage_error(cli: Cli) -> None:
    assert cli.run("--bogus", "extra")[0] == 2


@pytest.mark.parametrize("text", ["--shout", "-x", "--pro"])
def test_a_request_may_start_with_a_dash(cli: Cli, text: str) -> None:
    out = cli.ok("intake", text)
    assert out.startswith("T-0001: contract_proposed")
    with cli.ledger() as ledger:
        assert ledger.get_ticket("T-0001").request == text


def test_an_answer_may_start_with_a_dash_wherever_the_options_are(cli: Cli) -> None:
    cli.backend = ScriptedBackend(happy_scripts(critic=critique_out("Q1?", "Q2?", "Q3?")))
    cli.backend.add("spec-writer", [EmitOutput(spec_out())])
    cli.backend.add("ambiguity-critic", [EmitOutput(critique_out())])
    cli.ok("intake", REQUEST)
    cli.ok("answer", "T-0001", "Q-1", "--project", "cli-target", "free text first")
    cli.ok("answer", "T-0001", "Q-2", "--project", "cli-target", "--loud")
    cli.ok("answer", "T-0001", "Q-3", "-1", "--project", "cli-target")
    with cli.ledger() as ledger:
        answers = [q.answer for q in questions(ledger, "T-0001")]
    assert answers == ["free text first", "--loud", "-1"]


def test_options_and_text_in_any_order(cli: Cli) -> None:
    out = cli.ok("intake", "--project", "cli-target", REQUEST)
    assert out.startswith("T-0001: contract_proposed")
    assert " --project cli-target" in out  # the next commands to run keep the project


def test_an_abbreviated_option_is_not_accepted(cli: Cli) -> None:
    assert cli.run("show", "T-0001", "--hold")[0] == 2


# ------------------------------------------------------------------ failures


def test_a_broken_factory_config_is_an_error(cli: Cli) -> None:
    (cli.home / "config.toml").write_text("this = is not [toml")
    code, _, err = cli.run("questions", "T-0001")
    assert code == 1 and err.startswith("error:") and "config.toml" in err


def test_a_site_that_cannot_start_is_an_error_and_the_ticket_can_be_resumed(cli: Cli) -> None:
    cli.site_error = SandboxError("docker is not running")
    code, out, err = cli.run("intake", REQUEST, "--project", "cli-target")
    assert code == 1 and out == ""
    assert "error: docker is not running" in err
    assert "T-0001 was recorded; retry with: factory intake --resume T-0001" in err
    assert "--project cli-target" in err
    cli.site_error = None
    assert cli.ok("intake", "--resume", "T-0001").startswith("T-0001: contract_proposed")


def test_a_resume_that_cannot_start_does_not_claim_a_new_ticket(cli: Cli) -> None:
    cli.site_error = SandboxError("docker is not running")
    with cli.ledger() as ledger:
        receive(ledger, REQUEST)
    code, _, err = cli.run("intake", "--resume", "T-0001")
    assert code == 1 and "was recorded" not in err


def test_an_oracle_that_never_passes_exits_1_with_the_problems(cli: Cli) -> None:
    rigged = oracle_script(files={"greet.py": "x = 1\n"})
    cli.backend.scripts["test-architect"] = [rigged] * 3
    code, out, err = cli.run("intake", REQUEST)
    assert code == 1 and out.splitlines() == ["T-0001: oracle_failed"]
    assert err.startswith("error: the tests were rejected 3 times") and "greet.py" in err
    assert "factory intake --resume" not in out + err  # only a halt can be resumed


def test_a_halt_says_how_to_resume(cli: Cli) -> None:
    cli.backend.scripts["triage"].insert(0, [RaiseHalt(HaltReason.RATE_LIMIT)])
    code, out, err = cli.run("intake", REQUEST, "--project", "cli-target")
    assert code == 1
    assert "Resume with: factory intake --resume T-0001 --project cli-target" in out
    assert "rate_limit" in err
    assert cli.ok("intake", "--resume", "T-0001").startswith("T-0001: contract_proposed")


def test_a_ticket_that_is_not_waiting_cannot_be_answered(cli: Cli) -> None:
    cli.ok("intake", REQUEST)
    code, _, err = cli.run("answer", "T-0001", "Q-1", "x")
    assert code == 1 and "not waiting for answers" in err


def test_a_ticket_past_intake_cannot_be_resumed(cli: Cli) -> None:
    cli.ok("intake", REQUEST)
    cli.ok("approve", "T-0001")
    code, _, err = cli.run("intake", "--resume", "T-0001")
    assert code == 1 and "approved" in err


# ------------------------------------------------------------------ what is printed


def test_a_waiting_ticket_prints_its_questions_and_how_to_answer(cli: Cli) -> None:
    cli.backend = ScriptedBackend(happy_scripts(critic=critique_out("Which flag?")))
    out = cli.ok("intake", REQUEST, "--project", "cli-target")
    assert out.splitlines()[:2] == ["T-0001: needs_info", "Q-1: Which flag?"]
    assert "why it blocks: A wrong guess is rejected." in out
    assert 'factory answer T-0001 QUESTION "TEXT" --project cli-target' in out
    again = cli.ok("intake", "--resume", "T-0001")  # nothing to run: still waiting
    assert again.startswith("T-0001: needs_info\nQ-1: Which flag?")
    assert len(cli.backend.calls) == 3  # triage, spec writer and critic once; none to resume


def test_an_unanswered_question_is_open_in_show_and_questions(cli: Cli) -> None:
    cli.backend = ScriptedBackend(happy_scripts(critic=critique_out("Which flag?")))
    cli.backend.add("spec-writer", [EmitOutput(spec_out())])
    cli.backend.add("ambiguity-critic", [EmitOutput(critique_out())])
    cli.ok("intake", REQUEST)
    assert "Q-1 [open] Which flag?" in cli.ok("questions", "T-0001")
    out = cli.ok("show", "T-0001")
    assert "Q-1: Which flag?" in out and "answer: (open)" in out
    assert "contract" not in out  # nothing proposed yet
    out = cli.ok("answer", "T-0001", "Q-1", "--shout")
    assert out.splitlines()[0] == "Q-1 answered"
    assert "Q-1 [answered] Which flag?\n    --shout" in cli.ok("questions", "T-0001")


def test_a_ticket_without_questions_says_so(cli: Cli) -> None:
    cli.ok("intake", REQUEST)
    assert cli.ok("questions", "T-0001") == "no questions\n"


def test_show_prints_the_whole_contract_but_never_the_probes(cli: Cli) -> None:
    cli.ok("intake", REQUEST)
    out = cli.ok("show", "T-0001")
    assert out.splitlines()[:5] == [
        "T-0001: Add --shout to greet.py",
        "status: contract_proposed",
        "type: feature",
        "risk: R1",
        "request:",
    ]
    for line in (
        f"  AC-1 [e2e, change] {EARS}",
        f"      visible test: {SHOUT_ID}",
        "      holdout probes: 1",
        "assumptions:",
        "  - Names are plain ASCII.",
        "  - Upper case means str.upper().",
        "out of scope:",
        "migration impact: none",
    ):
        assert line in out, line
    assert "oracle version: sha256:" in out and "contract hash" not in out  # not frozen yet
    assert "holdout (sealed" not in out and "HP-1" not in out
    cli.ok("approve", "T-0001")
    assert "contract hash: sha256:" in cli.ok("show", "T-0001")


def test_show_prints_each_criterions_kind(cli: Cli) -> None:
    cli.backend = ScriptedBackend(keep_scripts())
    cli.ok("intake", REQUEST)
    out = cli.ok("show", "T-0001")
    assert f"  AC-1 [e2e, change] {EARS}" in out
    assert f"  AC-2 [e2e, keep] {KEEP_EARS}" in out
    assert f"      visible test: {KEEP_ID}" in out


def test_show_holdout_prints_each_probe(cli: Cli) -> None:
    stdin_probe: dict[str, JsonValue] = {
        "criteria": ["AC-1"],
        "argv": ["python3", "greet.py", "--name", "Ada", "--shout"],
        "stdin": "ignored\n",
        "expect": {"stdout_contains": ["ADA"], "exit_code": 0},
    }
    cli.backend.scripts["test-architect"] = [
        oracle_script(out=manifest(holdout=[shout_probe(), stdin_probe]))
    ]
    cli.ok("intake", REQUEST)
    out = cli.ok("show", "T-0001", "--holdout")
    assert "holdout (sealed, never shown to implementers):" in out
    assert '  HP-1 [AC-1] argv: ["python3", "greet.py", "--name", "Grace Hopper", "--shout"]' in out
    assert '      expect: {"exit_code": 0, "stdout": "HELLO, GRACE HOPPER!"}' in out
    assert '      stdin: "ignored\\n"' in out
    assert '      expect: {"exit_code": 0, "stdout_contains": ["ADA"]}' in out


def test_show_holdout_without_a_record_says_so(cli: Cli) -> None:
    with cli.ledger() as ledger:
        receive(ledger, REQUEST)
    out = cli.ok("show", "T-0001", "--holdout")
    assert out.startswith("T-0001: ") and "status: received" in out
    assert "holdout (sealed, never shown to implementers):\n  none recorded" in out


def test_show_holdout_with_a_damaged_record_is_an_error(cli: Cli) -> None:
    cli.ok("intake", REQUEST)
    record = project_paths(FactoryConfig(home=cli.home), "cli-target").holdout / "T-0001.json"
    record.write_text("damaged")
    code, _, err = cli.run("show", "T-0001", "--holdout")
    assert code == 1 and err.startswith("error:")


# ------------------------------------------------------------------ the services


class RecordingRuntime:
    instances: ClassVar[list[RecordingRuntime]] = []

    def __init__(self, *, docker: str = "docker") -> None:
        self.docker = docker
        self.cleaned = 0
        self.instances.append(self)

    def cleanup(self) -> None:
        self.cleaned += 1


def test_the_default_services_use_the_claude_backend(tmp_path: Path) -> None:
    config = FactoryConfig(home=tmp_path)
    backend = default_services().backend(config, tmp_path / "transcripts")
    assert isinstance(backend, ClaudeAPIBackend) and backend.name == "claude"
    assert default_services().clock is utc_now


@pytest.mark.parametrize("fails", [False, True])
def test_the_default_site_is_the_box_and_cleans_up_after_itself(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fails: bool
) -> None:
    RecordingRuntime.instances.clear()
    monkeypatch.setattr("local_factory.cli.DockerRuntime", RecordingRuntime)
    config = FactoryConfig(home=tmp_path, sandbox=SandboxConfig(docker="/opt/docker"))
    try:
        with default_services().site(config) as site:
            assert site.run_limit_s == config.sandbox.run_timeout_s  # the box site's limit
            if fails:
                raise KeyboardInterrupt
    except KeyboardInterrupt:
        assert fails
    [runtime] = RecordingRuntime.instances
    assert (runtime.docker, runtime.cleaned) == ("/opt/docker", 1)


def test_a_question_the_factory_asks_has_no_reason_line(cli: Cli) -> None:
    cli.backend.scripts["triage"] = [[EmitOutput(triage_out(size="XL", rationale="Too much."))]]
    out = cli.ok("intake", REQUEST)
    assert out.splitlines()[0] == "T-0001: needs_info"
    assert "Q-1: Triage thinks this is too large for one ticket: Too much." in out
    assert "why it blocks" not in out


def test_show_prints_a_contract_that_has_no_oracle_yet(cli: Cli) -> None:
    contract = Contract(
        version=1,
        summary="Only the words.",
        criteria=(Criterion(id="AC-1", ears="When x, the system shall y.", layer=TestLayer.UNIT),),
    )
    with cli.ledger() as ledger:
        ticket = receive(ledger, REQUEST)
        ledger.propose_contract(ticket.id, contract, actor="human")
    out = cli.ok("show", "T-0001")
    assert "contract (version 1): Only the words." in out
    assert "      holdout probes: 0" in out and "oracle version" not in out


def test_a_role_that_failed_is_named_in_the_error(cli: Cli) -> None:
    cli.backend.scripts["spec-writer"] = [[Fail("spec writer fell over")]]
    code, out, err = cli.run("intake", REQUEST)
    assert code == 1 and out.splitlines() == ["T-0001: agent_error"]
    assert err == "error: spec-writer: spec writer fell over\n"
