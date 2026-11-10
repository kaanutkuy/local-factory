"""Integration: the ledger alone (events and runs) explains every way a task ends badly."""

from __future__ import annotations

import pytest

from local_factory.agents import Action, Fail, RaiseHalt, RunCommand, WriteFile
from local_factory.models import HaltReason, TaskStatus, TicketStatus, Violation

from .conftest import IMPLEMENT_SHOUT, Git, World

pytestmark = pytest.mark.integration

EVIL_LINK = RunCommand(("ln", "-s", "/etc/passwd", "evil"))
EDIT_CONFIG = WriteFile("factory.toml", '[project]\nname = "x"\n')


def violations_from_ledger(world: World, task_id: str) -> tuple[Violation, ...]:
    raw = world.task_ends()[task_id].detail.get("violations", [])
    assert isinstance(raw, list)
    return tuple(Violation.model_validate(item) for item in raw)


@pytest.mark.parametrize(
    ("script", "rejection"),
    [
        ([], "no_changes"),
        ([*IMPLEMENT_SHOUT, EVIL_LINK], "symlink_escape"),
        ([*IMPLEMENT_SHOUT, EDIT_CONFIG], "policy"),
        ([RunCommand(("rm", "-rf", ".git"))], "worker_git_error"),
    ],
    ids=["no-changes", "bundle-code", "policy", "worker-git-error"],
)
def test_the_ledger_records_each_rejection(
    world: World, script: list[Action], rejection: str
) -> None:
    outcome = world.run_scripts(script)

    assert (outcome.status, outcome.rejection) == ("rejected", rejection)
    end = world.task_ends()[outcome.task.id]
    assert (end.status, end.reason) == (TaskStatus.REJECTED, rejection)
    assert violations_from_ledger(world, outcome.task.id) == outcome.violations
    assert outcome.ticket.status is TicketStatus.BUILDING


def test_no_changes_names_the_base(world: World) -> None:
    outcome = world.run_scripts([])
    assert world.task_ends()[outcome.task.id].detail == {"error": f"no commits beyond {world.main}"}


def test_a_bundle_rejection_keeps_the_importers_detail(world: World) -> None:
    outcome = world.run_scripts([*IMPLEMENT_SHOUT, EVIL_LINK])
    error = world.task_ends()[outcome.task.id].detail["error"]
    assert isinstance(error, str)
    assert "evil" in error


def test_policy_violations_round_trip_through_the_ledger(world: World) -> None:
    outcome = world.run_scripts([*IMPLEMENT_SHOUT, EDIT_CONFIG])
    violations = violations_from_ledger(world, outcome.task.id)
    assert [(v.rule, v.path) for v in violations] == [("protected_path", "factory.toml")]
    assert all(v.message for v in violations)


def test_an_agent_error_is_in_the_task_event_and_the_run(world: World) -> None:
    outcome = world.run_scripts([*IMPLEMENT_SHOUT, Fail("cannot find the CLI parser")])

    assert (outcome.status, outcome.rejection) == ("agent_error", None)
    end = world.task_ends()[outcome.task.id]
    assert (end.status, end.reason) == (TaskStatus.REJECTED, "agent_error")
    assert end.detail == {"error": "cannot find the CLI parser"}
    (run,) = world.ledger.runs(world.ticket_id)
    assert (run.status, run.error) == ("error", "cannot find the CLI parser")


def test_successful_runs_record_no_error(world: World) -> None:
    outcome = world.run_scripts(IMPLEMENT_SHOUT)
    assert outcome.status == "candidate_verified"
    assert [r.error for r in world.ledger.runs(world.ticket_id)] == [None] * 3
    assert world.task_ends() == {}


@pytest.mark.parametrize(
    ("breakage", "failing_command"),
    [
        ("delete-git-dir", "git add"),  # the harness commit fails
        ("rename-task-branch", "git rev-list"),  # the commit check fails
    ],
)
def test_a_worker_that_breaks_its_repository_is_rejected(
    world: World, git: Git, breakage: str, failing_command: str
) -> None:
    script: list[Action] = (
        [RunCommand(("rm", "-rf", ".git"))]
        if breakage == "delete-git-dir"
        else [*IMPLEMENT_SHOUT, git.as_agent("branch", "-m", "task", "elsewhere")]
    )
    outcome = world.run_scripts(script)

    assert (outcome.status, outcome.rejection) == ("rejected", "worker_git_error")
    assert outcome.task.status is TaskStatus.REJECTED
    assert outcome.ticket.status is TicketStatus.BUILDING
    error = world.task_ends()[outcome.task.id].detail["error"]
    assert isinstance(error, str)
    assert error.startswith(failing_command)
    assert world.refs("refs/factory/") == []
    world.assert_boundaries()


def test_a_halted_task_records_the_halt(world: World) -> None:
    outcome = world.run_scripts([RaiseHalt(HaltReason.BUDGET)])

    assert outcome.status == "halted"
    end = world.task_ends()[outcome.task.id]
    assert (end.status, end.reason) == (TaskStatus.HALTED, "budget")
    assert end.detail == {"halt": {"reason": "budget", "resume_at": None, "detail": ""}}
    (run,) = world.ledger.runs(world.ticket_id)
    assert (run.status, run.error) == ("halted", None)
