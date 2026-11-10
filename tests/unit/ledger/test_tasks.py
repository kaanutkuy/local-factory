"""Unit tests for tasks."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import pytest
from pydantic import JsonValue, ValidationError

from local_factory.ledger import Ledger, LedgerError, TaskNotFound, TicketNotFound
from local_factory.models import Task, TaskStatus, Tier

from .helpers import SHA_A, SHA_B, SHA_C, FakeClock, new_ticket


def _task(
    ledger: Ledger,
    ticket_id: str,
    *,
    goal: str = "g",
    role: str = "implementer",
    criteria_served: Sequence[str] = (),
    depends_on: Sequence[str] = (),
) -> Task:
    return ledger.create_task(
        ticket_id,
        goal=goal,
        role=role,
        tier=Tier.STANDARD,
        criteria_served=criteria_served,
        depends_on=depends_on,
        actor="orch",
    )


def test_create_task_defaults_and_event(ledger: Ledger, clock: FakeClock) -> None:
    ticket = new_ticket(ledger)
    task = ledger.create_task(
        ticket.id,
        goal="add tag model",
        role="implementer",
        tier=Tier.FRONTIER,
        criteria_served=["AC-1", "AC-2"],
        actor="orch",
    )
    assert task == Task(
        id=f"{ticket.id}/task-1",
        ticket_id=ticket.id,
        goal="add tag model",
        criteria_served=("AC-1", "AC-2"),
        role="implementer",
        tier=Tier.FRONTIER,
    )
    assert task.status is TaskStatus.PENDING
    assert ledger.get_task(task.id) == task
    event = ledger.events(ticket.id)[-1]
    assert (event.kind, event.actor, event.at) == ("task_created", "orch", clock.now)
    assert event.data == {
        "task_id": task.id,
        "role": "implementer",
        "tier": "frontier",
        "depends_on": [],
    }


def test_task_numbers_are_numeric_and_survive_reopen(db_path: Path, clock: FakeClock) -> None:
    with Ledger(db_path, clock=clock) as led:
        ticket = new_ticket(led)
        for _ in range(10):
            _task(led, ticket.id)
    with Ledger(db_path, clock=clock) as led:
        eleventh = _task(led, ticket.id)
        assert eleventh.id == f"{ticket.id}/task-11"
        ids = [t.id for t in led.list_tasks(ticket.id)]
        assert ids == [f"{ticket.id}/task-{n}" for n in range(1, 12)]


def test_dependencies_must_exist_in_the_same_ticket(ledger: Ledger) -> None:
    a, b = new_ticket(ledger, "a"), new_ticket(ledger, "b")
    first = _task(ledger, a.id)
    other = _task(ledger, b.id)
    second = _task(ledger, a.id, depends_on=[first.id])
    assert second.depends_on == (first.id,)
    assert ledger.events(a.id)[-1].data["depends_on"] == [first.id]
    with pytest.raises(TaskNotFound, match="task-9"):
        _task(ledger, a.id, depends_on=[f"{a.id}/task-9"])
    with pytest.raises(LedgerError, match="another ticket"):
        _task(ledger, a.id, depends_on=[other.id])
    assert [t.id for t in ledger.list_tasks(a.id)] == [first.id, second.id]


def test_a_bare_string_is_not_a_sequence_of_ids(ledger: Ledger) -> None:
    ticket = new_ticket(ledger)
    with pytest.raises(ValidationError):
        _task(ledger, ticket.id, criteria_served="AC-1")
    assert ledger.list_tasks(ticket.id) == []


def test_invalid_task_fields_are_rejected(ledger: Ledger) -> None:
    ticket = new_ticket(ledger)
    with pytest.raises(ValidationError):
        _task(ledger, ticket.id, goal="")
    with pytest.raises(ValidationError):
        _task(ledger, ticket.id, role="")
    assert ledger.list_tasks(ticket.id) == []


def test_task_operations_on_unknown_ids(ledger: Ledger) -> None:
    with pytest.raises(TicketNotFound):
        _task(ledger, "T-0404")
    with pytest.raises(TicketNotFound):
        ledger.list_tasks("T-0404")
    with pytest.raises(TaskNotFound, match="T-0404/task-1"):
        ledger.get_task("T-0404/task-1")
    with pytest.raises(TaskNotFound):
        ledger.update_task("T-0404/task-1", actor="factory", status=TaskStatus.RUNNING)
    assert ledger.events() == []


def test_list_tasks_of_a_ticket_without_tasks(ledger: Ledger) -> None:
    assert ledger.list_tasks(new_ticket(ledger).id) == []


def test_update_task_changes_only_given_fields(ledger: Ledger, clock: FakeClock) -> None:
    task = _task(ledger, new_ticket(ledger).id)
    with_base = ledger.update_task(task.id, actor="factory", base_sha=SHA_A)
    assert (with_base.base_sha, with_base.head_sha, with_base.status) == (
        SHA_A,
        None,
        TaskStatus.PENDING,
    )
    running = ledger.update_task(task.id, actor="factory", status=TaskStatus.RUNNING)
    assert (running.base_sha, running.status) == (SHA_A, TaskStatus.RUNNING)
    done = ledger.update_task(task.id, actor="factory", status=TaskStatus.ACCEPTED, head_sha=SHA_C)
    assert ledger.get_task(task.id) == done
    assert (done.base_sha, done.head_sha, done.status) == (SHA_A, SHA_C, TaskStatus.ACCEPTED)
    event = ledger.events(task.ticket_id)[-1]
    assert (event.kind, event.actor, event.at) == ("task_updated", "factory", clock.now)
    assert event.data == {
        "task_id": task.id,
        "changes": {"status": "accepted", "head_sha": SHA_C},
        "reason": "",
        "detail": {},
    }


def test_update_task_requires_a_change(ledger: Ledger) -> None:
    task = _task(ledger, new_ticket(ledger).id)
    events = ledger.events()
    with pytest.raises(LedgerError, match="needs status, base_sha or head_sha"):
        ledger.update_task(task.id, actor="factory")
    assert ledger.events() == events


@pytest.mark.parametrize("bad", ["A" * 40, "a" * 39, "a" * 41, "g" * 40, ""])
def test_update_task_rejects_malformed_shas(ledger: Ledger, bad: str) -> None:
    task = _task(ledger, new_ticket(ledger).id)
    events = ledger.events()
    with pytest.raises(ValidationError):
        ledger.update_task(task.id, actor="factory", head_sha=bad, base_sha=SHA_B)
    assert ledger.get_task(task.id) == task
    assert ledger.events() == events


def test_update_task_records_reason_and_detail(db_path: Path, clock: FakeClock) -> None:
    detail: dict[str, JsonValue] = {
        "violations": [{"rule": "protected_path", "path": "factory.toml", "line": None}],
        "bundle": {"code": "unsafe_path", "ok": False, "commits": 3, "ratio": 0.5},
    }
    with Ledger(db_path, clock=clock) as led:
        task = _task(led, new_ticket(led).id)
        led.update_task(
            task.id,
            actor="factory",
            status=TaskStatus.REJECTED,
            reason="diff policy violations",
            data=detail,
        )
        detail["mutated_after_the_call"] = True  # the ledger keeps its own copy
    with Ledger(db_path, clock=clock) as led:
        assert led.get_task(task.id).status is TaskStatus.REJECTED
        event = led.events(task.ticket_id)[-1]
        assert event.data == {
            "task_id": task.id,
            "changes": {"status": "rejected"},
            "reason": "diff policy violations",
            "detail": {
                "violations": [{"rule": "protected_path", "path": "factory.toml", "line": None}],
                "bundle": {"code": "unsafe_path", "ok": False, "commits": 3, "ratio": 0.5},
            },
        }


def test_reason_and_detail_alone_are_not_an_update(ledger: Ledger) -> None:
    task = _task(ledger, new_ticket(ledger).id)
    events = ledger.events()
    with pytest.raises(LedgerError, match="needs status, base_sha or head_sha"):
        ledger.update_task(task.id, actor="factory", reason="why", data={"k": "v"})
    assert ledger.get_task(task.id) == task
    assert ledger.events() == events


@pytest.mark.parametrize("bad", [object(), math.nan, math.inf])
def test_non_json_detail_rolls_back_the_task_change(ledger: Ledger, bad: object) -> None:
    task = _task(ledger, new_ticket(ledger).id)
    events = ledger.events()
    with pytest.raises(ValidationError):
        ledger.update_task(
            task.id,
            actor="factory",
            status=TaskStatus.REJECTED,
            data=cast("dict[str, JsonValue]", {"value": bad}),
        )
    assert ledger.get_task(task.id) == task
    assert ledger.events() == events
