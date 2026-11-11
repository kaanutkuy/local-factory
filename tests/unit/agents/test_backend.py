"""Unit tests for the backend contract types in local_factory.agents.backend."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from local_factory.agents.backend import (
    EFFORT_LEVELS,
    AgentBackend,
    AgentResult,
    AgentTask,
    Capability,
    RoleSpec,
    WorkerHandle,
)
from local_factory.agents.local import LocalExecutor
from local_factory.agents.tools import ToolExecutor
from local_factory.models import Halt, HaltReason, Tier, Usage


class Out(BaseModel):
    ok: bool


def assign(obj: Any, attribute: str, value: Any) -> None:
    """Attempt an attribute assignment the type checker would (rightly) reject."""
    setattr(obj, attribute, value)


# --------------------------------------------------------------------------- Capability / Effort


def test_capability_values() -> None:
    assert {c.value for c in Capability} == {
        "native_fs_tools",
        "hooks",
        "structured_output",
        "sessions",
        "subagents",
    }
    assert Capability("hooks") is Capability.HOOKS


def test_effort_levels_in_order() -> None:
    assert EFFORT_LEVELS == ("low", "medium", "high", "xhigh", "max")


# --------------------------------------------------------------------------- RoleSpec


def test_role_spec_defaults() -> None:
    role = RoleSpec(name="planner", prompt="plan", tier=Tier.FRONTIER)
    assert (role.effort, role.tools, role.max_turns, role.output_schema) == ("high", (), 50, None)


def test_role_spec_is_frozen_hashable_and_compared_by_value() -> None:
    role = RoleSpec(name="r", prompt="p", tier=Tier.FAST, tools=("Read",), output_schema=Out)
    same = RoleSpec(name="r", prompt="p", tier=Tier.FAST, tools=("Read",), output_schema=Out)
    assert role == same
    assert hash(role) == hash(same)
    with pytest.raises(dataclasses.FrozenInstanceError):
        assign(role, "name", "other")


@pytest.mark.parametrize("effort", EFFORT_LEVELS)
def test_role_spec_accepts_every_effort_level(effort: Any) -> None:
    assert RoleSpec(name="r", prompt="p", tier=Tier.FAST, effort=effort).effort == effort


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"name": ""}, "name"),
        ({"effort": "extreme"}, "effort"),
        ({"max_turns": 0}, "max_turns"),
        ({"max_turns": -3}, "max_turns"),
    ],
)
def test_role_spec_rejects_invalid_values(kwargs: dict[str, Any], message: str) -> None:
    fields: dict[str, Any] = {"name": "r", "prompt": "p", "tier": Tier.FAST, **kwargs}
    with pytest.raises(ValueError, match=message):
        RoleSpec(**fields)


# --------------------------------------------------------------------------- AgentTask


def test_agent_task_defaults_and_independent_context() -> None:
    first = AgentTask(ticket_id="T-0001", instructions="x")
    second = AgentTask(ticket_id="T-0001", instructions="x")
    assert first.task_id is None
    assert first.context == {}
    assert first.context is not second.context


def test_agent_task_accepts_json_context() -> None:
    task = AgentTask(
        ticket_id="T-0042",
        task_id="T-0042/task-3",
        instructions="x",
        context={"files": ["a.py"], "depth": 2, "flags": {"fast": True}, "note": None},
    )
    assert task.context["files"] == ["a.py"]


@pytest.mark.parametrize(
    "fields",
    [
        {"ticket_id": "T-1"},
        {"ticket_id": "ticket"},
        {"task_id": "task-1"},
        {"task_id": "T-0001/step-1"},
        {"unexpected": "field"},
    ],
)
def test_agent_task_rejects_malformed_ids_and_extra_fields(fields: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        AgentTask.model_validate({"ticket_id": "T-0001", "instructions": "x", **fields})


def test_agent_task_is_frozen() -> None:
    task = AgentTask(ticket_id="T-0001", instructions="x")
    with pytest.raises(ValidationError):
        assign(task, "instructions", "y")


# --------------------------------------------------------------------------- WorkerHandle


def test_worker_handle_is_frozen(tmp_path: Path) -> None:
    handle = WorkerHandle(work_dir=tmp_path)
    with pytest.raises(dataclasses.FrozenInstanceError):
        assign(handle, "work_dir", tmp_path / "x")
    with pytest.raises(dataclasses.FrozenInstanceError):
        assign(handle, "executor", None)


def test_worker_handle_tools_default_to_a_local_executor_on_work_dir(tmp_path: Path) -> None:
    handle = WorkerHandle(work_dir=tmp_path)
    assert handle.executor is None
    tools = handle.tools()
    assert isinstance(tools, LocalExecutor)
    assert tools.root == tmp_path
    assert isinstance(tools, ToolExecutor)


def test_worker_handle_tools_are_the_given_executor() -> None:
    executor = LocalExecutor(Path("/elsewhere"))
    handle = WorkerHandle(work_dir=Path("/work"), executor=executor)
    assert handle.tools() is executor


# --------------------------------------------------------------------------- AgentResult


def test_agent_result_ok_defaults() -> None:
    result = AgentResult(status="ok")
    assert result.halt is None
    assert result.structured_output is None
    assert result.text is None
    assert result.usage == Usage()
    assert result.session_id is None
    assert result.transcript_path is None
    assert result.permission_denials == ()
    assert result.error is None


def test_agent_result_halted_with_halt_is_valid() -> None:
    halt = Halt(reason=HaltReason.BILLING, detail="card declined")
    assert AgentResult(status="halted", halt=halt).halt == halt


@pytest.mark.parametrize(
    "fields",
    [
        {"status": "halted"},
        {"status": "ok", "halt": Halt(reason=HaltReason.BUDGET)},
        {"status": "error", "error": "boom", "halt": Halt(reason=HaltReason.MAX_TURNS)},
        {"status": "error"},
        {"status": "error", "error": ""},
        {"status": "done"},
        {"status": "ok", "unexpected": 1},
    ],
)
def test_agent_result_rejects_inconsistent_fields(fields: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        AgentResult.model_validate(fields)


def test_agent_result_round_trips_through_json(tmp_path: Path) -> None:
    result = AgentResult(
        status="ok",
        structured_output={"summary": "done", "files": ["a.py"], "n": 2},
        text="all good",
        usage=Usage(input_tokens=5, output_tokens=6, cache_read_tokens=1, cost_usd=0.5),
        session_id="s-1",
        transcript_path=tmp_path / "t.jsonl",
        permission_denials=({"tool": "Bash", "input": {"command": "curl x"}},),
    )
    assert AgentResult.model_validate_json(result.model_dump_json()) == result


def test_agent_result_rejects_non_json_structured_output() -> None:
    with pytest.raises(ValidationError):
        AgentResult.model_validate({"status": "ok", "structured_output": {"s": {1, 2}}})


# --------------------------------------------------------------------------- AgentBackend


class MinimalBackend:
    name: str = "minimal"
    capabilities: frozenset[Capability] = frozenset({Capability.SESSIONS})

    async def run(self, role: RoleSpec, task: AgentTask, worker: WorkerHandle) -> AgentResult:
        return AgentResult(status="ok")


def test_protocol_accepts_any_structurally_matching_backend() -> None:
    backend: AgentBackend = MinimalBackend()
    assert isinstance(backend, AgentBackend)
    assert not isinstance(object(), AgentBackend)
