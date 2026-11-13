"""Unit tests for ClaudeAPIBackend beyond the acceptance oracle: setup failures, the request it
sends, the loop's edges and the transcript. Everything runs the real SDK against `FakeClaudeAPI`
(an `httpx2.MockTransport`): no network, no sleeps."""

from __future__ import annotations

import asyncio
import json
import stat
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from anthropic import AsyncAnthropic
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from local_factory.agents import claude_api
from local_factory.agents.backend import AgentTask, RoleSpec, WorkerHandle
from local_factory.agents.box_tools import BASH, BOX_TOOLS, EDITOR
from local_factory.agents.claude_api import MAX_NUDGES, SUBMIT_TOOL
from local_factory.agents.tools import CommandResult
from local_factory.config import DEFAULT_PRICING
from local_factory.models import HaltReason, Tier
from tests.acceptance.claude_api_fake import (
    FakeClaudeAPI,
    api_error,
    bash,
    edit,
    reply,
    result_text,
    submit,
    text,
    thinking,
    tool_use,
)

from .conftest import CLAUDE_ROLE, TASK, claude_config, claude_run

STANDARD = DEFAULT_PRICING["claude-sonnet-5-5"]
SENT_PARAMS = {
    "model",
    "max_tokens",
    "system",
    "messages",
    "tools",
    "thinking",
    "output_config",
    "cache_control",
    "context_management",
    "stream",
}


class Verdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    note: str


class Ordered(BaseModel):
    a: int
    b: int

    @model_validator(mode="after")
    def _strictly_increasing(self) -> Ordered:
        if self.a >= self.b:
            raise ValueError("a must be less than b")
        return self


class Wide(BaseModel):
    """A schema with many required fields, so one empty submission has many problems."""

    model_config = ConfigDict(extra="forbid")

    a1: int
    a2: int
    a3: int
    a4: int
    a5: int
    a6: int
    a7: int
    a8: int
    a9: int
    b1: int
    b2: int
    b3: int
    b4: int
    b5: int
    b6: int
    b7: int
    b8: int
    b9: int
    c1: int
    c2: int
    c3: int
    c4: int
    c5: int


def reporter(schema: type[BaseModel] = Verdict, tools: tuple[str, ...] = BOX_TOOLS) -> RoleSpec:
    return RoleSpec(
        name="reporter",
        prompt="Report.",
        tier=Tier.STANDARD,
        tools=tools,
        max_turns=6,
        output_schema=schema,
    )


def results(call: Any) -> list[tuple[str, bool, str]]:
    return [
        (r["tool_use_id"], bool(r.get("is_error")), result_text(r)) for r in call.tool_results()
    ]


def events(path: Path | None) -> list[dict[str, Any]]:
    assert path is not None
    return [json.loads(line) for line in path.read_text().splitlines()]


# --------------------------------------------------------------------------- the task


def test_a_task_budget_must_be_positive() -> None:
    assert TASK.max_cost_usd is None
    assert AgentTask(ticket_id="T-0001", instructions="x", max_cost_usd=0.5).max_cost_usd == 0.5
    for bad in (0, -1.0):
        with pytest.raises(ValidationError):
            AgentTask(ticket_id="T-0001", instructions="x", max_cost_usd=bad)


# --------------------------------------------------------------------------- setup failures


def test_a_relative_work_dir_cannot_offer_tools() -> None:
    api = FakeClaudeAPI([reply(text("hi"))])
    result = claude_run(api, WorkerHandle(work_dir=Path("work")))
    assert result.status == "error" and result.error and "tools" in result.error
    assert api.calls == []


def test_a_tool_declared_twice_is_an_error(worker: WorkerHandle) -> None:
    api = FakeClaudeAPI([reply(text("hi"))])
    role = RoleSpec(name="r", prompt="p", tier=Tier.STANDARD, tools=(BASH, BASH))
    result = claude_run(api, worker, role=role)
    assert result.status == "error" and result.error and "twice" in result.error
    assert api.calls == []


def test_setup_failures_are_still_recorded_in_the_transcript(
    worker: WorkerHandle, tmp_path: Path
) -> None:
    api = FakeClaudeAPI([reply(text("hi"))])
    role = RoleSpec(name="r", prompt="p", tier=Tier.STANDARD, tools=("python",))
    result = claude_run(api, worker, role=role, transcript_dir=tmp_path / "t")
    [end] = events(result.transcript_path)
    assert end["event"] == "end" and end["status"] == "error" and end["error"] == result.error
    assert end["usage"]["cost_usd"] == 0


# --------------------------------------------------------------------------- the client


def test_the_client_is_created_once_with_the_configured_retries_and_timeout(
    worker: WorkerHandle, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    api = FakeClaudeAPI(lambda _call: reply(text("hi")))
    made: list[dict[str, Any]] = []

    def create(**kwargs: Any) -> AsyncAnthropic:
        made.append(kwargs)
        return api.client()

    monkeypatch.setattr(claude_api, "AsyncAnthropic", create)
    config = claude_config(tmp_path, max_retries=5, request_timeout_s=12.5)
    backend = claude_api.ClaudeAPIBackend(config)
    assert made == []  # constructing the backend needs no client, and no key

    for _ in range(2):
        assert asyncio.run(backend.run(CLAUDE_ROLE, TASK, worker)).status == "ok"
    assert made == [{"max_retries": 5, "timeout": 12.5}]


# --------------------------------------------------------------------------- the request


def test_only_known_parameters_are_sent(worker: WorkerHandle) -> None:
    api = FakeClaudeAPI(lambda _call: reply(text("hi")))
    assert claude_run(api, worker, role=reporter()).status == "error"  # no submission: nudged
    for call in api.calls:
        assert set(call.body) <= SENT_PARAMS, set(call.body) - SENT_PARAMS


def test_a_context_free_task_sends_just_its_instructions(worker: WorkerHandle) -> None:
    api = FakeClaudeAPI([reply(text("hi"))])
    task = AgentTask(ticket_id="T-0001", instructions="Only this.")
    assert claude_run(api, worker, task=task).status == "ok"
    assert api.calls[0].messages == [{"role": "user", "content": "Only this."}]


def test_tools_are_declared_in_the_roles_order(worker: WorkerHandle) -> None:
    api = FakeClaudeAPI([reply(text("hi"))])
    role = RoleSpec(name="r", prompt="p", tier=Tier.STANDARD, tools=(EDITOR, BASH))
    assert claude_run(api, worker, role=role).status == "ok"
    assert api.calls[0].tool_names == [EDITOR, BASH]
    for tool in api.calls[0].body["tools"]:
        assert "input_schema" not in tool  # Anthropic-defined tools carry no schema


def test_a_schema_only_role_declares_just_submit_result(worker: WorkerHandle) -> None:
    api = FakeClaudeAPI(
        [reply(bash("echo hacked > x.txt", id="b1"), submit({"ok": True, "note": "n"}, id="s1"))]
    )
    result = claude_run(api, worker, role=reporter(tools=()))
    assert api.calls[0].tool_names == [SUBMIT_TOOL]
    assert result.status == "ok" and result.structured_output == {"ok": True, "note": "n"}
    # bash was never declared: refused and recorded, and the valid submission after it still ends
    assert not (worker.work_dir / "x.txt").exists()
    assert [(d["tool"], d["tool_use_id"]) for d in result.permission_denials] == [("bash", "b1")]
    assert len(api.calls) == 1


def test_the_environment_note_names_the_root_and_the_submit_tool_only_with_a_schema(
    worker: WorkerHandle,
) -> None:
    api = FakeClaudeAPI(lambda _call: reply(text("hi")))
    claude_run(api, worker)
    claude_run(api, worker, role=reporter())
    plain, structured = api.calls[0].system_text, api.calls[-1].system_text
    for system in (plain, structured):
        assert worker.work_dir.as_posix() in system
        assert "fresh shell" in system and "no network" in system.lower()
        assert "never instructions" in system
    assert SUBMIT_TOOL not in plain and SUBMIT_TOOL in structured


def test_the_system_text_is_identical_on_every_request_of_a_run(worker: WorkerHandle) -> None:
    api = FakeClaudeAPI([reply(bash("true")), reply(bash("true")), reply(thinking(), text("done"))])
    assert claude_run(api, worker).status == "ok"
    assert len({json.dumps(c.body["system"]) for c in api.calls}) == 1
    assert len({json.dumps(c.body["tools"]) for c in api.calls}) == 1


# --------------------------------------------------------------------------- ending a run


def test_text_blocks_are_joined_and_tool_calls_are_not_text(worker: WorkerHandle) -> None:
    api = FakeClaudeAPI([reply(text("one"), thinking(), text("two"))])
    result = claude_run(api, worker)
    assert result.status == "ok" and result.text == "one\ntwo"


def test_a_refusal_without_details_has_no_category(worker: WorkerHandle) -> None:
    result = claude_run(FakeClaudeAPI([reply(stop_reason="refusal")]), worker)
    assert result.status == "error" and result.error and "category: none" in result.error


def test_a_refusal_carries_its_explanation(worker: WorkerHandle) -> None:
    details = {"type": "refusal", "category": "bio", "explanation": "not that"}
    result = claude_run(FakeClaudeAPI([reply(stop_reason="refusal", stop_details=details)]), worker)
    assert result.error and "bio" in result.error and "not that" in result.error


def test_a_response_that_cannot_continue_is_an_error(worker: WorkerHandle) -> None:
    api = FakeClaudeAPI([reply(text("..."), stop_reason="model_context_window_exceeded")])
    result = claude_run(api, worker)
    assert result.status == "error" and result.error
    assert "model_context_window_exceeded" in result.error


def test_pause_turn_uses_up_turns_like_any_request(worker: WorkerHandle) -> None:
    role = RoleSpec(name="r", prompt="p", tier=Tier.STANDARD, max_turns=2)
    api = FakeClaudeAPI(lambda _call: reply(text("..."), stop_reason="pause_turn"))
    result = claude_run(api, worker, role=role)
    assert result.halt is not None and result.halt.reason is HaltReason.MAX_TURNS
    assert len(api.calls) == 2


def test_the_last_turn_may_still_end_the_run(worker: WorkerHandle) -> None:
    role = RoleSpec(name="r", prompt="p", tier=Tier.STANDARD, tools=BOX_TOOLS, max_turns=2)
    api = FakeClaudeAPI([reply(bash("true")), reply(text("fine"))])
    assert claude_run(api, worker, role=role).status == "ok"


def test_a_nudge_uses_a_turn(worker: WorkerHandle) -> None:
    role = RoleSpec(name="r", prompt="p", tier=Tier.STANDARD, max_turns=2, output_schema=Verdict)
    api = FakeClaudeAPI(lambda _call: reply(text("done?")))
    result = claude_run(api, worker, role=role)
    assert result.halt is not None and result.halt.reason is HaltReason.MAX_TURNS
    assert len(api.calls) == 2 and MAX_NUDGES == 2


# --------------------------------------------------------------------------- submissions


def test_an_invalid_submission_lists_the_problems_without_the_input(worker: WorkerHandle) -> None:
    secret = "SECRET-VALUE-" + "x" * 50
    api = FakeClaudeAPI(
        [
            reply(submit({"ok": "maybe", "note": secret, "extra": 1}, id="s1")),
            reply(submit({"ok": True, "note": "n"}, id="s2")),
        ]
    )
    result = claude_run(api, worker, role=reporter())
    assert result.status == "ok"
    [(tid, failed, content)] = results(api.calls[1])
    assert (tid, failed) == ("s1", True)
    assert content.startswith("Error:") and "ok" in content and "extra" in content
    assert secret not in content


def test_a_problem_with_the_whole_input_has_no_field_to_name(worker: WorkerHandle) -> None:
    api = FakeClaudeAPI(
        [reply(submit({"a": 1, "b": 1}, id="s1")), reply(submit({"a": 1, "b": 2}, id="s2"))]
    )
    result = claude_run(api, worker, role=reporter(Ordered))
    assert result.status == "ok" and result.structured_output == {"a": 1, "b": 2}
    [(tid, failed, content)] = results(api.calls[1])
    assert (tid, failed) == ("s1", True) and "(the whole input): " in content


def test_many_problems_are_summarized(worker: WorkerHandle) -> None:
    api = FakeClaudeAPI([reply(submit({}, id="s1")), reply(text("giving up"))])
    claude_run(api, worker, role=reporter(Wide))
    [(_, failed, content)] = results(api.calls[1])
    assert failed and "and 3 more" in content
    assert content.count("\n- ") == claude_api.MAX_VALIDATION_ERRORS + 1  # + the summary line


def test_a_valid_submission_is_recorded_in_the_transcript(
    worker: WorkerHandle, tmp_path: Path
) -> None:
    api = FakeClaudeAPI(
        [reply(bash("echo ran", id="b1"), submit({"ok": True, "note": "n"}, id="s1"))]
    )
    result = claude_run(api, worker, role=reporter(), transcript_dir=tmp_path / "t")
    assert result.status == "ok"
    sent = [e for e in events(result.transcript_path) if e["event"] == "tool_results"]
    assert [r["tool_use_id"] for r in sent[0]["content"]] == ["b1", "s1"]


def test_an_undeclared_tool_name_is_clipped_in_the_record(worker: WorkerHandle) -> None:
    name = "weird_" * 100
    api = FakeClaudeAPI([reply(tool_use(name, {}, id="w1")), reply(text("ok"))])
    result = claude_run(api, worker)
    [denial] = result.permission_denials
    assert isinstance(denial["tool"], str) and len(denial["tool"]) <= claude_api.MAX_NAME_CHARS
    assert denial["tool_use_id"] == "w1"
    [(_, failed, content)] = results(api.calls[1])
    assert failed and len(content) < 300


def test_a_path_denial_from_the_box_tools_is_recorded_with_its_reason(
    worker: WorkerHandle,
) -> None:
    api = FakeClaudeAPI(
        [reply(edit(command="view", path="/etc/passwd", id="e1")), reply(text("ok"))]
    )
    result = claude_run(api, worker)
    [denial] = result.permission_denials
    assert denial["tool"] == EDITOR and denial["tool_use_id"] == "e1"
    assert denial["reason"] == results(api.calls[1])[0][2]


# --------------------------------------------------------------------------- metering


def test_a_response_without_iterations_is_metered_from_its_own_counts(
    worker: WorkerHandle,
) -> None:
    usage: dict[str, Any] = {
        "input_tokens": 100,
        "output_tokens": 50,
        "cache_read_input_tokens": None,
        "cache_creation_input_tokens": 20,
        "iterations": [],
    }
    result = claude_run(FakeClaudeAPI([reply(text("hi"), usage=usage)]), worker)
    assert (result.usage.input_tokens, result.usage.output_tokens) == (100, 50)
    assert (result.usage.cache_read_tokens, result.usage.cache_write_tokens) == (0, 20)
    expected = STANDARD.cost(input_tokens=100, output_tokens=50, cache_write_tokens=20)
    assert result.usage.cost_usd == pytest.approx(expected)


def test_long_context_rates_are_applied_per_iteration(worker: WorkerHandle, tmp_path: Path) -> None:
    """Two 60k-token iterations are two standard-rate requests, not one 120k long-context one."""
    haiku = DEFAULT_PRICING["claude-haiku-5-5"]
    assert haiku.long_context_over == 100_000
    usage = {
        "input_tokens": 60_000,
        "output_tokens": 10,
        "iterations": [
            {"type": "compaction", "input_tokens": 60_000, "output_tokens": 10},
            {"type": "message", "input_tokens": 60_000, "output_tokens": 10},
        ],
    }
    role = RoleSpec(name="r", prompt="p", tier=Tier.FAST)
    result = claude_run(FakeClaudeAPI([reply(text("hi"), usage=usage)]), worker, role=role)
    one = haiku.cost(input_tokens=60_000, output_tokens=10)
    assert result.usage.cost_usd == pytest.approx(2 * one)
    assert result.usage.cost_usd < haiku.cost(input_tokens=120_000, output_tokens=20)


# --------------------------------------------------------------------------- budgets


def test_reaching_the_budget_exactly_halts(worker: WorkerHandle) -> None:
    task = TASK.model_copy(update={"max_cost_usd": 0.2})
    usage = {"input_tokens": 100_000, "output_tokens": 0}  # exactly $0.20 on Sonnet 5.5
    result = claude_run(FakeClaudeAPI([reply(bash("true"), usage=usage)]), worker, task=task)
    assert result.halt is not None and result.halt.reason is HaltReason.BUDGET
    assert "$0.2000" in result.halt.detail


def test_a_response_that_ends_the_run_anyway_ends_it_at_the_budget(worker: WorkerHandle) -> None:
    """Over budget nothing more runs, but a refusal or a cut-off is the run's end all the same."""
    task = TASK.model_copy(update={"max_cost_usd": 0.05})
    usage = {"input_tokens": 100_000, "output_tokens": 0}
    refusal = reply(stop_reason="refusal", usage=usage)
    result = claude_run(FakeClaudeAPI([refusal]), worker, task=task)
    assert result.status == "error" and result.error and "refused" in result.error
    cut_off = reply(bash("echo x > x.txt"), stop_reason="max_tokens", usage=usage)
    result = claude_run(FakeClaudeAPI([cut_off]), worker, task=task)
    assert result.status == "error" and result.error and "max_tokens" in result.error
    assert not (worker.work_dir / "x.txt").exists()


def test_an_invalid_first_submission_at_the_budget_halts(worker: WorkerHandle) -> None:
    task = TASK.model_copy(update={"max_cost_usd": 0.05})
    usage = {"input_tokens": 100_000, "output_tokens": 0}
    invalid = reply(submit({"ok": "maybe", "note": "n"}), usage=usage)
    api = FakeClaudeAPI([invalid, reply(submit({"ok": True, "note": "n"}))])
    result = claude_run(api, worker, role=reporter(), task=task)
    assert result.halt is not None and result.halt.reason is HaltReason.BUDGET
    assert result.structured_output is None and len(api.calls) == 1


def test_a_submission_kept_at_the_budget_is_in_the_transcript(
    worker: WorkerHandle, tmp_path: Path
) -> None:
    task = TASK.model_copy(update={"max_cost_usd": 0.05})
    usage = {"input_tokens": 100_000, "output_tokens": 0}
    api = FakeClaudeAPI([reply(submit({"ok": True, "note": "n"}, id="s1"), usage=usage)])
    result = claude_run(api, worker, role=reporter(), task=task, transcript_dir=tmp_path / "logs")
    assert result.status == "ok" and result.structured_output == {"ok": True, "note": "n"}
    log = events(result.transcript_path)
    assert [e["event"] for e in log] == ["request", "response", "tool_results", "end"]
    assert log[-1]["status"] == "ok"


# --------------------------------------------------------------------------- transcript


def test_the_transcript_records_every_event_and_the_end(
    worker: WorkerHandle, tmp_path: Path
) -> None:
    api = FakeClaudeAPI(
        [reply(thinking("sig-1", "hm"), bash("echo hi", id="t1")), reply(text("ok"))]
    )
    result = claude_run(api, worker, transcript_dir=tmp_path / "deep" / "er")
    log = events(result.transcript_path)
    assert [(e["event"], e.get("turn")) for e in log] == [
        ("request", 1),
        ("response", 1),
        ("tool_results", 1),
        ("request", 2),
        ("response", 2),
        ("end", None),
    ]
    assert log[0]["model"] == "claude-sonnet-5-5"
    assert log[1]["message"]["content"][0]["signature"] == "sig-1"
    assert log[2]["content"][0]["tool_use_id"] == "t1"
    assert log[-1]["usage"]["input_tokens"] == 20 and "error" not in log[-1]
    assert "halt" not in log[-1]
    assert result.transcript_path is not None
    assert stat.S_IMODE(result.transcript_path.stat().st_mode) == 0o600


def test_a_halt_is_recorded_in_the_transcript_end(worker: WorkerHandle, tmp_path: Path) -> None:
    api = FakeClaudeAPI([api_error(429, "rate_limit_error", headers={"retry-after": "7"})])
    result = claude_run(api, worker, transcript_dir=tmp_path / "logs")
    [_request, end] = events(result.transcript_path)
    assert end["status"] == "halted"
    assert end["halt"]["reason"] == "rate_limit"
    assert end["halt"]["resume_at"].startswith("2026-01-01T00:00:07")


def test_every_run_gets_its_own_transcript(worker: WorkerHandle, tmp_path: Path) -> None:
    api = FakeClaudeAPI(lambda _call: reply(text("hi")))
    logs = tmp_path / "logs"
    first = claude_run(api, worker, transcript_dir=logs)
    second = claude_run(api, worker, transcript_dir=logs)
    assert first.session_id != second.session_id
    assert first.transcript_path != second.transcript_path
    assert sorted(p.name for p in logs.iterdir()) == sorted(
        [f"{first.session_id}.jsonl", f"{second.session_id}.jsonl"]
    )


def test_a_factory_failure_leaves_a_transcript_without_an_end(
    worker: WorkerHandle, tmp_path: Path
) -> None:
    class BoxDownError(Exception):
        pass

    class Broken:
        async def write_file(self, path: str, content: str) -> None:
            raise BoxDownError

        async def read_file(self, path: str) -> str:
            raise BoxDownError

        async def delete_file(self, path: str) -> None:
            raise BoxDownError

        async def run_command(self, argv: Sequence[str], *, timeout_s: float) -> CommandResult:
            raise BoxDownError

    broken = WorkerHandle(work_dir=worker.work_dir, executor=Broken())
    api = FakeClaudeAPI([reply(bash("ls")), reply(text("never"))])
    with pytest.raises(BoxDownError):
        claude_run(api, broken, transcript_dir=tmp_path / "logs")
    [path] = list((tmp_path / "logs").iterdir())
    kinds = [json.loads(line)["event"] for line in path.read_text().splitlines()]
    assert kinds == ["request", "response"]  # the crash is not dressed up as a result
