"""Acceptance: ClaudeAPIBackend, the host-side model loop (docs/specs/phase-2a.md § claude).
Protected. Runs the real `anthropic` SDK against `FakeClaudeAPI` (no network) on a
`LocalExecutor` worker in `tmp_path`; the box variant is in test_claude_slice.py and the red-team.
"""

from __future__ import annotations

import asyncio
import json
import re
import stat
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import httpx2
import pytest
from anthropic import transform_schema
from pydantic import BaseModel, ConfigDict, Field

from local_factory.agents.backend import (
    AgentBackend,
    AgentResult,
    AgentTask,
    Capability,
    RoleSpec,
    WorkerHandle,
)
from local_factory.agents.box_tools import BASH, BOX_TOOLS, EDITOR
from local_factory.agents.claude_api import (
    COMPACTION_BETA,
    MAX_ERROR_CHARS,
    MAX_NUDGES,
    SUBMIT_TOOL,
    TASK_BUDGET_BETA,
    ClaudeAPIBackend,
)
from local_factory.agents.tools import CommandResult
from local_factory.config import DEFAULT_PRICING, ClaudeConfig, FactoryConfig
from local_factory.models import HaltReason, Tier

from .claude_api_fake import (
    Call,
    FakeClaudeAPI,
    Scripted,
    api_error,
    bash,
    compaction,
    edit,
    reply,
    result_text,
    submit,
    text,
    thinking,
    tool_use,
)

T0 = datetime(2026, 1, 1, tzinfo=UTC)
STANDARD = "claude-sonnet-5-5"
PRICE = DEFAULT_PRICING[STANDARD]
KEY = "sk-ant-acceptance-key-7f3c"
FORBIDDEN_PARAMS = ("temperature", "top_p", "top_k", "tool_choice", "fallbacks", "stop_sequences")

ROLE = RoleSpec(
    name="implementer",
    prompt="You are the implementer. Make the change the task describes.",
    tier=Tier.STANDARD,
    effort="high",
    tools=BOX_TOOLS,
    max_turns=8,
)
TASK = AgentTask(
    ticket_id="T-0001",
    task_id="T-0001/task-1",
    instructions="Add a script hello.py that prints hi.",
    context={"criteria": [{"id": "AC-1", "ears": "When run, hello.py shall print hi."}]},
)


class Report(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str
    files: list[str]
    verdict: Literal["done", "blocked"]
    ticket: str = Field(pattern=r"^T-[0-9]{4}$")


REPORTER = RoleSpec(
    name="reporter",
    prompt="You report on the work.",
    tier=Tier.STANDARD,
    effort="medium",
    tools=BOX_TOOLS,
    max_turns=8,
    output_schema=Report,
)
VALID_REPORT = {"summary": "s", "files": ["a.py"], "verdict": "done", "ticket": "T-0001"}


@pytest.fixture
def root(tmp_path: Path) -> Path:
    work = tmp_path / "work"
    work.mkdir()
    return work


def factory_config(tmp_path: Path, **claude: Any) -> FactoryConfig:
    return FactoryConfig(home=tmp_path / "home", claude=ClaudeConfig(**claude))


def go(
    api: FakeClaudeAPI,
    root: Path,
    *,
    role: RoleSpec = ROLE,
    task: AgentTask = TASK,
    config: FactoryConfig | None = None,
    transcript_dir: Path | None = None,
    worker: WorkerHandle | None = None,
) -> AgentResult:
    backend = ClaudeAPIBackend(
        config or factory_config(root.parent),
        client=api.client(api_key=KEY),
        transcript_dir=transcript_dir,
        clock=lambda: T0,
    )
    return asyncio.run(backend.run(role, task, worker or WorkerHandle(work_dir=root)))


def strip_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: strip_none(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [strip_none(v) for v in value]
    return value


def message_text(message: dict[str, Any]) -> str:
    content = message["content"]
    if isinstance(content, str):
        return content
    return "".join(block.get("text", "") for block in content if block.get("type") == "text")


def results_of(call: Call) -> list[tuple[str, bool, str]]:
    return [
        (r["tool_use_id"], bool(r.get("is_error")), result_text(r)) for r in call.tool_results()
    ]


def done(*blocks: dict[str, Any]) -> Scripted:
    return reply(*(blocks or (text("Done."),)))


# --------------------------------------------------------------------------- identity


def test_identity() -> None:
    backend = ClaudeAPIBackend(FactoryConfig(home=Path("/nonexistent")))  # no key needed yet
    assert isinstance(backend, AgentBackend)
    assert backend.name == "claude"
    assert Capability.STRUCTURED_OUTPUT in backend.capabilities
    assert SUBMIT_TOOL == "submit_result"
    assert (COMPACTION_BETA, TASK_BUDGET_BETA) == ("compact-2026-01-12", "task-budgets-2026-03-13")
    assert MAX_NUDGES == 2 and MAX_ERROR_CHARS == 4_000


def test_the_agents_package_never_imports_the_sdk_eagerly() -> None:
    code = (
        "import sys, local_factory.agents, local_factory.agents.local, "
        "local_factory.agents.box_tools; "
        "assert 'anthropic' not in sys.modules, 'anthropic imported'"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


# --------------------------------------------------------------------------- the request


def test_request_shape(root: Path, tmp_path: Path) -> None:
    api = FakeClaudeAPI([reply(bash("true")), done()])
    result = go(api, root)
    assert result.status == "ok", result
    first = api.calls[0]
    body = first.body
    assert body["model"] == STANDARD
    assert body["stream"] is True
    assert body["max_tokens"] == 64_000
    assert body["tools"] == [
        {"type": "bash_20250124", "name": "bash"},
        {
            "type": "text_editor_20250728",
            "name": "str_replace_based_edit_tool",
            "max_characters": 40_000,
        },
    ]
    assert body["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert body["output_config"] == {"effort": "high"}
    assert body["cache_control"] == {"type": "ephemeral"}
    for name in FORBIDDEN_PARAMS:
        assert name not in body, name

    system = body["system"]
    assert isinstance(system, list) and len(system) == 1
    assert system[0]["type"] == "text"
    assert system[0]["cache_control"] == {"type": "ephemeral"}
    assert system[0]["text"].startswith(ROLE.prompt)
    assert root.as_posix() in system[0]["text"]
    assert "network" in system[0]["text"]

    edits = body["context_management"]["edits"]
    assert len(edits) == 1
    assert edits[0]["type"] == "compact_20260112"
    assert edits[0]["trigger"] == {"type": "input_tokens", "value": 150_000}
    assert isinstance(edits[0]["instructions"], str) and edits[0]["instructions"]
    assert COMPACTION_BETA in first.betas

    assert len(body["messages"]) == 1
    opening = body["messages"][0]
    assert opening["role"] == "user"
    words = message_text(opening)
    assert TASK.instructions in words
    assert "Context (JSON):" in words
    assert json.dumps(TASK.context, indent=2, sort_keys=True) in words

    assert first.headers["authorization"] == f"Bearer {KEY}"
    second = api.calls[1].body
    assert second["system"] == body["system"]  # a stable prefix: prompt caching depends on it
    assert second["tools"] == body["tools"]


def test_compaction_off_and_a_task_budget(root: Path, tmp_path: Path) -> None:
    api = FakeClaudeAPI([done()])
    config = factory_config(tmp_path, compaction=False, task_budget_tokens=40_000)
    assert go(api, root, config=config).status == "ok"
    call = api.calls[0]
    assert "context_management" not in call.body
    assert COMPACTION_BETA not in call.betas
    assert TASK_BUDGET_BETA in call.betas
    assert call.body["output_config"] == {
        "effort": "high",
        "task_budget": {"type": "tokens", "total": 40_000},
    }


def test_no_betas_when_none_are_needed(root: Path, tmp_path: Path) -> None:
    api = FakeClaudeAPI([done()])
    assert go(api, root, config=factory_config(tmp_path, compaction=False)).status == "ok"
    assert api.calls[0].betas == []


def test_tools_follow_the_role(root: Path) -> None:
    api = FakeClaudeAPI([done(), done()])
    no_tools = RoleSpec(name="writer", prompt="Write.", tier=Tier.STANDARD)
    assert go(api, root, role=no_tools).status == "ok"
    assert "tools" not in api.calls[0].body
    only_bash = RoleSpec(name="runner", prompt="Run.", tier=Tier.FRONTIER, tools=(BASH,))
    assert go(api, root, role=only_bash).status == "ok"
    assert api.calls[1].body["tools"] == [{"type": "bash_20250124", "name": "bash"}]
    assert api.calls[1].body["model"] == "claude-opus-5-5"
    assert api.calls[1].body["output_config"] == {"effort": "high"}


@pytest.mark.parametrize("tools", [("python",), (BASH, "submit_result"), ("Bash",)])
def test_unknown_role_tools_fail_before_any_request(root: Path, tools: tuple[str, ...]) -> None:
    api = FakeClaudeAPI([done()])
    role = RoleSpec(name="r", prompt="p", tier=Tier.STANDARD, tools=tools)
    result = go(api, root, role=role)
    assert result.status == "error" and result.error
    assert api.calls == []


def test_unresolvable_or_unpriced_models_fail_before_any_request(
    root: Path, tmp_path: Path
) -> None:
    api = FakeClaudeAPI([done()])
    apex = RoleSpec(name="r", prompt="p", tier=Tier.APEX)
    assert go(api, root, role=apex).status == "error"  # apex is disabled by default
    unpriced = FactoryConfig(
        home=tmp_path / "home", tiers={"claude": {Tier.STANDARD: "claude-unpriced-9"}}
    )
    result = go(api, root, config=unpriced)
    assert result.status == "error" and result.error and "claude-unpriced-9" in result.error
    assert api.calls == []


# --------------------------------------------------------------------------- the loop


def test_tool_loop_acts_on_the_worker(root: Path) -> None:
    api = FakeClaudeAPI(
        [
            reply(
                thinking("sig-A"),
                text("Creating it."),
                edit(command="create", path="hello.py", file_text="print('hi')\n", id="t1"),
            ),
            reply(bash("python3 hello.py", id="t2")),
            reply(text("All done.")),
        ]
    )
    result = go(api, root)
    assert result.status == "ok", result
    assert result.text == "All done."
    assert result.structured_output is None
    assert (root / "hello.py").read_text() == "print('hi')\n"

    (tid, failed, content), *rest = results_of(api.calls[1])
    assert (tid, failed, rest) == ("t1", False, [])
    assert content == f"File created successfully at: {root}/hello.py"
    (tid, failed, content), *rest = results_of(api.calls[2])
    assert (tid, failed, rest) == ("t2", False, [])
    assert "hi" in content

    assert result.usage.input_tokens == 30 and result.usage.output_tokens == 15
    assert result.usage.cost_usd == pytest.approx(3 * PRICE.cost(input_tokens=10, output_tokens=5))
    assert result.halt is None and result.error is None


def test_assistant_turns_are_echoed_unchanged(root: Path) -> None:
    blocks = [
        compaction("<summary>so far</summary>"),
        thinking("sig-B", "summary of thoughts"),
        text("Looking."),
        bash("ls", id="e1"),
    ]
    api = FakeClaudeAPI([reply(*blocks), done()])
    assert go(api, root).status == "ok"
    messages = api.calls[1].messages
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert api.calls[1].messages[0] == api.calls[0].messages[0]
    assert strip_none(messages[1]["content"]) == blocks


def test_parallel_calls_run_in_order_and_answer_in_one_message(root: Path) -> None:
    api = FakeClaudeAPI(
        [
            reply(
                bash("echo one >> order.txt", id="p1"),
                bash("echo two >> order.txt", id="p2"),
                edit(command="view", path="missing.txt", id="p3"),
            ),
            done(),
        ]
    )
    assert go(api, root).status == "ok"
    assert (root / "order.txt").read_text() == "one\ntwo\n"
    second = api.calls[1]
    assert [m["role"] for m in second.messages] == ["user", "assistant", "user"]
    assert [(tid, failed) for tid, failed, _ in results_of(second)] == [
        ("p1", False),
        ("p2", False),
        ("p3", True),  # a ToolError is the model's to recover from, not the run's end
    ]


def test_undeclared_tools_are_refused_and_recorded(root: Path) -> None:
    only_bash = RoleSpec(name="runner", prompt="Run.", tier=Tier.STANDARD, tools=(BASH,))
    api = FakeClaudeAPI(
        [
            reply(
                edit(command="create", path="x.txt", file_text="x", id="u1"),
                tool_use("python", {"code": "open('/tmp/pwned', 'w')"}, id="u2"),
                bash("echo still-here", id="u3"),
            ),
            done(),
        ]
    )
    result = go(api, root, role=only_bash)
    assert result.status == "ok"
    assert not (root / "x.txt").exists()
    outcomes = results_of(api.calls[1])
    assert [(tid, failed) for tid, failed, _ in outcomes] == [
        ("u1", True),
        ("u2", True),
        ("u3", False),
    ]
    denied = [(d["tool"], d["tool_use_id"]) for d in result.permission_denials]
    assert denied == [(EDITOR, "u1"), ("python", "u2")]
    assert all(isinstance(d["reason"], str) and d["reason"] for d in result.permission_denials)


def test_path_escapes_are_denials_and_malformed_input_is_an_error(root: Path) -> None:
    api = FakeClaudeAPI(
        [
            reply(
                edit(command="create", path="/etc/x.txt", file_text="x", id="d1"),
                edit(command="view", path="../../x", id="d2"),
                tool_use(BASH, {"command": 5}, id="m1"),
                tool_use(EDITOR, {"command": "undo_edit", "path": "a"}, id="m2"),
            ),
            done(),
        ]
    )
    result = go(api, root)
    assert result.status == "ok"
    assert all(failed for _, failed, _ in results_of(api.calls[1]))
    assert [d["tool_use_id"] for d in result.permission_denials] == ["d1", "d2"]


# --------------------------------------------------------------------------- stop reasons


def test_refusal_is_an_error_with_its_category(root: Path) -> None:
    refusal = {"type": "refusal", "category": "cyber", "explanation": "declined"}
    api = FakeClaudeAPI([reply(stop_reason="refusal", stop_details=refusal)])
    result = go(api, root)
    assert result.status == "error" and result.error
    assert "refus" in result.error.lower() and "cyber" in result.error


def test_max_tokens_runs_none_of_the_truncated_calls(root: Path) -> None:
    api = FakeClaudeAPI(
        [
            reply(
                edit(command="create", path="y.txt", file_text="partial", id="x1"),
                stop_reason="max_tokens",
            )
        ]
    )
    result = go(api, root)
    assert result.status == "error" and result.error and "max_tokens" in result.error
    assert not (root / "y.txt").exists()
    assert len(api.calls) == 1


def test_pause_turn_is_resumed_as_is(root: Path) -> None:
    api = FakeClaudeAPI([reply(text("thinking..."), stop_reason="pause_turn"), done()])
    assert go(api, root).status == "ok"
    resumed = api.calls[1].messages
    assert [m["role"] for m in resumed] == ["user", "assistant"]


@pytest.mark.parametrize(
    "scripted",
    [reply(text("?"), stop_reason="something_new"), reply(text("?"), stop_reason="tool_use")],
)
def test_unexpected_stops_are_errors(root: Path, scripted: Scripted) -> None:
    result = go(FakeClaudeAPI([scripted]), root)
    assert result.status == "error" and result.error


def test_max_turns_halts(root: Path) -> None:
    role = RoleSpec(name="r", prompt="p", tier=Tier.STANDARD, tools=BOX_TOOLS, max_turns=3)
    api = FakeClaudeAPI(lambda _call: reply(bash("true")))
    result = go(api, root, role=role)
    assert result.status == "halted"
    assert result.halt is not None and result.halt.reason is HaltReason.MAX_TURNS
    assert len(api.calls) == 3
    assert result.usage.input_tokens == 30


# --------------------------------------------------------------------------- structured output


def test_submit_result_tool_and_output(root: Path) -> None:
    api = FakeClaudeAPI([reply(text("Reporting."), submit(VALID_REPORT, id="s1"))])
    result = go(api, root, role=REPORTER)
    assert result.status == "ok", result
    assert result.structured_output == VALID_REPORT
    assert len(api.calls) == 1
    tools = api.calls[0].body["tools"]
    assert [t["name"] for t in tools] == [BASH, EDITOR, SUBMIT_TOOL]
    submit_tool = tools[-1]
    assert submit_tool["strict"] is True
    assert submit_tool["input_schema"] == transform_schema(Report)
    assert isinstance(submit_tool["description"], str) and submit_tool["description"]
    assert "type" not in submit_tool
    assert SUBMIT_TOOL in api.calls[0].system_text


def test_an_invalid_submission_is_returned_for_correction(root: Path) -> None:
    bad = dict(VALID_REPORT, ticket="nope")
    api = FakeClaudeAPI([reply(submit(bad, id="s1")), reply(submit(VALID_REPORT, id="s2"))])
    result = go(api, root, role=REPORTER)
    assert result.status == "ok"
    assert result.structured_output == VALID_REPORT
    [(tid, failed, content)] = results_of(api.calls[1])
    assert (tid, failed) == ("s1", True)
    assert "ticket" in content


def test_end_turn_without_a_submission_is_nudged_then_fails(root: Path) -> None:
    api = FakeClaudeAPI(lambda _call: reply(text("I think I am done.")))
    result = go(api, root, role=REPORTER)
    assert result.status == "error" and result.error
    assert len(api.calls) == 1 + MAX_NUDGES
    nudge = api.calls[1].messages[-1]
    assert nudge["role"] == "user" and SUBMIT_TOOL in message_text(nudge)


def test_a_nudge_can_succeed(root: Path) -> None:
    api = FakeClaudeAPI([reply(text("Done.")), reply(submit(VALID_REPORT))])
    result = go(api, root, role=REPORTER)
    assert result.status == "ok" and result.structured_output == VALID_REPORT


def test_calls_after_a_valid_submission_do_not_run(root: Path) -> None:
    api = FakeClaudeAPI(
        [
            reply(
                bash("echo a > a.txt"),
                submit(VALID_REPORT),
                bash("echo b > b.txt"),
            )
        ]
    )
    result = go(api, root, role=REPORTER)
    assert result.status == "ok"
    assert (root / "a.txt").exists() and not (root / "b.txt").exists()
    assert len(api.calls) == 1


def test_submit_result_is_undeclared_without_a_schema(root: Path) -> None:
    api = FakeClaudeAPI([reply(submit(VALID_REPORT, id="s1")), done()])
    result = go(api, root)
    assert result.status == "ok" and result.structured_output is None
    [(tid, failed, _)] = results_of(api.calls[1])
    assert (tid, failed) == ("s1", True)
    assert [d["tool"] for d in result.permission_denials] == [SUBMIT_TOOL]


# --------------------------------------------------------------------------- usage and budget


def test_usage_is_metered_over_iterations(root: Path) -> None:
    first = {
        "input_tokens": 1_000,
        "output_tokens": 500,
        "cache_read_input_tokens": 2_000,
        "cache_creation_input_tokens": 400,
    }
    compacted = {
        "input_tokens": 23_000,
        "output_tokens": 1_000,
        "iterations": [
            {"type": "compaction", "input_tokens": 180_000, "output_tokens": 3_500},
            {"type": "message", "input_tokens": 23_000, "output_tokens": 1_000},
        ],
    }
    api = FakeClaudeAPI([reply(bash("true"), usage=first), reply(text("ok"), usage=compacted)])
    result = go(api, root)
    assert result.status == "ok"
    usage = result.usage
    assert usage.input_tokens == 1_000 + 180_000 + 23_000
    assert usage.output_tokens == 500 + 3_500 + 1_000
    assert (usage.cache_read_tokens, usage.cache_write_tokens) == (2_000, 400)
    expected = (1_000 * 2.0 + 500 * 10.0 + 2_000 * 0.10 + 400 * 2.50) / 1e6
    expected += (180_000 * 2.0 + 3_500 * 10.0) / 1e6 + (23_000 * 2.0 + 1_000 * 10.0) / 1e6
    assert usage.cost_usd == pytest.approx(expected)


def test_the_task_budget_halts_before_more_tool_calls(root: Path) -> None:
    task = TASK.model_copy(update={"max_cost_usd": 0.05})
    expensive = {"input_tokens": 100_000, "output_tokens": 0}  # $0.20 on Sonnet 5.5
    api = FakeClaudeAPI(
        [reply(edit(command="create", path="z.txt", file_text="z"), usage=expensive), done()]
    )
    result = go(api, root, task=task)
    assert result.status == "halted"
    assert result.halt is not None and result.halt.reason is HaltReason.BUDGET
    assert not (root / "z.txt").exists()
    assert len(api.calls) == 1
    assert result.usage.cost_usd == pytest.approx(0.2)


def test_a_finished_answer_that_reaches_the_budget_is_kept(root: Path) -> None:
    task = TASK.model_copy(update={"max_cost_usd": 0.05})
    expensive = {"input_tokens": 100_000, "output_tokens": 0}  # $0.20 on Sonnet 5.5
    result = go(FakeClaudeAPI([reply(text("The answer."), usage=expensive)]), root, task=task)
    assert result.status == "ok", result
    assert result.text == "The answer."
    api = FakeClaudeAPI(
        [reply(submit(VALID_REPORT), bash("echo late > late.txt"), usage=expensive)]
    )
    result = go(api, root, role=REPORTER, task=task)
    assert result.status == "ok", result
    assert result.structured_output == VALID_REPORT
    assert not (root / "late.txt").exists()


def test_work_still_to_do_at_the_budget_halts(root: Path) -> None:
    task = TASK.model_copy(update={"max_cost_usd": 0.05})
    expensive = {"input_tokens": 100_000, "output_tokens": 0}
    for scripted, role in (
        (reply(text("Done, I think."), usage=expensive), REPORTER),  # would need a nudge
        (reply(text("..."), stop_reason="pause_turn", usage=expensive), ROLE),
        (reply(bash("echo x > x.txt"), submit(VALID_REPORT), usage=expensive), REPORTER),
    ):
        api = FakeClaudeAPI([scripted, done()])
        result = go(api, root, role=role, task=task)
        assert result.halt is not None and result.halt.reason is HaltReason.BUDGET, result
        assert len(api.calls) == 1
    assert not (root / "x.txt").exists()


def test_the_configured_run_budget_applies_when_the_task_has_none(
    root: Path, tmp_path: Path
) -> None:
    api = FakeClaudeAPI(lambda _call: reply(bash("true"), usage={"input_tokens": 50_000}))
    result = go(api, root, config=factory_config(tmp_path, max_run_cost_usd=0.25))
    assert result.halt is not None and result.halt.reason is HaltReason.BUDGET
    assert len(api.calls) == 3  # $0.10 per response: halts once $0.25 is reached


def test_the_task_budget_wins_over_the_configured_one(root: Path, tmp_path: Path) -> None:
    task = TASK.model_copy(update={"max_cost_usd": 10.0})
    api = FakeClaudeAPI(
        [reply(bash("true"), usage={"input_tokens": 50_000, "output_tokens": 0}), done()]
    )
    result = go(api, root, task=task, config=factory_config(tmp_path, max_run_cost_usd=0.05))
    assert result.status == "ok"


# --------------------------------------------------------------------------- API errors


@pytest.mark.parametrize(
    ("scripted", "reason"),
    [
        (api_error(429, "rate_limit_error", headers={"retry-after": "30"}), HaltReason.RATE_LIMIT),
        (api_error(402, "billing_error"), HaltReason.BILLING),
        (api_error(401, "authentication_error"), HaltReason.AUTH),
        (api_error(403, "permission_error"), HaltReason.AUTH),
        (api_error(529, "overloaded_error"), HaltReason.UNAVAILABLE),
        (api_error(500, "api_error"), HaltReason.UNAVAILABLE),
        (api_error(503, "api_error"), HaltReason.UNAVAILABLE),
        (httpx2.ConnectError("connection refused"), HaltReason.UNAVAILABLE),
        (httpx2.ReadTimeout("read timed out"), HaltReason.UNAVAILABLE),
    ],
)
def test_api_failures_halt_with_the_usage_so_far(
    root: Path, scripted: Scripted, reason: HaltReason
) -> None:
    api = FakeClaudeAPI([reply(bash("true")), scripted])
    result = go(api, root)
    assert result.status == "halted", result
    assert result.halt is not None and result.halt.reason is reason
    assert result.usage.input_tokens == 10  # the first response is still metered


def test_rate_limit_resume_time(root: Path) -> None:
    def halt_for(headers: dict[str, str]) -> datetime | None:
        result = go(FakeClaudeAPI([api_error(429, "rate_limit_error", headers=headers)]), root)
        assert result.halt is not None
        return result.halt.resume_at

    assert halt_for({"retry-after": "30"}) == T0 + timedelta(seconds=30)
    assert halt_for({"retry-after": "2.5"}) == T0 + timedelta(seconds=2.5)
    assert halt_for({}) is None
    assert halt_for({"retry-after": "soon"}) is None


@pytest.mark.parametrize(
    "scripted",
    [
        api_error(400, "invalid_request_error", "bad request"),
        api_error(404, "not_found_error", "model: claude-nope"),
        api_error(413, "request_too_large", "too big"),
        api_error(422, "invalid_request_error", "unprocessable"),
    ],
)
def test_other_api_errors_are_run_errors(root: Path, scripted: Scripted) -> None:
    result = go(FakeClaudeAPI([scripted]), root)
    assert result.status == "error" and result.error
    assert str(getattr(scripted, "status", "")) in result.error


def test_error_text_is_bounded(root: Path) -> None:
    result = go(FakeClaudeAPI([api_error(400, "invalid_request_error", "x" * 50_000)]), root)
    assert result.status == "error" and result.error
    assert len(result.error) <= MAX_ERROR_CHARS


def test_factory_failures_propagate(tmp_path: Path) -> None:
    class BoxDownError(Exception):
        pass

    class Broken:
        async def write_file(self, path: str, content: str) -> None:
            raise BoxDownError("docker daemon is gone")

        async def read_file(self, path: str) -> str:
            raise BoxDownError("docker daemon is gone")

        async def delete_file(self, path: str) -> None:
            raise BoxDownError("docker daemon is gone")

        async def run_command(self, argv: Sequence[str], *, timeout_s: float) -> CommandResult:
            raise BoxDownError("docker daemon is gone")

    api = FakeClaudeAPI([reply(bash("ls")), done()])
    worker = WorkerHandle(work_dir=Path("/work"), executor=Broken())
    with pytest.raises(BoxDownError):
        go(api, tmp_path, worker=worker)


# --------------------------------------------------------------------------- transcript, key


def test_transcript(root: Path, tmp_path: Path) -> None:
    transcripts = tmp_path / "transcripts"
    api = FakeClaudeAPI([reply(bash("echo hi", id="t1")), reply(text("Bye."))])
    result = go(api, root, transcript_dir=transcripts)
    assert result.status == "ok"
    assert result.session_id is not None
    assert re.fullmatch(r"claude-[0-9a-f]{32}", result.session_id)
    path = result.transcript_path
    assert path == transcripts / f"{result.session_id}.jsonl"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    raw = path.read_text()
    assert KEY not in raw and "authorization" not in raw.lower()
    events = [json.loads(line) for line in raw.splitlines()]
    kinds = [e["event"] for e in events]
    assert kinds[0] == "request" and kinds[-1] == "end"
    assert {"request", "response", "tool_results", "end"} <= set(kinds)
    assert events[-1]["status"] == "ok"


def test_no_transcript_by_default(root: Path) -> None:
    result = go(FakeClaudeAPI([done()]), root)
    assert result.transcript_path is None


def test_the_key_never_reaches_a_local_worker(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_key = "sk-ant-env-canary-91b2"
    monkeypatch.setenv("ANTHROPIC_API_KEY", env_key)
    api = FakeClaudeAPI([reply(bash("env; cat /proc/self/environ | tr '\\0' '\\n'")), done()])
    assert go(api, root).status == "ok"
    for _, _, content in results_of(api.calls[1]):
        assert KEY not in content and env_key not in content
