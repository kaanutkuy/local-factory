"""Acceptance: the Phase 1b slice driven by ClaudeAPIBackend (docs/specs/phase-2a.md § Exit check).
Protected. Needs Docker and the sandbox image. The model is `FakeClaudeAPI`: $0, no network; the
backend, the SDK and every tool call through the box are real.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from local_factory.agents.claude_api import ClaudeAPIBackend
from local_factory.config import ClaudeConfig, FactoryConfig
from local_factory.pipeline.flow import IMPLEMENTER_ROLE, VERIFIER_ROLE
from local_factory.pipeline.sandboxed_slice import run_sandboxed_slice

from .claude_api_fake import (
    Call,
    FakeClaudeAPI,
    Scripted,
    api_error,
    bash,
    edit,
    reply,
    result_text,
    submit,
    text,
)
from .conftest import T0, docker_objects
from .test_host_slice import SHOUT_GREET, SHOUT_TEST
from .test_sandboxed_slice import World, only_factory_bundles, world  # noqa: F401 - a fixture

pytestmark = [pytest.mark.container, pytest.mark.integration]

USAGE = {
    "input_tokens": 1_200,
    "output_tokens": 300,
    "cache_read_input_tokens": 800,
    "cache_creation_input_tokens": 100,
}
PASS_REPORT = {"criteria": [{"id": "AC-1", "status": "pass", "evidence": "HELLO, ADA!"}]}


def turn(call: Call) -> int:
    return sum(1 for m in call.messages if m["role"] == "assistant")


def implementer(step: int) -> Scripted:
    script: list[Scripted] = [
        reply(edit(command="view", path="/work"), usage=USAGE),
        reply(
            text("Adding --shout and a test."),
            edit(command="create", path="/work/greet.py", file_text=SHOUT_GREET),
            edit(command="create", path="tests/test_shout.py", file_text=SHOUT_TEST),
            usage=USAGE,
        ),
        reply(bash("python3 -m unittest discover -s tests -q"), usage=USAGE),
        reply(text("Implemented --shout with a test."), usage=USAGE),
    ]
    return script[step]


def verifier(step: int) -> Scripted:
    script: list[Scripted] = [
        reply(bash("python3 greet.py --name Ada --shout"), usage=USAGE),
        reply(submit(PASS_REPORT), usage=USAGE),
    ]
    return script[step]


def model(call: Call) -> Scripted:
    if call.system_text.startswith(IMPLEMENTER_ROLE.prompt):
        return implementer(turn(call))
    if call.system_text.startswith(VERIFIER_ROLE.prompt):
        return verifier(turn(call))
    return api_error(400, "invalid_request_error", "fake: unknown role")


@pytest.fixture
def ticket(world: World) -> World:  # noqa: F811 - the imported pytest fixture
    return world


def run(w: World, api: FakeClaudeAPI, tmp_path: Path, **claude: Any) -> Any:
    config = FactoryConfig(home=tmp_path / "home", claude=ClaudeConfig(**claude))
    backend = ClaudeAPIBackend(
        config, client=api.client(), transcript_dir=tmp_path / "transcripts", clock=lambda: T0
    )
    return asyncio.run(
        run_sandboxed_slice(
            ledger=w.ledger,
            repo=w.repo,
            project=w.project,
            backend=backend,
            sandbox=w.sandbox(),
            ticket_id=w.ticket_id,
            work_root=w.work_root,
            key=w.key,
            clock=lambda: T0,
            factory_config=config,
        )
    )


def test_the_roles_declare_the_box_tools() -> None:
    expected = ("bash", "str_replace_based_edit_tool")
    assert IMPLEMENTER_ROLE.tools == expected
    assert VERIFIER_ROLE.tools == expected


def test_a_model_driven_ticket_in_the_box(ticket: World, tmp_path: Path) -> None:
    api = FakeClaudeAPI(model)
    outcome = run(ticket, api, tmp_path)

    assert outcome.status == "candidate_verified", outcome
    assert outcome.candidate_sha is not None
    assert b"--shout" in ticket.repo.read_file(outcome.candidate_sha, "greet.py")
    assert ticket.repo.read_file(outcome.candidate_sha, "tests/test_shout.py") == (
        SHOUT_TEST.encode()
    )

    implementer_calls = [c for c in api.calls if c.system_text.startswith(IMPLEMENTER_ROLE.prompt)]
    assert len(implementer_calls) == 4
    assert "/work" in implementer_calls[0].system_text
    listing = result_text(implementer_calls[1].tool_results()[0])
    assert "/work/greet.py" in listing and "/work/.git" not in listing
    unit = implementer_calls[3].tool_results()[0]
    assert not unit.get("is_error")
    assert "OK" in result_text(unit)  # the target's tests really ran, in the box
    verifier_calls = [c for c in api.calls if c.system_text.startswith(VERIFIER_ROLE.prompt)]
    assert len(verifier_calls) == 4  # two rounds: the task head, then the candidate
    assert "HELLO, ADA!" in result_text(verifier_calls[1].tool_results()[0])
    assert verifier_calls[0].body["model"] == "claude-opus-5-5"
    assert implementer_calls[0].body["model"] == "claude-sonnet-5-5"

    runs = ticket.ledger.runs(ticket.ticket_id)
    assert [(r.role, r.backend, r.model, r.status) for r in runs] == [
        ("implementer", "claude", "claude-sonnet-5-5", "ok"),
        ("verifier", "claude", "claude-opus-5-5", "ok"),
        ("verifier", "claude", "claude-opus-5-5", "ok"),
    ]
    assert runs[0].usage.input_tokens == 4 * 1_200
    assert runs[0].usage.cache_read_tokens == 4 * 800
    assert all(r.usage.cost_usd > 0 for r in runs)
    total = sum(r.usage.cost_usd for r in runs)
    assert ticket.ledger.ticket_cost_usd(ticket.ticket_id) == pytest.approx(total)
    for record in runs:
        assert record.transcript_path is not None and record.transcript_path.is_file()
        events = [json.loads(line) for line in record.transcript_path.read_text().splitlines()]
        assert events[-1]["event"] == "end"

    evidence = outcome.evidence
    assert evidence is not None and evidence.passed
    assert evidence.models == {"implementer": "claude-sonnet-5-5", "verifier": "claude-opus-5-5"}
    assert only_factory_bundles(ticket.work_root) == []
    assert docker_objects(ticket.runtime.instance) == ([], [])


def test_a_rate_limited_implementer_halts_the_ticket(ticket: World, tmp_path: Path) -> None:
    api = FakeClaudeAPI(
        lambda _call: api_error(429, "rate_limit_error", headers={"retry-after": "60"})
    )
    outcome = run(ticket, api, tmp_path)
    assert outcome.status == "halted", outcome
    assert outcome.task.status.value == "halted"
    assert docker_objects(ticket.runtime.instance) == ([], [])


def test_a_budget_halt_leaves_nothing_behind(ticket: World, tmp_path: Path) -> None:
    api = FakeClaudeAPI(model)
    outcome = run(ticket, api, tmp_path, max_run_cost_usd=0.001)
    assert outcome.status == "halted", outcome
    assert len(api.calls) == 1
    assert docker_objects(ticket.runtime.instance) == ([], [])
