"""Unit tests for how ClaudeAPIBackend turns API failures into results (docs/specs/phase-2a.md §
API errors), beyond the cases in the acceptance oracle."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from datetime import timedelta
from pathlib import Path

import anthropic
import httpx2
import pytest

from local_factory.agents.backend import WorkerHandle
from local_factory.agents.claude_api import MAX_ERROR_CHARS
from local_factory.models import HaltReason
from tests.acceptance.claude_api_fake import (
    BASE_URL,
    FakeClaudeAPI,
    Scripted,
    api_error,
    bash,
    reply,
    text,
)

from .conftest import T0, FailingClient, claude_run


def test_a_missing_key_halts_with_auth_and_makes_no_request(
    worker: WorkerHandle, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The SDK raises a bare TypeError when no key or token is configured at all."""
    for name in [n for n in os.environ if n.startswith("ANTHROPIC_")]:
        monkeypatch.delenv(name)
    monkeypatch.setenv("HOME", str(tmp_path))  # no ~/.config/anthropic credentials file either
    api = FakeClaudeAPI([reply(text("hi"))])
    keyless = anthropic.AsyncAnthropic(
        base_url=BASE_URL,
        max_retries=0,
        http_client=anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(api.handle)),
    )
    result = claude_run(api, worker, client=keyless)
    assert result.status == "halted"
    assert result.halt is not None and result.halt.reason is HaltReason.AUTH
    assert api.calls == []


def test_another_type_error_is_a_bug_and_propagates(worker: WorkerHandle) -> None:
    failing = FailingClient(TypeError("unsupported operand"))
    with pytest.raises(TypeError, match="unsupported operand"):
        claude_run(FakeClaudeAPI([]), worker, client=failing.as_client())


def test_credential_errors_halt_with_auth(worker: WorkerHandle) -> None:
    failing = FailingClient(anthropic.CredentialsError("profile file is unreadable"))
    result = claude_run(FakeClaudeAPI([]), worker, client=failing.as_client())
    assert result.halt is not None and result.halt.reason is HaltReason.AUTH
    assert "profile file is unreadable" in result.halt.detail


def test_other_sdk_errors_are_run_errors(worker: WorkerHandle) -> None:
    failing = FailingClient(anthropic.AnthropicError("something the SDK did not expect"))
    result = claude_run(FakeClaudeAPI([]), worker, client=failing.as_client())
    assert result.status == "error" and result.error
    assert result.error.startswith("Claude API error: ") and "did not expect" in result.error


def test_a_billing_error_halts_whatever_its_status(worker: WorkerHandle) -> None:
    result = claude_run(FakeClaudeAPI([api_error(400, "billing_error")]), worker)
    assert result.halt is not None and result.halt.reason is HaltReason.BILLING


@pytest.mark.parametrize(
    "scripted",
    [
        api_error(429, "rate_limit_error", headers={"retry-after": "9" * 400}),  # not finite
        api_error(429, "rate_limit_error", headers={"retry-after": "9" * 20}),  # past year 9999
        api_error(429, "rate_limit_error", headers={"retry-after": "-5"}),
        api_error(429, "rate_limit_error", headers={"retry-after": "inf"}),
        api_error(429, "rate_limit_error", headers={"retry-after": "nan"}),
        api_error(429, "rate_limit_error", headers={"retry-after": "1_0"}),
        api_error(
            429, "rate_limit_error", headers={"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}
        ),
        api_error(429, "rate_limit_error", headers={"retry-after": ""}),
    ],
)
def test_a_retry_after_that_is_not_seconds_gives_no_resume_time(
    worker: WorkerHandle, scripted: Scripted
) -> None:
    result = claude_run(FakeClaudeAPI([scripted]), worker)
    assert result.halt is not None and result.halt.reason is HaltReason.RATE_LIMIT
    assert result.halt.resume_at is None


def test_a_zero_retry_after_resumes_at_once(worker: WorkerHandle) -> None:
    api = FakeClaudeAPI([api_error(429, "rate_limit_error", headers={"retry-after": "0"})])
    result = claude_run(api, worker)
    assert result.halt is not None and result.halt.resume_at == T0
    assert result.halt.resume_at - T0 == timedelta(0)


def test_halt_details_are_bounded(worker: WorkerHandle) -> None:
    api = FakeClaudeAPI([api_error(529, "overloaded_error", "busy " * 20_000)])
    result = claude_run(api, worker)
    assert result.halt is not None and result.halt.reason is HaltReason.UNAVAILABLE
    assert 0 < len(result.halt.detail) <= MAX_ERROR_CHARS


def test_a_failure_after_work_reports_the_usage_so_far(worker: WorkerHandle) -> None:
    api = FakeClaudeAPI([reply(bash("true")), api_error(400, "invalid_request_error", "bad")])
    result = claude_run(api, worker)
    assert result.status == "error" and result.error and "400" in result.error
    assert result.usage.input_tokens == 10 and result.usage.cost_usd > 0


class Dies(httpx2.AsyncByteStream):
    """A response body that starts well and then fails, as a dropped connection does."""

    def __init__(self, error: Exception) -> None:
        self.error = error

    async def __aiter__(self) -> AsyncIterator[bytes]:
        start = {
            "type": "message_start",
            "message": {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "claude-sonnet-5-5",
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        }
        yield f"event: message_start\ndata: {json.dumps(start)}\n\n".encode()
        raise self.error


@pytest.mark.parametrize(
    "error",
    [httpx2.ReadError("connection reset"), httpx2.ReadTimeout("no data for 900s")],
)
def test_a_connection_lost_mid_stream_halts_as_unavailable(
    worker: WorkerHandle, error: Exception
) -> None:
    def handler(_request: httpx2.Request) -> httpx2.Response:
        stream = Dies(error)
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)

    client = anthropic.AsyncAnthropic(
        api_key="sk-ant-test",
        base_url=BASE_URL,
        max_retries=0,
        http_client=anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(handler)),
    )
    result = claude_run(FakeClaudeAPI([]), worker, client=client)
    assert result.halt is not None and result.halt.reason is HaltReason.UNAVAILABLE
    assert type(error).__name__ in result.halt.detail
