"""Unit tests for the shared flow's backend limit (the rest of the flow is exercised end to end
by the host and sandboxed slice tests)."""

from __future__ import annotations

import asyncio

import pytest

from local_factory.agents import AgentResult
from local_factory.pipeline.flow import run_limited

OK = AgentResult(status="ok", text="done")


async def finishes() -> AgentResult:
    return OK


async def crashes() -> AgentResult:
    raise RuntimeError("the backend crashed")


def test_a_run_without_a_limit_just_runs() -> None:
    assert asyncio.run(run_limited(finishes(), None)) is OK


def test_a_run_within_the_limit_returns_its_result() -> None:
    assert asyncio.run(run_limited(finishes(), 30.0)) is OK


def test_a_run_over_the_limit_becomes_a_timeout_error_and_is_stopped() -> None:
    stopped: list[bool] = []

    async def never_ends() -> AgentResult:
        try:
            await asyncio.Event().wait()
        finally:
            stopped.append(True)
        raise AssertionError("a run that never ends cannot return")

    result = asyncio.run(run_limited(never_ends(), 0.05))

    assert result.status == "error"
    assert result.error == "timeout: the run exceeded 0.05s"
    assert stopped == [True]  # the run itself was cancelled, not just abandoned


def test_other_errors_are_not_swallowed() -> None:
    with pytest.raises(RuntimeError, match="the backend crashed"):
        asyncio.run(run_limited(crashes(), 30.0))
    with pytest.raises(RuntimeError, match="the backend crashed"):
        asyncio.run(run_limited(crashes(), None))
