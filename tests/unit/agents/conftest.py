"""Shared fixtures and helpers for the agents unit tests."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest
from anthropic import AsyncAnthropic

from local_factory.agents.backend import AgentResult, AgentTask, RoleSpec, WorkerHandle
from local_factory.agents.box_tools import BOX_TOOLS
from local_factory.agents.claude_api import ClaudeAPIBackend
from local_factory.agents.fake import Action, FakeBackend
from local_factory.config import ClaudeConfig, FactoryConfig
from local_factory.models import Tier
from tests.acceptance.claude_api_fake import FakeClaudeAPI

ROLE = RoleSpec(name="implementer", prompt="build it", tier=Tier.STANDARD)
TASK = AgentTask(ticket_id="T-0001", task_id="T-0001/task-1", instructions="do the thing")

# ClaudeAPIBackend: a role with both box tools, a fixed clock, and the in-process fake API.
CLAUDE_ROLE = RoleSpec(
    name="implementer", prompt="Implement.", tier=Tier.STANDARD, tools=BOX_TOOLS, max_turns=8
)
T0 = datetime(2026, 1, 1, tzinfo=UTC)


def run(
    backend: FakeBackend, worker: WorkerHandle, role: RoleSpec = ROLE, task: AgentTask = TASK
) -> AgentResult:
    return asyncio.run(backend.run(role, task, worker))


def run_script(worker: WorkerHandle, *actions: Action, role: RoleSpec = ROLE) -> AgentResult:
    """Run `actions` as the script for `role`."""
    return run(FakeBackend({role.name: actions}), worker, role=role)


def claude_config(home: Path, **claude: Any) -> FactoryConfig:
    return FactoryConfig(home=home / "factory-home", claude=ClaudeConfig(**claude))


def claude_run(
    api: FakeClaudeAPI,
    worker: WorkerHandle,
    *,
    role: RoleSpec = CLAUDE_ROLE,
    task: AgentTask = TASK,
    config: FactoryConfig | None = None,
    transcript_dir: Path | None = None,
    client: AsyncAnthropic | None = None,
) -> AgentResult:
    """One ClaudeAPIBackend run against `api` (a fake: no network) on `worker`."""
    backend = ClaudeAPIBackend(
        config or claude_config(worker.work_dir.parent),
        client=client or api.client(),
        transcript_dir=transcript_dir,
        clock=lambda: T0,
    )
    return asyncio.run(backend.run(role, task, worker))


class FailingClient:
    """Stands in for an `AsyncAnthropic` whose every request fails with `error`."""

    def __init__(self, error: BaseException) -> None:
        self.error = error
        self.beta = self
        self.messages = self

    def stream(self, **_params: object) -> object:
        raise self.error

    def as_client(self) -> AsyncAnthropic:
        return cast(AsyncAnthropic, self)


@pytest.fixture
def worker(tmp_path: Path) -> WorkerHandle:
    work = tmp_path / "work"
    work.mkdir()
    return WorkerHandle(work_dir=work)


@pytest.fixture
def outside(tmp_path: Path) -> Path:
    """A directory next to the worker that scripts must never touch."""
    path = tmp_path / "outside"
    path.mkdir()
    return path
