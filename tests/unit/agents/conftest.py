"""Shared fixtures and helpers for the agents unit tests."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from local_factory.agents.backend import AgentResult, AgentTask, RoleSpec, WorkerHandle
from local_factory.agents.fake import Action, FakeBackend
from local_factory.models import Tier

ROLE = RoleSpec(name="implementer", prompt="build it", tier=Tier.STANDARD)
TASK = AgentTask(ticket_id="T-0001", task_id="T-0001/task-1", instructions="do the thing")


def run(
    backend: FakeBackend, worker: WorkerHandle, role: RoleSpec = ROLE, task: AgentTask = TASK
) -> AgentResult:
    return asyncio.run(backend.run(role, task, worker))


def run_script(worker: WorkerHandle, *actions: Action, role: RoleSpec = ROLE) -> AgentResult:
    """Run `actions` as the script for `role`."""
    return run(FakeBackend({role.name: actions}), worker, role=role)


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
