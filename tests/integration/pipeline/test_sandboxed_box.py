"""Integration: what the sandboxed slice must never leave behind, with the real box.

Few and purposeful: ``tests/acceptance/test_sandboxed_slice.py`` and the red-team suite cover the
outcomes, and ``test_sandboxed_flow.py`` the flow. These need Docker and the sandbox image
(``scripts/build-images``) and fail, never skip, without them. Each test has a runtime of its own
whose containers and volumes must all be gone when it ends.
"""

from __future__ import annotations

import asyncio
import subprocess

import pytest

from local_factory.agents import AgentResult, AgentTask, FakeBackend, RoleSpec, WorkerHandle
from local_factory.sandbox import Sandbox
from local_factory.sandbox.runtime import DockerRuntime, SandboxError
from tests.acceptance.conftest import docker_objects, small_sandbox_config

from .conftest import IMPLEMENT_SHOUT, VERIFIER_PASS, World

pytestmark = [pytest.mark.container, pytest.mark.integration]

LINGER = ("sleep", "119")  # a command that outlives any test


class LingeringBackend(FakeBackend):
    """Leaves a file, starts a long command in the box, then crashes or waits to be cancelled."""

    def __init__(self, *, crash: bool) -> None:
        super().__init__({})
        self.crash = crash

    async def run(self, role: RoleSpec, task: AgentTask, worker: WorkerHandle) -> AgentResult:
        tools = worker.tools()
        await tools.write_file("scratch.txt", "half done")
        if self.crash:
            raise RuntimeError("the backend crashed")
        await tools.run_command(LINGER, timeout_s=120)
        raise AssertionError("the command outlives the test: it is only ever cancelled")


def lingering_command_runs(instance: str) -> bool:
    """Whether the instance has a live container running ``LINGER``, not just a workspace holder."""
    out = subprocess.run(
        [
            "docker",
            "ps",
            "--no-trunc",
            "--format",
            "{{.Command}}",
            "--filter",
            f"label=local-factory.instance={instance}",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return any(" ".join(LINGER) in line for line in out.stdout.splitlines())


def test_a_crashing_backend_leaves_no_container_or_volume(
    world: World, real_sandbox: Sandbox, docker_runtime: DockerRuntime
) -> None:
    with pytest.raises(RuntimeError, match="the backend crashed"):
        world.run(LingeringBackend(crash=True), sandbox=real_sandbox)

    assert docker_objects(docker_runtime.instance) == ([], [])
    assert world.work_root_oddities() == []


def test_a_cancelled_run_leaves_no_container_or_volume(
    world: World, real_sandbox: Sandbox, docker_runtime: DockerRuntime
) -> None:
    """Cancelled while its command is running in the box: the command's container goes, and so
    do the workspace's holder and volume."""

    async def scenario() -> None:
        running = asyncio.create_task(
            world.start(LingeringBackend(crash=False), sandbox=real_sandbox)
        )
        for _ in range(600):  # polled with the docker CLI off the event loop, until it is there
            if await asyncio.to_thread(lingering_command_runs, docker_runtime.instance):
                break
            assert not running.done(), "the slice ended before its command started"
        else:
            pytest.fail("the command never started in the box")
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running

    asyncio.run(scenario())

    assert docker_objects(docker_runtime.instance) == ([], [])
    assert world.work_root_oddities() == []


def test_a_missing_image_is_refused_before_any_change(
    world: World, docker_runtime: DockerRuntime
) -> None:
    sandbox = Sandbox(
        docker_runtime, small_sandbox_config(image="local-factory/sandbox:no-such-tag")
    )
    backend = FakeBackend({"implementer": IMPLEMENT_SHOUT, "verifier": VERIFIER_PASS})

    with pytest.raises(SandboxError, match="no-such-tag"):
        world.run(backend, sandbox=sandbox)

    assert len(world.ledger.events(world.ticket_id)) == world.initial_events
    assert backend.calls == []
    assert not world.work_root.exists()
    assert world.refs("refs/heads/factory/") == []
