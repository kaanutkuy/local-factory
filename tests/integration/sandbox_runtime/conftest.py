"""Fixtures for runtime container tests: a per-test DockerRuntime that must leave nothing behind.

Without Docker or the sandbox image these tests fail (never skip). Leftovers are checked with
the docker CLI directly, not through the runtime under test.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterator

import pytest

from local_factory.config import SandboxConfig
from local_factory.sandbox.runtime import DockerRuntime, SandboxError

IMAGE = "local-factory/sandbox:v1"


def leftovers(instance: str) -> tuple[list[str], list[str]]:
    """Containers and volumes labelled with ``instance``, asked of docker directly."""
    label = f"label=local-factory.instance={instance}"

    def listing(*args: str) -> list[str]:
        out = subprocess.run(
            ["docker", *args, "--filter", label], capture_output=True, text=True, check=True
        )
        return out.stdout.split()

    return listing("ps", "-aq"), listing("volume", "ls", "-q")


@pytest.fixture
def runtime() -> Iterator[DockerRuntime]:
    rt = DockerRuntime()
    try:
        rt.check(IMAGE)
    except SandboxError as exc:
        pytest.fail(f"container tests need Docker and {IMAGE} (scripts/build-images): {exc}")
    yield rt
    rt.cleanup()
    assert leftovers(rt.instance) == ([], []), "the runtime left objects behind"


@pytest.fixture
def config() -> SandboxConfig:
    return SandboxConfig(
        image=IMAGE,
        memory_mb=256,
        cpus=1.0,
        pids_limit=64,
        work_size_mb=64,
        tmp_size_mb=32,
        command_timeout_s=60.0,
    )
