"""A scripted stand-in for DockerRuntime: records every container spec and returns canned results.

The host-side box code (executor, facade) only builds specs and interprets results; these unit
tests check both without Docker. The real thing is exercised by the container tests.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, cast

import pytest

from local_factory.config import SandboxConfig
from local_factory.sandbox.runtime import ContainerResult, DockerRuntime
from local_factory.sandbox.spec import ContainerSpec, VolumeMount
from local_factory.sandbox.workspace import TaskWorkspace


def result(
    exit_code: int | None = 0,
    *,
    stdout: bytes = b"",
    stderr: str = "",
    timed_out: bool = False,
    oom_killed: bool = False,
    output_exceeded: bool = False,
    duration_s: float = 0.25,
) -> ContainerResult:
    return ContainerResult(
        exit_code=exit_code,
        timed_out=timed_out,
        oom_killed=oom_killed,
        output_exceeded=output_exceeded,
        stdout=stdout,
        stderr=stderr,
        duration_s=duration_s,
    )


@dataclass
class Run:
    spec: ContainerSpec
    kwargs: dict[str, Any]


@dataclass
class Step:
    """What the fake runtime does for one run: write ``sink`` to the stdout sink, then return or
    raise."""

    outcome: ContainerResult | BaseException
    sink: bytes = b""


@dataclass
class FakeRuntime:
    steps: list[Step] = field(default_factory=list)
    runs: list[Run] = field(default_factory=list)
    instance: str = "lftest"

    def push(self, outcome: ContainerResult | BaseException, *, sink: bytes = b"") -> None:
        self.steps.append(Step(outcome, sink))

    async def run(self, spec: ContainerSpec, **kwargs: Any) -> ContainerResult:
        self.runs.append(Run(spec, kwargs))
        step = self.steps.pop(0)
        sink: BinaryIO | None = kwargs.get("stdout_sink")
        if step.sink:
            assert sink is not None
            sink.write(step.sink)
        if isinstance(step.outcome, BaseException):
            raise step.outcome
        return step.outcome

    def check(self, image: str) -> str:
        return f"sha256:checked-{image}"

    def image_id(self, image: str) -> str:
        return f"sha256:id-{image}"

    @property
    def last(self) -> Run:
        return self.runs[-1]


class FakeWorkspace:
    volume = "lf-lftest-volume0000"

    def mount(self, *, read_only: bool = False) -> VolumeMount:
        return VolumeMount(self.volume, "/work", read_only=read_only)


@pytest.fixture
def fake() -> Iterator[FakeRuntime]:
    runtime = FakeRuntime()
    yield runtime
    assert runtime.steps == [], "a scripted container run was never used"


@pytest.fixture
def runtime(fake: FakeRuntime) -> DockerRuntime:
    return cast("DockerRuntime", fake)


@pytest.fixture
def workspace() -> TaskWorkspace:
    return cast("TaskWorkspace", FakeWorkspace())


@pytest.fixture
def config() -> SandboxConfig:
    return SandboxConfig(
        memory_mb=256,
        pids_limit=64,
        work_size_mb=64,
        tmp_size_mb=32,
        command_timeout_s=60.0,
        max_bundle_bytes=5_000,
        max_output_bytes=2_048,
    )


@pytest.fixture
def factory_src(tmp_path: Path) -> Path:
    """A world-readable stand-in for the factory package (bind sources must exist)."""
    src = tmp_path / "factory" / "local_factory"
    src.mkdir(parents=True)
    (src / "__init__.py").write_text("")
    for path in (src.parent, src):
        path.chmod(0o755)
    (src / "__init__.py").chmod(0o644)
    return src
