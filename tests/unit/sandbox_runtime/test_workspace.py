"""Unit tests for TaskWorkspace against a recording runtime (no Docker needed)."""

from __future__ import annotations

import pytest

from local_factory.config import SandboxConfig
from local_factory.sandbox.runtime import DockerRuntime, SandboxError
from local_factory.sandbox.spec import BOX_WORK, ContainerSpec, VolumeMount
from local_factory.sandbox.workspace import HOLDER_ARGV, TaskWorkspace

CONFIG = SandboxConfig(work_size_mb=96, uid=20001, gid=20002, memory_mb=128, pids_limit=32)


class Recorder(DockerRuntime):
    """Records what a workspace asks of the runtime; never runs docker."""

    def __init__(self, *, fail_start: bool = False, fail_remove_volume: bool = False) -> None:
        super().__init__(docker="/nonexistent/docker", instance="lftest")
        self.events: list[tuple[str, str]] = []  # (what, object name)
        self.volumes: list[tuple[int, int, int]] = []  # (size_mb, uid, gid) per volume
        self.holders: list[ContainerSpec] = []
        self.fail_start = fail_start
        self.fail_remove_volume = fail_remove_volume

    def create_volume(self, *, size_mb: int, uid: int, gid: int) -> str:
        name = f"lf-lftest-v{len(self.events):011d}"
        self.events.append(("create_volume", name))
        self.volumes.append((size_mb, uid, gid))
        return name

    def start_detached(self, spec: ContainerSpec) -> str:
        if self.fail_start:
            raise SandboxError("holder would not start")
        self.holders.append(spec)
        name = f"lf-lftest-c{len(self.events):011d}"
        self.events.append(("start_detached", name))
        return name

    def remove_container(self, name: str) -> None:
        self.events.append(("remove_container", name))

    def remove_volume(self, name: str) -> None:
        self.events.append(("remove_volume", name))
        if self.fail_remove_volume:
            raise SandboxError("volume busy")


def test_open_creates_a_volume_and_a_read_only_holder() -> None:
    runtime = Recorder()
    ws = TaskWorkspace(runtime, CONFIG)
    assert ws.open() is ws
    (kind, volume), (start, holder) = runtime.events
    assert (kind, start) == ("create_volume", "start_detached")
    assert runtime.volumes == [(96, 20001, 20002)]
    assert ws.volume == volume
    (spec,) = runtime.holders
    assert spec.argv == HOLDER_ARGV
    assert spec.volumes == (VolumeMount(volume, BOX_WORK, read_only=True),)
    assert (spec.image, spec.uid, spec.gid) == (CONFIG.image, 20001, 20002)
    assert (spec.memory_mb, spec.pids_limit) == (128, 32)
    assert ws.mount() == VolumeMount(volume, BOX_WORK, read_only=False)
    assert ws.mount(read_only=True) == VolumeMount(volume, BOX_WORK, read_only=True)
    assert repr(ws) == f"TaskWorkspace(volume={volume!r}, holder={holder!r})"


def test_open_and_close_are_idempotent_and_close_in_order() -> None:
    runtime = Recorder()
    ws = TaskWorkspace(runtime, CONFIG)
    ws.open()
    ws.open()
    volume = ws.volume
    holder = runtime.events[1][1]
    ws.close()
    ws.close()
    assert runtime.events[2:] == [("remove_container", holder), ("remove_volume", volume)]
    assert [kind for kind, _ in runtime.events].count("create_volume") == 1


def test_the_volume_exists_only_while_open() -> None:
    ws = TaskWorkspace(Recorder(), CONFIG)
    with pytest.raises(RuntimeError, match="not open"):
        _ = ws.volume
    with pytest.raises(RuntimeError, match="not open"):
        ws.mount()
    with ws as opened:
        assert opened is ws
        assert ws.volume.startswith("lf-lftest-")
    with pytest.raises(RuntimeError, match="not open"):
        _ = ws.volume


def test_a_holder_that_fails_takes_its_volume_with_it() -> None:
    runtime = Recorder(fail_start=True)
    ws = TaskWorkspace(runtime, CONFIG)
    with pytest.raises(SandboxError, match="holder would not start"):
        ws.open()
    (created, removed) = runtime.events
    assert created[0] == "create_volume" and removed[0] == "remove_volume"
    assert removed[1] == created[1]
    with pytest.raises(RuntimeError, match="not open"):
        _ = ws.volume
    ws.close()  # nothing left to do
    assert len(runtime.events) == 2


def test_the_holder_error_wins_over_a_failed_volume_removal() -> None:
    runtime = Recorder(fail_start=True, fail_remove_volume=True)
    with pytest.raises(SandboxError, match="holder would not start"):
        TaskWorkspace(runtime, CONFIG).open()


def test_a_failed_close_can_be_retried() -> None:
    runtime = Recorder(fail_remove_volume=True)
    ws = TaskWorkspace(runtime, CONFIG).open()
    with pytest.raises(SandboxError, match="volume busy"):
        ws.close()
    runtime.fail_remove_volume = False
    ws.close()
    kinds = [kind for kind, _ in runtime.events]
    assert kinds == [
        "create_volume",
        "start_detached",
        "remove_container",
        "remove_volume",
        "remove_volume",
    ]
