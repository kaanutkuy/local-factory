"""Sandbox facade: construction checks, materialize, export and gates against a scripted runtime.

No Docker: the fake runtime records each ContainerSpec and plays back canned results, so these
tests check what the facade asks of the box and how it reads the answers.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from collections.abc import Iterator
from pathlib import Path

import pytest

import local_factory
from local_factory.config import CacheMount, SandboxConfig
from local_factory.models import GateResult, GateStatus
from local_factory.sandbox import ExportOutcome, Sandbox, SandboxExecutor
from local_factory.sandbox.protocol import LF_ENTRY, MAX_DETAIL
from local_factory.sandbox.runtime import ContainerResult, DockerRuntime, SandboxError
from local_factory.sandbox.spec import BindMount, TmpfsMount, VolumeMount
from local_factory.sandbox.workspace import TaskWorkspace
from local_factory.verification.gates import GateSpec, run_gates

from .conftest import FakeRuntime, result

BASE = "a" * 40
FACTORY_TARGET = "/opt/local-factory/local_factory"
G = GateStatus


@pytest.fixture
def sandbox(runtime: DockerRuntime, config: SandboxConfig, factory_src: Path) -> Sandbox:
    return Sandbox(runtime, config, factory_src=factory_src)


@pytest.fixture
def bundle(tmp_path: Path) -> Path:
    path = tmp_path / "bundles" / "base.bundle"
    path.parent.mkdir()
    path.write_bytes(b"# v2 git bundle\n")
    path.chmod(0o600)
    return path


@contextlib.contextmanager
def umask(mask: int) -> Iterator[None]:
    previous = os.umask(mask)
    try:
        yield
    finally:
        os.umask(previous)


# --------------------------------------------------------------------------- construction


def test_never_the_host_user(runtime: DockerRuntime, factory_src: Path) -> None:
    for override in ({"uid": os.getuid()}, {"gid": os.getgid()}):
        with pytest.raises(SandboxError, match="host user"):
            Sandbox(runtime, SandboxConfig(**override), factory_src=factory_src)


def test_factory_src_defaults_to_the_imported_package(runtime: DockerRuntime) -> None:
    sandbox = Sandbox(runtime, SandboxConfig())
    assert sandbox.factory_src == Path(local_factory.__file__).resolve().parent
    assert sandbox.factory_src.is_absolute()


def test_factory_src_is_resolved(runtime: DockerRuntime, factory_src: Path) -> None:
    link = factory_src.parent / "link"
    link.symlink_to(factory_src)
    assert Sandbox(runtime, SandboxConfig(), factory_src=link).factory_src == factory_src.resolve()


@pytest.mark.parametrize(
    ("entry", "mode"),
    [
        ("__init__.py", 0o600),  # a file the box user cannot read
        ("pkg", 0o750),  # a directory it cannot read
        ("pkg", 0o744),  # a directory it cannot enter
        (".", 0o711),  # the package directory itself
    ],
)
def test_factory_src_must_be_world_readable(
    runtime: DockerRuntime, factory_src: Path, entry: str, mode: int
) -> None:
    (factory_src / "pkg").mkdir(mode=0o755)
    (factory_src / "pkg" / "mod.py").write_text("")
    (factory_src / "pkg" / "mod.py").chmod(0o644)
    (factory_src / "pkg" / "dangling").symlink_to("/nonexistent")  # symlinks are fine
    target = factory_src / entry
    target.chmod(mode)
    try:
        with pytest.raises(SandboxError, match="chmod -R a\\+rX"):
            Sandbox(runtime, SandboxConfig(), factory_src=factory_src)
    finally:
        target.chmod(0o755)


def test_pycache_directories_are_exempt(runtime: DockerRuntime, factory_src: Path) -> None:
    """The host interpreter writes caches with the user's umask; the box compiles from source."""
    for parent in (factory_src, factory_src / "pkg"):
        parent.mkdir(exist_ok=True, mode=0o755)
        cache = parent / "__pycache__"
        cache.mkdir(mode=0o700)
        (cache / "mod.cpython-312.pyc").write_bytes(b"pyc")
        (cache / "mod.cpython-312.pyc").chmod(0o600)
    (factory_src / "pkg" / "__pycache__").chmod(0o000)  # not even listable: never walked
    try:
        assert Sandbox(runtime, SandboxConfig(), factory_src=factory_src).factory_src
        (factory_src / "pkg" / "mod.py").write_text("")
        (factory_src / "pkg" / "mod.py").chmod(0o600)  # everything else is still checked
        with pytest.raises(SandboxError, match="cannot read 1 entries"):
            Sandbox(runtime, SandboxConfig(), factory_src=factory_src)
    finally:
        (factory_src / "pkg" / "__pycache__").chmod(0o700)


def test_a_file_named_pycache_is_not_exempt(runtime: DockerRuntime, factory_src: Path) -> None:
    (factory_src / "__pycache__").write_text("")
    (factory_src / "__pycache__").chmod(0o600)
    with pytest.raises(SandboxError, match="__pycache__"):
        Sandbox(runtime, SandboxConfig(), factory_src=factory_src)


def test_unreadable_factory_src_lists_a_few_entries(
    runtime: DockerRuntime, factory_src: Path
) -> None:
    for i in range(7):
        (factory_src / f"m{i}.py").write_text("")
        (factory_src / f"m{i}.py").chmod(0o640)
    with pytest.raises(SandboxError, match=r"cannot read 7 entries \(.* \.\.\.\)"):
        Sandbox(runtime, SandboxConfig(), factory_src=factory_src)


def test_factory_src_that_cannot_be_walked(runtime: DockerRuntime, factory_src: Path) -> None:
    locked = factory_src / "locked"
    locked.mkdir(mode=0o000)
    try:
        with pytest.raises(SandboxError, match="cannot be checked"):
            Sandbox(runtime, SandboxConfig(), factory_src=factory_src)
    finally:
        locked.chmod(0o755)


def test_factory_src_must_be_a_directory(runtime: DockerRuntime, tmp_path: Path) -> None:
    with pytest.raises(SandboxError, match="not a directory"):
        Sandbox(runtime, SandboxConfig(), factory_src=tmp_path / "missing")
    (tmp_path / "file").write_text("")
    with pytest.raises(SandboxError, match="not a directory"):
        Sandbox(runtime, SandboxConfig(), factory_src=tmp_path / "file")


def test_check_image_workspace_and_executor(
    sandbox: Sandbox, runtime: DockerRuntime, config: SandboxConfig, workspace: TaskWorkspace
) -> None:
    assert sandbox.check() == f"sha256:checked-{config.image}"
    assert sandbox.image_id() == f"sha256:id-{config.image}"
    ws = sandbox.workspace()
    assert isinstance(ws, TaskWorkspace)
    tools = sandbox.executor(workspace)
    assert isinstance(tools, SandboxExecutor)
    assert (tools.runtime, tools.config, tools.workspace) == (runtime, config, workspace)
    assert tools.factory_src == sandbox.factory_src


# --------------------------------------------------------------------------- materialize


def test_materialize_mounts_the_bundle_read_only(
    sandbox: Sandbox, fake: FakeRuntime, workspace: TaskWorkspace, bundle: Path
) -> None:
    fake.push(result(0))
    asyncio.run(sandbox.materialize(workspace, bundle=bundle, ref="refs/heads/factory-base-1"))
    spec = fake.last.spec
    assert spec.argv == (
        LF_ENTRY,
        "materialize",
        "--bundle",
        "/in/base.bundle",
        "--ref",
        "refs/heads/factory-base-1",
        "--dest",
        "/work",
    )
    assert spec.binds == (
        BindMount(bundle.resolve(), "/in/base.bundle"),
        BindMount(sandbox.factory_src, FACTORY_TARGET),
    )
    assert all(b.read_only for b in spec.binds)
    assert spec.volumes == (VolumeMount("lf-lftest-volume0000", "/work", read_only=False),)
    assert bundle.stat().st_mode & 0o777 == 0o644  # readable by the box user


@pytest.mark.parametrize(
    ("box", "message"),
    [
        (result(70, stderr="error: materialize: git fetch failed\n"), "git fetch failed"),
        (result(None, timed_out=True), "timed out after 60s"),
        (result(137, oom_killed=True), "ran out of memory"),
        (result(1, stderr="Traceback\n"), "exit status 1: Traceback"),
        (result(125), "exit status 125$"),
    ],
)
def test_materialize_failures_are_factory_errors(
    sandbox: Sandbox,
    fake: FakeRuntime,
    workspace: TaskWorkspace,
    bundle: Path,
    box: ContainerResult,
    message: str,
) -> None:
    fake.push(box)
    with pytest.raises(SandboxError, match=message) as failed:
        asyncio.run(sandbox.materialize(workspace, bundle=bundle, ref="refs/heads/x"))
    assert len(str(failed.value)) <= MAX_DETAIL


def test_materialize_needs_a_bundle_file(
    sandbox: Sandbox, workspace: TaskWorkspace, tmp_path: Path
) -> None:
    with pytest.raises(SandboxError, match="cannot be handed to the box"):
        asyncio.run(sandbox.materialize(workspace, bundle=tmp_path / "nope", ref="refs/heads/x"))
    with pytest.raises(SandboxError, match="not a regular file"):
        asyncio.run(sandbox.materialize(workspace, bundle=tmp_path, ref="refs/heads/x"))


# --------------------------------------------------------------------------- export


def export(
    sandbox: Sandbox,
    workspace: TaskWorkspace,
    out: Path,
    *,
    base_sha: str = BASE,
    message: str = "T-0001/task-1: change",
) -> ExportOutcome:
    return asyncio.run(sandbox.export(workspace, base_sha=base_sha, message=message, out=out))


def test_export_streams_into_a_private_file(
    sandbox: Sandbox, fake: FakeRuntime, workspace: TaskWorkspace, tmp_path: Path
) -> None:
    out = tmp_path / "task.bundle"
    fake.push(result(0), sink=b"# v2 git bundle\nDATA")
    with umask(0o277):  # the file must end up exactly 0600 whatever the umask
        assert export(sandbox, workspace, out) == ExportOutcome("exported")
    assert out.read_bytes() == b"# v2 git bundle\nDATA"
    assert out.stat().st_mode & 0o777 == 0o600
    run = fake.last
    assert run.spec.argv == (
        LF_ENTRY,
        "export",
        "--work",
        "/work",
        "--base",
        BASE,
        "--message",
        "T-0001/task-1: change",
    )
    assert run.spec.volumes == (VolumeMount("lf-lftest-volume0000", "/work", read_only=False),)
    assert run.spec.binds == (BindMount(sandbox.factory_src, FACTORY_TARGET),)
    assert run.spec.workdir == "/work"
    assert run.kwargs["max_stdout_bytes"] == sandbox.config.max_bundle_bytes
    assert run.kwargs["timeout_s"] == sandbox.config.command_timeout_s
    assert run.kwargs.get("output", "head") == "head"


@pytest.mark.parametrize(
    ("box", "sink", "status", "detail"),
    [
        (result(3), b"", "no_changes", "no changes"),
        (result(3), b"garbage", "no_changes", "no changes"),
        (result(None, output_exceeded=True), b"x" * 100, "bundle_too_large", "5000 bytes"),
        (result(0), b"", "worker_git_error", "empty bundle"),
        (result(4, stderr="error: git add --all failed (exit 128): fatal: bad\n"), b"", "", "bad"),
        (result(4), b"partial", "worker_git_error", "export failed: exit status 4"),
        (result(1, stderr="Traceback\n"), b"", "worker_git_error", "exit status 1: Traceback"),
        (result(None, timed_out=True), b"part", "worker_git_error", "timed out after 60s"),
        (result(137, oom_killed=True), b"part", "worker_git_error", "ran out of memory"),
        (result(0, oom_killed=True), b"bundle", "worker_git_error", "ran out of memory"),
    ],
)
def test_export_outcomes(
    sandbox: Sandbox,
    fake: FakeRuntime,
    workspace: TaskWorkspace,
    tmp_path: Path,
    box: ContainerResult,
    sink: bytes,
    status: str,
    detail: str,
) -> None:
    out = tmp_path / "task.bundle"
    fake.push(box, sink=sink)
    outcome = export(sandbox, workspace, out)
    assert outcome.status == (status or "worker_git_error")
    assert detail in outcome.detail
    assert len(outcome.detail) <= MAX_DETAIL
    assert not out.exists()


def test_export_detail_is_bounded(
    sandbox: Sandbox, fake: FakeRuntime, workspace: TaskWorkspace, tmp_path: Path
) -> None:
    fake.push(result(4, stderr="error: " + "w" * 1990 + "\n"))
    outcome = export(sandbox, workspace, tmp_path / "task.bundle")
    assert outcome.status == "worker_git_error"
    assert len(outcome.detail) <= MAX_DETAIL
    fake.push(result(9, stderr="\n".join(["x" * 100] * 500)))
    assert len(export(sandbox, workspace, tmp_path / "t2.bundle").detail) <= MAX_DETAIL


def test_export_never_overwrites(
    sandbox: Sandbox, workspace: TaskWorkspace, tmp_path: Path
) -> None:
    out = tmp_path / "task.bundle"
    out.write_bytes(b"precious")
    with pytest.raises(FileExistsError):
        export(sandbox, workspace, out)
    assert out.read_bytes() == b"precious"
    link = tmp_path / "link.bundle"
    link.symlink_to(tmp_path / "elsewhere")
    with pytest.raises(FileExistsError):
        export(sandbox, workspace, link)
    assert not (tmp_path / "elsewhere").exists()


def test_factory_failures_during_export_propagate_and_clean_up(
    sandbox: Sandbox, fake: FakeRuntime, workspace: TaskWorkspace, tmp_path: Path
) -> None:
    out = tmp_path / "task.bundle"
    fake.push(SandboxError("docker went away"), sink=b"half")
    with pytest.raises(SandboxError, match="docker went away"):
        export(sandbox, workspace, out)
    assert not out.exists()


@pytest.mark.parametrize(
    ("kw", "error"),
    [
        ({"base_sha": "HEAD"}, "full object name"),
        ({"message": "  \n"}, "message"),
    ],
)
def test_export_rejects_bad_factory_input(
    sandbox: Sandbox, workspace: TaskWorkspace, tmp_path: Path, kw: dict[str, str], error: str
) -> None:
    out = tmp_path / "task.bundle"
    with pytest.raises(ValueError, match=error):
        export(sandbox, workspace, out, **kw)
    assert not out.exists()


# --------------------------------------------------------------------------- gates


def gates(
    sandbox: Sandbox,
    bundle: Path,
    specs: list[GateSpec],
    *,
    fail_fast: bool = False,
    tail_bytes: int = 16_000,
) -> list[GateResult]:
    run = sandbox.run_gates(
        specs,
        bundle=bundle,
        ref="refs/heads/factory-base-2",
        fail_fast=fail_fast,
        tail_bytes=tail_bytes,
    )
    return asyncio.run(run)


def test_each_gate_is_a_fresh_box_on_the_tree(
    runtime: DockerRuntime,
    fake: FakeRuntime,
    config: SandboxConfig,
    factory_src: Path,
    bundle: Path,
    tmp_path: Path,
) -> None:
    cache = tmp_path / "uv-cache"
    cache.mkdir()
    config = config.model_copy(
        update={
            "cache_mounts": (CacheMount(source=cache, target="/cache/uv"),),
            "gate_env": {"UV_CACHE_DIR": "/cache/uv"},
        }
    )
    sandbox = Sandbox(runtime, config, factory_src=factory_src)
    fake.push(result(0, stdout=b"linted\n", duration_s=1.5))
    fake.push(result(0))
    results = gates(
        sandbox, bundle, [GateSpec("lint", "ruff check ."), GateSpec("unit", "pytest", 30.0)]
    )
    assert [(r.name, r.status, r.exit_code) for r in results] == [
        ("lint", G.PASS, 0),
        ("unit", G.PASS, 0),
    ]
    assert results[0].output_tail == "linted\n"
    assert results[0].duration_s == 1.5
    first, second = fake.runs
    assert first.spec.argv[-3:] == ("bash", "-c", "ruff check .")
    assert second.spec.argv == (
        LF_ENTRY,
        "materialize",
        "--bundle",
        "/in/tree.bundle",
        "--ref",
        "refs/heads/factory-base-2",
        "--dest",
        "/work",
        "--",
        "bash",
        "-c",
        "pytest",
    )
    assert second.spec.binds == (
        BindMount(bundle.resolve(), "/in/tree.bundle"),
        BindMount(sandbox.factory_src, FACTORY_TARGET),
        BindMount(cache, "/cache/uv"),
    )
    assert all(b.read_only for b in second.spec.binds)
    assert second.spec.volumes == ()  # never the task workspace
    assert second.spec.tmpfs == (TmpfsMount("/work", config.work_size_mb, executable=True),)
    assert second.spec.env == {"UV_CACHE_DIR": "/cache/uv"}
    assert (second.kwargs["timeout_s"], second.kwargs["output"]) == (30.0, "tail")
    assert second.kwargs["max_stdout_bytes"] == 16_001
    assert bundle.stat().st_mode & 0o777 == 0o644


def test_gate_statuses(sandbox: Sandbox, fake: FakeRuntime, bundle: Path) -> None:
    fake.push(result(3, stdout=b"failing\n"))
    fake.push(result(None, stdout=b"zzz", timed_out=True))
    fake.push(result(70, stdout=b"", stderr="error: materialize: git fetch failed\n"))
    fake.push(result(70, stdout=b"my own exit 70\n"))  # the gate itself exited 70
    fake.push(result(None))
    fake.push(result(0, stdout=b"ok\n", oom_killed=True))
    fake.push(result(137, oom_killed=True))
    specs = [
        GateSpec("unit", "exit 3"),
        GateSpec("slow", "sleep 60", timeout_s=1.5),
        GateSpec("broken", "true"),
        GateSpec("seventy", "exit 70"),
        GateSpec("odd", "true"),
        GateSpec("leaky", "true"),
        GateSpec("hog", "true"),
    ]
    results = gates(sandbox, bundle, specs)
    assert [(r.name, r.status, r.exit_code) for r in results] == [
        ("unit", G.FAIL, 3),
        ("slow", G.ERROR, None),
        ("broken", G.ERROR, None),
        ("seventy", G.FAIL, 70),
        ("odd", G.ERROR, None),
        ("leaky", G.PASS, 0),
        ("hog", G.FAIL, 137),
    ]
    tails = {r.name: r.output_tail for r in results}
    assert tails["unit"] == "failing\n"
    assert tails["slow"].startswith("zzz\n[local-factory] gate 'slow' timed out after 1.5s")
    assert tails["broken"] == (
        "\n[local-factory] gate 'broken' could not be started: materialize: git fetch failed\n"
    )
    assert tails["seventy"] == "my own exit 70\n"
    assert "without an exit status" in tails["odd"]
    assert tails["leaky"] == "ok\n\n[local-factory] gate 'leaky' killed: out of memory\n"
    assert "out of memory" in tails["hog"]


def test_fail_fast_skips_like_the_host_runner(
    sandbox: Sandbox, fake: FakeRuntime, bundle: Path, tmp_path: Path
) -> None:
    specs = [GateSpec("lint", "exit 1"), GateSpec("unit", "echo ran"), GateSpec("e2e", "true")]
    fake.push(result(1))
    in_box = gates(sandbox, bundle, specs, fail_fast=True)
    on_host = run_gates(specs, cwd=tmp_path, fail_fast=True)
    assert len(fake.runs) == 1  # the skipped gates never ran
    assert [r.status for r in in_box] == [G.FAIL, G.SKIPPED, G.SKIPPED]
    assert [(r.status, r.output_tail) for r in in_box[1:]] == [
        (r.status, r.output_tail) for r in on_host[1:]
    ]
    assert in_box[1].output_tail.startswith("[local-factory] not run")


def test_without_fail_fast_every_gate_runs(
    sandbox: Sandbox, fake: FakeRuntime, bundle: Path
) -> None:
    fake.push(result(1))
    fake.push(result(0))
    results = gates(sandbox, bundle, [GateSpec("a", "exit 1"), GateSpec("b", "true")])
    assert [r.status for r in results] == [G.FAIL, G.PASS]


@pytest.mark.parametrize("tail_bytes", [0, 1, 7, 100])
def test_output_tail_never_exceeds_tail_bytes(
    sandbox: Sandbox, fake: FakeRuntime, bundle: Path, tail_bytes: int
) -> None:
    data = ("é" * 200).encode() + b"LAST"
    fake.push(result(None, stdout=data[-(tail_bytes + 1) :], timed_out=True))
    (gate,) = gates(sandbox, bundle, [GateSpec("loud", "x")], tail_bytes=tail_bytes)
    assert len(gate.output_tail) <= tail_bytes
    assert "�" not in gate.output_tail
    assert fake.last.kwargs["max_stdout_bytes"] == tail_bytes + 1


def test_gate_arguments_are_checked(sandbox: Sandbox, bundle: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="tail_bytes"):
        gates(sandbox, bundle, [GateSpec("a", "true")], tail_bytes=-1)
    with pytest.raises(SandboxError, match="cannot be handed to the box"):
        gates(sandbox, tmp_path / "missing.bundle", [GateSpec("a", "true")])
    assert gates(sandbox, bundle, []) == []


def test_gate_factory_failures_propagate(sandbox: Sandbox, fake: FakeRuntime, bundle: Path) -> None:
    fake.push(SandboxError("image missing"))
    with pytest.raises(SandboxError, match="image missing"):
        gates(sandbox, bundle, [GateSpec("a", "true")])
