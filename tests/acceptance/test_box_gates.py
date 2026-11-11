"""Acceptance: gates run in the box, one fresh tree per gate (docs/specs/phase-1b.md § box).

Protected. Needs Docker and the sandbox image.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path

import pytest

from local_factory.config import CacheMount
from local_factory.models import GateResult, GateStatus
from local_factory.sandbox import Sandbox
from local_factory.sandbox.runtime import DockerRuntime
from local_factory.verification.gates import GateSpec

from .conftest import Seeded, docker_objects, small_sandbox_config

pytestmark = [pytest.mark.container, pytest.mark.integration]


def gates(
    sandbox: Sandbox,
    seeded: Seeded,
    specs: Sequence[GateSpec],
    *,
    fail_fast: bool = False,
    tail_bytes: int = 16_000,
) -> list[GateResult]:
    return asyncio.run(
        sandbox.run_gates(
            specs, bundle=seeded.bundle, ref=seeded.ref, fail_fast=fail_fast, tail_bytes=tail_bytes
        )
    )


def test_exit_status_decides(sandbox: Sandbox, seeded: Seeded) -> None:
    results = gates(
        sandbox,
        seeded,
        [
            GateSpec("lint", "test -f src/app.py && echo linted"),
            GateSpec("unit", "echo failing >&2; exit 3"),
            GateSpec("slow", "sleep 60", timeout_s=1.5),
        ],
    )
    assert [(r.name, r.status, r.exit_code) for r in results] == [
        ("lint", GateStatus.PASS, 0),
        ("unit", GateStatus.FAIL, 3),
        ("slow", GateStatus.ERROR, None),
    ]
    assert "linted" in results[0].output_tail
    assert "failing" in results[1].output_tail  # stderr is part of the tail
    assert "[local-factory] " in results[2].output_tail
    assert docker_objects(sandbox.runtime.instance) == ([], [])


def test_gates_see_the_exact_committed_tree(sandbox: Sandbox, seeded: Seeded) -> None:
    (result,) = gates(
        sandbox,
        seeded,
        [GateSpec("tree", "git rev-parse HEAD; git status --porcelain; cat README.md")],
    )
    assert result.status is GateStatus.PASS
    lines = result.output_tail.splitlines()
    assert lines[0] == seeded.main
    assert "# seed" in lines


def test_every_gate_gets_a_fresh_tree(sandbox: Sandbox, seeded: Seeded) -> None:
    results = gates(
        sandbox,
        seeded,
        [
            GateSpec("vandal", "echo tampered > README.md && rm -rf tests && touch planted"),
            GateSpec("check", "grep -qx '# seed' README.md && test -d tests && test ! -e planted"),
        ],
    )
    assert [r.status for r in results] == [GateStatus.PASS, GateStatus.PASS]


def test_fail_fast_skips_without_running(sandbox: Sandbox, seeded: Seeded) -> None:
    results = gates(
        sandbox,
        seeded,
        [GateSpec("lint", "exit 1"), GateSpec("unit", "echo ran")],
        fail_fast=True,
    )
    assert [r.status for r in results] == [GateStatus.FAIL, GateStatus.SKIPPED]
    assert "ran" not in results[1].output_tail
    assert results[1].output_tail.startswith("[local-factory] not run")


def test_output_tail_is_bounded(sandbox: Sandbox, seeded: Seeded) -> None:
    (result,) = gates(
        sandbox,
        seeded,
        [GateSpec("loud", "python3 -c \"print('n' * 300000 + 'LAST')\"")],
        tail_bytes=2_000,
    )
    assert result.status is GateStatus.PASS
    assert len(result.output_tail) <= 2_000
    assert result.output_tail.rstrip().endswith("LAST")


def test_cache_mounts_are_read_only_and_env_is_explicit(
    docker_runtime: DockerRuntime, seeded: Seeded, tmp_path: Path
) -> None:
    cache = tmp_path / "uv-cache"
    cache.mkdir()
    (cache / "wheel.txt").write_text("cached wheel\n")
    cache.chmod(0o755)
    (cache / "wheel.txt").chmod(0o644)
    config = small_sandbox_config(
        cache_mounts=[CacheMount(source=cache, target="/cache/uv")],
        gate_env={"UV_CACHE_DIR": "/cache/uv"},
    )
    sandbox = Sandbox(docker_runtime, config)
    results = gates(
        sandbox,
        seeded,
        [
            GateSpec("reads", 'cat "$UV_CACHE_DIR/wheel.txt"'),
            GateSpec("writes", "echo poison > /cache/uv/wheel.txt"),
        ],
    )
    assert results[0].status is GateStatus.PASS
    assert "cached wheel" in results[0].output_tail
    assert results[1].status is GateStatus.FAIL
    assert (cache / "wheel.txt").read_text() == "cached wheel\n"


def test_gates_have_no_network_and_no_host_env(
    sandbox: Sandbox, seeded: Seeded, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LF_CANARY_GATE", "canary-77ac21")
    (result,) = gates(
        sandbox,
        seeded,
        [GateSpec("probe", "ls /sys/class/net; env")],
    )
    assert result.status is GateStatus.PASS
    assert result.output_tail.splitlines()[0] == "lo"
    assert "canary-77ac21" not in result.output_tail
