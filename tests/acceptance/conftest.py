"""Shared helpers for acceptance tests. Orchestrator-owned and protected.

The git helpers deliberately do NOT use local_factory.gitops, so the gitops tests check the
implementation against an independent git harness. The Phase 1b box fixtures build their inputs
with the (1a-verified) BareRepo and check for leftovers with the docker CLI directly.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from local_factory.config import SandboxConfig
from local_factory.gitops.bare import BareRepo
from local_factory.gitops.bundles import export_base_bundle
from local_factory.models import Contract, Criterion, TestLayer

if TYPE_CHECKING:
    from local_factory.sandbox import Sandbox
    from local_factory.sandbox.runtime import DockerRuntime

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def git_env(home: Path) -> dict[str, str]:
    return {
        "PATH": os.environ["PATH"],
        "LANG": "C",
        "HOME": str(home),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_AUTHOR_NAME": "tester",
        "GIT_AUTHOR_EMAIL": "tester@localhost",
        "GIT_COMMITTER_NAME": "tester",
        "GIT_COMMITTER_EMAIL": "tester@localhost",
    }


class Git:
    """Minimal independent git harness for building test fixtures."""

    def __init__(self, home: Path) -> None:
        self.env = git_env(home)

    def __call__(self, *args: str, cwd: Path, input: str | None = None) -> str:
        result = subprocess.run(
            ["git", "-c", "init.defaultBranch=main", *args],
            cwd=cwd,
            env=self.env,
            input=input,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()


@pytest.fixture
def git(tmp_path: Path) -> Git:
    home = tmp_path / "home"
    home.mkdir()
    return Git(home)


@pytest.fixture
def seed_repo(tmp_path: Path, git: Git) -> Path:
    """A normal (non-bare) repo with one commit on main."""
    repo = tmp_path / "seed"
    (repo / "src").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "README.md").write_text("# seed\n")
    (repo / "src" / "app.py").write_text("def add(a, b):\n    return a + b\n")
    (repo / "tests" / "test_app.py").write_text("def test_add():\n    assert True\n")
    git("init", "-q", cwd=repo)
    git("add", "-A", cwd=repo)
    git("commit", "-q", "-m", "seed", cwd=repo)
    return repo


def make_contract(n: int = 1) -> Contract:
    return Contract(
        version=1,
        summary="tag notes",
        criteria=tuple(
            Criterion(id=f"AC-{i}", ears=f"When {i}, the system shall {i}.", layer=TestLayer.E2E)
            for i in range(1, n + 1)
        ),
    )


# --------------------------------------------------------------------------- Phase 1b: the box
# Container fixtures. The sandbox modules are imported lazily so that the non-container acceptance
# tests still collect while the sandbox is being built.

BOX_IMAGE = "local-factory/sandbox:v1"


def small_sandbox_config(**overrides: object) -> SandboxConfig:
    """Tight limits so container tests are fast and resource checks are cheap to trigger."""
    values: dict[str, object] = {
        "memory_mb": 256,
        "cpus": 1.0,
        "pids_limit": 64,
        "work_size_mb": 64,
        "tmp_size_mb": 32,
        "command_timeout_s": 60.0,
        "run_timeout_s": 120.0,
        "max_bundle_bytes": 5_000_000,
        "max_output_bytes": 64_000,
    }
    values.update(overrides)
    return SandboxConfig.model_validate(values)


def docker_objects(instance: str) -> tuple[list[str], list[str]]:
    """Containers and volumes carrying `instance`'s label, asked of docker directly."""
    label = f"label=local-factory.instance={instance}"

    def listing(*args: str) -> list[str]:
        out = subprocess.run(
            ["docker", *args, "--filter", label], capture_output=True, text=True, check=True
        )
        return out.stdout.split()

    return listing("ps", "-aq"), listing("volume", "ls", "-q")


@pytest.fixture
def docker_runtime() -> Iterator[DockerRuntime]:
    from local_factory.sandbox.runtime import DockerRuntime, SandboxError

    runtime = DockerRuntime()
    try:
        runtime.check(BOX_IMAGE)
    except SandboxError as exc:
        pytest.fail(f"container tests need Docker and {BOX_IMAGE} (scripts/build-images): {exc}")
    yield runtime
    runtime.cleanup()
    assert docker_objects(runtime.instance) == ([], []), "the runtime left objects behind"


@pytest.fixture
def box_config() -> SandboxConfig:
    return small_sandbox_config()


@pytest.fixture
def sandbox(docker_runtime: DockerRuntime, box_config: SandboxConfig) -> Sandbox:
    from local_factory.sandbox import Sandbox

    return Sandbox(docker_runtime, box_config)


@dataclass
class Seeded:
    """A factory bare repo with one commit on main, and a base bundle of it."""

    repo: BareRepo
    main: str
    bundle: Path
    ref: str


@pytest.fixture
def seeded(tmp_path: Path, seed_repo: Path) -> Seeded:
    repo = BareRepo.init(tmp_path / "state" / "repo.git")
    repo.fetch_from(seed_repo, ["refs/heads/main:refs/heads/main"])
    main = repo.rev_parse("refs/heads/main")
    bundle = tmp_path / "bundles" / "base.bundle"
    bundle.parent.mkdir()
    ref = export_base_bundle(repo, sha=main, out=bundle)
    return Seeded(repo=repo, main=main, bundle=bundle, ref=ref)
