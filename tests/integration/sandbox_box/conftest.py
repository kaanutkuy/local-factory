"""Fixtures for the box's integration tests: real git in tmp_path, and per-test Docker runtimes.

Fixture repositories are built with an independent git harness (not local_factory.gitops).
Container fixtures fail (never skip) without Docker and the sandbox image, and every runtime is
cleaned up and checked for leftovers when its test ends.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from local_factory.config import SandboxConfig
from local_factory.gitops.bare import BareRepo
from local_factory.gitops.bundles import export_base_bundle
from local_factory.gitops.worker_side import init_from_bundle
from local_factory.sandbox import Sandbox
from local_factory.sandbox.protocol import MAX_DETAIL
from local_factory.sandbox.runtime import DockerRuntime, SandboxError

BOX_IMAGE = "local-factory/sandbox:v1"
DEST_REF = "refs/factory/tasks/T-0001/task-1"


class Git:
    """An independent git harness for fixtures and checks."""

    def __init__(self, home: Path) -> None:
        self.env = {
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

    def __call__(self, *args: str, cwd: Path) -> str:
        result = subprocess.run(
            ["git", "-c", "init.defaultBranch=main", *args],
            cwd=cwd,
            env=self.env,
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
def repo(tmp_path: Path, git: Git) -> BareRepo:
    seed = tmp_path / "seed"
    seed.mkdir()
    (seed / "README.md").write_text("# seed\n")
    git("init", "-q", cwd=seed)
    git("add", "-A", cwd=seed)
    git("commit", "-q", "-m", "seed", cwd=seed)
    bare = BareRepo.init(tmp_path / "repo.git")
    bare.fetch_from(seed, ["refs/heads/main:refs/heads/main"])
    return bare


@pytest.fixture
def base(repo: BareRepo) -> str:
    return repo.rev_parse("refs/heads/main")


@dataclass(frozen=True)
class Bundle:
    path: Path
    ref: str


@pytest.fixture
def bundle(repo: BareRepo, base: str, tmp_path: Path) -> Bundle:
    out = tmp_path / "bundles" / "base.bundle"
    out.parent.mkdir()
    return Bundle(out, export_base_bundle(repo, sha=base, out=out))


@pytest.fixture
def work(bundle: Bundle, tmp_path: Path) -> Path:
    dest = tmp_path / "work"
    init_from_bundle(bundle.path, bundle.ref, dest)
    return dest


def last_error(stderr: str) -> str:
    line = stderr.splitlines()[-1]
    assert line.startswith("error: ") and len(line) <= MAX_DETAIL
    return line


# --------------------------------------------------------------------------- containers


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
def docker_runtime() -> Iterator[DockerRuntime]:
    runtime = DockerRuntime()
    try:
        runtime.check(BOX_IMAGE)
    except SandboxError as exc:
        pytest.fail(f"container tests need Docker and {BOX_IMAGE} (scripts/build-images): {exc}")
    yield runtime
    runtime.cleanup()
    assert leftovers(runtime.instance) == ([], []), "the runtime left objects behind"


def box_config(**overrides: object) -> SandboxConfig:
    values: dict[str, object] = {
        "image": BOX_IMAGE,
        "memory_mb": 256,
        "cpus": 1.0,
        "pids_limit": 64,
        "work_size_mb": 64,
        "tmp_size_mb": 32,
        "command_timeout_s": 60.0,
        "max_bundle_bytes": 5_000_000,
        "max_output_bytes": 64_000,
    }
    values.update(overrides)
    return SandboxConfig.model_validate(values)


@pytest.fixture
def sandbox(docker_runtime: DockerRuntime) -> Sandbox:
    return Sandbox(docker_runtime, box_config())
