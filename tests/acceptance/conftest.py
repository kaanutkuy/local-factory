"""Shared helpers for acceptance tests. Orchestrator-owned and protected.

These helpers deliberately do NOT use local_factory.gitops, so the gitops tests check the
implementation against an independent git harness.
"""

from __future__ import annotations

import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

from local_factory.models import Contract, Criterion, TestLayer

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
