"""Fixtures for gitops integration tests: real git in tmp_path.

Fixture repositories are built with an independent git harness (not local_factory.gitops), so
the tests check the implementation against git itself.
"""

from __future__ import annotations

import binascii
import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from local_factory.gitops.bare import BareRepo
from local_factory.gitops.bundles import export_base_bundle, import_task_bundle
from local_factory.gitops.worker_side import commit_all, export_task_bundle, init_from_bundle

DEST = "refs/factory/tasks/T-0001/task-1"


class Git:
    """Minimal independent git harness for building fixtures."""

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

    def raw(self, *args: str, cwd: Path, input: bytes | None = None) -> bytes:
        result = subprocess.run(
            ["git", "-c", "init.defaultBranch=main", *args],
            cwd=cwd,
            env=self.env,
            input=input,
            capture_output=True,
            check=True,
        )
        return result.stdout

    def __call__(self, *args: str, cwd: Path, input: str | bytes | None = None) -> str:
        data = input.encode() if isinstance(input, str) else input
        return self.raw(*args, cwd=cwd, input=data).decode().strip()

    def blob(self, content: bytes, *, cwd: Path) -> str:
        return self("hash-object", "-w", "--stdin", cwd=cwd, input=content)

    def tree(self, entries: Sequence[tuple[str, bytes, str]], *, cwd: Path) -> str:
        """Write a raw tree object (bypassing mktree's checks): (mode, name, sha) entries."""

        def key(item: tuple[str, bytes, str]) -> bytes:
            mode, name, _ = item
            return name + b"/" if mode == "40000" else name

        data = b"".join(
            mode.encode() + b" " + name + b"\0" + binascii.unhexlify(sha)
            for mode, name, sha in sorted(entries, key=key)
        )
        return self(
            "hash-object", "-t", "tree", "-w", "--stdin", "--literally", cwd=cwd, input=data
        )

    def tree_entries(self, rev: str, *, cwd: Path) -> list[tuple[str, bytes, str]]:
        """Top-level entries of ``rev``'s tree in the form :meth:`tree` takes."""
        out = self.raw("ls-tree", "-z", rev, cwd=cwd)
        entries = []
        for record in filter(None, out.split(b"\0")):
            meta, name = record.split(b"\t", 1)
            mode, _, sha = meta.decode().split()
            entries.append(("40000" if mode == "040000" else mode, name, sha))
        return entries


@pytest.fixture
def git(tmp_path: Path) -> Git:
    home = tmp_path / "home"
    home.mkdir()
    return Git(home)


def make_seed(git: Git, root: Path, files: Mapping[str, str | bytes]) -> Path:
    """A normal repository with one commit on main containing ``files``."""
    root.mkdir(parents=True)
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, str):
            path.write_text(content)
        else:
            path.write_bytes(content)
    git("init", "-q", cwd=root)
    git("add", "-A", cwd=root)
    git("commit", "-q", "-m", "seed", cwd=root)
    return root


SEED_FILES: dict[str, str | bytes] = {
    "README.md": "# seed\n",
    "src/app.py": "def add(a, b):\n    return a + b\n",
    "tests/test_app.py": "def test_add():\n    assert True\n",
}


@pytest.fixture
def seed_repo(tmp_path: Path, git: Git) -> Path:
    return make_seed(git, tmp_path / "seed", SEED_FILES)


def bare_from(seed: Path, path: Path) -> BareRepo:
    repo = BareRepo.init(path)
    repo.fetch_from(seed, ["refs/heads/main:refs/heads/main"])
    return repo


@pytest.fixture
def repo(tmp_path: Path, seed_repo: Path) -> BareRepo:
    return bare_from(seed_repo, tmp_path / "repo.git")


@pytest.fixture
def base(repo: BareRepo) -> str:
    return repo.rev_parse("refs/heads/main")


def make_work(repo: BareRepo, sha: str, tmp_path: Path, name: str = "work") -> Path:
    bundle = tmp_path / f"{name}-base.bundle"
    ref = export_base_bundle(repo, sha=sha, out=bundle)
    work = tmp_path / name
    init_from_bundle(bundle, ref, work)
    return work


@pytest.fixture
def work(repo: BareRepo, base: str, tmp_path: Path) -> Path:
    return make_work(repo, base, tmp_path)


def write(work: Path, files: Mapping[str, str | bytes]) -> None:
    for rel, content in files.items():
        path = work / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, str):
            path.write_text(content)
        else:
            path.write_bytes(content)


def bundle_of(work: Path, base: str, tmp_path: Path, name: str = "task.bundle") -> Path:
    out = tmp_path / name
    export_task_bundle(work, out, base_sha=base)
    return out


def set_task(git: Git, work: Path, tree: str, parent: str, message: str = "crafted") -> str:
    """Make a commit of ``tree`` on top of ``parent`` the worker's task head."""
    commit = git("commit-tree", tree, "-p", parent, "-m", message, cwd=work)
    git("update-ref", "refs/heads/task", commit, cwd=work)
    return commit


def all_refs(repo: BareRepo, git: Git) -> list[str]:
    return git("for-each-ref", "--format=%(refname)", cwd=repo.path).splitlines()


def import_files(
    repo: BareRepo,
    base: str,
    tmp_path: Path,
    files: Mapping[str, str | bytes],
    *,
    name: str = "task",
    dest_ref: str = DEST,
) -> str:
    """Build a task that writes ``files`` on ``base``, import it, return its head."""
    work = make_work(repo, base, tmp_path, name)
    write(work, files)
    commit_all(work, name)
    bundle = bundle_of(work, base, tmp_path, f"{name}.bundle")
    return import_task_bundle(repo, bundle, base_sha=base, dest_ref=dest_ref).head_sha
