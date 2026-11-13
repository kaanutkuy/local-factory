"""Worker trees ignore tool caches (docs/specs/phase-2b.md § gitops/worker_side, amendment C).

Run in-process, like every container run of ``worker_side`` is also unit-tested (the box's runs
are invisible to host coverage). Real git in ``tmp_path``.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest

from local_factory.gitops import BareRepo, commit_all, export_base_bundle, init_from_bundle
from local_factory.gitops.worker_side import WORKER_EXCLUDES
from tests.acceptance.conftest import Git

pytestmark = pytest.mark.integration

TRACKED_CACHE = "pkg/__pycache__/legacy.cpython-312.pyc"  # committed by the project, on purpose

CACHES = {
    "__pycache__/top.cpython-312.pyc": "bytecode",
    "pkg/__pycache__/mod.cpython-312.pyc": "bytecode",
    "loose.pyc": "bytecode",
    "pkg/deep/loose.pyc": "bytecode",
    ".pytest_cache/v/cache/lastfailed": "{}",
    "pkg/.pytest_cache/README.md": "cache",
    ".mypy_cache/3.12/app.meta.json": "{}",
    ".ruff_cache/CACHEDIR.TAG": "Signature: 8a477f597d28d172789f06886806bc55",
}


@pytest.fixture
def git(tmp_path: Path) -> Git:
    (tmp_path / "githome").mkdir()
    return Git(tmp_path / "githome")


def worker(tmp_path: Path, git: Git, files: Mapping[str, str] | None = None) -> tuple[Path, str]:
    """A worker tree made from a seed repository (with ``files`` added to it): ``(tree, base)``."""
    seed = tmp_path / "seed"
    seed.mkdir()
    for path, content in {"app.py": "print(1)\n", **(files or {})}.items():
        (seed / path).parent.mkdir(parents=True, exist_ok=True)
        (seed / path).write_text(content)
    git("init", "-q", cwd=seed)
    git("add", "-A", "-f", cwd=seed)  # -f: a project may track a cache file
    git("commit", "-q", "-m", "seed", cwd=seed)
    repo = BareRepo.init(tmp_path / "repo.git")
    repo.fetch_from(seed, ["refs/heads/main:refs/heads/main"])
    base = repo.rev_parse("refs/heads/main")
    bundle = tmp_path / "base.bundle"
    ref = export_base_bundle(repo, sha=base, out=bundle)
    tree = tmp_path / "work"
    init_from_bundle(bundle, ref, tree)
    return tree, base


def write(tree: Path, files: Mapping[str, str]) -> None:
    for path, content in files.items():
        (tree / path).parent.mkdir(parents=True, exist_ok=True)
        (tree / path).write_text(content)


def committed(git: Git, tree: Path, base: str) -> list[str]:
    """What the worker's task branch changed against ``base``, by path."""
    return git("diff", "--name-only", base, "refs/heads/task", cwd=tree).splitlines()


def test_the_excludes_are_the_python_tool_caches() -> None:
    assert WORKER_EXCLUDES == (
        "__pycache__/",
        "*.pyc",
        ".pytest_cache/",
        ".mypy_cache/",
        ".ruff_cache/",
    )


def test_a_new_worker_repository_excludes_them_one_per_line(tmp_path: Path, git: Git) -> None:
    tree, _ = worker(tmp_path, git)
    assert (tree / ".git" / "info" / "exclude").read_text().splitlines() == list(WORKER_EXCLUDES)


def test_into_an_existing_empty_directory_too(tmp_path: Path, git: Git) -> None:
    seed = tmp_path / "seed"
    seed.mkdir()
    (seed / "app.py").write_text("x = 1\n")
    git("init", "-q", cwd=seed)
    git("add", "-A", cwd=seed)
    git("commit", "-q", "-m", "seed", cwd=seed)
    repo = BareRepo.init(tmp_path / "repo.git")
    repo.fetch_from(seed, ["refs/heads/main:refs/heads/main"])
    bundle = tmp_path / "base.bundle"
    ref = export_base_bundle(repo, sha=repo.rev_parse("refs/heads/main"), out=bundle)
    dest = tmp_path / "work"
    dest.mkdir()
    init_from_bundle(bundle, ref, dest)
    assert (dest / ".git" / "info" / "exclude").read_text().splitlines() == list(WORKER_EXCLUDES)


def test_untracked_caches_are_not_staged(tmp_path: Path, git: Git) -> None:
    tree, base = worker(tmp_path, git)
    write(tree, {**CACHES, "tests/test_new.py": "def test_x():\n    pass\n"})
    assert commit_all(tree, "tests and caches") is not None
    assert committed(git, tree, base) == ["tests/test_new.py"]
    status = git("status", "--porcelain", "--ignored", cwd=tree)
    assert "!!" in status and "??" not in status  # the caches are still there, and ignored


def test_a_worker_that_only_left_caches_has_nothing_to_commit(tmp_path: Path, git: Git) -> None:
    tree, base = worker(tmp_path, git)
    write(tree, CACHES)
    assert commit_all(tree, "caches only") is None
    assert committed(git, tree, base) == []


def test_changes_to_a_tracked_cache_file_are_still_staged(tmp_path: Path, git: Git) -> None:
    tree, base = worker(tmp_path, git, {TRACKED_CACHE: "old bytecode"})
    write(tree, {TRACKED_CACHE: "new bytecode", "pkg/__pycache__/untracked.pyc": "bytecode"})
    assert commit_all(tree, "tracked cache changed") is not None
    assert committed(git, tree, base) == [TRACKED_CACHE]


def test_a_tracked_cache_file_that_is_deleted_is_a_staged_deletion(
    tmp_path: Path, git: Git
) -> None:
    tree, base = worker(tmp_path, git, {TRACKED_CACHE: "old bytecode"})
    (tree / TRACKED_CACHE).unlink()
    assert commit_all(tree, "removed") is not None
    assert committed(git, tree, base) == [TRACKED_CACHE]


def test_ordinary_files_next_to_the_caches_are_staged(tmp_path: Path, git: Git) -> None:
    tree, base = worker(tmp_path, git)
    write(tree, {**CACHES, "app.py": "print(2)\n", "pkg/mod.py": "x = 1\n", "cache.txt": "c\n"})
    commit_all(tree, "real work")
    assert committed(git, tree, base) == ["app.py", "cache.txt", "pkg/mod.py"]
