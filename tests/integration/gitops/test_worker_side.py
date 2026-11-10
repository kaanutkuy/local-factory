"""worker_side against real git (in production these run inside the sandbox)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from local_factory.gitops._git import FACTORY_EMAIL, FACTORY_NAME, GitError, run_git
from local_factory.gitops.bare import BareRepo
from local_factory.gitops.bundles import export_base_bundle
from local_factory.gitops.worker_side import (
    TASK_BRANCH,
    commit_all,
    export_task_bundle,
    has_task_commits,
    init_from_bundle,
)

from .conftest import Git, make_work, write

pytestmark = pytest.mark.integration


def test_init_into_existing_empty_dir(repo: BareRepo, base: str, tmp_path: Path, git: Git) -> None:
    bundle = tmp_path / "base.bundle"
    ref = export_base_bundle(repo, sha=base, out=bundle)
    dest = tmp_path / "work"
    dest.mkdir()
    init_from_bundle(bundle, ref, dest)
    assert git("rev-parse", "HEAD", cwd=dest) == base
    assert git("for-each-ref", "--format=%(refname)", cwd=dest) == f"refs/heads/{TASK_BRANCH}"
    assert git("status", "--porcelain", cwd=dest) == ""
    assert not (dest / ".git" / "objects" / "info" / "alternates").exists()  # standalone


def test_init_with_unknown_ref_fails(repo: BareRepo, base: str, tmp_path: Path) -> None:
    bundle = tmp_path / "base.bundle"
    export_base_bundle(repo, sha=base, out=bundle)
    with pytest.raises(GitError):
        init_from_bundle(bundle, "refs/heads/not-in-bundle", tmp_path / "work")


def test_commit_all_ignores_worker_hooks_and_signing(work: Path, tmp_path: Path, git: Git) -> None:
    marker = tmp_path / "pwned"
    hooks = work / ".git" / "hooks"
    for hook in ("pre-commit", "commit-msg", "post-commit", "prepare-commit-msg"):
        (hooks / hook).write_text(f"#!/bin/sh\ntouch {marker}\nexit 1\n")
        (hooks / hook).chmod(0o755)
    git("config", "commit.gpgSign", "true", cwd=work)
    git("config", "gpg.program", str(tmp_path / "evil-gpg"), cwd=work)
    git("config", "core.fsmonitor", f"touch {marker}; true", cwd=work)

    write(work, {"new.txt": "n\n"})
    message = "-leading dash\n\n# not a comment\nbody line\n"
    sha = commit_all(work, message)
    assert sha is not None and sha == git("rev-parse", "HEAD", cwd=work)
    assert not marker.exists()
    assert git("log", "-1", "--format=%B", cwd=work) == message.strip()
    assert git("log", "-1", "--format=%an <%ae>|%cn <%ce>", cwd=work) == (
        f"{FACTORY_NAME} <{FACTORY_EMAIL}>|{FACTORY_NAME} <{FACTORY_EMAIL}>"
    )
    assert git("show", "--name-only", "--format=", "HEAD", cwd=work) == "new.txt"


def test_commit_all_stages_deletions_and_renames(work: Path, git: Git) -> None:
    (work / "README.md").unlink()
    (work / "src" / "app.py").rename(work / "src" / "main.py")
    sha = commit_all(work, "move")
    assert sha is not None
    assert git("status", "--porcelain", cwd=work) == ""
    assert commit_all(work, "again") is None


def test_commit_all_rejects_empty_message(work: Path) -> None:
    write(work, {"x": "x\n"})
    with pytest.raises(GitError):
        commit_all(work, "")


def test_commit_all_surfaces_unexpected_diff_failure(
    work: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = run_git

    def broken(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        if args[:2] == ["diff", "--cached"]:
            return subprocess.CompletedProcess(args, 128, b"", b"fatal: index corrupt")
        return real(args, **kwargs)

    monkeypatch.setattr("local_factory.gitops.worker_side.run_git", broken)
    write(work, {"x": "x\n"})
    with pytest.raises(GitError) as exc:
        commit_all(work, "x")
    assert (exc.value.returncode, exc.value.stderr) == (128, "fatal: index corrupt")


def test_export_task_bundle(
    repo: BareRepo, base: str, work: Path, tmp_path: Path, git: Git
) -> None:
    out = tmp_path / "empty.bundle"
    with pytest.raises(GitError, match="empty bundle"):
        export_task_bundle(work, out, base_sha=base)
    write(work, {"x": "x\n"})
    head = commit_all(work, "x")
    export_task_bundle(work, out, base_sha=base)
    assert git("bundle", "list-heads", str(out), cwd=tmp_path) == f"{head} refs/heads/task"
    verify = subprocess.run(
        ["git", "bundle", "verify", str(out)],
        cwd=repo.path,
        env=git.env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert base in verify.stdout  # the base is the bundle's only prerequisite


def test_two_workers_are_independent(repo: BareRepo, base: str, tmp_path: Path) -> None:
    one = make_work(repo, base, tmp_path, "one")
    two = make_work(repo, base, tmp_path, "two")
    write(one, {"a.txt": "a\n"})
    assert commit_all(one, "a") is not None
    assert commit_all(two, "nothing") is None
    assert not (two / "a.txt").exists()


# --------------------------------------------------------------------------- has_task_commits


def test_has_task_commits_tracks_the_task_branch(work: Path, base: str, git: Git) -> None:
    assert has_task_commits(work, base_sha=base) is False  # fresh clone: task == base
    write(work, {"x.txt": "x\n"})
    commit_all(work, "one")
    assert has_task_commits(work, base_sha=base) is True
    write(work, {"y.txt": "y\n"})
    commit_all(work, "two")
    assert has_task_commits(work, base_sha=base) is True
    git("reset", "-q", "--hard", base, cwd=work)  # the worker threw its commits away
    assert has_task_commits(work, base_sha=base) is False


def test_has_task_commits_ignores_other_branches_and_head(work: Path, base: str, git: Git) -> None:
    git("checkout", "-q", "-b", "side", cwd=work)
    write(work, {"x.txt": "x\n"})
    git("add", "-A", cwd=work)
    git("commit", "-q", "-m", "side work", cwd=work)
    assert has_task_commits(work, base_sha=base) is False  # only refs/heads/task counts


def test_has_task_commits_base_ahead_of_task(
    repo: BareRepo, base: str, work: Path, tmp_path: Path
) -> None:
    write(work, {"x.txt": "x\n"})
    head = commit_all(work, "x")
    assert head is not None
    later = make_work(repo, base, tmp_path, "later")
    # Measured against its own head, the task has nothing new.
    assert has_task_commits(work, base_sha=head) is False
    assert has_task_commits(later, base_sha=base) is False


def test_has_task_commits_raises_on_broken_repos(work: Path, base: str, git: Git) -> None:
    git("branch", "-m", TASK_BRANCH, "renamed", cwd=work)
    with pytest.raises(GitError):  # no task branch: broken, not "no commits"
        has_task_commits(work, base_sha=base)
    git("branch", "-m", "renamed", TASK_BRANCH, cwd=work)
    with pytest.raises(GitError):  # base unknown to the worker repo
        has_task_commits(work, base_sha="0123456789" * 4)
    assert has_task_commits(work, base_sha=base) is False
    shutil.rmtree(work / ".git")
    with pytest.raises(GitError):  # no repository at all
        has_task_commits(work, base_sha=base)
