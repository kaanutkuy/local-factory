"""lf-entry's git commands in-process, against real git in tmp_path.

In production ``materialize`` and ``export`` run inside the box; their container runs are invisible
to host coverage, so they are exercised here directly through ``entry.main``.
"""

from __future__ import annotations

import os
import shutil
import signal
from collections.abc import Iterator
from pathlib import Path
from typing import NoReturn

import pytest

from local_factory.gitops._git import FACTORY_EMAIL, FACTORY_NAME
from local_factory.gitops.bare import BareRepo
from local_factory.gitops.bundles import import_task_bundle
from local_factory.sandbox import entry
from local_factory.sandbox.protocol import (
    EXIT_GIT_FAILED,
    EXIT_MATERIALIZE_FAILED,
    EXIT_NO_CHANGES,
    EXIT_OK,
)

from .conftest import DEST_REF, Bundle, Git, last_error

pytestmark = pytest.mark.integration


# --------------------------------------------------------------------------- materialize


def test_materialize_checks_out_the_ref_as_task(
    bundle: Bundle, base: str, tmp_path: Path, git: Git
) -> None:
    path, ref = bundle.path, bundle.ref
    dest = tmp_path / "dest"
    dest.mkdir()  # an existing empty directory, like the box's /work mount point
    assert (
        entry.main(["materialize", "--bundle", str(path), "--ref", ref, "--dest", str(dest)]) == 0
    )
    assert git("rev-parse", "HEAD", cwd=dest) == base
    assert git("branch", "--show-current", cwd=dest) == "task"
    assert (dest / "README.md").read_text() == "# seed\n"


@pytest.mark.parametrize("problem", ["junk bundle", "unknown ref", "bad ref", "dest not empty"])
def test_materialize_failures_exit_70(
    problem: str, bundle: Bundle, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path, ref = bundle.path, bundle.ref
    dest = tmp_path / "dest"
    if problem == "junk bundle":
        path = tmp_path / "junk.bundle"
        path.write_bytes(b"not a bundle\n")
    elif problem == "unknown ref":
        ref = "refs/heads/not-in-bundle"
    elif problem == "bad ref":
        ref = "main"
    else:
        dest.mkdir()
        (dest / "occupied").touch()
    argv = ["materialize", "--bundle", str(path), "--ref", ref, "--dest", str(dest)]
    assert entry.main([*argv, "--", "true"]) == EXIT_MATERIALIZE_FAILED
    assert last_error(capsys.readouterr().err).startswith("error: materialize: ")


class Replaced(BaseException):
    """Stands in for a successful exec."""


@pytest.fixture
def restore_signals() -> Iterator[None]:
    """A "replaced" process never restores the signals exec reset: do it after the test."""
    before = {signum: signal.getsignal(signum) for signum in entry._RESTORED_SIGNALS}
    yield
    for signum, handler in before.items():
        signal.signal(signum, handler)


def test_materialize_then_exec_runs_in_the_tree(
    bundle: Bundle,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    restore_signals: None,
) -> None:
    path, ref = bundle.path, bundle.ref
    dest = tmp_path / "dest"
    monkeypatch.chdir(tmp_path)  # restored afterwards: materialize changes into dest
    seen: dict[str, object] = {}
    real_dup = os.dup
    duplicated: list[int] = []

    def dup(fd: int) -> int:
        duplicated.append(real_dup(fd))
        return duplicated[-1]

    def execvp(file: str, args: list[str]) -> NoReturn:
        seen["call"] = (file, list(args))
        seen["cwd"] = Path.cwd()
        raise Replaced

    monkeypatch.setattr(os, "dup", dup)
    monkeypatch.setattr(os, "execvp", execvp)
    argv = ["materialize", "--bundle", str(path), "--ref", ref, "--dest", str(dest)]
    try:
        with pytest.raises(Replaced):
            entry.main([*argv, "--", "bash", "-c", "pytest -q"])
    finally:
        for fd in duplicated:
            os.dup2(fd, 2)  # the "replaced" process would not restore stderr: do it here
            os.close(fd)
    assert seen == {"call": ("bash", ["bash", "-c", "pytest -q"]), "cwd": dest.resolve()}


def test_materialize_reports_an_exec_failure_on_the_original_stderr(
    bundle: Bundle,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    path, ref = bundle.path, bundle.ref
    monkeypatch.chdir(tmp_path)
    argv = ["materialize", "--bundle", str(path), "--ref", ref, "--dest", str(tmp_path / "d")]
    assert entry.main([*argv, "--", "lf-no-such-tool-x"]) == EXIT_MATERIALIZE_FAILED
    captured = capfd.readouterr()
    assert captured.out == ""
    assert "cannot run 'lf-no-such-tool-x'" in last_error(captured.err)


# --------------------------------------------------------------------------- export


def export(work: Path, base: str, message: str = "T-0001/task-1: change") -> int:
    return entry.main(["export", "--work", str(work), "--base", base, "--message", message])


def test_export_streams_an_importable_bundle(
    work: Path,
    repo: BareRepo,
    base: str,
    tmp_path: Path,
    git: Git,
    capfdbinary: pytest.CaptureFixture[bytes],
) -> None:
    (work / "feature.txt").write_text("new\n")
    (work / "README.md").unlink()
    assert export(work, base, "-dash first\n\nbody") == EXIT_OK
    captured = capfdbinary.readouterr()
    assert captured.err == b""
    assert captured.out.startswith(b"# v2 git bundle\n")
    out = tmp_path / "task.bundle"
    out.write_bytes(captured.out)
    imported = import_task_bundle(repo, out, base_sha=base, dest_ref=DEST_REF)
    assert sorted(c.path for c in imported.changes) == ["README.md", "feature.txt"]
    assert (
        git("log", "-1", "--format=%B", imported.head_sha, cwd=repo.path) == "-dash first\n\nbody"
    )
    author = git("log", "-1", "--format=%an <%ae>", imported.head_sha, cwd=repo.path)
    assert author == f"{FACTORY_NAME} <{FACTORY_EMAIL}>"


def test_export_includes_the_workers_own_commits(
    work: Path, base: str, git: Git, capfdbinary: pytest.CaptureFixture[bytes]
) -> None:
    (work / "a.txt").write_text("a\n")
    git("add", "a.txt", cwd=work)
    git("commit", "-q", "-m", "worker commit", cwd=work)
    assert export(work, base) == EXIT_OK  # nothing left to stage: the worker's commit is exported
    assert capfdbinary.readouterr().out.startswith(b"# v2 git bundle\n")


def test_export_without_changes_exits_3(
    work: Path, base: str, capfdbinary: pytest.CaptureFixture[bytes]
) -> None:
    assert export(work, base) == EXIT_NO_CHANGES
    assert capfdbinary.readouterr() == (b"", b"")


@pytest.mark.parametrize("damage", ["no repository", "no objects", "no task branch", "bad base"])
def test_export_of_a_broken_repo_exits_4(
    damage: str, work: Path, base: str, git: Git, capfdbinary: pytest.CaptureFixture[bytes]
) -> None:
    (work / "x.txt").write_text("x\n")
    if damage == "no repository":
        shutil.rmtree(work / ".git")
    elif damage == "no objects":
        shutil.rmtree(work / ".git" / "objects")
        (work / ".git" / "objects").mkdir()
    elif damage == "no task branch":
        git("branch", "-m", "task", "elsewhere", cwd=work)
    else:
        base = "not-a-sha"
    assert export(work, base) == EXIT_GIT_FAILED
    captured = capfdbinary.readouterr()
    assert captured.out == b""
    last_error(captured.err.decode())
