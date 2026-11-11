"""Unit tests for the git environment, GitError and run_git's argv/cwd/stdin handling.

run_git is exercised with subprocess.run replaced, so these tests check exactly what would be
executed without running git.
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path
from typing import Any

import pytest

from local_factory.gitops import _git
from local_factory.gitops._git import (
    FACTORY_EMAIL,
    FACTORY_NAME,
    HARDENING_FLAGS,
    GitError,
    git_env,
    neutral_dir,
    run_git,
)

# --------------------------------------------------------------------------- environment


def test_git_env_is_minimal_and_ignores_inherited_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inherited = ("GIT_DIR", "GIT_WORK_TREE", "GIT_CONFIG_COUNT", "GIT_SSH_COMMAND", "GIT_EXEC_PATH")
    for name in inherited:
        monkeypatch.setenv(name, "/tmp/evil")
    monkeypatch.setenv("XDG_CONFIG_HOME", "/tmp/evil-config")
    monkeypatch.setenv("LD_PRELOAD", "/tmp/evil.so")
    env = git_env()
    assert set(env) == {
        "PATH",
        "LANG",
        "HOME",
        "GIT_CONFIG_NOSYSTEM",
        "GIT_CONFIG_GLOBAL",
        "GIT_TERMINAL_PROMPT",
        "GIT_ATTR_NOSYSTEM",
        "GIT_NO_REPLACE_OBJECTS",
        "GIT_ALLOW_PROTOCOL",
        "GIT_AUTHOR_NAME",
        "GIT_AUTHOR_EMAIL",
        "GIT_COMMITTER_NAME",
        "GIT_COMMITTER_EMAIL",
    }
    assert env["PATH"] == os.environ["PATH"]
    assert env["LANG"] == "C"
    assert env["GIT_ALLOW_PROTOCOL"] == "file"
    assert env["GIT_AUTHOR_NAME"] == FACTORY_NAME and env["GIT_AUTHOR_EMAIL"] == FACTORY_EMAIL


def test_git_env_home_is_empty_and_read_only() -> None:
    home = Path(git_env()["HOME"])
    assert home == neutral_dir()
    assert home.is_dir() and list(home.iterdir()) == []
    assert stat.S_IMODE(home.stat().st_mode) & 0o222 == 0
    assert home != Path.home()


def test_git_env_without_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PATH")
    assert git_env()["PATH"] == os.defpath


def test_git_env_extra_allowed() -> None:
    env = git_env({"GIT_AUTHOR_DATE": "2026-01-01T00:00:00Z", "TZ": "UTC"})
    assert env["GIT_AUTHOR_DATE"] == "2026-01-01T00:00:00Z"
    assert env["TZ"] == "UTC"
    assert env["GIT_CONFIG_GLOBAL"] == os.devnull


@pytest.mark.parametrize(
    "key",
    [
        "GIT_DIR",
        "GIT_CONFIG_GLOBAL",
        "GIT_CONFIG_NOSYSTEM",
        "GIT_CONFIG_PARAMETERS",
        "GIT_CONFIG_COUNT",
        "GIT_ALLOW_PROTOCOL",
        "GIT_TERMINAL_PROMPT",
        "GIT_SSH_COMMAND",
        "HOME",
        "XDG_CONFIG_HOME",
    ],
)
def test_git_env_extra_cannot_weaken_isolation(key: str) -> None:
    with pytest.raises(ValueError, match=key):
        git_env({key: "x"})


def test_neutral_dir_is_recreated_if_removed() -> None:
    first = neutral_dir()
    first.rmdir()
    try:
        second = neutral_dir()
        assert second != first and second.is_dir()
    finally:
        assert neutral_dir().is_dir()


def test_remove_dir_tolerates_missing(tmp_path: Path) -> None:
    gone = tmp_path / "gone"
    _git._remove_dir(gone)  # no exception
    gone.mkdir()
    _git._remove_dir(gone)
    assert not gone.exists()


# --------------------------------------------------------------------------- GitError


def test_git_error_fields_and_message() -> None:
    error = GitError(["rev-parse", "nope"], 128, "fatal: bad revision\n")
    assert error.args == ("rev-parse", "nope")
    assert error.returncode == 128
    assert error.stderr == "fatal: bad revision\n"
    assert str(error) == "git rev-parse nope failed (exit 128): fatal: bad revision"


def test_git_error_without_stderr() -> None:
    assert str(GitError(("status",), 1, "  ")) == "git status failed (exit 1): no error output"


def test_git_error_reduces_to_its_constructor_arguments() -> None:
    # What pickle (e.g. multiprocessing) uses to rebuild the exception.
    error = GitError(["fetch", "x"], 1, "boom")
    factory, arguments = error.__reduce__()
    clone = factory(*arguments)
    assert (clone.args, clone.returncode, clone.stderr) == (error.args, 1, "boom")


# --------------------------------------------------------------------------- run_git


class Recorder:
    def __init__(self, returncode: int = 0, stdout: bytes = b"out", stderr: bytes = b"") -> None:
        self.calls: list[tuple[list[str], dict[str, Any]]] = []
        self.result = (returncode, stdout, stderr)

    def __call__(self, argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        self.calls.append((argv, kwargs))
        code, out, err = self.result
        return subprocess.CompletedProcess(argv, code, out, err)


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    rec = Recorder()
    monkeypatch.setattr(subprocess, "run", rec)
    return rec


def test_run_git_prefixes_hardening_flags(recorder: Recorder) -> None:
    result = run_git(["status"])
    [(argv, kwargs)] = recorder.calls
    assert argv == ["git", *HARDENING_FLAGS, "status"]
    assert HARDENING_FLAGS == (
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "protocol.allow=never",
        "-c",
        "protocol.file.allow=always",
    )
    assert kwargs["cwd"] == neutral_dir()  # never the process's cwd
    assert kwargs["stdin"] is subprocess.DEVNULL and kwargs["input"] is None
    assert kwargs["env"] == git_env()
    assert kwargs["capture_output"] is True and kwargs["check"] is False
    assert result.stdout == b"out"


def test_run_git_git_dir_is_absolute(
    recorder: Recorder, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    run_git(["log"], git_dir=Path("repo.git"))
    argv = recorder.calls[0][0]
    assert argv[len(HARDENING_FLAGS) + 1] == f"--git-dir={tmp_path / 'repo.git'}"
    assert argv[-1] == "log"


def test_run_git_passes_cwd_and_input(recorder: Recorder, tmp_path: Path) -> None:
    run_git(["hash-object", "--stdin"], cwd=tmp_path, input=b"data")
    kwargs = recorder.calls[0][1]
    assert kwargs["cwd"] == tmp_path
    assert kwargs["input"] == b"data" and kwargs["stdin"] is None


def test_run_git_check_raises_with_decoded_stderr(recorder: Recorder) -> None:
    recorder.result = (128, b"", b"fatal: caf\xc3\xa9 \xff\n")
    with pytest.raises(GitError) as exc:
        run_git(["fetch", "x"])
    assert exc.value.args == ("fetch", "x")
    assert exc.value.returncode == 128
    assert exc.value.stderr == "fatal: café �\n"


def test_run_git_without_check_returns_failure(recorder: Recorder) -> None:
    recorder.result = (1, b"", b"nope")
    assert run_git(["diff", "--quiet"], check=False).returncode == 1


def test_run_git_streams_stdout_to_a_file_descriptor(recorder: Recorder, tmp_path: Path) -> None:
    recorder.result = (0, b"", b"")
    result = run_git(["bundle", "create", "-", "HEAD"], git_dir=tmp_path / "w.git", stdout_fd=7)
    [(argv, kwargs)] = recorder.calls
    assert argv == [
        "git",
        *HARDENING_FLAGS,
        f"--git-dir={tmp_path / 'w.git'}",
        "bundle",
        "create",
        "-",
        "HEAD",
    ]
    assert kwargs["stdout"] == 7 and kwargs["stderr"] is subprocess.PIPE
    assert "capture_output" not in kwargs
    assert kwargs["cwd"] == neutral_dir()
    assert kwargs["stdin"] is subprocess.DEVNULL and kwargs["input"] is None
    assert kwargs["env"] == git_env() and kwargs["check"] is False
    assert result.returncode == 0


def test_run_git_streamed_failure_raises_with_stderr(recorder: Recorder) -> None:
    recorder.result = (128, b"", b"fatal: Refusing to create empty bundle.\n")
    with pytest.raises(GitError) as exc:
        run_git(["bundle", "create", "-", "HEAD"], stdout_fd=1)
    assert (exc.value.returncode, exc.value.stderr) == (
        128,
        "fatal: Refusing to create empty bundle.\n",
    )
    recorder.result = (128, b"", b"x")
    assert run_git(["bundle", "create", "-", "HEAD"], stdout_fd=1, check=False).returncode == 128


def test_run_git_streams_for_real(tmp_path: Path) -> None:
    out = tmp_path / "version.txt"
    with out.open("wb") as fh:
        result = run_git(["--version"], stdout_fd=fh.fileno())
    assert result.stdout is None
    assert out.read_text().startswith("git version ")
