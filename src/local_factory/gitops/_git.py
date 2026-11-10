"""The single entry point for running git: a sanitized environment and mandatory hardening flags.

Every git invocation in local-factory goes through :func:`run_git`. It never inherits the user's
git configuration or any ``GIT_*`` variable, never prompts, never runs hooks or an fsmonitor, and
only speaks the ``file`` transport (local paths and bundles), so no network access is possible.

When no ``cwd`` is given, git runs in a private, empty, read-only directory -- never in the
process's current directory, which might happen to be inside a worker-controlled tree.
"""

from __future__ import annotations

import atexit
import contextlib
import os
import subprocess
import tempfile
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path

FACTORY_NAME = "local-factory"
FACTORY_EMAIL = "local-factory@localhost"

NULL_SHA = "0" * 40
EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"

# Prefixed to every invocation. Command-line config has the highest precedence, so these override
# anything a repository's own config file says (e.g. a worker-planted core.fsmonitor).
HARDENING_FLAGS: tuple[str, ...] = (
    "-c",
    "core.hooksPath=/dev/null",
    "-c",
    "core.fsmonitor=false",
    "-c",
    "protocol.allow=never",
    "-c",
    "protocol.file.allow=always",
)

# GIT_* variables a caller may set through ``git_env(extra=...)``. Everything else under GIT_*,
# and HOME / XDG_CONFIG_HOME (which locate attribute and ignore files), would weaken the sandbox.
_EXTRA_GIT_ALLOWED = frozenset(
    {
        "GIT_AUTHOR_NAME",
        "GIT_AUTHOR_EMAIL",
        "GIT_AUTHOR_DATE",
        "GIT_COMMITTER_NAME",
        "GIT_COMMITTER_EMAIL",
        "GIT_COMMITTER_DATE",
        "GIT_INDEX_FILE",
    }
)
_EXTRA_FORBIDDEN = frozenset({"HOME", "XDG_CONFIG_HOME"})


class GitError(Exception):
    """A git command failed (or produced output the factory refuses to trust).

    ``args`` are the arguments passed to :func:`run_git` (without the hardening prefix),
    ``returncode`` is git's exit status (0 when git succeeded but its result was unacceptable,
    e.g. "not a bare repository"), ``stderr`` is git's decoded error output or our explanation.
    """

    args: tuple[str, ...]
    returncode: int
    stderr: str

    def __init__(self, args: Sequence[str], returncode: int, stderr: str) -> None:
        super().__init__(*args)
        self.args = tuple(args)
        self.returncode = returncode
        self.stderr = stderr

    def __str__(self) -> str:
        command = " ".join(("git", *self.args))
        message = self.stderr.strip() or "no error output"
        return f"{command} failed (exit {self.returncode}): {message}"

    def __reduce__(self) -> tuple[type[GitError], tuple[tuple[str, ...], int, str]]:
        return (type(self), (self.args, self.returncode, self.stderr))


_neutral_lock = threading.Lock()
_neutral_dir: Path | None = None


def _remove_dir(path: Path) -> None:
    with contextlib.suppress(OSError):
        path.rmdir()


def neutral_dir() -> Path:
    """A private, empty, read-only directory: git's HOME and default working directory."""
    global _neutral_dir
    with _neutral_lock:
        if _neutral_dir is None or not _neutral_dir.is_dir():
            created = Path(tempfile.mkdtemp(prefix="local-factory-git-"))
            created.chmod(0o500)  # nothing (not even git) can create files in it
            atexit.register(_remove_dir, created)
            _neutral_dir = created
        return _neutral_dir


def git_env(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """The complete environment for a git subprocess. Nothing is inherited except PATH.

    ``extra`` may add variables, but not ones that would undo the isolation (any ``GIT_*``
    outside a small identity/index allowlist, ``HOME``, ``XDG_CONFIG_HOME``): those raise
    ``ValueError``.
    """
    home = str(neutral_dir())
    env = {
        "PATH": os.environ.get("PATH", os.defpath),
        "LANG": "C",
        "HOME": home,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_ALLOW_PROTOCOL": "file",
        "GIT_AUTHOR_NAME": FACTORY_NAME,
        "GIT_AUTHOR_EMAIL": FACTORY_EMAIL,
        "GIT_COMMITTER_NAME": FACTORY_NAME,
        "GIT_COMMITTER_EMAIL": FACTORY_EMAIL,
    }
    for key, value in (extra or {}).items():
        if key in _EXTRA_FORBIDDEN or (key.startswith("GIT_") and key not in _EXTRA_GIT_ALLOWED):
            raise ValueError(f"git_env: refusing to override {key}: it would weaken isolation")
        env[key] = value
    return env


def run_git(
    args: Sequence[str],
    *,
    git_dir: Path | None = None,
    cwd: Path | None = None,
    input: bytes | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    """Run ``git <hardening flags> [--git-dir=<git_dir>] <args>`` and capture its output.

    ``git_dir`` is passed as an absolute ``--git-dir`` (no repository discovery). ``cwd`` defaults
    to :func:`neutral_dir`. stdin is ``input`` or empty, so git can never block on a terminal.
    With ``check`` (the default) a non-zero exit raises :class:`GitError`.
    """
    argv = ["git", *HARDENING_FLAGS]
    if git_dir is not None:
        argv.append(f"--git-dir={Path(git_dir).absolute()}")
    argv.extend(args)
    result = subprocess.run(
        argv,
        cwd=cwd if cwd is not None else neutral_dir(),
        env=git_env(),
        input=input,
        stdin=subprocess.DEVNULL if input is None else None,
        capture_output=True,
        check=False,
    )
    if check and result.returncode != 0:
        raise GitError(args, result.returncode, result.stderr.decode("utf-8", "replace"))
    return result
