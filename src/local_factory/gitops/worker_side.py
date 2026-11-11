"""The only code that runs git inside a worker directory.

In production these functions execute inside the box (``lf-entry materialize`` / ``export``,
:mod:`local_factory.sandbox.entry`, Python 3.12); tests also run them in ``tmp_path``. The host
process must never call them on a worker directory: host-side code talks to workers only through
bundle files (:mod:`.bundles`), and a task bundle leaves the box as a stream
(:func:`export_task_bundle_to_stdout`).

The hardening flags from :mod:`._git` still apply here (no hooks, no fsmonitor, no signing), but
a worker repository is worker-controlled by design: its own config and attributes (e.g. clean
filters) are only contained by the sandbox.
"""

from __future__ import annotations

import re
from pathlib import Path

from ._git import GitError, run_git
from .bare import absolute, check_ref_name

TASK_BRANCH = "task"
TASK_REF = f"refs/heads/{TASK_BRANCH}"
_SHA = re.compile(r"^[0-9a-f]{40}([0-9a-f]{24})?$")
STDOUT_FD = 1


def _git_dir(work: Path) -> Path:
    return absolute(work) / ".git"


def init_from_bundle(bundle: Path, ref: str, dest: Path) -> None:
    """Create a standalone repository at ``dest`` with ``ref`` from ``bundle`` checked out as
    branch ``task``. Bundles carry no HEAD, hence init + fetch + switch rather than clone.

    ``dest`` must not exist or be an empty directory (``FileExistsError`` otherwise).
    """
    check_ref_name(ref)
    target = absolute(dest)
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise FileExistsError(f"worker destination exists and is not empty: {target}")
    run_git(["init", "--quiet", "--initial-branch=main", str(target)])
    fetch = ["fetch", "--quiet", "--no-tags", "--no-write-fetch-head", "--no-recurse-submodules"]
    fetch += ["--end-of-options", str(absolute(bundle)), f"{ref}:{TASK_REF}"]
    run_git(fetch, git_dir=_git_dir(target), cwd=target)
    run_git(["switch", "--quiet", TASK_BRANCH], git_dir=_git_dir(target), cwd=target)


def commit_all(work: Path, message: str) -> str | None:
    """Stage everything (``add -A``) and commit; returns the new sha, or None if nothing changed."""
    git_dir, cwd = _git_dir(work), absolute(work)
    run_git(["add", "--all"], git_dir=git_dir, cwd=cwd)
    diff_args = ["diff", "--cached", "--quiet", "--no-ext-diff", "--no-textconv"]
    staged = run_git(diff_args, git_dir=git_dir, cwd=cwd, check=False)
    if staged.returncode == 0:
        return None
    if staged.returncode != 1:
        raise GitError(diff_args, staged.returncode, staged.stderr.decode("utf-8", "replace"))
    run_git(
        ["commit", "--quiet", "--no-verify", "--no-gpg-sign", "--file=-"],
        git_dir=git_dir,
        cwd=cwd,
        input=message.encode(),
    )
    head = run_git(["rev-parse", "--verify", "HEAD"], git_dir=git_dir, cwd=cwd)
    return head.stdout.decode().strip()


def check_sha(base_sha: str) -> None:
    """``ValueError`` unless ``base_sha`` is a full (SHA-1 or SHA-256) object name."""
    if not _SHA.match(base_sha):
        raise ValueError(f"base_sha must be a full object name: {base_sha!r}")


def has_task_commits(work: Path, *, base_sha: str) -> bool:
    """True iff ``refs/heads/task`` has at least one commit not reachable from ``base_sha``.

    Raises ``GitError`` if the worker repository is broken (no ``.git``, no task branch, or
    ``base_sha`` unknown to it) rather than reporting "no commits".
    """
    check_sha(base_sha)
    result = run_git(
        ["rev-list", "--count", "--end-of-options", TASK_REF, f"^{base_sha}"],
        git_dir=_git_dir(work),
        cwd=absolute(work),
    )
    return int(result.stdout) > 0


def export_task_bundle(work: Path, out: Path, *, base_sha: str) -> None:
    """Write ``refs/heads/task ^base_sha`` to the bundle ``out``.

    ``base_sha`` must be a full object name. Fails with ``GitError`` if ``task`` has no commits
    beyond ``base_sha`` (git refuses to create an empty bundle).
    """
    check_sha(base_sha)
    run_git(
        ["bundle", "create", "--quiet", str(absolute(out)), TASK_REF, f"^{base_sha}"],
        git_dir=_git_dir(work),
        cwd=absolute(work),
    )


def export_task_bundle_to_stdout(work: Path, *, base_sha: str, fd: int = STDOUT_FD) -> None:
    """Stream the bundle ``refs/heads/task ^base_sha`` to the file descriptor ``fd`` (default:
    this process's standard output).

    This is how a bundle leaves the box: git itself writes it (``git bundle create -``) straight
    to ``fd``, so nothing is buffered or staged in the box, and the reader on the other end (the
    host) enforces the size cap. Same preconditions and errors as :func:`export_task_bundle`; a
    failure can leave a partial bundle on ``fd``, so the reader must discard it.
    """
    check_sha(base_sha)
    run_git(
        ["bundle", "create", "--quiet", "-", TASK_REF, f"^{base_sha}"],
        git_dir=_git_dir(work),
        cwd=absolute(work),
        stdout_fd=fd,
    )
