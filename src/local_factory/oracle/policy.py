"""What the test architect may change: new test files under its ticket's acceptance directory.

The test architect's only legitimate output is a set of new files under
``tests/acceptance/<ticket>/`` (docs/specs/phase-2b.md § oracle). Everything else in its bundle
is a violation, so it cannot touch the app, ``factory.toml``, another ticket's tests, workflow
files or a symlink that makes the tree read something outside itself.
"""

from __future__ import annotations

from collections.abc import Sequence

from local_factory.models import (
    MODE_EXECUTABLE,
    MODE_REGULAR,
    MODE_SYMLINK,
    ChangedFile,
    ChangeKind,
    Violation,
)
from local_factory.oracle._ids import check_ticket_id

__all__ = [
    "MAX_ORACLE_FILES",
    "MAX_ORACLE_FILE_BYTES",
    "ORACLE_BINARY",
    "ORACLE_FILES",
    "ORACLE_PATH",
    "ORACLE_SIZE",
    "ORACLE_SYMLINK",
    "acceptance_dir",
    "check_oracle_changes",
]

ORACLE_PATH = "oracle_path"
ORACLE_SYMLINK = "oracle_symlink"
ORACLE_BINARY = "oracle_binary"
ORACLE_SIZE = "oracle_size"
ORACLE_FILES = "oracle_files"

MAX_ORACLE_FILES = 50
MAX_ORACLE_FILE_BYTES = 200_000
_REGULAR_MODES = (MODE_REGULAR, MODE_EXECUTABLE)


def acceptance_dir(ticket_id: str) -> str:
    """The only directory the test architect may add files to: ``tests/acceptance/T-0001/``."""
    return f"tests/acceptance/{check_ticket_id(ticket_id)}/"


def check_oracle_changes(changes: Sequence[ChangedFile], ticket_id: str) -> list[Violation]:
    """Violations of the oracle's rules, one per offending change, sorted by path.

    Every change must be an added regular file (mode 100644 or 100755) under
    :func:`acceptance_dir`, text, and at most ``MAX_ORACLE_FILE_BYTES``. More than
    ``MAX_ORACLE_FILES`` changes also gives one ``oracle_files`` violation (path ``""``). No
    changes at all is not a violation here: it is the caller's ``no_changes``.
    """
    directory = acceptance_dir(ticket_id)
    found = [v for change in changes if (v := _check_change(change, directory)) is not None]
    if len(changes) > MAX_ORACLE_FILES:
        found.append(
            Violation(
                rule=ORACLE_FILES,
                path="",
                message=(
                    f"{len(changes)} files changed, but at most {MAX_ORACLE_FILES} are allowed. "
                    "Write fewer, smaller test files."
                ),
            )
        )
    return sorted(found, key=lambda violation: (violation.path, violation.rule))


def _check_change(change: ChangedFile, directory: str) -> Violation | None:
    if change.mode == MODE_SYMLINK:
        return _violation(
            ORACLE_SYMLINK,
            change,
            "Symlinks are not allowed in the acceptance tests: a link could make the tests read "
            "files from outside the repository. Write a regular file instead.",
        )
    if change.kind is not ChangeKind.ADDED or change.mode not in _REGULAR_MODES:
        return _violation(ORACLE_PATH, change, _not_a_new_file(change, directory))
    if not _inside(change.path, directory):
        return _violation(
            ORACLE_PATH,
            change,
            f"Only new files under {directory} may be added; this one is outside it. Everything "
            "else in the repository belongs to the implementers and the factory.",
        )
    if change.is_binary:
        return _violation(
            ORACLE_BINARY,
            change,
            "Binary files are not allowed in the acceptance tests. Write the test as text.",
        )
    if change.size is not None and change.size > MAX_ORACLE_FILE_BYTES:
        return _violation(
            ORACLE_SIZE,
            change,
            f"This file is {change.size:,} bytes; the limit is {MAX_ORACLE_FILE_BYTES:,}. "
            "Keep test files small.",
        )
    return None


def _not_a_new_file(change: ChangedFile, directory: str) -> str:
    return (
        f"Only new regular files may be added, under {directory}; this change is "
        f"{change.kind.name.lower().replace('_', ' ')} (mode {change.mode}). Existing files, "
        "other modes (submodules, links) and deletions are not allowed."
    )


def _inside(path: str, directory: str) -> bool:
    """``path`` is a file below ``directory``, with no ``.`` or ``..`` or empty component."""
    if not path.startswith(directory):
        return False
    return all(part not in ("", ".", "..") for part in path[len(directory) :].split("/"))


def _violation(rule: str, change: ChangedFile, message: str) -> Violation:
    return Violation(rule=rule, path=change.path, message=message)
