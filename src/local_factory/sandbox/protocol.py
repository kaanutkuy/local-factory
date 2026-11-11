"""The contract between the host and ``lf-entry`` in the box (docs/specs/phase-1b.md § box).

Imported on both sides: by :mod:`.entry` inside the box (Python 3.12, stdlib only) and by the
host-side :mod:`.executor` and :mod:`.box`, so the exit codes and the error line are defined once.

A failing ``lf-entry`` command reports exactly one reason, as the last line of its stderr:
``error: <reason>``, a single printable line of at most :data:`MAX_DETAIL` characters. Everything
that crosses the boundary is worker-influenced text, so the host bounds it again on reading.
"""

from __future__ import annotations

LF_ENTRY = "/usr/local/bin/lf-entry"  # the launcher baked into the image (docker/sandbox/lf-entry)

EXIT_OK = 0
EXIT_REFUSED = 2  # write/read/delete: refused or failed
EXIT_NO_CHANGES = 3  # export: the task branch has no commits beyond the base
EXIT_GIT_FAILED = 4  # export: any git/OS failure
EXIT_USAGE = 64  # EX_USAGE: unknown command or bad arguments
EXIT_MATERIALIZE_FAILED = 70  # EX_SOFTWARE: materialize failed before the command was executed
EXIT_CANNOT_EXECUTE = 126  # exec: the command exists but cannot be executed (shell convention)
EXIT_NOT_FOUND = 127  # exec: the command was not found (shell convention)

DEFAULT_MAX_BYTES = 1_000_000  # read: the default --max-bytes
MAX_DETAIL = 2_000  # the longest error line, and the longest detail the host keeps
ERROR_PREFIX = "error: "
_ELLIPSIS = "..."


def bounded(text: str, limit: int = MAX_DETAIL) -> str:
    """``text`` as one printable line of at most ``limit`` characters.

    Control characters (newlines, escape sequences) become spaces and runs of whitespace collapse,
    so worker-influenced text can neither forge extra lines nor carry terminal escapes.
    """
    if limit < len(_ELLIPSIS):
        raise ValueError(f"limit must be at least {len(_ELLIPSIS)}, got {limit}")
    line = " ".join("".join(c if c.isprintable() else " " for c in text).split())
    if len(line) > limit:
        line = line[: limit - len(_ELLIPSIS)] + _ELLIPSIS
    return line


def error_line(reason: str) -> str:
    """The stderr line ``lf-entry`` ends with when it fails (at most :data:`MAX_DETAIL` chars)."""
    return bounded(ERROR_PREFIX + (bounded(reason) or "unknown error"))


def error_reason(stderr: str) -> str | None:
    """The reason from the last ``error:`` line of ``stderr``, or None if there is none."""
    for line in reversed(stderr.splitlines()):
        if line.startswith(ERROR_PREFIX):
            return bounded(line[len(ERROR_PREFIX) :]) or None
    return None
