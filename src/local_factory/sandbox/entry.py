"""The factory's code inside the box: ``lf-entry <command> [--option value ...] [-- CMD...]``.

``/usr/local/bin/lf-entry`` runs :func:`main` under ``python3 -I`` (Python 3.12) with the factory's
own ``local_factory`` package mounted read-only. Every command acts on worker-controlled data, so
failures never surface as tracebacks: each one is an exit code plus one bounded ``error: <reason>``
line at the end of stderr (:mod:`.protocol`).

- ``materialize --bundle B --ref R --dest D [-- CMD...]``: ``worker_side.init_from_bundle(B, R,
  D)``; with ``CMD``: ``chdir(D)``, stderr joined into stdout, ``execvp(CMD)``. Any failure before
  or at the exec: 70.
- ``write --root R --path P``: stdin (UTF-8 text) to ``P``. 0; refused or failed: 2.
- ``read --root R --path P [--max-bytes N]``: the file's bytes to stdout. 0; refused/failed: 2.
- ``delete --root R --path P``: remove the file or symlink. 0; refused or failed: 2.
- ``exec -- CMD...``: stderr joined into stdout, ``execvp(CMD)`` in the current directory. The
  command's status; not found: 127, not executable: 126 (the reason is part of the output).
- ``export --work W --base SHA --message M``: commit everything, then stream the bundle
  ``refs/heads/task ^SHA`` to stdout. 0 bundle written; 3 no commits; 4 any git/OS failure.

Unknown commands and bad arguments exit 64. An option takes the next argument as its value
verbatim (a path or a message may start with ``-``); ``--`` ends the options.
"""

from __future__ import annotations

import os
import signal
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from local_factory.agents.local import delete_file, read_file, write_file
from local_factory.gitops.worker_side import (
    commit_all,
    export_task_bundle_to_stdout,
    has_task_commits,
    init_from_bundle,
)
from local_factory.sandbox.protocol import (
    DEFAULT_MAX_BYTES,
    EXIT_CANNOT_EXECUTE,
    EXIT_GIT_FAILED,
    EXIT_MATERIALIZE_FAILED,
    EXIT_NO_CHANGES,
    EXIT_NOT_FOUND,
    EXIT_OK,
    EXIT_REFUSED,
    EXIT_USAGE,
    error_line,
)

# Python ignores these at startup and an ignored disposition survives exec: restore the defaults
# so the command sees a normal process (e.g. `yes | head` ends with SIGPIPE).
_RESTORED_SIGNALS = tuple(
    getattr(signal, name) for name in ("SIGPIPE", "SIGXFZ", "SIGXFSZ") if hasattr(signal, name)
)
_END_OF_OPTIONS = "--"


class _UsageError(Exception):
    """Unknown command or bad arguments (exit 64)."""


@dataclass(frozen=True)
class _Syntax:
    required: tuple[str, ...]
    optional: tuple[str, ...] = ()
    command: str = "none"  # "none" | "optional" | "required": a trailing `-- CMD...`


@dataclass(frozen=True)
class _Call:
    """One parsed invocation: the command name, its options and the argv after ``--``."""

    name: str
    options: dict[str, str] = field(default_factory=dict)
    command: list[str] = field(default_factory=list)  # empty when there is none

    def path(self, key: str) -> Path:
        return Path(self.options[key])


_SYNTAX: dict[str, _Syntax] = {
    "materialize": _Syntax(("bundle", "ref", "dest"), command="optional"),
    "write": _Syntax(("root", "path")),
    "read": _Syntax(("root", "path"), ("max-bytes",)),
    "delete": _Syntax(("root", "path")),
    "exec": _Syntax((), command="required"),
    "export": _Syntax(("work", "base", "message")),
}
_USAGE = "lf-entry {" + ",".join(_SYNTAX) + "} [--option value ...] [-- CMD...]"


def main(argv: Sequence[str]) -> int:
    """Run one ``lf-entry`` command; returns its exit status (a successful exec never returns)."""
    try:
        call = _parse(argv)
    except _UsageError as exc:
        return _fail(EXIT_USAGE, f"{exc} (usage: {_USAGE})")
    return _HANDLERS[call.name](call)


# --------------------------------------------------------------------------- arguments


def _parse(argv: Sequence[str]) -> _Call:
    if not argv:
        raise _UsageError("no command given")
    name, rest = argv[0], list(argv[1:])
    syntax = _SYNTAX.get(name)
    if syntax is None:
        raise _UsageError(f"unknown command {name!r}")
    allowed = {*syntax.required, *syntax.optional}
    options: dict[str, str] = {}
    command: list[str] = []
    index = 0
    while index < len(rest):
        arg = rest[index]
        if arg == _END_OF_OPTIONS:
            command = _trailing_command(name, syntax, rest[index + 1 :])
            break
        key = arg[2:] if arg.startswith("--") else ""
        if key not in allowed:
            raise _UsageError(f"{name}: unexpected argument {arg!r}")
        if key in options:
            raise _UsageError(f"{name}: --{key} given twice")
        if index + 1 == len(rest):
            raise _UsageError(f"{name}: --{key} needs a value")
        options[key] = rest[index + 1]
        index += 2
    missing = [f"--{key}" for key in syntax.required if key not in options]
    if missing:
        raise _UsageError(f"{name}: missing {', '.join(missing)}")
    if syntax.command == "required" and not command:
        raise _UsageError(f"{name}: needs -- CMD...")
    count = options.get("max-bytes", "0")
    if not (count.isascii() and count.isdigit()):
        raise _UsageError(f"{name}: --max-bytes must be a non-negative integer, got {count!r}")
    return _Call(name, options, command)


def _trailing_command(name: str, syntax: _Syntax, command: list[str]) -> list[str]:
    if syntax.command == "none":
        raise _UsageError(f"{name}: takes no command")
    if not command:
        raise _UsageError(f"{name}: empty command after --")
    return command


# --------------------------------------------------------------------------- commands


def _materialize(call: _Call) -> int:
    dest = call.path("dest")
    try:
        init_from_bundle(call.path("bundle"), call.options["ref"], dest)
        if call.command:
            os.chdir(dest)
    except Exception as exc:  # whatever went wrong, the factory reports it as its own failure
        return _fail(EXIT_MATERIALIZE_FAILED, f"materialize: {_describe(exc)}")
    if not call.command:
        return EXIT_OK
    error = _exec(call.command, keep_stderr=True)
    return _fail(EXIT_MATERIALIZE_FAILED, f"materialize: cannot run {call.command[0]!r}: {error}")


def _write(call: _Call) -> int:
    try:
        content = sys.stdin.buffer.read().decode("utf-8")
        write_file(call.path("root"), call.options["path"], content)
    except UnicodeDecodeError:
        return _fail(EXIT_REFUSED, "the content is not valid UTF-8")
    except Exception as exc:  # refused (ToolError) or failed (e.g. the workspace is full)
        return _fail(EXIT_REFUSED, _describe(exc))
    return EXIT_OK


def _read(call: _Call) -> int:
    max_bytes = int(call.options.get("max-bytes", DEFAULT_MAX_BYTES))
    try:
        data = read_file(call.path("root"), call.options["path"], max_bytes=max_bytes)
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.flush()
    except Exception as exc:  # refused (ToolError) or failed
        return _fail(EXIT_REFUSED, _describe(exc))
    return EXIT_OK


def _delete(call: _Call) -> int:
    try:
        delete_file(call.path("root"), call.options["path"])
    except Exception as exc:  # refused (ToolError) or failed
        return _fail(EXIT_REFUSED, _describe(exc))
    return EXIT_OK


def _exec_command(call: _Call) -> int:
    error = _exec(call.command, keep_stderr=False)  # non-empty: _parse requires `-- CMD...`
    code = EXIT_NOT_FOUND if isinstance(error, FileNotFoundError) else EXIT_CANNOT_EXECUTE
    # fd 2 is the command's output stream by now, so the reason reaches whoever reads it.
    return _fail(code, f"cannot run {call.command[0]!r}: {error}", fd=2)


def _export(call: _Call) -> int:
    work, base_sha = call.path("work"), call.options["base"]
    try:
        commit_all(work, call.options["message"])
        if not has_task_commits(work, base_sha=base_sha):
            return EXIT_NO_CHANGES
        sys.stdout.flush()  # nothing may precede the bundle on stdout
        export_task_bundle_to_stdout(work, base_sha=base_sha)
    except Exception as exc:  # the worker controls its repository: any failure is a git failure
        return _fail(EXIT_GIT_FAILED, _describe(exc))
    return EXIT_OK


_HANDLERS: dict[str, Callable[[_Call], int]] = {
    "materialize": _materialize,
    "write": _write,
    "read": _read,
    "delete": _delete,
    "exec": _exec_command,
    "export": _export,
}


# --------------------------------------------------------------------------- helpers


def _exec(command: list[str], *, keep_stderr: bool) -> OSError:
    """Replace this process with ``command``, its stderr joined into stdout.

    Returns only if the exec failed, with the error. ``keep_stderr`` restores the original stderr
    first, so the failure is reported there rather than in the command's output stream.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    saved_stderr = os.dup(2)  # not inheritable: a successful exec closes it
    previous = {signum: signal.signal(signum, signal.SIG_DFL) for signum in _RESTORED_SIGNALS}
    os.dup2(1, 2)
    try:
        # Replacing this process with the requested argv (no shell) is the whole point of `exec`.
        os.execvp(command[0], command)  # noqa: S606 - list argv, no shell; the spec's execvp
    except OSError as exc:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        if keep_stderr:
            os.dup2(saved_stderr, 2)
        os.close(saved_stderr)
        return exc


def _describe(exc: BaseException) -> str:
    message = str(exc)
    if type(exc).__module__.startswith("local_factory."):
        return message  # ToolError, GitError: already a complete sentence
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


def _fail(code: int, reason: str, *, fd: int | None = None) -> int:
    """Report ``reason`` as the final ``error:`` line on ``sys.stderr`` (or straight to ``fd``)."""
    line = error_line(reason) + "\n"
    if fd is None:
        sys.stderr.write(line)
        sys.stderr.flush()
    else:
        os.write(fd, line.encode("utf-8", "backslashreplace"))
    return code
