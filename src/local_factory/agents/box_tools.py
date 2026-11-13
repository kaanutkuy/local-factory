"""The semantics of Anthropic's two client tools, carried out through a ``ToolExecutor``.

``bash`` and ``str_replace_based_edit_tool`` are *client* tools: the model emits a call, the client
carries it out and sends the result back. :class:`BoxTools` is the client side, and it acts only
through a :class:`~local_factory.agents.tools.ToolExecutor`; in production that executor is the box
(no network, not the host user, only the task workspace writable), so nothing a model asks for
ever runs on the host.

Provider-neutral on purpose: it knows nothing of the Anthropic SDK, pydantic or the domain models,
so any model backend can reuse it. Standard library plus ``agents.tools`` and ``agents.local``
only.

Nothing a model sends can raise out of :meth:`BoxTools.call`: malformed input, a refused path or a
:class:`~local_factory.agents.tools.ToolError` from the executor all become a
:class:`ToolOutcome` with ``is_error`` set and content starting ``Error: ``. Any other exception
(a ``SandboxError``: the factory's failure, never the worker's) propagates.
"""

from __future__ import annotations

import math
import posixpath
from collections.abc import Awaitable, Callable
from collections.abc import Set as AbstractSet
from dataclasses import dataclass

from local_factory.agents.local import clip
from local_factory.agents.tools import CommandResult, ToolError, ToolExecutor

__all__ = [
    "BASH",
    "BOX_TOOLS",
    "EDITOR",
    "MAX_COMMAND_CHARS",
    "RESTART_NOTE",
    "BoxTools",
    "ToolOutcome",
]

BASH = "bash"
EDITOR = "str_replace_based_edit_tool"
BOX_TOOLS: tuple[str, ...] = (BASH, EDITOR)
MAX_COMMAND_CHARS = 100_000
RESTART_NOTE = (
    "Nothing to restart: every command already runs in a fresh shell. The files in the "
    "workspace persist between commands, but the working directory, environment variables and "
    "background processes do not."
)

_MIN_RESULT_CHARS = 100
_CONTEXT_LINES = 4  # lines of context either side of an edit in the snippet sent back
_COUNT_CAP = 1_000  # str_replace stops counting matches here (bounds the work on huge inputs)
_NAME_CHARS = 60  # how much of a model-chosen name or value an error message echoes
_ERROR_PREFIX = "Error: "
_REPLACEMENT_CHAR = "�"
_OMITTED = "\n[... {} characters omitted ...]\n"
_OMITTED_OVERHEAD = len(_OMITTED.format(""))
_TRUNCATED = "[truncated: output cut at {} characters; use view_range to view the rest]"
_BASH_KEYS = frozenset({"command", "restart"})
_EDITOR_COMMANDS = ("view", "create", "str_replace", "insert")
_REQUIRED: dict[str, tuple[str, ...]] = {
    "view": ("path",),
    "create": ("path", "file_text"),
    "str_replace": ("path", "old_str"),
    "insert": ("path", "insert_line", "insert_text"),
}
_OPTIONAL: dict[str, tuple[str, ...]] = {
    "view": ("view_range",),
    "create": (),
    "str_replace": ("new_str",),
    "insert": (),
}

type _Args = dict[str, object]
type _Handler = Callable[[_Args], Awaitable[ToolOutcome]]


@dataclass(frozen=True)
class ToolOutcome:
    """What one tool call sends back to the model."""

    content: str  # never empty
    is_error: bool = False
    denied: bool = False  # the call asked for something outside the grant (a path escape)


class _RefusalError(Exception):
    """A call the factory will not carry out; becomes an error :class:`ToolOutcome`."""

    def __init__(self, message: str, *, denied: bool = False) -> None:
        super().__init__(message)
        self.denied = denied


class BoxTools:
    """Carry out ``bash`` and ``str_replace_based_edit_tool`` calls through ``executor``.

    ``root`` is the workspace root as the model sees it (``WorkerHandle.work_dir``, e.g.
    ``/work``); the executor is rooted there and takes paths relative to it, and every message
    names paths as the model does (``root + "/" + relative``).
    """

    def __init__(
        self,
        executor: ToolExecutor,
        *,
        root: str,
        command_timeout_s: float,
        max_result_chars: int,
    ) -> None:
        _check_root(root)
        if not (math.isfinite(command_timeout_s) and command_timeout_s > 0):
            raise ValueError(
                f"command_timeout_s must be a positive number, got {command_timeout_s}"
            )
        if max_result_chars < _MIN_RESULT_CHARS:
            raise ValueError(
                f"max_result_chars must be >= {_MIN_RESULT_CHARS}, got {max_result_chars}"
            )
        self._executor = executor
        self._root = root
        self._prefix = root + "/"
        self._timeout = float(command_timeout_s)
        self._limit = max_result_chars
        self._handlers: dict[str, _Handler] = {
            "view": self._view,
            "create": self._create,
            "str_replace": self._str_replace,
            "insert": self._insert,
        }

    # ------------------------------------------------------------------ the public surface

    async def call(self, name: str, tool_input: object) -> ToolOutcome:
        """Dispatch by tool name; any name but ``bash`` and the editor is refused (``denied``)."""
        if name == BASH:
            return await self.bash(tool_input)
        if name == EDITOR:
            return await self.edit(tool_input)
        return self._error(f"unknown tool {_short(name)}", denied=True)

    async def bash(self, tool_input: object) -> ToolOutcome:
        return await self._guarded(self._bash(tool_input))

    async def edit(self, tool_input: object) -> ToolOutcome:
        return await self._guarded(self._edit(tool_input))

    async def _guarded(self, work: Awaitable[ToolOutcome]) -> ToolOutcome:
        try:
            return await work
        except _RefusalError as refusal:
            return self._error(str(refusal), denied=refusal.denied)
        except ToolError as exc:
            return self._error(str(exc))

    # ------------------------------------------------------------------ bash

    async def _bash(self, tool_input: object) -> ToolOutcome:
        args = _arguments(tool_input)
        _reject_unknown(args, _BASH_KEYS, BASH)
        restart = _restart_flag(args)
        if restart and "command" not in args:
            return self._ok(RESTART_NOTE)
        _require(args, ("command",))
        command = _text(args, "command", nonempty=True, allow_nul=False)
        if len(command) > MAX_COMMAND_CHARS:
            raise _RefusalError(f"command is longer than {MAX_COMMAND_CHARS} characters")
        result = await self._executor.run_command(["bash", "-c", command], timeout_s=self._timeout)
        if result.timed_out:
            budget = self._limit - len(_ERROR_PREFIX)
            return self._error(_timed_out(result, self._timeout, budget))
        return self._ok(_finished(result, self._limit))

    # ------------------------------------------------------------------ editor

    async def _edit(self, tool_input: object) -> ToolOutcome:
        args = _arguments(tool_input)
        command = _editor_command(args)
        allowed = {"command", *_REQUIRED[command], *_OPTIONAL[command]}
        _reject_unknown(args, allowed, f"{EDITOR} {command}")
        _require(args, _REQUIRED[command])
        return await self._handlers[command](args)

    async def _view(self, args: _Args) -> ToolOutcome:
        rel = self._path(args)
        view_range = _view_range(args)
        if rel == ".":
            return await self._list(rel, view_range)
        try:
            text = await self._executor.read_file(rel)
        except ToolError as exc:
            if "is a directory" not in str(exc):
                raise
            return await self._list(rel, view_range)
        return self._show(rel, text, view_range)

    def _show(self, rel: str, text: str, view_range: tuple[int, int] | None) -> ToolOutcome:
        lines = _split_lines(text)
        if not lines:
            return self._ok(f"The file {self._shown(rel)} is empty.")
        first, last = _bounds(view_range, len(lines))
        body = _numbered(lines[first - 1 : last], first)
        if len(body) > self._limit:
            note = _TRUNCATED.format(self._limit)
            body = f"{_cut(body, self._limit - len(note) - 1)}\n{note}"
        return self._ok(body)

    async def _list(self, rel: str, view_range: tuple[int, int] | None) -> ToolOutcome:
        shown = self._shown(rel)
        if view_range is not None:
            raise _RefusalError(f"view_range cannot be used on a directory ({shown})")
        start = "." if rel == "." else f"./{rel}"
        result = await self._executor.run_command(
            ["find", "-P", start, "-maxdepth", "2"], timeout_s=self._timeout
        )
        if not result.ok:
            reason = "timed out" if result.timed_out else f"exit code {result.exit_code}"
            raise _RefusalError(f"could not list {shown} ({reason}):\n{result.output}")
        entries = sorted(self._model_path(line) for line in _visible(result.output, start))
        header = (
            f"Here are the files and directories up to 2 levels deep in {shown}, "
            "excluding hidden items:"
        )
        return self._ok("\n".join([header, *entries]))

    async def _create(self, args: _Args) -> ToolOutcome:
        rel = self._file_path(args)
        content = _text(args, "file_text")
        await self._executor.write_file(rel, content)
        return self._ok(f"File created successfully at: {self._shown(rel)}")

    async def _str_replace(self, args: _Args) -> ToolOutcome:
        rel = self._file_path(args)
        old = _text(args, "old_str", nonempty=True)
        new = _text(args, "new_str") if "new_str" in args else ""
        text = await self._read_editable(rel)
        count = _count(text, old)
        shown = self._shown(rel)
        if count == 0:
            raise _RefusalError(
                f"No replacement was performed: old_str did not appear verbatim in {shown}."
            )
        if count > 1:
            times = f"at least {count}" if count >= _COUNT_CAP else str(count)
            raise _RefusalError(
                f"No replacement was performed: old_str appears {times} times in {shown}; "
                "add more surrounding context so that it matches exactly once."
            )
        at = text.index(old)
        updated = text[:at] + new + text[at + len(old) :]
        await self._executor.write_file(rel, updated)
        return self._edited(shown, updated, text.count("\n", 0, at), len(_split_lines(new)))

    async def _insert(self, args: _Args) -> ToolOutcome:
        rel = self._file_path(args)
        line = _integer(args, "insert_line")
        piece = _text(args, "insert_text")
        text = await self._read_editable(rel)
        count = len(_split_lines(text))
        if not 0 <= line <= count:
            raise _RefusalError(
                f"insert_line {line} is out of range; the file has {count} lines (0..{count})"
            )
        if not piece.endswith("\n"):
            piece += "\n"
        at = _offset_after(text, line)
        before = text[:at]
        if before and not before.endswith("\n"):
            before += "\n"  # inserting after a last line that has no newline
        updated = before + piece + text[at:]
        await self._executor.write_file(rel, updated)
        return self._edited(self._shown(rel), updated, line, piece.count("\n"))

    def _edited(self, shown: str, text: str, first: int, span: int) -> ToolOutcome:
        """The confirmation plus a numbered snippet of the ``span`` edited lines from ``first``."""
        lines = _split_lines(text)
        begin = max(0, first - _CONTEXT_LINES)
        snippet = _numbered(lines[begin : first + span + _CONTEXT_LINES], begin + 1)
        header = f"The file {shown} has been edited."
        return self._ok(f"{header}\n{snippet}" if snippet else header)

    async def _read_editable(self, rel: str) -> str:
        text = await self._executor.read_file(rel)
        if _REPLACEMENT_CHAR in text:
            raise _RefusalError(
                f"{self._shown(rel)} is not valid UTF-8 text (or contains U+FFFD), so it cannot "
                "be edited safely with this tool; use bash to change it"
            )
        return text

    # ------------------------------------------------------------------ paths

    def _file_path(self, args: _Args) -> str:
        """The relative path of a file argument; the workspace root itself is not a file."""
        rel = self._path(args)
        if rel == ".":
            raise _RefusalError(f"{self._root} is the workspace directory; give the path of a file")
        return rel

    def _path(self, args: _Args) -> str:
        return self._relative(_text(args, "path", nonempty=True, allow_nul=False))

    def _relative(self, raw: str) -> str:
        """``raw`` (absolute under the root, or relative to it) as a normalized relative path.

        ``"."`` is the root. Anything outside the workspace is refused as ``denied``; symlinks are
        the executor's business (it confines for real, in the box).
        """
        path = posixpath.normpath(raw)
        if raw.startswith("/"):
            if path == self._root:
                return "."
            if path.startswith(self._prefix):
                return path[len(self._prefix) :]
        elif path != ".." and not path.startswith("../"):
            return path
        raise _RefusalError(
            f"path {_short(raw)} is outside the workspace ({self._root}); use a path under it",
            denied=True,
        )

    def _shown(self, rel: str) -> str:
        return self._root if rel == "." else f"{self._root}/{rel}"

    def _model_path(self, find_line: str) -> str:
        """``.`` or ``./x`` as printed by ``find`` -> ``<root>`` or ``<root>/x``."""
        return self._root if find_line == "." else f"{self._root}/{find_line[2:]}"

    # ------------------------------------------------------------------ outcomes

    def _ok(self, content: str) -> ToolOutcome:
        return ToolOutcome(_clip(content, self._limit))

    def _error(self, message: str, *, denied: bool = False) -> ToolOutcome:
        return ToolOutcome(
            _clip(_ERROR_PREFIX + message, self._limit), is_error=True, denied=denied
        )


# --------------------------------------------------------------------------- construction


def _check_root(root: str) -> None:
    if (
        not root.startswith("/")
        or root == "/"
        or root.startswith("//")
        or "\x00" in root
        or posixpath.normpath(root) != root
    ):
        raise ValueError(f"root must be a normalized absolute POSIX path below /, got {root!r}")


# --------------------------------------------------------------------------- input validation


def _arguments(tool_input: object) -> _Args:
    if not isinstance(tool_input, dict):
        raise _RefusalError(f"tool input must be a JSON object, got {type(tool_input).__name__}")
    if not all(isinstance(key, str) for key in tool_input):
        raise _RefusalError("tool input keys must be strings")
    return dict(tool_input)


def _reject_unknown(args: _Args, allowed: AbstractSet[str], what: str) -> None:
    unknown = sorted(set(args) - allowed)
    if unknown:
        names = ", ".join(_short(key) for key in unknown)
        raise _RefusalError(f"unexpected key(s) for {what}: {names}")


def _require(args: _Args, keys: tuple[str, ...]) -> None:
    missing = [key for key in keys if key not in args]
    if missing:
        raise _RefusalError(f"missing required key(s): {', '.join(missing)}")


def _text(args: _Args, key: str, *, nonempty: bool = False, allow_nul: bool = True) -> str:
    value = args[key]
    if not isinstance(value, str):
        raise _RefusalError(f"{key} must be a string, got {type(value).__name__}")
    if nonempty and not value:
        raise _RefusalError(f"{key} must not be empty")
    if not allow_nul and "\x00" in value:
        raise _RefusalError(f"{key} must not contain a NUL character")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise _RefusalError(
            f"{key} contains a lone surrogate, which cannot be encoded as UTF-8"
        ) from None
    return value


def _integer(args: _Args, key: str) -> int:
    value = args[key]
    if not isinstance(value, int) or isinstance(value, bool):
        raise _RefusalError(f"{key} must be an integer, got {type(value).__name__}")
    return value


def _restart_flag(args: _Args) -> bool:
    value = args.get("restart", False)
    if not isinstance(value, bool):
        raise _RefusalError(f"restart must be a boolean, got {type(value).__name__}")
    return value


def _editor_command(args: _Args) -> str:
    value = args.get("command")
    if isinstance(value, str) and value in _EDITOR_COMMANDS:
        return value
    raise _RefusalError(
        f"command must be one of: {', '.join(_EDITOR_COMMANDS)} (got {_short(value)})"
    )


def _view_range(args: _Args) -> tuple[int, int] | None:
    """``view_range`` as ``(start, end)``; ``end == -1`` means the last line. Shape-checked only."""
    if "view_range" not in args:
        return None
    raw = args["view_range"]
    if not (
        isinstance(raw, list)
        and len(raw) == 2
        and all(isinstance(n, int) and not isinstance(n, bool) for n in raw)
    ):
        raise _RefusalError("view_range must be a list of two integers: [start, end]")
    start, end = raw
    if start < 1 or (end != -1 and end < start):
        raise _RefusalError("view_range needs 1 <= start and (end == -1 or start <= end)")
    return start, end


def _bounds(view_range: tuple[int, int] | None, count: int) -> tuple[int, int]:
    """The 1-based inclusive line range to show in a file of ``count`` (>= 1) lines."""
    if view_range is None:
        return 1, count
    start, end = view_range
    if start > count:
        raise _RefusalError(
            f"view_range start {start} is beyond the end of the file ({count} lines)"
        )
    return start, count if end == -1 else min(end, count)


# --------------------------------------------------------------------------- text helpers


def _short(value: object) -> str:
    """A model-chosen value, quoted and bounded, for an error message."""
    return clip(repr(value), _NAME_CHARS)


def _split_lines(text: str) -> list[str]:
    """Lines split on ``"\\n"`` only; a trailing newline does not add an empty last line."""
    if not text:
        return []
    lines = text.split("\n")
    if text.endswith("\n"):
        lines.pop()
    return lines


def _numbered(lines: list[str], start: int) -> str:
    return "\n".join(f"{n:6}\t{line}" for n, line in enumerate(lines, start))


def _count(text: str, needle: str) -> int:
    """Occurrences of ``needle`` in ``text``, overlapping ones included, up to ``_COUNT_CAP``."""
    count = start = 0
    while count < _COUNT_CAP and (at := text.find(needle, start)) >= 0:
        count += 1
        start = at + 1
    return count


def _offset_after(text: str, lines: int) -> int:
    """The offset just after the first ``lines`` lines of ``text`` (0 for none)."""
    offset = 0
    for _ in range(lines):
        newline = text.find("\n", offset)
        offset = len(text) if newline < 0 else newline + 1
    return offset


def _visible(output: str, start: str) -> list[str]:
    """The ``find`` lines to show: the start itself and entries with no hidden path component."""
    prefix = start + "/"
    return [
        line
        for line in output.splitlines()
        if line == start or (line.startswith(prefix) and not _hidden(line[len(prefix) :]))
    ]


def _hidden(below_start: str) -> bool:
    return any(part.startswith(".") for part in below_start.split("/"))


# --------------------------------------------------------------------------- clipping


def _clip(text: str, limit: int) -> str:
    """``text`` if it fits, else its head and tail around a ``[... N characters omitted ...]``
    line, in at most ``limit`` characters."""
    if len(text) <= limit:
        return text
    keep = limit - _OMITTED_OVERHEAD - len(str(len(text)))  # the worst case for N's digits
    if keep < 0:
        return text[:limit]  # too small for the marker: a plain cut
    tail = keep // 2
    head = keep - tail
    omitted = len(text) - keep
    return f"{text[:head]}{_OMITTED.format(omitted)}{text[len(text) - tail :]}"


def _cut(body: str, limit: int) -> str:
    """The first ``limit`` characters of ``body``, back to the last whole line if there is one."""
    cut = body[: max(limit, 0)]
    newline = cut.rfind("\n")
    return cut[:newline] if newline > 0 else cut


# --------------------------------------------------------------------------- bash results


def _finished(result: CommandResult, limit: int) -> str:
    """The content for a command that ran to the end: its output, then a non-zero exit code."""
    output = result.output
    if result.exit_code == 0:
        return _clip(output, limit) if output.strip() else "(no output)"
    line = f"[exit code: {result.exit_code}]"
    if not output.strip():
        return line
    separator = "" if output.endswith("\n") else "\n"
    return f"{_clip(output, limit - len(line) - 1)}{separator}{line}"


def _timed_out(result: CommandResult, seconds: float, limit: int) -> str:
    head = f"command timed out after {seconds:g}s and was killed."
    if not result.output.strip():
        return head
    return f"{head}\n{_clip(result.output, limit - len(head) - 1)}"
