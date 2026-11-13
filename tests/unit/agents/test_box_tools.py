"""Unit tests for local_factory.agents.box_tools: construction, dispatch, bash and clipping.

The editor is covered in test_box_tools_view.py and test_box_tools_edit.py. The oracle is
tests/acceptance/test_box_tool_semantics.py; these tests pin the rest of the behavior (every
branch, the exact messages the model sees, and "validation first, one executor path per command").
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Coroutine, Sequence
from typing import Any

import pytest

from local_factory.agents import box_tools
from local_factory.agents.box_tools import (
    BASH,
    EDITOR,
    MAX_COMMAND_CHARS,
    RESTART_NOTE,
    BoxTools,
    ToolOutcome,
)
from local_factory.agents.tools import CommandResult, ToolError, ToolExecutor

DIR_ERROR = "is a directory; only files can be read"


def run[T](coro: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coro)


class MemoryExecutor:
    """An in-memory ToolExecutor that records every call (name, arguments)."""

    def __init__(
        self,
        files: dict[str, str] | None = None,
        *,
        dirs: Sequence[str] = (),
        results: Sequence[CommandResult] = (),
        fail: dict[str, BaseException] | None = None,
    ) -> None:
        self.files = dict(files or {})
        self.dirs = set(dirs)
        self.results = list(results)
        self.fail = fail or {}
        self.calls: list[tuple[str, Any]] = []

    def _record(self, op: str, arg: Any) -> None:
        self.calls.append((op, arg))
        if op in self.fail:
            raise self.fail[op]

    async def write_file(self, path: str, content: str) -> None:
        self._record("write_file", (path, content))
        self.files[path] = content

    async def read_file(self, path: str) -> str:
        self._record("read_file", path)
        if path in self.dirs:
            raise ToolError(DIR_ERROR)
        if path not in self.files:
            raise ToolError("no such file")
        return self.files[path]

    async def delete_file(self, path: str) -> None:
        self._record("delete_file", path)
        del self.files[path]

    async def run_command(self, argv: Sequence[str], *, timeout_s: float) -> CommandResult:
        self._record("run_command", (list(argv), timeout_s))
        return self.results.pop(0) if self.results else CommandResult(0, "")


def make(
    executor: ToolExecutor | None = None, *, limit: int = 1000, root: str = "/work"
) -> BoxTools:
    return BoxTools(
        executor or MemoryExecutor(), root=root, command_timeout_s=5, max_result_chars=limit
    )


def err(outcome: ToolOutcome, *, denied: bool = False) -> str:
    assert outcome.is_error and outcome.denied is denied, outcome
    assert outcome.content.startswith("Error: ")
    return outcome.content


def good(outcome: ToolOutcome) -> str:
    assert not outcome.is_error and not outcome.denied, outcome
    assert outcome.content
    return outcome.content


# --------------------------------------------------------------------------- construction


@pytest.mark.parametrize(
    "kwargs",
    [
        {"root": ""},
        {"root": "/"},
        {"root": "//work"},
        {"root": "/work/./x"},
        {"root": "/work//x"},
        {"root": "/wo\x00rk"},
        {"command_timeout_s": -1.0},
        {"command_timeout_s": float("nan")},
        {"command_timeout_s": float("-inf")},
        {"max_result_chars": 0},
    ],
)
def test_construction_is_validated(kwargs: dict[str, Any]) -> None:
    values: dict[str, Any] = {"root": "/work", "command_timeout_s": 1.0, "max_result_chars": 1000}
    values.update(kwargs)
    with pytest.raises(ValueError, match=r"."):
        BoxTools(MemoryExecutor(), **values)


@pytest.mark.parametrize("root", ["/work", "/a/b-c/d.e", "/work/.hidden"])
def test_normalized_absolute_roots_are_accepted(root: str) -> None:
    BoxTools(MemoryExecutor(), root=root, command_timeout_s=0.5, max_result_chars=100)


def test_the_module_surface() -> None:
    assert box_tools.BOX_TOOLS == (BASH, EDITOR)
    assert sorted(box_tools.__all__) == sorted(
        [
            "BASH",
            "BOX_TOOLS",
            "EDITOR",
            "MAX_COMMAND_CHARS",
            "RESTART_NOTE",
            "BoxTools",
            "ToolOutcome",
        ]
    )
    assert ToolOutcome("x") == ToolOutcome("x", is_error=False, denied=False)


# --------------------------------------------------------------------------- dispatch


def test_call_dispatches_by_name() -> None:
    executor = MemoryExecutor({"a.txt": "hi\n"})
    tools = make(executor)
    assert good(run(tools.call(BASH, {"restart": True}))) == RESTART_NOTE
    assert "     1\thi" in good(run(tools.call(EDITOR, {"command": "view", "path": "a.txt"})))


@pytest.mark.parametrize("name", ["python", "Bash", "text_editor", "", "bash ", "x" * 500])
def test_an_unknown_tool_is_denied_and_runs_nothing(name: str) -> None:
    executor = MemoryExecutor()
    content = err(run(make(executor).call(name, {"command": "id"})), denied=True)
    assert "unknown tool" in content
    assert len(content) < 200  # an absurd name is not echoed whole
    assert executor.calls == []


# --------------------------------------------------------------------------- bash: input


@pytest.mark.parametrize(
    ("tool_input", "needle"),
    [
        ("ls", "JSON object"),
        (None, "NoneType"),
        ([{"command": "ls"}], "list"),
        ({1: "ls"}, "keys must be strings"),
        ({}, "missing required key(s): command"),
        ({"restart": False}, "missing required key(s): command"),
        ({"restart": 1}, "restart must be a boolean"),
        ({"restart": None}, "restart must be a boolean"),
        ({"command": "ls", "restart": "no"}, "restart must be a boolean"),
        ({"command": "ls", "cwd": "/"}, "cwd"),
        ({"command": "ls", "b": 1, "a": 2}, "'a', 'b'"),
        ({"command": None}, "command must be a string"),
        ({"command": ["ls"]}, "command must be a string, got list"),
        ({"command": ""}, "must not be empty"),
        ({"command": "ls\x00 -l"}, "NUL"),
        ({"command": "echo \ud800"}, "UTF-8"),
        ({"command": "x" * (MAX_COMMAND_CHARS + 1)}, str(MAX_COMMAND_CHARS)),
        ({"restart": True, "command": 5}, "command must be a string"),
    ],
)
def test_malformed_bash_input_is_refused_before_anything_runs(
    tool_input: object, needle: str
) -> None:
    executor = MemoryExecutor()
    assert needle in err(run(make(executor).bash(tool_input)))
    assert executor.calls == []


def test_a_command_of_exactly_the_maximum_length_runs() -> None:
    executor = MemoryExecutor()
    good(run(make(executor).bash({"command": "x" * MAX_COMMAND_CHARS})))
    assert len(executor.calls) == 1


# --------------------------------------------------------------------------- bash: running


def test_restart_runs_nothing_and_is_not_an_error() -> None:
    executor = MemoryExecutor()
    assert good(run(make(executor).bash({"restart": True}))) == RESTART_NOTE
    assert executor.calls == []


@pytest.mark.parametrize("restart", [True, False])
def test_a_command_next_to_restart_still_runs(restart: bool) -> None:
    executor = MemoryExecutor(results=[CommandResult(0, "ran\n")])
    content = good(run(make(executor).bash({"command": "echo ran", "restart": restart})))
    assert content == "ran\n"
    assert executor.calls == [("run_command", (["bash", "-c", "echo ran"], 5.0))]


def test_one_command_is_one_run_in_a_fresh_bash() -> None:
    executor = MemoryExecutor(results=[CommandResult(0, "a\n")])
    tools = BoxTools(executor, root="/work", command_timeout_s=12.5, max_result_chars=1000)
    script = "cd /tmp && FOO=1 env | sort; echo $FOO"
    run(tools.bash({"command": script}))
    assert executor.calls == [("run_command", (["bash", "-c", script], 12.5))]


def test_output_is_returned_as_is_on_success() -> None:
    executor = MemoryExecutor(results=[CommandResult(0, "  indented\n\nlast line\n")])
    assert good(run(make(executor).bash({"command": "x"}))) == "  indented\n\nlast line\n"


@pytest.mark.parametrize("output", ["", "\n", "  \n\t\n"])
def test_blank_output_with_success_is_no_output(output: str) -> None:
    executor = MemoryExecutor(results=[CommandResult(0, output)])
    assert good(run(make(executor).bash({"command": "true"}))) == "(no output)"


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("failed\n", "failed\n[exit code: 2]"),
        ("failed", "failed\n[exit code: 2]"),
        ("", "[exit code: 2]"),
        ("\n\n", "[exit code: 2]"),
    ],
)
def test_a_failing_command_reports_its_exit_code_without_an_error(
    output: str, expected: str
) -> None:
    executor = MemoryExecutor(results=[CommandResult(2, output)])
    assert good(run(make(executor).bash({"command": "false"}))) == expected


def test_a_signal_exit_code_is_shown_as_is() -> None:
    executor = MemoryExecutor(results=[CommandResult(-9, "killed\n")])
    assert good(run(make(executor).bash({"command": "x"}))).endswith("[exit code: -9]")


def test_the_exit_code_line_survives_clipping() -> None:
    executor = MemoryExecutor(results=[CommandResult(7, "HEAD" + "x" * 5000 + "TAIL")])
    content = good(run(make(executor, limit=300).bash({"command": "x"})))
    assert len(content) <= 300
    assert content.startswith("HEAD")
    assert content.endswith("TAIL\n[exit code: 7]")
    assert "characters omitted" in content


def test_success_output_is_clipped_to_the_limit() -> None:
    executor = MemoryExecutor(results=[CommandResult(0, "a" * 5000)])
    content = good(run(make(executor, limit=250).bash({"command": "x"})))
    assert len(content) <= 250 and "characters omitted" in content


def test_a_timeout_is_an_error_naming_the_time_with_the_output() -> None:
    executor = MemoryExecutor(results=[CommandResult(None, "so far\n", timed_out=True)])
    tools = BoxTools(executor, root="/work", command_timeout_s=12.5, max_result_chars=1000)
    assert err(run(tools.bash({"command": "sleep 99"}))) == (
        "Error: command timed out after 12.5s and was killed.\nso far\n"
    )


def test_a_whole_number_of_seconds_has_no_decimals() -> None:
    executor = MemoryExecutor(results=[CommandResult(None, "", timed_out=True)])
    tools = BoxTools(executor, root="/work", command_timeout_s=30, max_result_chars=1000)
    assert err(run(tools.bash({"command": "sleep 99"}))) == (
        "Error: command timed out after 30s and was killed."
    )


def test_a_timeout_keeps_within_the_limit() -> None:
    executor = MemoryExecutor(results=[CommandResult(None, "z" * 9000, timed_out=True)])
    content = err(run(make(executor, limit=100).bash({"command": "x"})))
    assert len(content) <= 100 and "timed out" in content


def test_an_executor_tool_error_is_an_error_result() -> None:
    executor = MemoryExecutor(fail={"run_command": ToolError("empty argv")})
    assert err(run(make(executor).bash({"command": "ls"}))) == "Error: empty argv"


def test_a_long_error_message_is_clipped() -> None:
    executor = MemoryExecutor(fail={"run_command": ToolError("e" * 5000)})
    content = err(run(make(executor, limit=200).bash({"command": "ls"})))
    assert len(content) <= 200


def test_other_executor_failures_propagate() -> None:
    class BoxGoneError(Exception):
        pass

    executor = MemoryExecutor(fail={"run_command": BoxGoneError("docker")})
    with pytest.raises(BoxGoneError):
        run(make(executor).bash({"command": "ls"}))
    with pytest.raises(BoxGoneError):
        run(make(executor).call(BASH, {"command": "ls"}))


def test_cancellation_propagates() -> None:
    executor = MemoryExecutor(fail={"run_command": asyncio.CancelledError()})
    with pytest.raises(asyncio.CancelledError):
        run(make(executor).bash({"command": "ls"}))


# --------------------------------------------------------------------------- clipping


@pytest.mark.parametrize("limit", [100, 137, 1000])
@pytest.mark.parametrize("size", [101, 999, 1000, 1001, 1234, 9999, 10_000, 10_001, 123_456])
def test_clip_never_exceeds_the_limit_and_counts_what_it_omitted(limit: int, size: int) -> None:
    text = "".join(chr(ord("a") + i % 26) for i in range(size))
    clipped = box_tools._clip(text, limit)
    if size <= limit:
        assert clipped == text
        return
    assert len(clipped) <= limit
    marker = re.search(r"\n\[\.\.\. (\d+) characters omitted \.\.\.\]\n", clipped)
    assert marker
    head, tail = clipped.split(marker.group(0))
    assert text.startswith(head) and text.endswith(tail)
    assert len(text) - len(head) - len(tail) == int(marker.group(1))
    assert head and tail  # both ends are kept


def test_clip_keeps_text_that_fits() -> None:
    assert box_tools._clip("abc", 3) == "abc"
    assert box_tools._clip("", 100) == ""


def test_clip_cuts_plainly_when_the_marker_cannot_fit() -> None:
    assert box_tools._clip("x" * 500, 20) == "x" * 20
    assert box_tools._clip("x" * 500, 0) == ""


def test_cut_prefers_the_last_whole_line() -> None:
    body = "line 1\nline 2\nline 3"
    assert box_tools._cut(body, 16) == "line 1\nline 2"
    assert box_tools._cut(body, 5) == "line "  # no newline in reach: a plain cut
    assert box_tools._cut(body, -3) == ""
