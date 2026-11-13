"""Acceptance: Anthropic's bash and text-editor tools carried out through a ToolExecutor
(docs/specs/phase-2a.md § tools). Protected. In-process: `LocalExecutor` on `tmp_path` and small
recording executors; the same `BoxTools` runs against the box in the container tests.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine, Sequence
from pathlib import Path
from typing import Any

import pytest

from local_factory.agents.box_tools import (
    BASH,
    BOX_TOOLS,
    EDITOR,
    MAX_COMMAND_CHARS,
    RESTART_NOTE,
    BoxTools,
    ToolOutcome,
)
from local_factory.agents.local import LocalExecutor
from local_factory.agents.tools import CommandResult, ToolError

TIMEOUT = 30.0


def run[T](coro: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coro)


class Recorder:
    """A ToolExecutor that records calls and answers run_command from a fixed result."""

    def __init__(self, result: CommandResult | None = None, *, fail: Exception | None = None):
        self.result = result or CommandResult(0, "")
        self.fail = fail
        self.commands: list[tuple[list[str], float]] = []
        self.files: dict[str, str] = {}

    async def write_file(self, path: str, content: str) -> None:
        self.files[path] = content

    async def read_file(self, path: str) -> str:
        if self.fail is not None:
            raise self.fail
        if path not in self.files:
            raise ToolError("no such file")
        return self.files[path]

    async def delete_file(self, path: str) -> None:
        self.files.pop(path)

    async def run_command(self, argv: Sequence[str], *, timeout_s: float) -> CommandResult:
        self.commands.append((list(argv), timeout_s))
        if self.fail is not None:
            raise self.fail
        return self.result


@pytest.fixture
def root(tmp_path: Path) -> Path:
    work = tmp_path / "work"
    work.mkdir()
    return work


def tools_on(root: Path, *, max_result_chars: int = 40_000) -> BoxTools:
    return BoxTools(
        LocalExecutor(root),
        root=root.as_posix(),
        command_timeout_s=TIMEOUT,
        max_result_chars=max_result_chars,
    )


@pytest.fixture
def tools(root: Path) -> BoxTools:
    return tools_on(root)


def ok(outcome: ToolOutcome) -> str:
    assert isinstance(outcome, ToolOutcome)
    assert not outcome.is_error, outcome.content
    assert not outcome.denied
    assert outcome.content
    return outcome.content


def error(outcome: ToolOutcome, *, denied: bool = False) -> str:
    assert outcome.is_error, outcome.content
    assert outcome.denied is denied, outcome
    assert outcome.content.startswith("Error: ")
    return outcome.content


# --------------------------------------------------------------------------- construction


def test_names() -> None:
    assert (BASH, EDITOR) == ("bash", "str_replace_based_edit_tool")
    assert BOX_TOOLS == (BASH, EDITOR)
    assert MAX_COMMAND_CHARS == 100_000


@pytest.mark.parametrize(
    "kwargs",
    [
        {"root": "work"},
        {"root": "/work/../etc"},
        {"root": "/work/"},
        {"command_timeout_s": 0.0},
        {"command_timeout_s": float("inf")},
        {"max_result_chars": 99},
    ],
)
def test_construction_is_validated(kwargs: dict[str, Any]) -> None:
    values: dict[str, Any] = {"root": "/work", "command_timeout_s": 1.0, "max_result_chars": 1000}
    values.update(kwargs)
    with pytest.raises(ValueError, match=r"."):
        BoxTools(Recorder(), **values)


def test_unknown_tool_is_refused_and_runs_nothing() -> None:
    recorder = Recorder()
    tools = BoxTools(recorder, root="/work", command_timeout_s=5, max_result_chars=1000)
    for name in ("python", "computer", "Bash", "text_editor", "submit_result", ""):
        outcome = run(tools.call(name, {"command": "id"}))
        assert "unknown tool" in error(outcome, denied=True)
    assert recorder.commands == [] and recorder.files == {}


# --------------------------------------------------------------------------- bash


def test_bash_runs_the_command_in_a_fresh_bash() -> None:
    recorder = Recorder(CommandResult(0, "hello\n"))
    tools = BoxTools(recorder, root="/work", command_timeout_s=12.5, max_result_chars=1000)
    content = ok(run(tools.call(BASH, {"command": "echo hello | tr a-z A-Z"})))
    assert recorder.commands == [(["bash", "-c", "echo hello | tr a-z A-Z"], 12.5)]
    assert "hello" in content
    assert "exit code" not in content


def test_bash_for_real(tools: BoxTools, root: Path) -> None:
    content = ok(run(tools.bash({"command": "echo out; echo err >&2; printf x > made.txt"})))
    assert "out" in content and "err" in content
    assert (root / "made.txt").read_text() == "x"


def test_bash_reports_a_failing_exit_code_without_an_error(tools: BoxTools) -> None:
    content = ok(run(tools.bash({"command": "echo nope; exit 3"})))
    assert "nope" in content
    assert content.rstrip().endswith("[exit code: 3]")


def test_bash_with_no_output(tools: BoxTools) -> None:
    assert ok(run(tools.bash({"command": "true"}))) == "(no output)"


def test_bash_timeout_is_an_error_with_the_output() -> None:
    recorder = Recorder(CommandResult(None, "partial output", timed_out=True))
    tools = BoxTools(recorder, root="/work", command_timeout_s=7.0, max_result_chars=1000)
    content = error(run(tools.bash({"command": "sleep 100"})))
    assert "timed out" in content and "7" in content
    assert "partial output" in content


def test_restart_is_a_no_op() -> None:
    recorder = Recorder()
    tools = BoxTools(recorder, root="/work", command_timeout_s=5, max_result_chars=1000)
    assert ok(run(tools.bash({"restart": True}))) == RESTART_NOTE
    assert recorder.commands == []
    for word in ("files", "directory", "environment"):
        assert word in RESTART_NOTE


@pytest.mark.parametrize(
    "tool_input",
    [
        {},
        "ls",
        ["ls"],
        None,
        {"command": ""},
        {"command": 5},
        {"command": ["ls"]},
        {"command": "ls", "extra": 1},
        {"command": "ls\x00"},
        {"command": "x" * (MAX_COMMAND_CHARS + 1)},
        {"restart": "yes"},
        {"restart": False},
        {1: "ls"},
    ],
)
def test_malformed_bash_input_runs_nothing(tool_input: object) -> None:
    recorder = Recorder()
    tools = BoxTools(recorder, root="/work", command_timeout_s=5, max_result_chars=1000)
    error(run(tools.call(BASH, tool_input)))
    assert recorder.commands == []


def test_long_output_keeps_head_and_tail(root: Path) -> None:
    tools = tools_on(root, max_result_chars=1_000)
    script = "python3 -c \"print('HEAD' + 'x' * 20000 + 'TAIL')\""
    content = ok(run(tools.bash({"command": script})))
    assert len(content) <= 1_000
    assert content.startswith("HEAD")
    assert content.rstrip().endswith("TAIL")
    assert "characters omitted" in content


def test_executor_tool_error_becomes_an_error_result() -> None:
    tools = BoxTools(
        Recorder(fail=ToolError("empty argv")),
        root="/work",
        command_timeout_s=5,
        max_result_chars=1000,
    )
    assert "empty argv" in error(run(tools.bash({"command": "ls"})))


def test_factory_failures_propagate() -> None:
    class BoxDownError(Exception):
        pass

    tools = BoxTools(
        Recorder(fail=BoxDownError("docker is gone")),
        root="/work",
        command_timeout_s=5,
        max_result_chars=1000,
    )
    with pytest.raises(BoxDownError):
        run(tools.bash({"command": "ls"}))
    with pytest.raises(BoxDownError):
        run(tools.edit({"command": "view", "path": "a.txt"}))


# --------------------------------------------------------------------------- editor: view


def test_view_numbers_lines(tools: BoxTools, root: Path) -> None:
    (root / "a.py").write_text("one\ntwo\nthree\n")
    content = ok(run(tools.edit({"command": "view", "path": "a.py"})))
    assert content.splitlines()[-3:] == ["     1\tone", "     2\ttwo", "     3\tthree"]
    assert "     4\t" not in content


def test_view_accepts_paths_under_the_root(tools: BoxTools, root: Path) -> None:
    (root / "src").mkdir()
    (root / "src" / "m.py").write_text("x = 1")
    for path in (f"{root}/src/m.py", "src/m.py", "./src/m.py", "src/../src/m.py"):
        assert "     1\tx = 1" in ok(run(tools.edit({"command": "view", "path": path})))


def test_view_range(tools: BoxTools, root: Path) -> None:
    (root / "f.txt").write_text("".join(f"line {i}\n" for i in range(1, 11)))

    def view(view_range: object) -> ToolOutcome:
        return run(tools.edit({"command": "view", "path": "f.txt", "view_range": view_range}))

    shown = ok(view([3, 5])).splitlines()
    assert shown[-3:] == ["     3\tline 3", "     4\tline 4", "     5\tline 5"]
    assert "line 2" not in ok(view([3, 5])) and "line 6" not in ok(view([3, 5]))
    assert ok(view([9, -1])).splitlines()[-2:] == ["     9\tline 9", "    10\tline 10"]
    assert ok(view([9, 50])).splitlines()[-1] == "    10\tline 10"  # clamped
    for bad in ([0, 2], [5, 3], [11, -1], [1], [1, 2, 3], ["1", "2"], [True, 2], "1-2", 3):
        error(view(bad))


def test_view_of_an_empty_file(tools: BoxTools, root: Path) -> None:
    (root / "empty.txt").write_text("")
    assert "empty" in ok(run(tools.edit({"command": "view", "path": "empty.txt"})))


def test_view_truncates_at_max_result_chars(root: Path) -> None:
    tools = tools_on(root, max_result_chars=500)
    (root / "big.txt").write_text("".join(f"row {i:05d}\n" for i in range(1000)))
    content = ok(run(tools.edit({"command": "view", "path": "big.txt"})))
    assert "     1\trow 00000" in content
    assert "row 00999" not in content
    assert "truncated" in content and "view_range" in content
    assert len(content) <= 500 + 200  # the cut, plus one closing note


def test_view_lists_a_directory(tools: BoxTools, root: Path) -> None:
    (root / "src" / "pkg").mkdir(parents=True)
    (root / "src" / "pkg" / "deep").mkdir()
    (root / "src" / "pkg" / "deep" / "deeper.txt").write_text("")
    (root / "src" / "app.py").write_text("")
    (root / "README.md").write_text("")
    (root / ".git").mkdir()
    (root / ".git" / "config").write_text("")
    (root / "src" / ".cache").mkdir()
    (root / ".env").write_text("")
    for path in (".", root.as_posix(), f"{root}/"):
        content = ok(run(tools.edit({"command": "view", "path": path})))
        lines = content.splitlines()
        assert lines[0].startswith(
            f"Here are the files and directories up to 2 levels deep in {root}"
        )
        listed = set(lines[1:])
        assert {f"{root}/src", f"{root}/src/app.py", f"{root}/src/pkg", f"{root}/README.md"} <= (
            listed
        )
        assert not any("/." in line for line in lines[1:])  # no hidden items
        assert f"{root}/src/pkg/deep" not in listed  # three levels down
        assert lines[1:] == sorted(lines[1:])
    sub = ok(run(tools.edit({"command": "view", "path": "src"})))
    assert f"{root}/src/pkg/deep" in sub.splitlines()
    assert f"{root}/README.md" not in sub
    error(run(tools.edit({"command": "view", "path": "src", "view_range": [1, 2]})))


def test_view_of_a_missing_file(tools: BoxTools) -> None:
    error(run(tools.edit({"command": "view", "path": "nope.txt"})))


# --------------------------------------------------------------------------- editor: paths


@pytest.mark.parametrize(
    "path",
    ["/etc/passwd", "/tmp/x.txt", "/work2/x", "../x.txt", "a/../../x", "..", "/"],
)
def test_paths_outside_the_workspace_are_denied(tools: BoxTools, root: Path, path: str) -> None:
    before = sorted(p.name for p in root.parent.iterdir())
    for tool_input in (
        {"command": "view", "path": path},
        {"command": "create", "path": path, "file_text": "owned"},
        {"command": "str_replace", "path": path, "old_str": "a", "new_str": "b"},
        {"command": "insert", "path": path, "insert_line": 0, "insert_text": "x"},
    ):
        assert "outside the workspace" in error(run(tools.edit(tool_input)), denied=True)
    assert sorted(p.name for p in root.parent.iterdir()) == before


def test_a_symlink_out_is_refused_by_the_executor(tools: BoxTools, root: Path) -> None:
    outside = root.parent / "outside.txt"
    outside.write_text("secret")
    (root / "link").symlink_to(outside)
    for tool_input in (
        {"command": "view", "path": "link"},
        {"command": "create", "path": "link", "file_text": "owned"},
        {"command": "str_replace", "path": "link", "old_str": "secret", "new_str": "x"},
    ):
        content = error(run(tools.edit(tool_input)))
        assert "secret" not in content
    assert outside.read_text() == "secret"


# --------------------------------------------------------------------------- editor: writes


def test_create(tools: BoxTools, root: Path) -> None:
    content = ok(run(tools.edit({"command": "create", "path": "pkg/new.py", "file_text": "A=1\n"})))
    assert content == f"File created successfully at: {root}/pkg/new.py"
    assert (root / "pkg" / "new.py").read_text() == "A=1\n"
    ok(run(tools.edit({"command": "create", "path": f"{root}/pkg/new.py", "file_text": "B"})))
    assert (root / "pkg" / "new.py").read_text() == "B"
    error(run(tools.edit({"command": "create", "path": ".", "file_text": "x"})))
    error(run(tools.edit({"command": "create", "path": "x.txt"})))
    error(run(tools.edit({"command": "create", "path": "x.txt", "file_text": 5})))


def test_str_replace(tools: BoxTools, root: Path) -> None:
    target = root / "m.py"
    target.write_text("def f():\n    return 1\n\ndef g():\n    return 2\n")
    content = ok(
        run(
            tools.edit(
                {
                    "command": "str_replace",
                    "path": "m.py",
                    "old_str": "return 2",
                    "new_str": "return 22",
                }
            )
        )
    )
    assert target.read_text() == "def f():\n    return 1\n\ndef g():\n    return 22\n"
    assert content.startswith(f"The file {root}/m.py has been edited.")
    assert "     5\t    return 22" in content


def test_str_replace_needs_exactly_one_match(tools: BoxTools, root: Path) -> None:
    target = root / "m.txt"
    target.write_text("aaa\nbb bb\n")

    def replace(old: str) -> ToolOutcome:
        return run(tools.edit({"command": "str_replace", "path": "m.txt", "old_str": old}))

    assert "did not appear verbatim" in error(replace("zzz"))
    assert "2" in error(replace("bb"))
    assert "2" in error(replace("aa"))  # overlapping occurrences count
    error(replace(""))
    assert target.read_text() == "aaa\nbb bb\n"
    ok(replace("bb bb\n"))  # new_str defaults to "": a deletion
    assert target.read_text() == "aaa\n"


def test_edits_refuse_text_that_is_not_utf8(tools: BoxTools, root: Path) -> None:
    target = root / "bin.dat"
    target.write_bytes(b"ok\n\xff\xfe broken\n")
    for tool_input in (
        {"command": "str_replace", "path": "bin.dat", "old_str": "ok", "new_str": "no"},
        {"command": "insert", "path": "bin.dat", "insert_line": 0, "insert_text": "x"},
    ):
        assert "bash" in error(run(tools.edit(tool_input)))
    assert target.read_bytes() == b"ok\n\xff\xfe broken\n"


def test_insert(tools: BoxTools, root: Path) -> None:
    target = root / "i.txt"
    target.write_text("a\nb\n")

    def insert(line: object, value: str) -> ToolOutcome:
        return run(
            tools.edit(
                {"command": "insert", "path": "i.txt", "insert_line": line, "insert_text": value}
            )
        )

    assert ok(insert(0, "start")).startswith(f"The file {root}/i.txt has been edited.")
    assert target.read_text() == "start\na\nb\n"
    ok(insert(2, "middle\n"))
    assert target.read_text() == "start\na\nmiddle\nb\n"
    ok(insert(4, "end"))
    assert target.read_text() == "start\na\nmiddle\nb\nend\n"
    target.write_text("no newline")
    ok(insert(1, "after"))
    assert target.read_text() == "no newline\nafter\n"
    for bad in (-1, 3, True, "1", None):
        error(insert(bad, "x"))
    assert target.read_text() == "no newline\nafter\n"


@pytest.mark.parametrize(
    "tool_input",
    [
        {},
        {"path": "a.txt"},
        {"command": "undo_edit", "path": "a.txt"},
        {"command": "delete", "path": "a.txt"},
        {"command": "view"},
        {"command": "view", "path": ""},
        {"command": "view", "path": 7},
        {"command": "view", "path": "a.txt\x00"},
        {"command": "view", "path": "a.txt", "file_text": "x"},
        {"command": "create", "path": "a.txt", "file_text": "x", "old_str": "y"},
        {"command": "str_replace", "path": "a.txt", "old_str": "a", "insert_line": 1},
        {"command": "insert", "path": "a.txt", "insert_line": 0},
        {"command": "insert", "path": "a.txt", "insert_text": "x"},
        {"command": ["view"], "path": "a.txt"},
        "view a.txt",
    ],
)
def test_malformed_editor_input_changes_nothing(
    tools: BoxTools, root: Path, tool_input: object
) -> None:
    (root / "a.txt").write_text("a\n")
    content = error(run(tools.call(EDITOR, tool_input)))
    assert content
    assert sorted(p.name for p in root.iterdir()) == ["a.txt"]
    assert (root / "a.txt").read_text() == "a\n"


def test_unknown_commands_list_the_four(tools: BoxTools) -> None:
    content = error(run(tools.edit({"command": "undo_edit", "path": "a.txt"})))
    for name in ("view", "create", "str_replace", "insert"):
        assert name in content


def test_unknown_keys_are_named(tools: BoxTools, root: Path) -> None:
    (root / "a.txt").write_text("a\n")
    content = error(run(tools.edit({"command": "view", "path": "a.txt", "file_text": "x"})))
    assert "file_text" in content
