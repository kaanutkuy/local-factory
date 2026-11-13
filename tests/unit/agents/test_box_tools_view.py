"""Unit tests for the editor's `view` and for path handling in local_factory.agents.box_tools."""

from __future__ import annotations

from pathlib import Path

import pytest

from local_factory.agents.box_tools import BoxTools, ToolOutcome
from local_factory.agents.local import LocalExecutor
from local_factory.agents.tools import CommandResult, ToolError
from tests.unit.agents.test_box_tools import MemoryExecutor, err, good, make, run

HEADER = "Here are the files and directories up to 2 levels deep in {}, excluding hidden items:"


def view(tools: BoxTools, path: str, **extra: object) -> ToolOutcome:
    return run(tools.edit({"command": "view", "path": path, **extra}))


def real(root: Path, *, limit: int = 1000) -> BoxTools:
    return BoxTools(
        LocalExecutor(root), root=root.as_posix(), command_timeout_s=30, max_result_chars=limit
    )


# --------------------------------------------------------------------------- view: files


def test_a_file_is_numbered_from_one_without_a_phantom_last_line() -> None:
    tools = make(MemoryExecutor({"a.txt": "one\n\nthree\n"}))
    assert good(view(tools, "a.txt")) == "     1\tone\n     2\t\n     3\tthree"


def test_a_file_without_a_final_newline_shows_its_last_line() -> None:
    tools = make(MemoryExecutor({"a.txt": "one\ntwo"}))
    assert good(view(tools, "a.txt")) == "     1\tone\n     2\ttwo"


def test_a_file_that_is_one_newline_is_one_empty_line() -> None:
    tools = make(MemoryExecutor({"a.txt": "\n"}))
    assert good(view(tools, "a.txt")) == "     1\t"


def test_only_newline_separates_lines() -> None:
    tools = make(MemoryExecutor({"a.txt": "a\rb\x0cc\N{LINE SEPARATOR}d\r\ne\n"}))
    assert good(view(tools, "a.txt")) == "     1\ta\rb\x0cc\N{LINE SEPARATOR}d\r\n     2\te"


def test_an_empty_file_says_so_with_its_path() -> None:
    tools = make(MemoryExecutor({"a.txt": ""}))
    assert good(view(tools, "a.txt")) == "The file /work/a.txt is empty."
    assert "is empty" in good(view(tools, "a.txt", view_range=[1, 5]))


def test_view_reads_through_the_executor_by_relative_path() -> None:
    executor = MemoryExecutor({"src/m.py": "x = 1\n"})
    for path in ("/work/src/m.py", "src/m.py", "./src/m.py", "src//m.py", "src/../src/m.py"):
        good(view(make(executor), path))
    assert executor.calls == [("read_file", "src/m.py")] * 5


def test_a_missing_file_is_an_error() -> None:
    assert err(view(make(), "nope.txt")) == "Error: no such file"


def test_a_read_error_that_is_not_a_directory_is_not_listed() -> None:
    executor = MemoryExecutor(fail={"read_file": ToolError("file is larger than the limit")})
    assert "larger than the limit" in err(view(make(executor), "big.bin"))
    assert [op for op, _ in executor.calls] == ["read_file"]


# --------------------------------------------------------------------------- view: view_range


def ten_lines() -> BoxTools:
    return make(MemoryExecutor({"f.txt": "".join(f"line {i}\n" for i in range(1, 11))}))


@pytest.mark.parametrize(
    ("view_range", "first", "last"),
    [
        ([1, 10], 1, 10),
        ([1, 1], 1, 1),
        ([3, 5], 3, 5),
        ([10, 10], 10, 10),
        ([4, -1], 4, 10),
        ([1, -1], 1, 10),
        ([9, 99], 9, 10),
        ([10, 10_000], 10, 10),
    ],
)
def test_view_range_selects_inclusive_lines(view_range: list[int], first: int, last: int) -> None:
    expected = "\n".join(f"{n:6}\tline {n}" for n in range(first, last + 1))
    assert good(view(ten_lines(), "f.txt", view_range=view_range)) == expected


@pytest.mark.parametrize(
    ("view_range", "needle"),
    [
        ([0, 2], "start"),
        ([-1, 2], "start"),
        ([5, 3], "start <= end"),
        ([5, -2], "start <= end"),
        ([5, 0], "start <= end"),
        ([1], "two integers"),
        ([], "two integers"),
        ([1, 2, 3], "two integers"),
        (["1", "2"], "two integers"),
        ([1.0, 2], "two integers"),
        ([True, 2], "two integers"),
        ([1, False], "two integers"),
        ((1, 2), "two integers"),
        ("1-2", "two integers"),
        (3, "two integers"),
        (None, "two integers"),
        ({"start": 1}, "two integers"),
        ([11, -1], "beyond the end of the file (10 lines)"),
        ([11, 12], "beyond the end"),
    ],
)
def test_a_bad_view_range_is_an_error(view_range: object, needle: str) -> None:
    assert needle in err(view(ten_lines(), "f.txt", view_range=view_range))


# --------------------------------------------------------------------------- view: truncation


def test_a_long_view_is_cut_at_a_whole_line_and_says_so() -> None:
    text = "".join(f"row {i:05d}\n" for i in range(1000))
    tools = make(MemoryExecutor({"big.txt": text}), limit=500)
    content = good(view(tools, "big.txt"))
    assert len(content) <= 500
    note = "[truncated: output cut at 500 characters; use view_range to view the rest]"
    assert content.endswith("\n" + note)
    body = content[: -len(note) - 1].splitlines()
    assert body[0] == "     1\trow 00000"
    assert body == [f"{n:6}\trow {n - 1:05d}" for n in range(1, len(body) + 1)]  # whole lines only
    assert 20 < len(body) < 1000


def test_a_view_range_is_cut_the_same_way() -> None:
    text = "".join(f"row {i:05d}\n" for i in range(1000))
    tools = make(MemoryExecutor({"big.txt": text}), limit=300)
    content = good(view(tools, "big.txt", view_range=[500, -1]))
    assert content.startswith("   500\trow 00499")
    assert "truncated" in content and "view_range" in content and len(content) <= 300


def test_a_single_huge_line_is_cut_plainly() -> None:
    tools = make(MemoryExecutor({"min.js": "x" * 5000}), limit=200)
    content = good(view(tools, "min.js"))
    assert len(content) <= 200
    assert content.startswith("     1\txxx")
    assert content.endswith("view_range to view the rest]")


def test_a_view_that_just_fits_is_not_cut() -> None:
    line = "     1\t" + "y" * 93  # exactly 100 characters
    content = good(view(make(MemoryExecutor({"a": "y" * 93}), limit=100), "a"))
    assert content == line and len(content) == 100


# --------------------------------------------------------------------------- view: directories


def listing(*lines: str, code: int = 0) -> CommandResult:
    return CommandResult(code, "".join(f"{line}\n" for line in lines))


def test_the_root_is_listed_with_one_find_and_never_read() -> None:
    executor = MemoryExecutor()
    for path in (".", "/work", "/work/", "./", "a/.."):
        executor.results = [listing(".", "./README.md", "./src", "./src/app.py", "./src/pkg")]
        content = good(view(make(executor), path))
        assert content.splitlines() == [
            HEADER.format("/work"),
            "/work",
            "/work/README.md",
            "/work/src",
            "/work/src/app.py",
            "/work/src/pkg",
        ]
    assert executor.calls == [("run_command", (["find", "-P", ".", "-maxdepth", "2"], 5.0))] * 5


def test_a_subdirectory_is_found_when_the_read_says_it_is_a_directory() -> None:
    executor = MemoryExecutor(
        dirs=["src"],
        results=[listing("./src", "./src/pkg", "./src/pkg/deep", "./src/app.py")],
    )
    content = good(view(make(executor), "/work/src/"))
    assert content.splitlines() == [
        HEADER.format("/work/src"),
        "/work/src",
        "/work/src/app.py",
        "/work/src/pkg",
        "/work/src/pkg/deep",
    ]
    assert executor.calls == [
        ("read_file", "src"),
        ("run_command", (["find", "-P", "./src", "-maxdepth", "2"], 5.0)),
    ]


def test_hidden_entries_are_dropped_at_any_depth_below_the_viewed_directory() -> None:
    executor = MemoryExecutor(
        results=[
            listing(
                ".",
                "./.git",
                "./.git/config",
                "./.env",
                "./a",
                "./a/.cache",
                "./a/.cache/x",
                "./a/b",
                "./a/..x",
                "./.a/b",
                "./b.d",
                "./..",
            )
        ]
    )
    shown = good(view(make(executor), ".")).splitlines()[1:]
    assert shown == ["/work", "/work/a", "/work/a/b", "/work/b.d"]


def test_a_hidden_directory_can_be_viewed_but_not_its_hidden_children() -> None:
    executor = MemoryExecutor(
        dirs=[".github"],
        results=[
            listing(
                "./.github", "./.github/workflows", "./.github/workflows/ci.yml", "./.github/.x"
            )
        ],
    )
    shown = good(view(make(executor), ".github")).splitlines()
    assert shown[1:] == [
        "/work/.github",
        "/work/.github/workflows",
        "/work/.github/workflows/ci.yml",
    ]
    assert executor.calls[1][1][0] == ["find", "-P", "./.github", "-maxdepth", "2"]


def test_a_path_that_looks_like_a_find_option_is_anchored() -> None:
    executor = MemoryExecutor(dirs=["-delete"], results=[listing("./-delete")])
    good(view(make(executor), "-delete"))
    assert executor.calls[1][1][0] == ["find", "-P", "./-delete", "-maxdepth", "2"]


def test_lines_that_are_not_entries_are_ignored_and_the_rest_is_sorted() -> None:
    executor = MemoryExecutor(
        results=[
            CommandResult(
                0, "find: warning\n./z\n./a\n\n./a/b\nsomething else\n.\n./m-n\n./m/n\n./m\n"
            )
        ]
    )
    shown = good(view(make(executor), ".")).splitlines()[1:]
    assert shown == sorted(shown)
    assert shown == [
        "/work",
        "/work/a",
        "/work/a/b",
        "/work/m",
        "/work/m-n",
        "/work/m/n",
        "/work/z",
    ]


def test_an_empty_directory_lists_itself() -> None:
    executor = MemoryExecutor(dirs=["empty"], results=[listing("./empty")])
    content = good(view(make(executor), "empty"))
    assert content.splitlines() == [HEADER.format("/work/empty"), "/work/empty"]


def test_a_long_listing_is_clipped_to_the_limit() -> None:
    names = [f"./file-{i:05d}.txt" for i in range(2000)]
    executor = MemoryExecutor(results=[listing(".", *names)])
    content = good(view(make(executor, limit=800), "."))
    assert len(content) <= 800
    assert content.startswith("Here are the files")
    assert "characters omitted" in content


@pytest.mark.parametrize("path", [".", "src"])
def test_view_range_on_a_directory_is_an_error_and_runs_nothing_more(path: str) -> None:
    executor = MemoryExecutor(dirs=["src"])
    content = err(view(make(executor), path, view_range=[1, 2]))
    assert "view_range cannot be used on a directory" in content
    assert not [call for call in executor.calls if call[0] == "run_command"]


def test_a_failing_find_is_an_error_with_its_output() -> None:
    executor = MemoryExecutor(
        results=[CommandResult(1, "find: './x': Permission denied\n./y\n")],
    )
    content = err(view(make(executor), "."))
    assert (
        content
        == "Error: could not list /work (exit code 1):\nfind: './x': Permission denied\n./y\n"
    )


def test_a_find_that_times_out_is_an_error() -> None:
    executor = MemoryExecutor(dirs=["d"], results=[CommandResult(None, "./d\n", timed_out=True)])
    assert "could not list /work/d (timed out)" in err(view(make(executor), "d"))


def test_a_find_that_cannot_run_is_an_error() -> None:
    executor = MemoryExecutor(fail={"run_command": ToolError("no such command")})
    assert err(view(make(executor), ".")) == "Error: no such command"


def test_the_real_find_agrees(tmp_path: Path) -> None:
    root = tmp_path / "work"
    (root / "a" / "b" / "c").mkdir(parents=True)
    (root / "a" / ".h").mkdir()
    (root / ".top").write_text("")
    (root / "a" / "f.txt").write_text("")
    (root / "link").symlink_to(root / "a")
    lines = good(view(real(root), ".")).splitlines()
    assert lines[0] == HEADER.format(root)
    assert lines[1:] == sorted(
        [f"{root}", f"{root}/a", f"{root}/a/b", f"{root}/a/f.txt", f"{root}/link"]
    )
    sub = good(view(real(root), "a")).splitlines()[1:]
    assert sub == [f"{root}/a", f"{root}/a/b", f"{root}/a/b/c", f"{root}/a/f.txt"]
    # a symlink to a directory lists as itself: -P never follows it
    assert good(view(real(root), "link")).splitlines()[1:] == [f"{root}/link"]


# --------------------------------------------------------------------------- paths


@pytest.mark.parametrize(
    "path",
    [
        "/etc/passwd",
        "/",
        "//work/a.txt",
        "/work2/a.txt",
        "/wor",
        "/work/../etc/passwd",
        "/work/a/../../etc",
        "/work/..",
        "..",
        "../a.txt",
        "a/../../a.txt",
        "./../a.txt",
        "a/b/../../../a.txt",
        "../work/a.txt",
    ],
)
def test_paths_outside_the_workspace_are_denied_before_the_executor_is_touched(path: str) -> None:
    executor = MemoryExecutor({"a.txt": "a\n"})
    tools = make(executor)
    for tool_input in (
        {"command": "view", "path": path},
        {"command": "create", "path": path, "file_text": "x"},
        {"command": "str_replace", "path": path, "old_str": "a"},
        {"command": "insert", "path": path, "insert_line": 0, "insert_text": "x"},
    ):
        content = err(run(tools.edit(tool_input)), denied=True)
        assert "outside the workspace (/work)" in content
    assert executor.calls == []


def test_a_denied_path_is_echoed_bounded() -> None:
    content = err(view(make(), "/etc/" + "x" * 5000), denied=True)
    assert len(content) < 250


@pytest.mark.parametrize(
    ("path", "relative"),
    [
        ("a.txt", "a.txt"),
        ("./a.txt", "a.txt"),
        ("/work/a.txt", "a.txt"),
        ("/work//a.txt", "a.txt"),
        ("/work/./a.txt", "a.txt"),
        ("/work/x/../a.txt", "a.txt"),
        ("x/y/../../a.txt", "a.txt"),
        ("..a.txt", "..a.txt"),
        ("a/..b/c", "a/..b/c"),
        ("/work/a.txt/", "a.txt"),
    ],
)
def test_paths_under_the_workspace_reach_the_executor_relative_and_normalized(
    path: str, relative: str
) -> None:
    executor = MemoryExecutor({relative: "x\n"})
    good(view(make(executor), path))
    assert executor.calls == [("read_file", relative)]


def test_a_root_with_a_dot_and_a_deeper_root_work() -> None:
    executor = MemoryExecutor({"m.py": "x\n"})
    tools = make(executor, root="/srv/job-1/work")
    assert good(view(tools, "/srv/job-1/work/m.py")).endswith("x")
    err(view(tools, "/srv/job-1/work2/m.py"), denied=True)
    err(view(tools, "/srv/job-1"), denied=True)


@pytest.mark.parametrize(
    ("args", "needle"),
    [
        ({"command": "view", "path": ""}, "path must not be empty"),
        ({"command": "view", "path": "a\x00b"}, "NUL"),
        ({"command": "view", "path": 7}, "path must be a string, got int"),
        ({"command": "view", "path": None}, "path must be a string"),
        ({"command": "view", "path": ["a"]}, "path must be a string, got list"),
        ({"command": "view", "path": "a\ud800"}, "UTF-8"),
        ({"command": "view"}, "missing required key(s): path"),
        ({"command": "view", "path": "a", "file_text": "x"}, "'file_text'"),
        ({"command": "view", "path": "a", "view_range": [1, 2], "x": 1, "y": 2}, "'x', 'y'"),
    ],
)
def test_malformed_view_input_runs_nothing(args: dict[str, object], needle: str) -> None:
    executor = MemoryExecutor({"a": "a\n"})
    assert needle in err(run(make(executor).edit(args)))
    assert executor.calls == []


@pytest.mark.parametrize(
    ("tool_input", "needle"),
    [
        ("view a", "JSON object"),
        (None, "NoneType"),
        ({}, "command must be one of: view, create, str_replace, insert (got None)"),
        ({"path": "a"}, "command must be one of"),
        ({"command": "undo_edit", "path": "a"}, "(got 'undo_edit')"),
        ({"command": "VIEW", "path": "a"}, "command must be one of"),
        ({"command": ["view"], "path": "a"}, "command must be one of"),
        ({"command": 1, "path": "a"}, "(got 1)"),
        ({1: "view"}, "keys must be strings"),
    ],
)
def test_a_bad_editor_command_lists_the_four(tool_input: object, needle: str) -> None:
    executor = MemoryExecutor()
    content = err(run(make(executor).edit(tool_input)))
    assert needle in content
    assert executor.calls == []
