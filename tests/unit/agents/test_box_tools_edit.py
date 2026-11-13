"""Unit tests for the editor's `create`, `str_replace` and `insert` in box_tools."""

from __future__ import annotations

from pathlib import Path

import pytest

from local_factory.agents.box_tools import BoxTools, ToolOutcome
from local_factory.agents.local import LocalExecutor
from local_factory.agents.tools import ToolError
from tests.unit.agents.test_box_tools import MemoryExecutor, err, good, make, run


def create(tools: BoxTools, path: str, text: object) -> ToolOutcome:
    return run(tools.edit({"command": "create", "path": path, "file_text": text}))


def replace(tools: BoxTools, path: str, old: object, *new: object) -> ToolOutcome:
    extra = {"new_str": new[0]} if new else {}
    return run(tools.edit({"command": "str_replace", "path": path, "old_str": old, **extra}))


def insert(tools: BoxTools, path: str, line: object, text: object) -> ToolOutcome:
    args = {"command": "insert", "path": path, "insert_line": line, "insert_text": text}
    return run(tools.edit(args))


def numbered(first: int, *lines: str) -> str:
    return "\n".join(f"{n:6}\t{line}" for n, line in enumerate(lines, first))


def twenty() -> str:
    return "".join(f"line {i}\n" for i in range(1, 21))


# --------------------------------------------------------------------------- create


def test_create_writes_once_through_the_executor() -> None:
    executor = MemoryExecutor()
    content = good(create(make(executor), "/work/pkg/new.py", "A = 1\n"))
    assert content == "File created successfully at: /work/pkg/new.py"
    assert executor.calls == [("write_file", ("pkg/new.py", "A = 1\n"))]


def test_create_overwrites_and_accepts_an_empty_file() -> None:
    executor = MemoryExecutor({"a.txt": "old"})
    assert good(create(make(executor), "a.txt", "")) == "File created successfully at: /work/a.txt"
    assert executor.files["a.txt"] == ""


def test_create_keeps_text_exactly() -> None:
    executor = MemoryExecutor()
    text = "tab\t\r\n\x00 \N{LINE SEPARATOR} é中\U0001f600 no newline"
    good(create(make(executor), "t.txt", text))
    assert executor.files["t.txt"] == text


@pytest.mark.parametrize("root", [".", "/work", "/work/", "a/.."])
def test_create_on_the_root_is_an_error(root: str) -> None:
    executor = MemoryExecutor()
    content = err(create(make(executor), root, "x"))
    assert content == "Error: /work is the workspace directory; give the path of a file"
    assert executor.calls == []


@pytest.mark.parametrize(
    ("args", "needle"),
    [
        ({"command": "create", "path": "a"}, "missing required key(s): file_text"),
        ({"command": "create", "file_text": "x"}, "missing required key(s): path"),
        ({"command": "create", "path": "a", "file_text": 5}, "file_text must be a string, got int"),
        ({"command": "create", "path": "a", "file_text": None}, "file_text must be a string"),
        ({"command": "create", "path": "a", "file_text": "\ud83d"}, "UTF-8"),
        ({"command": "create", "path": "a", "file_text": "x", "old_str": "y"}, "'old_str'"),
        ({"command": "create", "path": "a", "file_text": "x", "new_str": ""}, "'new_str'"),
        ({"command": "create", "path": "a", "file_text": "x", "view_range": [1, 2]}, "view_range"),
    ],
)
def test_malformed_create_input_writes_nothing(args: dict[str, object], needle: str) -> None:
    executor = MemoryExecutor()
    assert needle in err(run(make(executor).edit(args)))
    assert executor.calls == []


def test_create_over_a_directory_is_the_executors_error(tmp_path: Path) -> None:
    root = tmp_path / "work"
    (root / "d").mkdir(parents=True)
    tools = BoxTools(LocalExecutor(root), root=str(root), command_timeout_s=5, max_result_chars=500)
    assert err(create(tools, "d", "x")) == "Error: is a directory; only files can be written"
    assert (root / "d").is_dir()


# --------------------------------------------------------------------------- str_replace


def test_a_replacement_reads_then_writes_once_and_shows_the_edited_lines() -> None:
    executor = MemoryExecutor({"m.py": "def f():\n    return 1\n\ndef g():\n    return 2\n"})
    content = good(replace(make(executor), "m.py", "return 2", "return 22"))
    assert content == (
        "The file /work/m.py has been edited.\n"
        + numbered(1, "def f():", "    return 1", "", "def g():", "    return 22")
    )
    assert executor.calls == [
        ("read_file", "m.py"),
        ("write_file", ("m.py", "def f():\n    return 1\n\ndef g():\n    return 22\n")),
    ]


def test_the_snippet_has_four_lines_of_context_either_side() -> None:
    executor = MemoryExecutor({"f.txt": twenty()})
    content = good(replace(make(executor), "f.txt", "line 10\n", "ten\n"))
    shown = content.splitlines()
    assert shown[0] == "The file /work/f.txt has been edited."
    assert shown[1:] == numbered(6, "line 6", "line 7", "line 8", "line 9", "ten").splitlines() + (
        numbered(11, "line 11", "line 12", "line 13", "line 14").splitlines()
    )


def test_the_snippet_covers_a_multi_line_replacement_and_its_context() -> None:
    executor = MemoryExecutor({"f.txt": twenty()})
    content = good(replace(make(executor), "f.txt", "line 10\n", "a\nb\nc\n"))
    lines = ["line 6", "line 7", "line 8", "line 9", "a", "b", "c"]
    lines += ["line 11", "line 12", "line 13", "line 14"]
    assert content.splitlines()[1:] == numbered(6, *lines).splitlines()


def test_the_snippet_is_clamped_at_both_ends_of_the_file() -> None:
    executor = MemoryExecutor({"f.txt": "a\nb\nc\n"})
    tools = make(executor)
    assert good(replace(tools, "f.txt", "a", "A")).splitlines()[1:] == [
        "     1\tA",
        "     2\tb",
        "     3\tc",
    ]
    assert good(replace(tools, "f.txt", "c", "C")).splitlines()[1:] == [
        "     1\tA",
        "     2\tb",
        "     3\tC",
    ]


def test_a_replacement_inside_a_line_shows_that_line() -> None:
    executor = MemoryExecutor({"f.txt": "alpha beta gamma\nnext\n"})
    content = good(replace(make(executor), "f.txt", "beta", "BETA"))
    assert content.splitlines()[1:] == ["     1\talpha BETA gamma", "     2\tnext"]


def test_new_str_defaults_to_a_deletion() -> None:
    executor = MemoryExecutor({"f.txt": "keep\ndrop\nkeep too\n"})
    content = good(replace(make(executor), "f.txt", "drop\n"))
    assert executor.files["f.txt"] == "keep\nkeep too\n"
    assert content.splitlines()[1:] == ["     1\tkeep", "     2\tkeep too"]


def test_an_explicit_empty_new_str_is_the_same_deletion() -> None:
    executor = MemoryExecutor({"f.txt": "keep\ndrop\n"})
    good(replace(make(executor), "f.txt", "drop\n", ""))
    assert executor.files["f.txt"] == "keep\n"


def test_deleting_the_whole_file_content_leaves_only_the_confirmation() -> None:
    executor = MemoryExecutor({"f.txt": "only\n"})
    assert (
        good(replace(make(executor), "f.txt", "only\n")) == "The file /work/f.txt has been edited."
    )
    assert executor.files["f.txt"] == ""


def test_old_str_is_matched_verbatim_including_whitespace_and_case() -> None:
    executor = MemoryExecutor({"f.txt": "  Name = 1\n"})
    tools = make(executor)
    for miss in ("name = 1", "Name  = 1", "Name = 1 ", "\tName"):
        assert "did not appear verbatim in /work/f.txt" in err(replace(tools, "f.txt", miss, "x"))
    good(replace(tools, "f.txt", "  Name = 1\n", "x"))
    assert [op for op, _ in executor.calls].count("write_file") == 1


def test_no_match_changes_nothing() -> None:
    executor = MemoryExecutor({"f.txt": "aaa\n"})
    content = err(replace(make(executor), "f.txt", "zzz", "y"))
    assert content.startswith("Error: No replacement was performed")
    assert [op for op, _ in executor.calls] == ["read_file"]


def test_several_matches_are_counted_including_overlapping_ones() -> None:
    executor = MemoryExecutor({"f.txt": "aaaa\nxx xx xx\n"})
    tools = make(executor)
    assert "appears 3 times in /work/f.txt" in err(replace(tools, "f.txt", "aa", "b"))
    assert "appears 3 times" in err(replace(tools, "f.txt", "xx", "y"))
    assert "more surrounding context" in err(replace(tools, "f.txt", "xx", "y"))
    assert [op for op, _ in executor.calls].count("write_file") == 0


def test_counting_stops_at_a_cap_and_says_so() -> None:
    executor = MemoryExecutor({"f.txt": "a" * 5000})
    assert "appears at least 1000 times" in err(replace(make(executor), "f.txt", "a", "b"))
    executor.files["f.txt"] = "a" * 999 + "b"
    assert "appears 999 times" in err(replace(make(executor), "f.txt", "a", "c"))


def test_the_replacement_is_applied_once_at_the_only_match() -> None:
    executor = MemoryExecutor({"f.txt": "a\nb\nc\n"})
    good(replace(make(executor), "f.txt", "b\nc", "B\nC\nD"))
    assert executor.files["f.txt"] == "a\nB\nC\nD\n"


def test_new_str_may_contain_what_old_str_matched() -> None:
    executor = MemoryExecutor({"f.txt": "x = 1\n"})
    good(replace(make(executor), "f.txt", "x", "xx"))
    assert executor.files["f.txt"] == "xx = 1\n"


@pytest.mark.parametrize("old", ["", None, 5, ["a"]])
def test_old_str_must_be_a_non_empty_string(old: object) -> None:
    executor = MemoryExecutor({"f.txt": "a\n"})
    assert "old_str" in err(replace(make(executor), "f.txt", old, "b"))
    assert executor.calls == []


@pytest.mark.parametrize("new", [None, 5, ["a"], "\ud800"])
def test_new_str_must_be_a_string_when_given(new: object) -> None:
    executor = MemoryExecutor({"f.txt": "a\n"})
    assert "new_str" in err(replace(make(executor), "f.txt", "a", new))
    assert executor.calls == []


def test_str_replace_on_the_root_is_an_error() -> None:
    executor = MemoryExecutor()
    assert "give the path of a file" in err(replace(make(executor), ".", "a", "b"))
    assert executor.calls == []


@pytest.mark.parametrize(
    "args",
    [
        {"command": "str_replace", "path": "f.txt"},
        {"command": "str_replace", "old_str": "a"},
        {"command": "str_replace", "path": "f.txt", "old_str": "a", "insert_line": 1},
        {"command": "str_replace", "path": "f.txt", "old_str": "a", "file_text": "x"},
    ],
)
def test_malformed_str_replace_input_runs_nothing(args: dict[str, object]) -> None:
    executor = MemoryExecutor({"f.txt": "a\n"})
    err(run(make(executor).edit(args)))
    assert executor.calls == []


def test_str_replace_refuses_files_that_contain_the_replacement_character() -> None:
    executor = MemoryExecutor({"bin": "ok �\n", "fine": "ok\n"})
    tools = make(executor)
    content = err(replace(tools, "bin", "ok", "no"))
    assert "/work/bin is not valid UTF-8" in content and "use bash" in content
    assert [op for op, _ in executor.calls] == ["read_file"]
    good(replace(tools, "fine", "ok", "�"))  # only the file's text is checked
    assert executor.files["fine"] == "�\n"


def test_str_replace_on_a_missing_file_is_an_error() -> None:
    executor = MemoryExecutor()
    assert err(replace(make(executor), "gone.txt", "a", "b")) == "Error: no such file"
    assert executor.calls == [("read_file", "gone.txt")]


def test_str_replace_on_a_directory_is_the_executors_error() -> None:
    executor = MemoryExecutor(dirs=["d"])
    assert "is a directory" in err(replace(make(executor), "d", "a", "b"))
    assert [op for op, _ in executor.calls] == ["read_file"]  # no listing for a write


def test_a_failing_write_is_an_error_after_one_read() -> None:
    executor = MemoryExecutor({"f.txt": "a\n"}, fail={"write_file": ToolError("disk is full")})
    assert err(replace(make(executor), "f.txt", "a", "b")) == "Error: disk is full"
    assert [op for op, _ in executor.calls] == ["read_file", "write_file"]


def test_a_long_snippet_is_clipped_to_the_limit() -> None:
    executor = MemoryExecutor({"f.txt": "a\n"})
    content = good(replace(make(executor, limit=300), "f.txt", "a", "b" * 5000))
    assert len(content) <= 300
    assert content.startswith("The file /work/f.txt has been edited.")
    assert "characters omitted" in content


def test_other_executor_failures_propagate_from_the_editor() -> None:
    class BoxGoneError(Exception):
        pass

    executor = MemoryExecutor({"f.txt": "a\n"}, fail={"read_file": BoxGoneError("docker")})
    for tool_input in (
        {"command": "str_replace", "path": "f.txt", "old_str": "a"},
        {"command": "insert", "path": "f.txt", "insert_line": 0, "insert_text": "x"},
        {"command": "view", "path": "f.txt"},
    ):
        with pytest.raises(BoxGoneError):
            run(make(executor).edit(tool_input))


# --------------------------------------------------------------------------- insert


@pytest.mark.parametrize(
    ("before", "line", "text", "after"),
    [
        ("a\nb\n", 0, "start", "start\na\nb\n"),
        ("a\nb\n", 1, "mid", "a\nmid\nb\n"),
        ("a\nb\n", 2, "end", "a\nb\nend\n"),
        ("a\nb\n", 1, "mid\n", "a\nmid\nb\n"),
        ("a\nb\n", 1, "two\nlines", "a\ntwo\nlines\nb\n"),
        ("a\nb\n", 1, "two\nlines\n", "a\ntwo\nlines\nb\n"),
        ("a\nb\n", 1, "", "a\n\nb\n"),
        ("a\nb\n", 1, "\n", "a\n\nb\n"),
        ("a\nb\n", 0, "\n\n", "\n\na\nb\n"),
        ("a\nb", 0, "x", "x\na\nb"),
        ("a\nb", 1, "x", "a\nx\nb"),
        ("a\nb", 2, "x", "a\nb\nx\n"),
        ("no newline", 1, "after", "no newline\nafter\n"),
        ("", 0, "first", "first\n"),
        ("", 0, "first\n", "first\n"),
        ("\n", 1, "y", "\ny\n"),
        ("\n", 0, "y", "y\n\n"),
        ("a\r\nb\r\n", 1, "x", "a\r\nx\nb\r\n"),
    ],
)
def test_insert_places_text_after_the_given_line(
    before: str, line: int, text: str, after: str
) -> None:
    executor = MemoryExecutor({"i.txt": before})
    content = good(insert(make(executor), "i.txt", line, text))
    assert content.startswith("The file /work/i.txt has been edited.")
    assert executor.files["i.txt"] == after
    assert executor.calls == [("read_file", "i.txt"), ("write_file", ("i.txt", after))]


def test_insert_shows_the_inserted_lines_with_context() -> None:
    executor = MemoryExecutor({"f.txt": twenty()})
    content = good(insert(make(executor), "f.txt", 10, "new a\nnew b"))
    lines = ["line 7", "line 8", "line 9", "line 10", "new a", "new b"]
    lines += ["line 11", "line 12", "line 13", "line 14"]
    assert content.splitlines()[1:] == numbered(7, *lines).splitlines()


def test_insert_at_the_start_and_end_clamp_the_snippet() -> None:
    executor = MemoryExecutor({"f.txt": "a\nb\n"})
    tools = make(executor)
    assert (
        good(insert(tools, "f.txt", 0, "z")).splitlines()[1:]
        == numbered(1, "z", "a", "b").splitlines()
    )
    assert (
        good(insert(tools, "f.txt", 3, "y")).splitlines()[1:]
        == numbered(1, "z", "a", "b", "y").splitlines()
    )


@pytest.mark.parametrize("line", [-1, 3, 100, True, False, "1", None, 1.0, [1]])
def test_insert_line_must_be_an_integer_within_the_file(line: object) -> None:
    executor = MemoryExecutor({"i.txt": "a\nb\n"})
    content = err(insert(make(executor), "i.txt", line, "x"))
    assert "insert_line" in content
    assert [op for op, _ in executor.calls] in ([], ["read_file"])  # the range needs the file
    assert executor.files["i.txt"] == "a\nb\n"


def test_an_out_of_range_line_names_the_range() -> None:
    executor = MemoryExecutor({"i.txt": "a\nb\n"})
    content = err(insert(make(executor), "i.txt", 3, "x"))
    assert content == "Error: insert_line 3 is out of range; the file has 2 lines (0..2)"


def test_inserting_into_an_empty_file_needs_line_zero() -> None:
    executor = MemoryExecutor({"e.txt": ""})
    assert "the file has 0 lines (0..0)" in err(insert(make(executor), "e.txt", 1, "x"))


@pytest.mark.parametrize("text", [None, 5, ["x"], "\udc00"])
def test_insert_text_must_be_a_string(text: object) -> None:
    executor = MemoryExecutor({"i.txt": "a\n"})
    assert "insert_text" in err(insert(make(executor), "i.txt", 0, text))
    assert executor.calls == []


def test_insert_refuses_files_that_contain_the_replacement_character() -> None:
    executor = MemoryExecutor({"bin": "ok �\n"})
    assert "use bash" in err(insert(make(executor), "bin", 0, "x"))
    assert [op for op, _ in executor.calls] == ["read_file"]


def test_insert_on_the_root_or_a_missing_file_is_an_error() -> None:
    executor = MemoryExecutor()
    assert "give the path of a file" in err(insert(make(executor), "/work", 0, "x"))
    assert executor.calls == []
    assert err(insert(make(executor), "gone", 0, "x")) == "Error: no such file"


@pytest.mark.parametrize(
    "args",
    [
        {"command": "insert", "path": "i.txt", "insert_line": 0},
        {"command": "insert", "path": "i.txt", "insert_text": "x"},
        {"command": "insert", "insert_line": 0, "insert_text": "x"},
        {
            "command": "insert",
            "path": "i.txt",
            "insert_line": 0,
            "insert_text": "x",
            "new_str": "y",
        },
    ],
)
def test_malformed_insert_input_runs_nothing(args: dict[str, object]) -> None:
    executor = MemoryExecutor({"i.txt": "a\n"})
    err(run(make(executor).edit(args)))
    assert executor.calls == []


# --------------------------------------------------------------------------- the real executor


def test_edits_round_trip_through_the_local_executor(tmp_path: Path) -> None:
    root = tmp_path / "work"
    root.mkdir()
    tools = BoxTools(
        LocalExecutor(root), root=str(root), command_timeout_s=5, max_result_chars=2000
    )
    good(create(tools, "pkg/m.py", "A = 1\nB = 2\n"))
    good(replace(tools, "pkg/m.py", "B = 2", "B = 3"))
    good(insert(tools, f"{root}/pkg/m.py", 1, "# note"))
    assert (root / "pkg" / "m.py").read_text() == "A = 1\n# note\nB = 3\n"
    assert good(run(tools.edit({"command": "view", "path": "pkg/m.py"}))) == numbered(
        1, "A = 1", "# note", "B = 3"
    )
