"""Unit tests for the pure diff parsers in local_factory.gitops._diff."""

from __future__ import annotations

import pytest

from local_factory.gitops._diff import (
    BINARY_SNIFF_BYTES,
    RawChange,
    all_lines,
    is_binary,
    lines_in_ranges,
    parse_hunk_ranges,
    parse_raw_diff,
    split_lines,
)
from local_factory.models import AddedLine, ChangeKind

A = "a" * 40
B = "b" * 40
Z = "0" * 40


def raw(*records: bytes) -> bytes:
    return b"".join(records)


# --------------------------------------------------------------------------- raw diff


def test_parse_raw_diff_all_kinds() -> None:
    out = raw(
        f":000000 100644 {Z} {A} A\0".encode(),
        b"new.txt\0",
        f":100644 100755 {A} {B} M\0".encode(),
        b"run.sh\0",
        f":100644 000000 {A} {Z} D\0".encode(),
        b"gone.txt\0",
        f":100644 100644 {A} {B} R087\0".encode(),
        b"old name.txt\0",
        b"new name.txt\0",
        f":100644 120000 {A} {B} T\0".encode(),
        b"link\0",
    )
    changes = parse_raw_diff(out)
    assert [(c.kind, c.path, c.old_path) for c in changes] == [
        (ChangeKind.ADDED, b"new.txt", None),
        (ChangeKind.MODIFIED, b"run.sh", None),
        (ChangeKind.DELETED, b"gone.txt", None),
        (ChangeKind.RENAMED, b"new name.txt", b"old name.txt"),
        (ChangeKind.TYPE_CHANGED, b"link", None),
    ]
    assert (changes[1].old_mode, changes[1].new_mode) == ("100644", "100755")
    assert (changes[1].old_sha, changes[1].new_sha) == (A, B)


def test_parse_raw_diff_keeps_exact_path_bytes() -> None:
    weird = "dir/café \t\nname".encode() + b"\xff"
    changes = parse_raw_diff(f":000000 100644 {Z} {A} A\0".encode() + weird + b"\0")
    assert changes[0].path == weird


def test_parse_raw_diff_empty_output() -> None:
    assert parse_raw_diff(b"") == []


def test_parse_raw_diff_without_trailing_nul() -> None:
    [change] = parse_raw_diff(f":000000 100644 {Z} {A} A\0x.txt".encode())
    assert change.path == b"x.txt"


def test_parse_raw_diff_rejects_malformed_record() -> None:
    with pytest.raises(ValueError, match="unexpected raw diff record"):
        parse_raw_diff(b"garbage\0path\0")
    with pytest.raises(ValueError, match="unexpected raw diff record"):
        parse_raw_diff(b":100644 100644 short M\0path\0")


@pytest.mark.parametrize("status", ["C075", "U", "X"])
def test_parse_raw_diff_rejects_unexpected_status(status: str) -> None:
    with pytest.raises(ValueError, match="unexpected diff status"):
        parse_raw_diff(f":100644 100644 {A} {B} {status}\0p\0q\0".encode())


@pytest.mark.parametrize(
    ("kind", "new_mode", "expected"),
    [
        (ChangeKind.ADDED, "100644", True),
        (ChangeKind.MODIFIED, "120000", True),
        (ChangeKind.TYPE_CHANGED, "100755", True),
        (ChangeKind.DELETED, "000000", False),
        (ChangeKind.ADDED, "160000", False),
    ],
)
def test_has_new_blob(kind: ChangeKind, new_mode: str, expected: bool) -> None:
    change = RawChange(kind, "100644", new_mode, A, B, b"p")
    assert change.has_new_blob is expected


# --------------------------------------------------------------------------- hunks


def test_parse_hunk_ranges_multiple_hunks_and_implicit_counts() -> None:
    patch = b"\n".join(
        [
            b"diff --git a/x b/y",
            b"index 1111111..2222222 100644",
            b"--- a/x",
            b"+++ b/y",
            b"@@ -1,0 +2 @@ def f():",
            b"+added",
            b"@@ -5,2 +6,3 @@",
            b"-old",
            b"-old",
            b"+new",
            b"+new",
            b"+new",
            b"@@ -9 +10,0 @@",
            b"-deleted only",
            b"\\ No newline at end of file",
        ]
    )
    assert parse_hunk_ranges(patch) == [(2, 1), (6, 3), (10, 0)]


def test_parse_hunk_ranges_ignores_content_that_looks_like_headers() -> None:
    patch = b"@@ -1 +1 @@\n-@@ -9 +9 @@ not a header\n+++ b/also content\n"
    assert parse_hunk_ranges(patch) == [(1, 1)]


def test_parse_hunk_ranges_no_hunks() -> None:
    assert parse_hunk_ranges(b"") == []
    assert parse_hunk_ranges(b"Binary files a/x and b/y differ\n") == []


# --------------------------------------------------------------------------- lines


@pytest.mark.parametrize(
    ("content", "lines"),
    [
        (b"", []),
        (b"\n", [""]),
        (b"a\nb\n", ["a", "b"]),
        (b"a\nb", ["a", "b"]),  # no trailing newline: last line still counts
        (b"a\r\nb\r\n", ["a", "b"]),  # CRLF terminators are stripped
        (b"a\r\nb\r", ["a", "b\r"]),  # a final \r without \n is content
        (b"a\rb\n", ["a\rb"]),  # a lone \r inside a line is content
        (b"\n\n\n", ["", "", ""]),
        ("café ✓\n".encode(), ["café ✓"]),
        (b"bad \xff byte\n", ["bad � byte"]),
    ],
)
def test_split_lines(content: bytes, lines: list[str]) -> None:
    assert split_lines(content) == lines


def test_all_lines_numbers_from_one() -> None:
    assert all_lines(b"x\ny") == (AddedLine(line_no=1, text="x"), AddedLine(line_no=2, text="y"))
    assert all_lines(b"") == ()


def test_lines_in_ranges_picks_new_side_lines() -> None:
    content = b"l1\nl2\nl3\nl4\nl5\n"
    added = lines_in_ranges(content, [(2, 1), (4, 2), (3, 0)])
    assert [(ln.line_no, ln.text) for ln in added] == [(2, "l2"), (4, "l4"), (5, "l5")]


def test_lines_in_ranges_accepts_range_ending_on_last_line() -> None:
    assert [ln.line_no for ln in lines_in_ranges(b"a\nb", [(2, 1)])] == [2]


def test_lines_in_ranges_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match="beyond the end"):
        lines_in_ranges(b"a\nb\n", [(2, 2)])


# --------------------------------------------------------------------------- binary


def test_is_binary_sniff_window_boundary() -> None:
    assert BINARY_SNIFF_BYTES == 8000
    assert is_binary(b"x" * (BINARY_SNIFF_BYTES - 1) + b"\0")
    assert not is_binary(b"x" * BINARY_SNIFF_BYTES + b"\0")
    assert not is_binary(b"")
    assert not is_binary("café\n".encode())
