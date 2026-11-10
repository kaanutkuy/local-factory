"""Pure parsers for git's diff output and line bookkeeping. No git is invoked here."""

from __future__ import annotations

import re
from dataclasses import dataclass

from local_factory.models import AddedLine, ChangeKind

NULL_MODE = "000000"
# git's own heuristic (xdiff buffer_is_binary): a NUL byte within the first 8000 bytes.
BINARY_SNIFF_BYTES = 8000

_STATUS = {
    "A": ChangeKind.ADDED,
    "M": ChangeKind.MODIFIED,
    "D": ChangeKind.DELETED,
    "R": ChangeKind.RENAMED,
    "T": ChangeKind.TYPE_CHANGED,
}
_HUNK = re.compile(rb"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


@dataclass(frozen=True)
class RawChange:
    """One entry of ``git diff-tree -r -z -M`` raw output. Paths are the exact bytes."""

    kind: ChangeKind
    old_mode: str
    new_mode: str
    old_sha: str
    new_sha: str
    path: bytes  # new path; the old path for deletions
    old_path: bytes | None = None  # renames only

    @property
    def has_new_blob(self) -> bool:
        """True if the new side is a blob in the repository (not a deletion or gitlink)."""
        return self.kind is not ChangeKind.DELETED and self.new_mode != "160000"


def parse_raw_diff(output: bytes) -> list[RawChange]:
    """Parse ``git diff-tree -r -z -M --raw`` output (full object names)."""
    tokens = output.split(b"\0")
    if tokens and tokens[-1] == b"":
        tokens.pop()
    changes: list[RawChange] = []
    index = 0
    while index < len(tokens):
        meta = tokens[index]
        fields = meta[1:].decode("ascii", "replace").split(" ")
        if not meta.startswith(b":") or len(fields) != 5:
            raise ValueError(f"unexpected raw diff record: {meta!r}")
        old_mode, new_mode, old_sha, new_sha, status = fields
        kind = _STATUS.get(status[:1])
        if kind is None:
            raise ValueError(f"unexpected diff status {status!r}")
        if kind is ChangeKind.RENAMED:
            old_path, path = tokens[index + 1], tokens[index + 2]
            index += 3
        else:
            old_path, path = None, tokens[index + 1]
            index += 2
        changes.append(RawChange(kind, old_mode, new_mode, old_sha, new_sha, path, old_path))
    return changes


def parse_hunk_ranges(patch: bytes) -> list[tuple[int, int]]:
    """``(new_start, new_count)`` of every hunk in ``git diff -U0`` output, in order.

    Only hunk headers are read: with ``-U0`` every content line starts with ``+``, ``-`` or
    ``\\``, so a line starting with ``@@`` can only be a header.
    """
    ranges: list[tuple[int, int]] = []
    for line in patch.split(b"\n"):
        match = _HUNK.match(line)
        if match:
            count = 1 if match[2] is None else int(match[2])
            ranges.append((int(match[1]), count))
    return ranges


def is_binary(content: bytes) -> bool:
    return b"\0" in content[:BINARY_SNIFF_BYTES]


def split_lines(content: bytes) -> list[str]:
    """Lines as git numbers them, without their terminator (``\\n`` or ``\\r\\n``).

    A final line without a newline is still a line. Text is decoded as UTF-8 with replacement.
    """
    parts = content.split(b"\n")
    last = parts.pop()  # b"" when content ends with a newline
    lines = [part.removesuffix(b"\r") for part in parts]
    if last:
        lines.append(last)
    return [line.decode("utf-8", "replace") for line in lines]


def all_lines(content: bytes) -> tuple[AddedLine, ...]:
    return tuple(
        AddedLine(line_no=number, text=text)
        for number, text in enumerate(split_lines(content), start=1)
    )


def lines_in_ranges(content: bytes, ranges: list[tuple[int, int]]) -> tuple[AddedLine, ...]:
    """The new file's lines covered by the hunk ranges (from :func:`parse_hunk_ranges`)."""
    lines = split_lines(content)
    added: list[AddedLine] = []
    for start, count in ranges:
        if start + count - 1 > len(lines):
            raise ValueError(f"hunk +{start},{count} is beyond the end of a {len(lines)}-line file")
        added.extend(
            AddedLine(line_no=number, text=lines[number - 1])
            for number in range(start, start + count)
        )
    return tuple(added)
