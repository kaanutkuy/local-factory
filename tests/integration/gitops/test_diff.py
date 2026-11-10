"""BareRepo.diff edge cases against real git histories."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest

from local_factory.gitops import bare as bare_module
from local_factory.gitops._git import GitError
from local_factory.gitops.bare import BareRepo
from local_factory.models import (
    MODE_EXECUTABLE,
    MODE_GITLINK,
    MODE_REGULAR,
    MODE_SYMLINK,
    ChangedFile,
    ChangeKind,
)

from .conftest import Git, write

pytestmark = pytest.mark.integration


class History:
    """A normal repository whose snapshots are fetched into a bare factory repo."""

    def __init__(self, git: Git, root: Path) -> None:
        self.git = git
        self.work = root / "hist"
        self.work.mkdir()
        git("init", "-q", cwd=self.work)
        self.repo = BareRepo.init(root / "hist.git")

    def snap(
        self, files: Mapping[str, str | bytes] | None = None, *, remove: tuple[str, ...] = ()
    ) -> str:
        write(self.work, files or {})
        for rel in remove:
            (self.work / rel).unlink()
        self.git("add", "-A", cwd=self.work)
        self.git("commit", "-q", "--allow-empty", "-m", "snap", cwd=self.work)
        self.repo.fetch_from(self.work, ["+refs/heads/main:refs/heads/main"])
        return self.repo.rev_parse("refs/heads/main")

    def diff(self, base: str, head: str) -> dict[str, ChangedFile]:
        changes = self.repo.diff(base, head)
        assert [c.path for c in changes] == sorted(c.path for c in changes)
        return {c.path: c for c in changes}


@pytest.fixture
def hist(git: Git, tmp_path: Path) -> History:
    return History(git, tmp_path)


def lines(change: ChangedFile) -> list[tuple[int, str]]:
    return [(ln.line_no, ln.text) for ln in change.added_lines]


def test_identical_commits_have_no_changes(hist: History) -> None:
    a = hist.snap({"x.txt": "x\n"})
    b = hist.snap()
    assert hist.repo.diff(a, b) == [] and hist.repo.diff(a, a) == []


def test_multiple_hunks_in_one_file(hist: History) -> None:
    original = "".join(f"line {i}\n" for i in range(1, 11))
    a = hist.snap({"f.txt": original})
    edited = original.replace("line 2\n", "LINE 2\n").replace("line 5\n", "line 5\ninserted\n")
    edited = edited.replace("line 9\n", "")
    b = hist.snap({"f.txt": edited + "tail\n"})
    change = hist.diff(a, b)["f.txt"]
    assert change.kind is ChangeKind.MODIFIED
    assert lines(change) == [(2, "LINE 2"), (6, "inserted"), (11, "tail")]


def test_pure_deletion_hunks_add_nothing(hist: History) -> None:
    a = hist.snap({"f.txt": "a\nb\nc\n"})
    b = hist.snap({"f.txt": "a\nc\n"})
    change = hist.diff(a, b)["f.txt"]
    assert (change.kind, change.added_lines, change.size) == (ChangeKind.MODIFIED, (), 4)


def test_missing_trailing_newline(hist: History) -> None:
    a = hist.snap({"f.txt": "a\nb"})
    b = hist.snap({"f.txt": "a\nb\nc"})
    # "b" gained a newline, so git reports it as changed too.
    assert lines(hist.diff(a, b)["f.txt"]) == [(2, "b"), (3, "c")]


def test_crlf_lines_are_stripped_of_their_terminator(hist: History) -> None:
    a = hist.snap({"f.txt": b"one\r\ntwo\r\n"})
    b = hist.snap({"f.txt": b"one\r\nnew\r\ntwo\r\n"})
    assert lines(hist.diff(a, b)["f.txt"]) == [(2, "new")]


def test_unicode_and_unusual_paths(hist: History) -> None:
    a = hist.snap({"x": "x\n"})
    names = ["café/ünï.txt", "with space.txt", "tab\tand\nnewline.txt", "-dash"]
    b = hist.snap(dict.fromkeys(names, "hi\n"))
    changes = hist.diff(a, b)
    assert sorted(changes) == sorted(names)
    assert all(lines(changes[n]) == [(1, "hi")] for n in names)


def test_non_utf8_path_is_displayed_with_replacement(hist: History, git: Git) -> None:
    a = hist.snap({"x": "x\n"})
    blob = git.blob(b"latin\n", cwd=hist.work)
    entries = [*git.tree_entries(a, cwd=hist.work), ("100644", b"caf\xe9.txt", blob)]
    tree = git.tree(entries, cwd=hist.work)
    commit = git("commit-tree", tree, "-p", a, "-m", "latin-1 name", cwd=hist.work)
    git("update-ref", "refs/heads/main", commit, cwd=hist.work)
    hist.repo.fetch_from(hist.work, ["+refs/heads/main:refs/heads/main"])
    [change] = hist.repo.diff(a, commit)
    assert change.path == "caf�.txt" and lines(change) == [(1, "latin")]
    change.model_dump_json()  # serializable


def test_empty_file(hist: History) -> None:
    a = hist.snap({"x": "x\n"})
    b = hist.snap({"empty.txt": ""})
    change = hist.diff(a, b)["empty.txt"]
    assert (change.kind, change.size, change.is_binary, change.added_lines) == (
        ChangeKind.ADDED,
        0,
        False,
        (),
    )
    assert (change.mode, change.old_mode) == (MODE_REGULAR, None)


def test_executable_bit_change_is_a_mode_change(hist: History) -> None:
    a = hist.snap({"run.sh": "#!/bin/sh\necho hi\n"})
    (hist.work / "run.sh").chmod(0o755)
    b = hist.snap()
    change = hist.diff(a, b)["run.sh"]
    assert (change.kind, change.old_mode, change.mode) == (
        ChangeKind.MODIFIED,
        MODE_REGULAR,
        MODE_EXECUTABLE,
    )
    assert (change.added_lines, change.size) == ((), 18)

    (hist.work / "run.sh").chmod(0o644)
    c = hist.snap({"run.sh": "#!/bin/sh\necho bye\n"})
    change = hist.diff(b, c)["run.sh"]
    assert (change.old_mode, change.mode, lines(change)) == (
        MODE_EXECUTABLE,
        MODE_REGULAR,
        [(2, "echo bye")],
    )


def test_rename_with_and_without_edits(hist: History, git: Git) -> None:
    body = "".join(f"line {i}\n" for i in range(20))
    a = hist.snap({"old.py": body, "keep.py": body + "keep\n"})
    git("mv", "old.py", "new.py", cwd=hist.work)
    git("mv", "keep.py", "moved.py", cwd=hist.work)
    b = hist.snap({"new.py": body + "# noqa\n"})
    changes = hist.diff(a, b)
    edited, pure = changes["new.py"], changes["moved.py"]
    assert (edited.kind, edited.old_path, lines(edited)) == (
        ChangeKind.RENAMED,
        "old.py",
        [(21, "# noqa")],
    )
    assert (pure.kind, pure.old_path, pure.added_lines) == (ChangeKind.RENAMED, "keep.py", ())
    assert pure.size == len(body) + 5


def test_symlink_changes(hist: History) -> None:
    a = hist.snap({"target.txt": "t\n", "other.txt": "o\n", "f": "regular\n"})
    (hist.work / "link").symlink_to("target.txt")
    b = hist.snap()
    added = hist.diff(a, b)["link"]
    assert (added.kind, added.mode, added.symlink_target, added.size, added.added_lines) == (
        ChangeKind.ADDED,
        MODE_SYMLINK,
        "target.txt",
        10,
        (),
    )

    (hist.work / "link").unlink()
    (hist.work / "link").symlink_to("other.txt")
    (hist.work / "f").unlink()
    (hist.work / "f").symlink_to("target.txt")
    c = hist.snap()
    changes = hist.diff(b, c)
    retargeted, to_link = changes["link"], changes["f"]
    assert (retargeted.kind, retargeted.symlink_target) == (ChangeKind.MODIFIED, "other.txt")
    assert (to_link.kind, to_link.old_mode, to_link.mode, to_link.symlink_target) == (
        ChangeKind.TYPE_CHANGED,
        MODE_REGULAR,
        MODE_SYMLINK,
        "target.txt",
    )

    (hist.work / "f").unlink()
    d = hist.snap({"f": "now a file\n# noqa\n"}, remove=("link",))
    changes = hist.diff(c, d)
    to_file, deleted = changes["f"], changes["link"]
    # A symlink turned into a file: every line of the file is new content.
    assert (to_file.kind, to_file.symlink_target, lines(to_file)) == (
        ChangeKind.TYPE_CHANGED,
        None,
        [(1, "now a file"), (2, "# noqa")],
    )
    assert (deleted.kind, deleted.mode, deleted.old_mode, deleted.symlink_target) == (
        ChangeKind.DELETED,
        None,
        MODE_SYMLINK,
        None,
    )


def test_binary_detection(hist: History) -> None:
    a = hist.snap({"img.bin": b"\x89PNG\0\0data", "late.txt": "x\n"})
    late_nul = b"y" * 8000 + b"\0tail\n"
    b = hist.snap({"img.bin": b"\x89PNG\0\0other", "late.txt": b"x\n" + late_nul})
    changes = hist.diff(a, b)
    assert (changes["img.bin"].is_binary, changes["img.bin"].added_lines) == (True, ())
    late = changes["late.txt"]
    assert not late.is_binary  # git's heuristic only sniffs the first 8000 bytes
    assert lines(late) == [(2, late_nul[:-1].decode())]


def test_attributes_cannot_hide_content(hist: History) -> None:
    attributes = "* -diff binary\n*.py diff=evil\n[attr]x binary\n"
    a = hist.snap({".gitattributes": attributes, "src/app.py": "a = 1\n"})
    b = hist.snap({"src/app.py": "a = 1\nb = 2  # type: ignore\n"})
    change = hist.diff(a, b)["src/app.py"]
    assert not change.is_binary
    assert lines(change) == [(2, "b = 2  # type: ignore")]


def test_gitlink_entries(hist: History, git: Git) -> None:
    a = hist.snap({"x": "x\n"})
    git("update-index", "--add", "--cacheinfo", f"160000,{a},vendor/sub", cwd=hist.work)
    git("commit", "-q", "-m", "sub", cwd=hist.work)
    hist.repo.fetch_from(hist.work, ["+refs/heads/main:refs/heads/main"])
    b = hist.repo.rev_parse("refs/heads/main")
    [change] = hist.repo.diff(a, b)
    assert (change.path, change.mode, change.size, change.added_lines) == (
        "vendor/sub",
        MODE_GITLINK,
        None,
        (),
    )


def test_deleted_files_report_old_path_and_mode(hist: History) -> None:
    a = hist.snap({"gone/script.sh": "#!/bin/sh\n", "keep": "k\n"})
    (hist.work / "gone" / "script.sh").chmod(0o755)
    b = hist.snap()
    c = hist.snap(remove=("gone/script.sh",))
    [change] = hist.repo.diff(b, c)
    assert (change.path, change.kind, change.mode, change.old_mode, change.size) == (
        "gone/script.sh",
        ChangeKind.DELETED,
        None,
        MODE_EXECUTABLE,
        None,
    )
    assert hist.repo.diff(a, a) == []


def test_missing_hunks_fail_closed(hist: History, monkeypatch: pytest.MonkeyPatch) -> None:
    a = hist.snap({"f.txt": "a\n"})
    b = hist.snap({"f.txt": "b\n"})
    monkeypatch.setattr(bare_module, "parse_hunk_ranges", lambda _patch: [])
    with pytest.raises(GitError, match="no hunks"):
        hist.repo.diff(a, b)


def test_type_confused_entry_fails_closed(hist: History, git: Git) -> None:
    a = hist.snap({"x": "x\n"})
    bare = hist.repo.path  # objects written straight into the factory repo
    inner = git.tree([("100644", b"config", git.blob(b"pwn\n", cwd=bare))], cwd=bare)
    tree = git.tree([*git.tree_entries(a, cwd=bare), ("100644", b"weird", inner)], cwd=bare)
    commit = git("commit-tree", tree, "-p", a, "-m", "file mode pointing at a tree", cwd=bare)
    with pytest.raises(GitError, match="expected a blob"):
        hist.repo.diff(a, commit)
