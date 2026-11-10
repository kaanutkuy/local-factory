"""Host-side operations on factory-owned bare repositories.

Every method runs git with ``--git-dir`` set to this bare repository and no ``cwd`` (so git runs
in the neutral directory from :mod:`._git`); nothing here ever runs inside a worker directory.
Content-derived results (diffs, merges) are computed with ``--attr-source`` set to the empty tree,
so a ``.gitattributes`` file in worker-controlled content can never change how the factory sees
that content.
"""

from __future__ import annotations

import contextlib
import os
import stat
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from subprocess import CompletedProcess
from typing import Literal

from local_factory.models import (
    MODE_EXECUTABLE,
    MODE_GITLINK,
    MODE_REGULAR,
    MODE_SYMLINK,
    AddedLine,
    ChangedFile,
    ChangeKind,
)

from ._diff import (
    RawChange,
    all_lines,
    is_binary,
    lines_in_ranges,
    parse_hunk_ranges,
    parse_raw_diff,
)
from ._git import EMPTY_TREE, GitError, run_git
from ._paths import decode_path, display, unsafe_path_reason

HARDENED_CONFIG: tuple[tuple[str, str], ...] = (
    ("core.hooksPath", "/dev/null"),
    ("core.fsmonitor", "false"),
    ("fetch.fsckObjects", "true"),
    ("receive.fsckObjects", "true"),
    ("transfer.fsckObjects", "true"),
    ("gc.auto", "0"),
    ("maintenance.auto", "false"),
)
# Passed on the command line of every fetch into a factory repo, whatever its config file says.
FSCK_FLAGS: tuple[str, ...] = ("-c", "fetch.fsckObjects=true", "-c", "transfer.fsckObjects=true")
FETCH_OPTIONS: tuple[str, ...] = (
    "--quiet",
    "--no-tags",
    "--no-write-fetch-head",
    "--no-recurse-submodules",
    "--no-auto-maintenance",
)
NO_ATTRIBUTES = f"--attr-source={EMPTY_TREE}"
REGULAR_MODES = frozenset({MODE_REGULAR, MODE_EXECUTABLE})
EXPORT_MODES = frozenset({MODE_REGULAR, MODE_EXECUTABLE, MODE_SYMLINK, MODE_GITLINK})
_EXPORT_CHUNK_BYTES = 32 * 1024 * 1024
_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_FILE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC


def absolute(path: Path | str) -> Path:
    """Absolute path without resolving symlinks (keeps paths comparable to what callers pass)."""
    return Path(path).absolute()


def check_ref_name(ref: str) -> None:
    """Refs the factory manipulates must be full names under ``refs/``."""
    if not ref.startswith("refs/") or any(ch in ref for ch in "\0\n"):
        raise ValueError(f"not a full ref name under refs/: {ref!r}")


@dataclass(frozen=True)
class TreeEntry:
    """One non-tree entry of ``git ls-tree -r -l``."""

    mode: str
    sha: str
    size: int | None  # None for gitlinks
    path: bytes


@dataclass(frozen=True)
class RefCommand:
    """One command of an atomic ``git update-ref --stdin`` transaction.

    ``old`` is the expected current value (``NULL_SHA`` = must not exist; None = no check).
    """

    verb: Literal["update", "delete"]
    ref: str
    new: str = ""
    old: str | None = None

    def encode(self) -> bytes:
        check_ref_name(self.ref)
        if self.verb == "update" and not self.new:
            raise ValueError(f"update of {self.ref} needs a new value")
        fields = [self.new] if self.verb == "update" else []
        fields.append(self.old or "")
        return f"{self.verb} {self.ref}\0".encode() + b"".join(
            f"{field}\0".encode() for field in fields
        )


@dataclass(frozen=True)
class BareRepo:
    path: Path

    # ------------------------------------------------------------------ lifecycle

    @classmethod
    def init(cls, path: Path) -> BareRepo:
        """Create (or re-harden) a bare repository with the factory's hardened config.

        A missing or empty directory is initialized; an existing bare repository has its
        hardening re-applied; anything else raises :class:`GitError`.
        """
        target = absolute(path)
        if target.is_dir() and any(target.iterdir()):
            repo = cls.open(target)
        else:
            run_git(["init", "--bare", "--quiet", "--initial-branch=main", str(target)])
            repo = cls(target)
        for key, value in HARDENED_CONFIG:
            repo.git(["config", key, value])
        return repo

    @classmethod
    def open(cls, path: Path) -> BareRepo:
        """Open an existing bare repository; :class:`GitError` if ``path`` is not one."""
        target = absolute(path)
        args = ["rev-parse", "--is-bare-repository"]
        result = run_git(args, git_dir=target)
        if result.stdout.strip() != b"true":
            raise GitError(args, result.returncode, f"not a bare repository: {target}")
        return cls(target)

    def git(
        self, args: Sequence[str], *, input: bytes | None = None, check: bool = True
    ) -> CompletedProcess[bytes]:
        """Run git against this bare repository (``--git-dir``, neutral cwd). The only way
        host-side code should run git."""
        return run_git(args, git_dir=self.path, input=input, check=check)

    # ------------------------------------------------------------------ refs

    def fetch_from(self, source: Path | str, refspecs: Sequence[str]) -> None:
        """Fetch ``refspecs`` from a *trusted* local repository or bundle (seeding).

        Only local paths and bundles work (the ``file`` transport); objects are fsck-checked and
        no tags, FETCH_HEAD or other refs are created beyond the refspecs.
        """
        if not refspecs:
            raise ValueError("fetch_from needs at least one refspec")
        location = str(absolute(source)) if isinstance(source, Path) else source
        self.git([*FSCK_FLAGS, "fetch", *FETCH_OPTIONS, "--end-of-options", location, *refspecs])

    def rev_parse(self, rev: str) -> str:
        """Full object name of ``rev``; :class:`GitError` if it does not resolve."""
        result = self.git(["rev-parse", "--verify", "--end-of-options", rev])
        return result.stdout.decode().strip()

    def ref_exists(self, ref: str) -> bool:
        check_ref_name(ref)
        args = ["show-ref", "--verify", "--quiet", ref]
        result = self.git(args, check=False)
        if result.returncode not in (0, 1):
            raise GitError(args, result.returncode, result.stderr.decode("utf-8", "replace"))
        return result.returncode == 0

    def ref_transaction(self, commands: Sequence[RefCommand]) -> None:
        """Apply all ``commands`` atomically: either every one succeeds or none does."""
        payload = b"".join(command.encode() for command in commands)
        self.git(["update-ref", "--stdin", "-z"], input=payload)

    def update_ref(self, ref: str, sha: str, *, old_sha: str | None = None) -> None:
        """Point ``ref`` at ``sha``. With ``old_sha`` this is a compare-and-swap: it fails with
        :class:`GitError` unless the ref currently has that value (``NULL_SHA``: does not exist)."""
        self.ref_transaction([RefCommand("update", ref, new=sha, old=old_sha)])

    def delete_ref(self, ref: str, *, old_sha: str | None = None) -> None:
        """Delete ``ref`` (a no-op if it does not exist, unless ``old_sha`` demands a value)."""
        self.ref_transaction([RefCommand("delete", ref, old=old_sha)])

    # ------------------------------------------------------------------ history

    def is_ancestor(self, ancestor: str, descendant: str) -> bool:
        args = ["merge-base", "--is-ancestor", "--end-of-options", ancestor, descendant]
        result = self.git(args, check=False)
        if result.returncode not in (0, 1):
            raise GitError(args, result.returncode, result.stderr.decode("utf-8", "replace"))
        return result.returncode == 0

    def list_commits(self, base: str, head: str) -> list[str]:
        """Commits in ``base..head``, oldest first (topological, so parents precede children)."""
        result = self.git(
            ["rev-list", "--reverse", "--topo-order", "--end-of-options", head, f"^{base}"]
        )
        return result.stdout.decode().split()

    def count_commits(self, base: str, head: str) -> int:
        result = self.git(["rev-list", "--count", "--end-of-options", head, f"^{base}"])
        return int(result.stdout)

    def new_objects(self, base: str, head: str) -> list[str]:
        """Every object reachable from ``head`` but not from ``base`` (commits, trees, blobs)."""
        result = self.git(
            ["rev-list", "--objects", "--no-object-names", "--end-of-options", head, f"^{base}"]
        )
        return result.stdout.decode().split()

    # ------------------------------------------------------------------ objects

    def tree_hash(self, rev: str) -> str:
        return self.rev_parse(f"{rev}^{{tree}}")

    def read_file(self, rev: str, path: str) -> bytes:
        return self.git(["cat-file", "blob", f"{rev}:{path}"]).stdout

    def object_info(self, shas: Iterable[str]) -> dict[str, tuple[str, int]]:
        """``sha -> (type, size)`` for each object that exists; missing objects are omitted."""
        unique = list(dict.fromkeys(shas))
        if not unique:
            return {}
        payload = "".join(f"{sha}\n" for sha in unique).encode()
        result = self.git(["cat-file", "--batch-check"], input=payload)
        info: dict[str, tuple[str, int]] = {}
        for line in result.stdout.decode().splitlines():
            fields = line.split(" ")
            if len(fields) == 3:
                info[fields[0]] = (fields[1], int(fields[2]))
        return info

    def read_blobs(self, shas: Iterable[str]) -> dict[str, bytes]:
        """``sha -> content`` for blobs; :class:`GitError` if any is missing or not a blob."""
        unique = list(dict.fromkeys(shas))
        if not unique:
            return {}
        args = ["cat-file", "--batch"]
        out = self.git(args, input="".join(f"{sha}\n" for sha in unique).encode()).stdout
        blobs: dict[str, bytes] = {}
        pos = 0
        for sha in unique:
            newline = out.index(b"\n", pos)
            header = out[pos:newline].decode().split(" ")
            if len(header) != 3 or header[1] != "blob":
                raise GitError(args, 0, f"expected a blob for {sha}, got: {' '.join(header)}")
            start = newline + 1
            end = start + int(header[2])
            blobs[sha] = out[start:end]
            pos = end + 1
        return blobs

    def ls_tree(self, rev: str) -> list[TreeEntry]:
        """Every non-tree entry of ``rev``'s tree, recursively, in git's order."""
        out = self.git(
            ["ls-tree", "-r", "-z", "-l", "--full-tree", "--end-of-options", f"{rev}^{{tree}}"]
        ).stdout
        entries: list[TreeEntry] = []
        for record in out.split(b"\0"):
            if not record:
                continue
            meta, path = record.split(b"\t", 1)
            mode, _type, sha, size = meta.decode().split()
            entries.append(TreeEntry(mode, sha, None if size == "-" else int(size), path))
        return entries

    # ------------------------------------------------------------------ diffs

    def raw_diff(self, base: str, head: str) -> list[RawChange]:
        args = [NO_ATTRIBUTES, "diff-tree", "-r", "-z", "-M", "--no-commit-id"]
        args += ["--end-of-options", base, head]
        return parse_raw_diff(self.git(args).stdout)

    def diff(self, base: str, head: str) -> list[ChangedFile]:
        """Changes from ``base`` to ``head`` (rename detection on), sorted by path.

        Line numbers and binary detection are computed from the exact blob bytes, never through
        attributes, textconv or external diff drivers. Paths that are not valid UTF-8 are
        reported with replacement characters.
        """
        return self.changes_from_raw(self.raw_diff(base, head))

    def changes_from_raw(self, raw: Sequence[RawChange]) -> list[ChangedFile]:
        contents = self.read_blobs(change.new_sha for change in raw if change.has_new_blob)
        changes = [self._changed_file(change, contents) for change in raw]
        return sorted(changes, key=lambda change: (change.path, change.kind))

    def _changed_file(self, raw: RawChange, contents: dict[str, bytes]) -> ChangedFile:
        path = display(decode_path(raw.path))
        if raw.kind is ChangeKind.DELETED:
            return ChangedFile(path=path, kind=raw.kind, old_mode=raw.old_mode)
        common = {
            "path": path,
            "kind": raw.kind,
            "old_path": None if raw.old_path is None else display(decode_path(raw.old_path)),
            "mode": raw.new_mode,
            "old_mode": None if raw.kind is ChangeKind.ADDED else raw.old_mode,
        }
        if raw.new_mode == MODE_GITLINK:
            return ChangedFile.model_validate(common)
        content = contents[raw.new_sha]
        if raw.new_mode == MODE_SYMLINK:
            target = content.decode("utf-8", "replace")
            return ChangedFile.model_validate(
                {**common, "size": len(content), "symlink_target": target}
            )
        binary = is_binary(content)
        added = () if binary else self._added_lines(raw, content)
        return ChangedFile.model_validate(
            {**common, "size": len(content), "is_binary": binary, "added_lines": added}
        )

    def _added_lines(self, raw: RawChange, content: bytes) -> tuple[AddedLine, ...]:
        old_is_file = raw.kind is not ChangeKind.ADDED and raw.old_mode in REGULAR_MODES
        if raw.kind is ChangeKind.TYPE_CHANGED or not old_is_file:
            return all_lines(content)  # new file, or replaces a non-file: every line is new
        if raw.old_sha == raw.new_sha:
            return ()  # pure rename or mode change
        args = [NO_ATTRIBUTES, "diff", "-U0", "--inter-hunk-context=0", "-a", "--no-color"]
        args += ["--no-ext-diff", "--no-textconv", "--diff-algorithm=myers"]
        args += ["--end-of-options", raw.old_sha, raw.new_sha]
        ranges = parse_hunk_ranges(self.git(args).stdout)
        if not ranges:  # fail closed: differing blobs always produce at least one hunk
            raise GitError(args, 0, "no hunks reported for differing blobs")
        return lines_in_ranges(content, ranges)

    # ------------------------------------------------------------------ export

    def export_tree(self, rev: str, dest: Path) -> None:
        """Materialize ``rev``'s tree into ``dest`` (created if missing; must be empty).

        Files are written byte-exact from the blobs (no attributes, filters or eol conversion),
        executables get mode 0777 & ~umask, symlinks are created verbatim (never followed), and
        gitlinks become empty directories, as with git checkout. Writing never follows a
        symlink (``O_NOFOLLOW``, directory file descriptors) and never replaces an existing
        entry. Raises ``FileExistsError`` if ``dest`` is not empty, ``NotADirectoryError`` if it
        is not a directory (or is a symlink), ``ValueError`` for unsafe paths or unknown modes.
        """
        entries = self.ls_tree(rev)
        for entry in entries:
            reason = unsafe_path_reason(decode_path(entry.path))
            if reason is not None:
                raise ValueError(f"refusing to export {entry.path!r}: {reason}")
            if entry.mode not in EXPORT_MODES:
                raise ValueError(f"refusing to export {entry.path!r}: unknown mode {entry.mode}")
        target = absolute(dest)
        _prepare_empty_dir(target)
        with _TreeWriter(target) as writer:
            for chunk in _chunks(entries, _EXPORT_CHUNK_BYTES):
                blobs = self.read_blobs(e.sha for e in chunk if e.mode != MODE_GITLINK)
                for entry in chunk:
                    writer.write(entry, blobs.get(entry.sha, b""))


def _prepare_empty_dir(target: Path) -> None:
    try:
        info = target.lstat()
    except FileNotFoundError:
        target.mkdir(parents=True)
        return
    if not stat.S_ISDIR(info.st_mode):
        raise NotADirectoryError(f"export destination is not a directory: {target}")
    if any(target.iterdir()):
        raise FileExistsError(f"export destination is not empty: {target}")


def _chunks(entries: Sequence[TreeEntry], budget: int) -> list[list[TreeEntry]]:
    chunks: list[list[TreeEntry]] = []
    current: list[TreeEntry] = []
    used = 0
    for entry in entries:
        size = entry.size or 0
        if current and used + size > budget:
            chunks.append(current)
            current, used = [], 0
        current.append(entry)
        used += size
    if current:
        chunks.append(current)
    return chunks


class _TreeWriter:
    """Writes tree entries below a root directory using only directory file descriptors."""

    def __init__(self, root: Path) -> None:
        self._root = os.open(root, _DIR_FLAGS)
        self._stack: list[tuple[bytes, int]] = []

    def __enter__(self) -> _TreeWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self._unwind(0)
        os.close(self._root)

    def _unwind(self, depth: int) -> None:
        while len(self._stack) > depth:
            os.close(self._stack.pop()[1])

    def _directory(self, parents: Sequence[bytes]) -> int:
        common = 0
        while (
            common < min(len(self._stack), len(parents))
            and self._stack[common][0] == parents[common]
        ):
            common += 1
        self._unwind(common)
        for name in parents[common:]:
            parent = self._stack[-1][1] if self._stack else self._root
            with contextlib.suppress(FileExistsError):
                os.mkdir(name, 0o777, dir_fd=parent)
            self._stack.append((name, os.open(name, _DIR_FLAGS, dir_fd=parent)))
        return self._stack[-1][1] if self._stack else self._root

    def write(self, entry: TreeEntry, content: bytes) -> None:
        *parents, name = entry.path.split(b"/")
        directory = self._directory(parents)
        if entry.mode == MODE_SYMLINK:
            os.symlink(content, name, dir_fd=directory)
        elif entry.mode == MODE_GITLINK:
            os.mkdir(name, 0o777, dir_fd=directory)
        else:
            permissions = 0o777 if entry.mode == MODE_EXECUTABLE else 0o666
            fd = os.open(name, _FILE_FLAGS, permissions, dir_fd=directory)
            with os.fdopen(fd, "wb") as handle:
                handle.write(content)
