"""Unit tests for the non-git logic in bare, bundles, integrate and worker_side."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from local_factory.gitops import bare, bundles
from local_factory.gitops._git import NULL_SHA
from local_factory.gitops.bare import RefCommand, TreeEntry, check_ref_name
from local_factory.gitops.bundles import BundleLimits, BundleRejected, ImportResult
from local_factory.gitops.integrate import MergeOutcome
from local_factory.gitops.worker_side import (
    export_task_bundle,
    has_task_commits,
    init_from_bundle,
)
from local_factory.models import MODE_EXECUTABLE, MODE_GITLINK, MODE_REGULAR, MODE_SYMLINK

SHA = "1" * 40
NOT_FOLLOWED = "Too many levels of symbolic links|Not a directory"  # ELOOP / ENOTDIR

# --------------------------------------------------------------------------- refs


@pytest.mark.parametrize("ref", ["refs/heads/x", "refs/factory/tasks/T-0001/task-1"])
def test_check_ref_name_accepts_full_refs(ref: str) -> None:
    check_ref_name(ref)


@pytest.mark.parametrize("ref", ["HEAD", "main", "heads/x", "-d", "refs/x\0y", "refs/x\nupdate"])
def test_check_ref_name_rejects(ref: str) -> None:
    with pytest.raises(ValueError, match="not a full ref name"):
        check_ref_name(ref)


def test_ref_command_encoding() -> None:
    assert RefCommand("update", "refs/a", new=SHA).encode() == f"update refs/a\0{SHA}\0\0".encode()
    assert (
        RefCommand("update", "refs/a", new=SHA, old=NULL_SHA).encode()
        == f"update refs/a\0{SHA}\0{NULL_SHA}\0".encode()
    )
    assert RefCommand("delete", "refs/a").encode() == b"delete refs/a\0\0"
    assert RefCommand("delete", "refs/a", old=SHA).encode() == f"delete refs/a\0{SHA}\0".encode()


def test_ref_command_validation() -> None:
    with pytest.raises(ValueError, match="not a full ref name"):
        RefCommand("delete", "HEAD").encode()
    with pytest.raises(ValueError, match="needs a new value"):
        RefCommand("update", "refs/a").encode()


# --------------------------------------------------------------------------- export helpers


def entry(path: str, size: int | None = 1, mode: str = MODE_REGULAR) -> TreeEntry:
    return TreeEntry(mode, SHA, size, path.encode())


def test_chunks_respect_budget_boundaries() -> None:
    entries = [entry("a", 4), entry("b", 6), entry("c", 1), entry("d", None), entry("e", 20)]
    chunks = bare._chunks(entries, 10)
    assert [[e.path for e in chunk] for chunk in chunks] == [
        [b"a", b"b"],  # exactly the budget fits
        [b"c", b"d"],  # gitlinks (size None) count as 0
        [b"e"],  # an entry larger than the budget gets a chunk of its own
    ]
    assert bare._chunks([], 10) == []


def test_prepare_empty_dir(tmp_path: Path) -> None:
    missing = tmp_path / "a" / "b"
    bare._prepare_empty_dir(missing)
    assert missing.is_dir()
    bare._prepare_empty_dir(missing)  # existing and empty: fine

    (missing / "x").write_text("x")
    with pytest.raises(FileExistsError, match="not empty"):
        bare._prepare_empty_dir(missing)

    file = tmp_path / "file"
    file.write_text("x")
    with pytest.raises(NotADirectoryError):
        bare._prepare_empty_dir(file)

    link = tmp_path / "link"
    link.symlink_to(tmp_path / "a")
    with pytest.raises(NotADirectoryError):
        bare._prepare_empty_dir(link)


def test_tree_writer_materializes_entries(tmp_path: Path) -> None:
    root = tmp_path / "out"
    root.mkdir()
    with bare._TreeWriter(root) as writer:
        writer.write(entry("a/b/c.txt"), b"c\r\n")
        writer.write(entry("a/b/d.sh", mode=MODE_EXECUTABLE), b"#!/bin/sh\n")
        writer.write(entry("a/e.txt"), b"e")
        writer.write(entry("link", mode=MODE_SYMLINK), b"/etc/passwd")
        writer.write(entry("vendor/sub", None, MODE_GITLINK), b"")
        writer.write(entry("z.txt"), b"")
    assert (root / "a" / "b" / "c.txt").read_bytes() == b"c\r\n"
    assert os.access(root / "a" / "b" / "d.sh", os.X_OK)
    assert not os.access(root / "a" / "e.txt", os.X_OK)
    assert (root / "link").readlink() == Path("/etc/passwd")  # verbatim, never followed
    assert (root / "vendor" / "sub").is_dir() and not any((root / "vendor" / "sub").iterdir())
    assert (root / "z.txt").read_bytes() == b""


def test_tree_writer_never_writes_through_symlinks(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "out"
    root.mkdir()
    with bare._TreeWriter(root) as writer:
        writer.write(entry("A", mode=MODE_SYMLINK), str(outside).encode())
        # On a case-insensitive filesystem "a/x" would land in "A"; simulate the collision.
        with pytest.raises(OSError, match="exists"):
            writer.write(entry("A"), b"overwrite")
        with pytest.raises(OSError, match=NOT_FOLLOWED):  # the symlink is not followed
            writer.write(entry("A/x"), b"pwned")
    assert list(outside.iterdir()) == []


def test_tree_writer_refuses_existing_entries(tmp_path: Path) -> None:
    root = tmp_path / "out"
    root.mkdir()
    with bare._TreeWriter(root) as writer:
        writer.write(entry("f"), b"1")
        with pytest.raises(FileExistsError):
            writer.write(entry("f"), b"2")
    assert (root / "f").read_bytes() == b"1"


def test_tree_writer_root_must_not_be_a_symlink(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    with pytest.raises(OSError, match=NOT_FOLLOWED):
        bare._TreeWriter(link)


# --------------------------------------------------------------------------- bundles


def test_bundle_limits_defaults() -> None:
    assert BundleLimits() == BundleLimits(50, 5_000_000, 50_000_000, 100_000_000)
    assert BundleLimits() == bundles.DEFAULT_LIMITS
    assert BundleLimits().max_bundle_bytes == 100_000_000


def test_bundle_rejected_fields() -> None:
    error = BundleRejected("gitlink", "vendor/sub: submodules are not allowed")
    assert (error.code, error.detail) == ("gitlink", "vendor/sub: submodules are not allowed")
    assert str(error) == "gitlink: vendor/sub: submodules are not allowed"
    assert error.args == ("gitlink", "vendor/sub: submodules are not allowed")


def test_import_result_is_frozen() -> None:
    result = ImportResult(head_sha=SHA, commits=(SHA,), changes=())
    with pytest.raises(AttributeError):
        result.head_sha = "x"  # type: ignore[misc]  # asserting immutability at runtime


CAP = bundles.DEFAULT_LIMITS.max_bundle_bytes


def _scratch(tmp_path: Path) -> Path:
    scratch = tmp_path / "scratch"
    scratch.mkdir(exist_ok=True)
    return scratch


def test_private_copy_copies_regular_file(tmp_path: Path) -> None:
    source = tmp_path / "task.bundle"
    source.write_bytes(b"bundle bytes")
    scratch = _scratch(tmp_path)
    copy = bundles._private_copy(source, scratch, max_bytes=CAP)
    assert copy.parent == scratch and copy.read_bytes() == b"bundle bytes"
    source.write_bytes(b"swapped later")
    assert copy.read_bytes() == b"bundle bytes"


def _rejected_copy(
    path: Path, tmp_path: Path, *, max_bytes: int = CAP, code: str = "invalid_bundle"
) -> BundleRejected:
    scratch = _scratch(tmp_path)
    with pytest.raises(BundleRejected) as exc:
        bundles._private_copy(path, scratch, max_bytes=max_bytes)
    assert exc.value.code == code
    assert list(scratch.iterdir()) == []
    return exc.value


def test_private_copy_refuses_symlink(tmp_path: Path) -> None:
    secret = tmp_path / "secret"
    secret.write_bytes(b"factory signing key")
    link = tmp_path / "task.bundle"
    link.symlink_to(secret)
    assert "cannot open" in _rejected_copy(link, tmp_path).detail


def test_private_copy_refuses_fifo_without_blocking(tmp_path: Path) -> None:
    fifo = tmp_path / "task.bundle"
    os.mkfifo(fifo)
    assert "not a regular file" in _rejected_copy(fifo, tmp_path).detail


def test_private_copy_refuses_directory_and_missing(tmp_path: Path) -> None:
    directory = tmp_path / "dir.bundle"
    directory.mkdir()
    assert "not a regular file" in _rejected_copy(directory, tmp_path).detail
    assert "cannot open" in _rejected_copy(tmp_path / "missing.bundle", tmp_path).detail


# --------------------------------------------------------------------------- bundle size cap


def test_private_copy_size_boundary(tmp_path: Path) -> None:
    source = tmp_path / "task.bundle"
    source.write_bytes(b"x" * 100)
    copy = bundles._private_copy(source, _scratch(tmp_path), max_bytes=100)
    assert copy.read_bytes() == b"x" * 100
    copy.unlink()
    rejection = _rejected_copy(source, tmp_path, max_bytes=99, code="bundle_too_large")
    assert rejection.detail == "bundle is 100 bytes, limit 99"  # nothing was copied


def test_an_unopenable_bundle_is_invalid_whatever_the_cap(tmp_path: Path) -> None:
    secret = tmp_path / "secret"
    secret.write_bytes(b"x" * 100)
    link = tmp_path / "task.bundle"
    link.symlink_to(secret)
    assert "cannot open" in _rejected_copy(link, tmp_path, max_bytes=0).detail
    missing = tmp_path / "missing.bundle"
    assert "cannot open" in _rejected_copy(missing, tmp_path, max_bytes=0).detail
    fifo = tmp_path / "fifo.bundle"
    os.mkfifo(fifo)  # size 0: within any cap, and still not a regular file
    assert "not a regular file" in _rejected_copy(fifo, tmp_path, max_bytes=0).detail


def test_a_bundle_that_grows_while_being_copied_is_cut_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """fstat claims the file fits under the cap; the copy itself still stops at the cap."""
    source = tmp_path / "task.bundle"
    source.write_bytes(b"x" * 100)
    real_fstat = os.fstat

    def stale_fstat(fd: int) -> os.stat_result:
        info = real_fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size != 100:
            return info
        fields = list(info[:10])
        fields[stat.ST_SIZE] = 0  # as it was before the worker kept writing
        return os.stat_result(fields)

    monkeypatch.setattr(os, "fstat", stale_fstat)
    scratch = _scratch(tmp_path)
    with pytest.raises(BundleRejected) as exc:
        bundles._private_copy(source, scratch, max_bytes=99)
    assert (exc.value.code, exc.value.detail) == (
        "bundle_too_large",
        "bundle is at least 100 bytes, limit 99",
    )
    (partial,) = scratch.iterdir()  # the caller's temporary directory removes it
    assert partial.stat().st_size <= 99


# --------------------------------------------------------------------------- integrate


def test_merge_outcome_invariant() -> None:
    assert MergeOutcome(SHA, ()).sha == SHA
    assert MergeOutcome(None, ("a",)).conflicts == ("a",)
    with pytest.raises(ValueError, match="exactly when"):
        MergeOutcome(None, ())
    with pytest.raises(ValueError, match="exactly when"):
        MergeOutcome(SHA, ("a",))


# --------------------------------------------------------------------------- worker side


@pytest.mark.parametrize("base_sha", ["HEAD", "abc", "--all", "^" + SHA, "A" * 40])
def test_export_task_bundle_requires_full_sha(tmp_path: Path, base_sha: str) -> None:
    with pytest.raises(ValueError, match="full object name"):
        export_task_bundle(tmp_path, tmp_path / "out.bundle", base_sha=base_sha)
    assert not (tmp_path / "out.bundle").exists()


@pytest.mark.parametrize("base_sha", ["HEAD", "abc", "--all", "^" + SHA, "A" * 40])
def test_has_task_commits_requires_full_sha(tmp_path: Path, base_sha: str) -> None:
    with pytest.raises(ValueError, match="full object name"):
        has_task_commits(tmp_path, base_sha=base_sha)


def test_init_from_bundle_refuses_non_empty_dest(tmp_path: Path) -> None:
    dest = tmp_path / "work"
    dest.mkdir()
    (dest / ".git").mkdir()  # e.g. a planted repository with hostile config
    with pytest.raises(FileExistsError):
        init_from_bundle(tmp_path / "base.bundle", "refs/heads/x", dest)
    file_dest = tmp_path / "file"
    file_dest.write_text("x")
    with pytest.raises(FileExistsError):
        init_from_bundle(tmp_path / "base.bundle", "refs/heads/x", file_dest)


def test_init_from_bundle_requires_full_ref(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not a full ref name"):
        init_from_bundle(tmp_path / "base.bundle", "main", tmp_path / "work")
    assert not (tmp_path / "work").exists()


def test_export_permissions_follow_umask(tmp_path: Path) -> None:
    root = tmp_path / "out"
    root.mkdir()
    previous = os.umask(0o027)
    try:
        with bare._TreeWriter(root) as writer:
            writer.write(entry("d/f"), b"x")
            writer.write(entry("d/x", mode=MODE_EXECUTABLE), b"x")
    finally:
        os.umask(previous)
    assert stat.S_IMODE((root / "d" / "f").stat().st_mode) == 0o640
    assert stat.S_IMODE((root / "d" / "x").stat().st_mode) == 0o750
    assert stat.S_IMODE((root / "d").stat().st_mode) == 0o750
