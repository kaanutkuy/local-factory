"""Host side of the worker boundary: base bundles out, task bundles in.

``import_task_bundle`` treats the bundle as hostile data. It never touches the worker's directory:
it checks the bundle file's size against ``BundleLimits.max_bundle_bytes`` before reading any of
it, copies it (refusing symlinks and non-regular files, and stopping at the cap) into a private
temporary directory and runs every git command against the factory's bare repository on that
copy. The bundle is fetched into a uniquely named quarantine ref and validated there; the
destination ref is only set, atomically with the quarantine ref's deletion, once every check has
passed. A ``finally`` clause deletes the quarantine ref on every other path, so a rejection (or
any unexpected error) leaves no ref behind.
"""

from __future__ import annotations

import os
import stat
import tempfile
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from local_factory.models import MODE_GITLINK, MODE_SYMLINK, ChangedFile, ChangeKind

from ._diff import RawChange
from ._git import NULL_SHA, GitError, run_git
from ._paths import (
    MAX_SYMLINK_TARGET,
    decode_path,
    display,
    symlink_escape_reason,
    unsafe_path_reason,
)
from .bare import FETCH_OPTIONS, FSCK_FLAGS, BareRepo, RefCommand, absolute, check_ref_name
from .worker_side import TASK_REF

QUARANTINE_PREFIX = "refs/factory/quarantine/"
BASE_REF_PREFIX = "refs/heads/factory-base-"
_BLOB_MODES = frozenset({"100644", "100755", MODE_SYMLINK})
_COPY_CHUNK = 1024 * 1024


@dataclass(frozen=True)
class BundleLimits:
    max_commits: int = 50
    max_file_bytes: int = 5_000_000
    max_total_bytes: int = 50_000_000
    max_bundle_bytes: int = 100_000_000  # the bundle file itself, checked before it is read


DEFAULT_LIMITS = BundleLimits()


class BundleRejected(Exception):  # noqa: N818 - the name is fixed by the phase spec
    """A task bundle failed validation. ``code`` is machine-readable, ``detail`` is for humans."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(code, detail)
        self.code = code
        self.detail = detail

    def __str__(self) -> str:
        return f"{self.code}: {self.detail}"


@dataclass(frozen=True)
class ImportResult:
    head_sha: str
    commits: tuple[str, ...]
    changes: tuple[ChangedFile, ...]


# --------------------------------------------------------------------------- export


def export_base_bundle(repo: BareRepo, *, sha: str, out: Path) -> str:
    """Write a bundle of ``sha`` and its history to ``out``; returns the bundle's only ref.

    The ref is a temporary branch created just for the bundle and deleted again afterwards.
    """
    commit = repo.rev_parse(f"{sha}^{{commit}}")
    ref = f"{BASE_REF_PREFIX}{uuid.uuid4().hex}"
    repo.update_ref(ref, commit, old_sha=NULL_SHA)
    try:
        run_git(["bundle", "create", "--quiet", str(absolute(out)), ref], git_dir=repo.path)
    finally:
        repo.delete_ref(ref)
    return ref


# --------------------------------------------------------------------------- import


def import_task_bundle(
    repo: BareRepo,
    bundle: Path,
    *,
    base_sha: str,
    dest_ref: str,
    limits: BundleLimits = DEFAULT_LIMITS,
) -> ImportResult:
    """Validate a worker's task bundle and, only if it passes, point ``dest_ref`` at its head.

    Raises :class:`BundleRejected` (see the phase spec for codes and their order) or, for caller
    errors such as an unknown ``base_sha``, :class:`GitError` / ``ValueError``.
    """
    check_ref_name(dest_ref)
    if dest_ref.startswith(QUARANTINE_PREFIX):
        raise ValueError(f"dest_ref must not be a quarantine ref: {dest_ref}")
    base = repo.rev_parse(f"{base_sha}^{{commit}}")
    with tempfile.TemporaryDirectory(prefix="local-factory-import-") as scratch:
        copy = _private_copy(bundle, Path(scratch), max_bytes=limits.max_bundle_bytes)
        _check_heads(repo, copy)
        _verify(repo, copy)
        quarantine = f"{QUARANTINE_PREFIX}{uuid.uuid4().hex}"
        try:
            head = _fetch(repo, copy, quarantine)
            result = _validate(repo, base, head, limits)
            repo.ref_transaction(
                [
                    RefCommand("update", dest_ref, new=head),
                    RefCommand("delete", quarantine, old=head),
                ]
            )
            return result
        finally:
            repo.delete_ref(quarantine)  # no-op after a successful transaction


def _private_copy(bundle: Path, scratch: Path, *, max_bytes: int) -> Path:
    """Copy the bundle into factory-owned storage, so nothing can swap it under us.

    The size cap comes first, on the opened file and before a single byte is copied; the copy
    stops at the cap too, in case the file grows while it is being read.
    """
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
    try:
        fd = os.open(bundle, flags)
    except OSError as exc:
        raise BundleRejected("invalid_bundle", f"cannot open bundle: {exc.strerror}") from exc
    try:
        info = os.fstat(fd)
        if info.st_size > max_bytes:
            raise _too_large(info.st_size, max_bytes)
        if not stat.S_ISREG(info.st_mode):
            raise BundleRejected("invalid_bundle", "bundle is not a regular file")
        copy = scratch / "task.bundle"
        copied = 0
        with copy.open("xb") as target:
            while chunk := os.read(fd, _COPY_CHUNK):
                copied += len(chunk)
                if copied > max_bytes:
                    raise _too_large(copied, max_bytes, at_least=True)
                target.write(chunk)
    finally:
        os.close(fd)
    return copy


def _too_large(size: int, max_bytes: int, *, at_least: bool = False) -> BundleRejected:
    shown = f"at least {size}" if at_least else str(size)
    return BundleRejected("bundle_too_large", f"bundle is {shown} bytes, limit {max_bytes}")


def _stderr(exc: GitError) -> str:
    return exc.stderr.strip() or str(exc)


def _check_heads(repo: BareRepo, bundle: Path) -> None:
    try:
        result = run_git(["bundle", "list-heads", str(bundle)], git_dir=repo.path)
    except GitError as exc:
        raise BundleRejected("invalid_bundle", _stderr(exc)) from exc
    heads = [line.split(" ", 1) for line in result.stdout.decode("utf-8", "replace").splitlines()]
    names = [head[-1] for head in heads]
    if names != [TASK_REF]:
        shown = ", ".join(names) or "none"
        raise BundleRejected("unexpected_refs", f"bundle must contain exactly {TASK_REF}: {shown}")


def _verify(repo: BareRepo, bundle: Path) -> None:
    try:
        run_git(["bundle", "verify", "--quiet", str(bundle)], git_dir=repo.path)
    except GitError as exc:
        raise BundleRejected("invalid_bundle", _stderr(exc)) from exc


def _fetch(repo: BareRepo, bundle: Path, quarantine: str) -> str:
    refspec = f"{TASK_REF}:{quarantine}"
    args = [*FSCK_FLAGS, "fetch", *FETCH_OPTIONS, "--end-of-options", str(bundle), refspec]
    try:
        run_git(args, git_dir=repo.path)
    except GitError as exc:
        raise BundleRejected("fetch_failed", _stderr(exc)) from exc
    return repo.rev_parse(quarantine)


def _path_of(change: RawChange) -> str:
    return display(decode_path(change.path))


def _validate(repo: BareRepo, base: str, head: str, limits: BundleLimits) -> ImportResult:
    raw = repo.raw_diff(base, head)
    _check_object_types(repo, raw)
    if not repo.is_ancestor(base, head):
        raise BundleRejected("base_not_ancestor", f"{base} is not an ancestor of {head}")
    if head == base:
        raise BundleRejected("no_changes", f"the task head is the base commit {base}")
    count = repo.count_commits(base, head)
    if count > limits.max_commits:
        raise BundleRejected(
            "too_many_commits", f"{count} commits in base..head, limit {limits.max_commits}"
        )
    for change in raw:
        if change.kind is not ChangeKind.DELETED and change.new_mode == MODE_GITLINK:
            raise BundleRejected("gitlink", f"{_path_of(change)}: submodules are not allowed")
    _check_symlinks(repo, head)
    for change in raw:
        for path in filter(None, (change.path, change.old_path)):
            reason = unsafe_path_reason(decode_path(path))
            if reason is not None:
                raise BundleRejected("unsafe_path", f"{display(decode_path(path))}: {reason}")
    _check_sizes(repo, base, head, raw, limits)
    return ImportResult(
        head_sha=head,
        commits=tuple(repo.list_commits(base, head)),
        changes=tuple(repo.changes_from_raw(raw)),
    )


def _check_object_types(repo: BareRepo, raw: Sequence[RawChange]) -> None:
    """Every changed file/symlink entry must point at a blob that exists.

    Fetch-time fsck does not catch a tree entry whose mode says blob but whose object is a tree,
    so this completes the "fsck errors" class of ``fetch_failed``.
    """
    info = repo.object_info(change.new_sha for change in raw if change.has_new_blob)
    for change in raw:
        if not change.has_new_blob:
            continue
        kind = info.get(change.new_sha, ("missing", 0))[0]
        if change.new_mode not in _BLOB_MODES or kind != "blob":
            raise BundleRejected(
                "fetch_failed",
                f"malformed tree entry {_path_of(change)}: mode {change.new_mode}, object {kind}",
            )


def _symlinks(repo: BareRepo, rev: str) -> dict[str, str | None]:
    """Every symlink in ``rev``'s tree -> its target (None if too long to be a real symlink)."""
    entries = [entry for entry in repo.ls_tree(rev) if entry.mode == MODE_SYMLINK]
    readable = [e for e in entries if e.size is not None and e.size <= MAX_SYMLINK_TARGET]
    blobs = repo.read_blobs(entry.sha for entry in readable)
    return {
        decode_path(entry.path): (decode_path(blobs[entry.sha]) if entry.sha in blobs else None)
        for entry in entries
    }


def _check_symlinks(repo: BareRepo, head: str) -> None:
    """The authoritative symlink check: every symlink in the head tree, changed or not, is
    resolved against every symlink in the head tree (see :func:`._paths.symlink_escape_reason`).

    Unchanged symlinks are checked too, because other changes can make them unsafe (``a -> .``
    replacing a directory ``a`` under an existing ``b -> a/..``). There is deliberately no
    exemption for unsafe symlinks that already exist in the base: whether such a link escapes
    physically depends on the rest of the tree, which the worker controls.
    """
    links = _symlinks(repo, head)
    for link in sorted(links):
        reason = symlink_escape_reason(link, links)
        if reason is not None:
            target = links[link]
            shown = "?" if target is None else display(target)
            raise BundleRejected("symlink_escape", f"{display(link)} -> {shown}: {reason}")


def _check_sizes(
    repo: BareRepo, base: str, head: str, raw: Sequence[RawChange], limits: BundleLimits
) -> None:
    """Limits apply to every blob the bundle introduces (reachable from head, not from base),
    including blobs that only exist in intermediate commits."""
    info = repo.object_info(repo.new_objects(base, head))
    blobs = {sha: size for sha, (kind, size) in info.items() if kind == "blob"}
    paths = {c.new_sha: _path_of(c) for c in raw if c.has_new_blob}
    for sha, size in sorted(blobs.items(), key=lambda item: paths.get(item[0], item[0])):
        if size > limits.max_file_bytes:
            where = paths.get(sha, f"blob {sha} (in an intermediate commit)")
            raise BundleRejected(
                "file_too_large", f"{where}: {size} bytes, limit {limits.max_file_bytes}"
            )
    total = sum(blobs.values())
    if total > limits.max_total_bytes:
        raise BundleRejected(
            "too_large", f"{total} new bytes in total, limit {limits.max_total_bytes}"
        )
