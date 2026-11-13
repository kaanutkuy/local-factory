"""The holdout store: one sealed record per ticket (docs/specs/phase-2b.md § oracle).

A record holds a ticket's black-box probes with their expectations. It lives only here: never in
a repository, a ledger event or a box. The store is a directory of mode 0700 holding one file of
mode 0600 per ticket, written atomically (an ``O_EXCL`` temporary file, then ``os.replace``).
Every access goes through a descriptor of the directory opened with ``O_NOFOLLOW``, so a symlink
planted as the directory or as a record is refused instead of followed.
"""

from __future__ import annotations

import contextlib
import os
import stat
import uuid
from pathlib import Path

from pydantic import ValidationError

from local_factory.hashing import canonical_json
from local_factory.models import Holdout
from local_factory.oracle._ids import check_ticket_id

__all__ = ["MAX_RECORD_BYTES", "HoldoutError", "HoldoutStore"]

MAX_RECORD_BYTES = 10_000_000  # a record is a few KB; anything near this is not one
_DIR_MODE = 0o700
_FILE_MODE = 0o600
_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_READ_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC  # NONBLOCK: a FIFO
_CREATE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC


class HoldoutError(Exception):
    """A holdout record or the store itself is unusable: symlinked, corrupt or misfiled."""


class HoldoutStore:
    """``<root>/<ticket id>.json``, 0600, in a 0700 directory."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def path(self, ticket_id: str) -> Path:
        return self.root / f"{check_ticket_id(ticket_id)}.json"

    # ------------------------------------------------------------------ write

    def save(self, holdout: Holdout) -> Path:
        """Seal ``holdout`` as its ticket's record, replacing any earlier one atomically."""
        path = self.path(holdout.ticket_id)
        payload = canonical_json(holdout)
        try:
            self.root.mkdir(mode=_DIR_MODE, parents=True, exist_ok=True)
        except OSError as exc:
            raise HoldoutError(f"cannot create the holdout store {self.root}: {exc}") from exc
        try:
            root_fd = os.open(self.root, _DIR_FLAGS)
        except OSError as exc:  # ELOOP: a symlink; ENOTDIR: a file; ...
            raise self._unusable(exc) from exc
        try:
            os.fchmod(root_fd, _DIR_MODE)  # tighten a directory that was created open
            _refuse_unusable_record(path.name, root_fd)
            _write_atomically(path.name, payload, root_fd)
        finally:
            os.close(root_fd)
        return path

    # ------------------------------------------------------------------ read

    def load(self, ticket_id: str) -> Holdout | None:
        """The ticket's record, or None if it has none. A bad record is a :class:`HoldoutError`."""
        path = self.path(ticket_id)
        try:
            root_fd = os.open(self.root, _DIR_FLAGS)
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise self._unusable(exc) from exc
        try:
            data = _read_record(path.name, root_fd)
        finally:
            os.close(root_fd)
        if data is None:
            return None
        try:
            record = Holdout.model_validate_json(data)
        except ValidationError as exc:
            raise HoldoutError(f"{path}: not a valid holdout record: {_problems(exc)}") from exc
        if record.ticket_id != ticket_id:
            raise HoldoutError(f"{path}: the record belongs to {record.ticket_id}")
        return record

    def _unusable(self, exc: OSError) -> HoldoutError:
        return HoldoutError(
            f"the holdout store {self.root} is not a usable directory: {exc.strerror}"
        )


def _problems(exc: ValidationError) -> str:
    """Where and why a record is invalid, without echoing its content (it is sealed data)."""
    errors = exc.errors(include_input=False, include_url=False, include_context=False)
    return "; ".join(f"{'.'.join(map(str, e['loc'])) or '<record>'}: {e['msg']}" for e in errors)


def _refuse_unusable_record(name: str, root_fd: int) -> None:
    """An existing record must be a regular file: never replace a symlink or a directory."""
    try:
        info = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    if not stat.S_ISREG(info.st_mode):
        raise HoldoutError(f"the holdout record {name} is not a regular file")


def _write_atomically(name: str, payload: bytes, root_fd: int) -> None:
    temp = f".{name}.{uuid.uuid4().hex}.tmp"
    fd = os.open(temp, _CREATE_FLAGS, _FILE_MODE, dir_fd=root_fd)
    try:
        with os.fdopen(fd, "wb") as out:  # takes ownership of fd
            out.write(payload)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temp, name, src_dir_fd=root_fd, dst_dir_fd=root_fd)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temp, dir_fd=root_fd)
        raise
    os.fsync(root_fd)


def _read_record(name: str, root_fd: int) -> bytes | None:
    try:
        fd = os.open(name, _READ_FLAGS, dir_fd=root_fd)
    except FileNotFoundError:
        return None
    except OSError as exc:  # ELOOP: the record is a symlink
        raise HoldoutError(f"the holdout record {name} cannot be opened: {exc.strerror}") from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise HoldoutError(f"the holdout record {name} is not a regular file")
        if info.st_size > MAX_RECORD_BYTES:
            raise HoldoutError(f"the holdout record {name} is {info.st_size} bytes: too large")
        with os.fdopen(fd, "rb", closefd=False) as record:
            return record.read()
    finally:
        os.close(fd)
