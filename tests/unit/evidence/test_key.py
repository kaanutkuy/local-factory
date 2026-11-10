"""Unit tests for load_or_create_key: creation, permissions, and refusal of unsafe key files."""

from __future__ import annotations

import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from local_factory.evidence import binding
from local_factory.evidence.binding import KEY_BYTES, EvidenceKeyError, load_or_create_key


def mode(path: Path) -> int:
    return stat.S_IMODE(path.lstat().st_mode)


def write_key(path: Path, data: bytes, file_mode: int = 0o600) -> None:
    path.write_bytes(data)
    path.chmod(file_mode)


@contextmanager
def umask(value: int) -> Iterator[None]:
    previous = os.umask(value)
    try:
        yield
    finally:
        os.umask(previous)


# --------------------------------------------------------------------------- creation


def test_creates_a_private_random_key_and_its_parent_directories(tmp_path: Path) -> None:
    path = tmp_path / "a" / "b" / "keys" / "evidence.key"
    key = load_or_create_key(path)
    assert len(key) == KEY_BYTES == 32
    assert path.read_bytes() == key
    assert mode(path) == 0o600
    assert mode(path.parent) == 0o700
    assert stat.S_ISREG(path.lstat().st_mode)


def test_creation_leaves_no_temporary_files(tmp_path: Path) -> None:
    load_or_create_key(tmp_path / "evidence.key")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["evidence.key"]


def test_key_is_stable_and_distinct_per_file(tmp_path: Path) -> None:
    first = load_or_create_key(tmp_path / "one.key")
    assert load_or_create_key(tmp_path / "one.key") == first
    assert load_or_create_key(tmp_path / "two.key") != first


def test_mode_is_exactly_0600_regardless_of_umask(tmp_path: Path) -> None:
    for name, value in (("lax.key", 0o000), ("strict.key", 0o277)):
        with umask(value):
            load_or_create_key(tmp_path / name)
        assert mode(tmp_path / name) == 0o600, name


def test_existing_parent_directory_keeps_its_mode(tmp_path: Path) -> None:
    keys = tmp_path / "keys"
    keys.mkdir()
    keys.chmod(0o755)
    load_or_create_key(keys / "evidence.key")
    assert mode(keys) == 0o755


def test_a_key_created_concurrently_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """If another process creates the key between our check and our link, we use theirs."""
    path = tmp_path / "evidence.key"
    theirs = bytes(range(32))
    write_key(path, theirs)
    real_read = binding._read_key
    calls: list[Path] = []

    def read_key_racing(target: Path) -> bytes:
        calls.append(target)
        if len(calls) == 1:
            raise FileNotFoundError(target)  # the key "did not exist yet" when first checked
        return real_read(target)

    monkeypatch.setattr(binding, "_read_key", read_key_racing)
    assert load_or_create_key(path) == theirs
    assert path.read_bytes() == theirs
    assert len(calls) == 2
    assert sorted(p.name for p in tmp_path.iterdir()) == ["evidence.key"]


# --------------------------------------------------------------------------- existing keys


def test_existing_owner_only_key_is_loaded(tmp_path: Path) -> None:
    path = tmp_path / "evidence.key"
    key = os.urandom(KEY_BYTES)
    write_key(path, key, 0o400)
    assert load_or_create_key(path) == key


@pytest.mark.parametrize("file_mode", [0o640, 0o604, 0o660, 0o644, 0o610, 0o601, 0o602, 0o620])
def test_group_or_other_bits_are_refused_and_the_key_is_kept(
    tmp_path: Path, file_mode: int
) -> None:
    path = tmp_path / "evidence.key"
    key = os.urandom(KEY_BYTES)
    write_key(path, key, file_mode)
    with pytest.raises(EvidenceKeyError, match=f"mode {file_mode:04o}"):
        load_or_create_key(path)
    assert path.read_bytes() == key
    assert mode(path) == file_mode


@pytest.mark.parametrize(
    ("length", "problem"),
    [(0, "truncated"), (16, "truncated"), (31, "truncated"), (33, "too long"), (4096, "too long")],
)
def test_key_of_the_wrong_length_is_refused(tmp_path: Path, length: int, problem: str) -> None:
    path = tmp_path / "evidence.key"
    write_key(path, b"k" * length)
    with pytest.raises(EvidenceKeyError, match=problem):
        load_or_create_key(path)
    assert path.read_bytes() == b"k" * length


def test_symlinked_key_file_is_refused(tmp_path: Path) -> None:
    real = tmp_path / "real.key"
    key = load_or_create_key(real)
    link = tmp_path / "evidence.key"
    link.symlink_to(real)
    with pytest.raises(EvidenceKeyError, match="symbolic link"):
        load_or_create_key(link)
    assert real.read_bytes() == key


def test_dangling_symlink_is_refused_and_not_followed(tmp_path: Path) -> None:
    link = tmp_path / "evidence.key"
    link.symlink_to(tmp_path / "elsewhere.key")
    with pytest.raises(EvidenceKeyError, match="symbolic link"):
        load_or_create_key(link)
    assert not (tmp_path / "elsewhere.key").exists()


def test_directory_at_the_key_path_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "evidence.key"
    path.mkdir(mode=0o700)
    with pytest.raises(EvidenceKeyError, match="not a regular file"):
        load_or_create_key(path)
    assert path.is_dir()


def test_fifo_at_the_key_path_is_refused_without_blocking(tmp_path: Path) -> None:
    path = tmp_path / "evidence.key"
    os.mkfifo(path, 0o600)
    with pytest.raises(EvidenceKeyError, match="not a regular file"):
        load_or_create_key(path)


def test_parent_that_is_a_file_is_refused(tmp_path: Path) -> None:
    parent = tmp_path / "keys"
    parent.write_text("not a directory")
    with pytest.raises(EvidenceKeyError, match="Not a directory"):
        load_or_create_key(parent / "evidence.key")
    assert parent.read_text() == "not a directory"
