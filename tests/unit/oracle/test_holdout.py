"""Unit tests for the holdout store's guarantees: sealed modes, atomic writes, refused symlinks,
bounded and non-leaking error handling."""

from __future__ import annotations

import json
import os
import stat
from collections.abc import Iterator
from pathlib import Path

import pytest

from local_factory.hashing import canonical_json
from local_factory.models import Holdout, HoldoutProbe, ProbeExpectation
from local_factory.oracle import HoldoutError, HoldoutStore
from local_factory.oracle.holdout import MAX_RECORD_BYTES

SHA = "a" * 40
CANARY = "SEALED-EXPECTATION-CANARY"


def record(ticket: str = "T-0001", stdout: str = "x") -> Holdout:
    probe = HoldoutProbe(
        id="HP-1",
        criteria=("AC-1",),
        argv=("python3", "greet.py"),
        expect=ProbeExpectation(stdout=stdout),
    )
    return Holdout(ticket_id=ticket, oracle_sha=SHA, probes=(probe,))


@pytest.fixture
def store(tmp_path: Path) -> HoldoutStore:
    return HoldoutStore(tmp_path / "state" / "holdout")


@pytest.fixture
def open_umask() -> Iterator[None]:
    """A umask that would make everything world-readable, to show the store does not rely on it."""
    old = os.umask(0)
    try:
        yield
    finally:
        os.umask(old)


@pytest.mark.usefixtures("open_umask")
def test_modes_do_not_depend_on_the_umask(store: HoldoutStore) -> None:
    path = store.save(record())
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(store.root.stat().st_mode) == 0o700


def test_save_creates_missing_parents(store: HoldoutStore) -> None:
    assert not store.root.parent.exists()
    store.save(record())
    assert store.load("T-0001") == record()


def test_the_record_is_the_canonical_json_with_no_temp_file_left(store: HoldoutStore) -> None:
    path = store.save(record())
    assert path.read_bytes() == canonical_json(record())
    assert [p.name for p in store.root.iterdir()] == ["T-0001.json"]


def test_records_of_other_tickets_are_untouched(store: HoldoutStore) -> None:
    store.save(record("T-0001", "one"))
    store.save(record("T-0002", "two"))
    store.save(record("T-0001", "uno"))
    assert store.load("T-0001") == record("T-0001", "uno")
    assert store.load("T-0002") == record("T-0002", "two")
    assert store.load("T-0003") is None


def test_a_missing_store_has_no_records(tmp_path: Path) -> None:
    assert HoldoutStore(tmp_path / "nope").load("T-0001") is None


def test_a_failed_write_leaves_the_old_record_and_no_temp_file(
    store: HoldoutStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    store.save(record(stdout="old"))

    def failing_replace(*args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", failing_replace)
    with pytest.raises(OSError, match="disk full"):
        store.save(record(stdout="new"))
    monkeypatch.undo()
    assert store.load("T-0001") == record(stdout="old")
    assert [p.name for p in store.root.iterdir()] == ["T-0001.json"]


def test_a_store_that_cannot_be_created_is_an_error(tmp_path: Path) -> None:
    (tmp_path / "file").write_text("x")
    with pytest.raises(HoldoutError, match="cannot create"):
        HoldoutStore(tmp_path / "file" / "holdout").save(record())


def test_a_store_that_is_a_file_is_an_error(tmp_path: Path) -> None:
    (tmp_path / "holdout").write_text("x")
    store = HoldoutStore(tmp_path / "holdout")
    with pytest.raises(HoldoutError, match="cannot create"):
        store.save(record())
    with pytest.raises(HoldoutError, match="not a usable directory"):
        store.load("T-0001")


def test_a_dangling_symlink_as_the_store_is_an_error(tmp_path: Path) -> None:
    (tmp_path / "holdout").symlink_to(tmp_path / "nowhere")
    with pytest.raises(HoldoutError):
        HoldoutStore(tmp_path / "holdout").save(record())


def test_a_symlinked_store_is_refused_for_reading_too(tmp_path: Path) -> None:
    real = HoldoutStore(tmp_path / "real")
    real.save(record())
    (tmp_path / "linked").symlink_to(real.root)
    with pytest.raises(HoldoutError):
        HoldoutStore(tmp_path / "linked").load("T-0001")


def test_a_record_that_is_a_directory_is_never_replaced(store: HoldoutStore) -> None:
    store.root.mkdir(parents=True)
    store.path("T-0001").mkdir()
    with pytest.raises(HoldoutError, match="not a regular file"):
        store.save(record())
    with pytest.raises(HoldoutError, match="not a regular file"):
        store.load("T-0001")
    assert store.path("T-0001").is_dir()


def test_a_fifo_record_is_refused_without_blocking(store: HoldoutStore) -> None:
    store.root.mkdir(parents=True)
    os.mkfifo(store.path("T-0001"))
    with pytest.raises(HoldoutError, match="not a regular file"):
        store.load("T-0001")


def test_an_oversized_record_is_refused_unread(store: HoldoutStore) -> None:
    path = store.save(record())
    os.truncate(path, MAX_RECORD_BYTES + 1)
    with pytest.raises(HoldoutError, match="too large"):
        store.load("T-0001")


def test_a_symlinked_record_is_refused_and_the_target_untouched(
    store: HoldoutStore, tmp_path: Path
) -> None:
    store.save(record("T-0002"))
    target = tmp_path / "target.json"
    target.write_bytes(b"keep")
    store.path("T-0001").symlink_to(target)
    with pytest.raises(HoldoutError, match="cannot be opened"):
        store.load("T-0001")
    with pytest.raises(HoldoutError, match="not a regular file"):
        store.save(record())
    assert target.read_bytes() == b"keep"


def test_an_invalid_record_is_an_error_that_does_not_echo_its_content(store: HoldoutStore) -> None:
    store.root.mkdir(parents=True)
    probe = {"id": "HP-1", "criteria": ["AC-1"], "argv": ["x"], "expect": {"stdout": CANARY}}
    store.path("T-0001").write_text(
        json.dumps({"ticket_id": "T-0001", "oracle_sha": "nothex", "probes": [probe]})
    )
    with pytest.raises(HoldoutError, match="oracle_sha") as caught:
        store.load("T-0001")
    assert CANARY not in str(caught.value)
    assert "nothex" not in str(caught.value)


def test_garbage_is_an_error(store: HoldoutStore) -> None:
    store.root.mkdir(parents=True)
    store.path("T-0001").write_bytes(b"\xff\xfe not json")
    with pytest.raises(HoldoutError, match="not a valid holdout record"):
        store.load("T-0001")


def test_a_record_filed_under_another_ticket_is_an_error(store: HoldoutStore) -> None:
    store.root.mkdir(parents=True)
    store.path("T-0002").write_bytes(canonical_json(record("T-0001")))
    with pytest.raises(HoldoutError, match="belongs to T-0001"):
        store.load("T-0002")


@pytest.mark.parametrize("ticket", ["T-0001\n", "T-١٢٣٤", "T-0001 ", "../T-0001", "T-0001.json"])
def test_only_exact_ticket_ids_make_paths(store: HoldoutStore, ticket: str) -> None:
    with pytest.raises(ValueError, match="not a ticket id"):
        store.path(ticket)
