"""Unit tests for the oracle policy beyond the acceptance cases: precedence, ordering, bounds."""

from __future__ import annotations

import pytest

from local_factory.models import (
    MODE_EXECUTABLE,
    MODE_REGULAR,
    MODE_SYMLINK,
    ChangedFile,
    ChangeKind,
)
from local_factory.oracle import (
    MAX_ORACLE_FILE_BYTES,
    MAX_ORACLE_FILES,
    ORACLE_BINARY,
    ORACLE_FILES,
    ORACLE_PATH,
    ORACLE_SIZE,
    ORACLE_SYMLINK,
    acceptance_dir,
    check_oracle_changes,
)

DIR = "tests/acceptance/T-0001/"


def change(
    path: str,
    kind: ChangeKind = ChangeKind.ADDED,
    mode: str | None = MODE_REGULAR,
    **fields: object,
) -> ChangedFile:
    return ChangedFile.model_validate({"path": path, "kind": kind, "mode": mode, **fields})


def rules(*changes: ChangedFile) -> list[tuple[str, str]]:
    return [(v.rule, v.path) for v in check_oracle_changes(changes, "T-0001")]


def test_the_directory_is_the_ticket_s_own() -> None:
    assert acceptance_dir("T-0001") == DIR
    assert acceptance_dir("T-12345") == "tests/acceptance/T-12345/"


@pytest.mark.parametrize("ticket", ["", "T-1", "../x", "T-0001/", "t-0001"])
def test_a_bad_ticket_id_is_a_programming_error(ticket: str) -> None:
    with pytest.raises(ValueError, match="not a ticket id"):
        acceptance_dir(ticket)
    with pytest.raises(ValueError, match="not a ticket id"):
        check_oracle_changes([], ticket)


def test_a_symlink_is_a_symlink_wherever_it_is() -> None:
    assert rules(
        change(f"{DIR}a", mode=MODE_SYMLINK),
        change("greet.py", mode=MODE_SYMLINK),
        change(f"{DIR}b", ChangeKind.MODIFIED, mode=MODE_SYMLINK),
        change(f"{DIR}c", ChangeKind.TYPE_CHANGED, mode=MODE_SYMLINK),
    ) == [
        (ORACLE_SYMLINK, "greet.py"),
        (ORACLE_SYMLINK, f"{DIR}a"),
        (ORACLE_SYMLINK, f"{DIR}b"),
        (ORACLE_SYMLINK, f"{DIR}c"),
    ]


def test_renames_and_type_changes_are_not_new_files() -> None:
    renamed = change(f"{DIR}b.py", ChangeKind.RENAMED, old_path="tests/old.py")
    assert rules(renamed, change(f"{DIR}c", ChangeKind.TYPE_CHANGED)) == [
        (ORACLE_PATH, f"{DIR}b.py"),
        (ORACLE_PATH, f"{DIR}c"),
    ]


def test_the_path_must_be_a_clean_file_below_the_directory() -> None:
    assert rules(
        change(DIR),
        change(f"{DIR}../T-0002/x.py"),
        change(f"{DIR}./x.py"),
        change(f"{DIR}a//b.py"),
        change(f"{DIR}.hidden/x.py"),
        change("tests/acceptance/T-0001"),
    ) == [
        (ORACLE_PATH, "tests/acceptance/T-0001"),
        (ORACLE_PATH, DIR),
        (ORACLE_PATH, f"{DIR}../T-0002/x.py"),
        (ORACLE_PATH, f"{DIR}./x.py"),
        (ORACLE_PATH, f"{DIR}a//b.py"),
    ]  # a hidden directory is a normal name


def test_binary_is_reported_before_size_and_one_violation_per_change() -> None:
    big_binary = change(f"{DIR}x.bin", is_binary=True, size=MAX_ORACLE_FILE_BYTES + 1)
    assert rules(big_binary) == [(ORACLE_BINARY, f"{DIR}x.bin")]


def test_the_size_limit_is_inclusive_and_an_unknown_size_is_fine() -> None:
    assert rules(change(f"{DIR}a.py", size=MAX_ORACLE_FILE_BYTES)) == []
    assert rules(change(f"{DIR}b.py", mode=MODE_EXECUTABLE)) == []
    assert rules(change(f"{DIR}c.py", size=MAX_ORACLE_FILE_BYTES + 1)) == [
        (ORACLE_SIZE, f"{DIR}c.py")
    ]


def test_too_many_files_sorts_first_next_to_the_other_violations() -> None:
    changes = [change(f"{DIR}test_{n:03}.py") for n in range(MAX_ORACLE_FILES)]
    assert check_oracle_changes(changes, "T-0001") == []  # exactly the maximum is fine
    over = [*changes, change("greet.py")]
    assert rules(*over) == [(ORACLE_FILES, ""), (ORACLE_PATH, "greet.py")]


def test_messages_tell_the_architect_what_to_do() -> None:
    [violation] = check_oracle_changes([change("greet.py", ChangeKind.MODIFIED)], "T-0001")
    assert DIR in violation.message and "modified" in violation.message
    [too_many] = check_oracle_changes(
        [change(f"{DIR}t{n}.py") for n in range(MAX_ORACLE_FILES + 1)], "T-0001"
    )
    assert str(MAX_ORACLE_FILES + 1) in too_many.message
