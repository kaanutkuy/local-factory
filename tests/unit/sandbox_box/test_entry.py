"""lf-entry in-process: argument parsing, the file commands and exec (no git needed).

Inside the box these run under ``python3 -I`` in a container, invisible to host coverage; here
``entry.main`` is called directly on ``tmp_path``.
"""

from __future__ import annotations

import io
import os
import signal
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import NoReturn

import pytest

from local_factory.sandbox import entry
from local_factory.sandbox.protocol import (
    EXIT_CANNOT_EXECUTE,
    EXIT_NOT_FOUND,
    EXIT_OK,
    EXIT_REFUSED,
    EXIT_USAGE,
    MAX_DETAIL,
)


def last_error(stderr: str) -> str:
    line = stderr.splitlines()[-1]
    assert line.startswith("error: ")
    assert len(line) <= MAX_DETAIL
    return line


def feed_stdin(monkeypatch: pytest.MonkeyPatch, data: bytes) -> None:
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(data)))


@pytest.fixture
def root(tmp_path: Path) -> Path:
    work = tmp_path / "work"
    work.mkdir()
    return work


@pytest.fixture(autouse=True)
def keep_signals() -> Iterator[None]:
    """exec resets SIGPIPE/SIGXFSZ; make sure no test leaves them changed for pytest."""
    before = {signum: signal.getsignal(signum) for signum in entry._RESTORED_SIGNALS}
    yield
    for signum, handler in before.items():
        assert signal.getsignal(signum) == handler
        signal.signal(signum, handler)


# --------------------------------------------------------------------------- arguments


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        ([], "no command given"),
        (["frobnicate"], "unknown command 'frobnicate'"),
        (["write", "--root"], "--root needs a value"),
        (["write", "--root", "/w", "--path", "a", "--path", "b"], "--path given twice"),
        (["write", "--root", "/w", "positional"], "unexpected argument 'positional'"),
        (["write", "--root", "/w", "--bogus", "x"], "unexpected argument '--bogus'"),
        (["write", "--root", "/w"], "missing --path"),
        (["materialize", "--ref", "r"], "missing --bundle, --dest"),
        (["write", "--root", "/w", "--path", "p", "--", "ls"], "takes no command"),
        (["exec"], "needs -- CMD"),
        (["exec", "--"], "empty command after --"),
        (["materialize", "--bundle", "b", "--ref", "r", "--dest", "d", "--"], "empty command"),
        (["read", "--root", "/w", "--path", "p", "--max-bytes", "-1"], "non-negative integer"),
        (["read", "--root", "/w", "--path", "p", "--max-bytes", "1e3"], "non-negative integer"),
        (["read", "--root", "/w", "--path", "p", "--max-bytes", "²"], "non-negative"),
    ],
)
def test_bad_arguments_exit_64(
    argv: list[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert entry.main(argv) == EXIT_USAGE
    line = last_error(capsys.readouterr().err)
    assert message in line
    assert "usage: lf-entry" in line


def test_option_values_are_taken_verbatim(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    feed_stdin(monkeypatch, b"dash\n")
    assert entry.main(["write", "--root", str(root), "--path", "--root"]) == EXIT_OK
    assert (root / "--root").read_text() == "dash\n"
    feed_stdin(monkeypatch, b"minus\n")
    assert entry.main(["write", "--path", "-x", "--root", str(root)]) == EXIT_OK
    assert (root / "-x").read_text() == "minus\n"


# --------------------------------------------------------------------------- write / read / delete


def test_write_creates_parents_and_keeps_text_exact(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = "VALUE = 'ünïcode'\r\nsecond line\n"
    feed_stdin(monkeypatch, content.encode())
    assert entry.main(["write", "--root", str(root), "--path", "src/new/mod.py"]) == EXIT_OK
    assert (root / "src" / "new" / "mod.py").read_bytes() == content.encode()


def test_write_refuses_invalid_utf8(
    root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    feed_stdin(monkeypatch, b"\xff\xfe")
    assert entry.main(["write", "--root", str(root), "--path", "x.txt"]) == EXIT_REFUSED
    assert last_error(capsys.readouterr().err) == "error: the content is not valid UTF-8"
    assert not (root / "x.txt").exists()


@pytest.mark.parametrize("bad", ["../escape.txt", "/etc/passwd", "src/../../x", "a\x00b"])
def test_file_commands_refuse_paths_outside_the_root(
    root: Path, bad: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    feed_stdin(monkeypatch, b"pwned")
    for command in ("write", "read", "delete"):
        assert entry.main([command, "--root", str(root), "--path", bad]) == EXIT_REFUSED
        assert last_error(capsys.readouterr().err)
    assert not (root.parent / "escape.txt").exists()


def test_read_writes_the_bytes_to_stdout(
    root: Path, capfdbinary: pytest.CaptureFixture[bytes]
) -> None:
    (root / "data.bin").write_bytes(b"\x00\xffraw\n")
    assert entry.main(["read", "--root", str(root), "--path", "data.bin"]) == EXIT_OK
    assert capfdbinary.readouterr().out == b"\x00\xffraw\n"


def test_read_enforces_max_bytes(root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (root / "big.txt").write_bytes(b"b" * 11)
    args = ["read", "--root", str(root), "--path", "big.txt"]
    assert entry.main([*args, "--max-bytes", "11"]) == EXIT_OK
    assert capsys.readouterr().out == "b" * 11
    assert entry.main([*args, "--max-bytes", "10"]) == EXIT_REFUSED
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "larger than the limit of 10 bytes" in last_error(captured.err)


def test_read_defaults_to_one_megabyte(root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (root / "big.txt").write_bytes(b"b" * 1_000_001)
    assert entry.main(["read", "--root", str(root), "--path", "big.txt"]) == EXIT_REFUSED
    assert "1000000 bytes" in last_error(capsys.readouterr().err)


def test_delete_removes_files_but_not_directories(
    root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (root / "src").mkdir()
    (root / "gone.txt").write_text("x")
    assert entry.main(["delete", "--root", str(root), "--path", "gone.txt"]) == EXIT_OK
    assert not (root / "gone.txt").exists()
    assert entry.main(["delete", "--root", str(root), "--path", "gone.txt"]) == EXIT_REFUSED
    assert "no such file" in last_error(capsys.readouterr().err)
    assert entry.main(["delete", "--root", str(root), "--path", "src"]) == EXIT_REFUSED
    assert "directory" in last_error(capsys.readouterr().err)
    assert (root / "src").is_dir()


def test_unexpected_failures_are_reported_not_raised(
    root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def boom(*args: object, **kwargs: object) -> NoReturn:
        raise RuntimeError("boom\nsecond line")

    def silent(*args: object, **kwargs: object) -> NoReturn:
        raise RuntimeError

    monkeypatch.setattr(entry, "read_file", boom)
    monkeypatch.setattr(entry, "delete_file", silent)
    monkeypatch.setattr(entry, "write_file", boom)
    feed_stdin(monkeypatch, b"x")
    assert entry.main(["read", "--root", str(root), "--path", "p"]) == EXIT_REFUSED
    assert last_error(capsys.readouterr().err) == "error: RuntimeError: boom second line"
    assert entry.main(["delete", "--root", str(root), "--path", "p"]) == EXIT_REFUSED
    assert last_error(capsys.readouterr().err) == "error: RuntimeError"
    assert entry.main(["write", "--root", str(root), "--path", "p"]) == EXIT_REFUSED
    assert last_error(capsys.readouterr().err) == "error: RuntimeError: boom second line"


def test_failure_lines_are_bounded(root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    long_name = "d" * 3000
    assert entry.main(["delete", "--root", str(root), "--path", long_name]) == EXIT_REFUSED
    assert len(last_error(capsys.readouterr().err)) <= MAX_DETAIL


# --------------------------------------------------------------------------- exec


class Replaced(BaseException):
    """Stands in for a successful exec: the process image is gone, nothing after it runs."""


def test_exec_replaces_the_process_with_stderr_joined(
    monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    seen: dict[str, object] = {}
    before = {signum: signal.getsignal(signum) for signum in entry._RESTORED_SIGNALS}
    real_dup = os.dup
    duplicated: list[int] = []

    def dup(fd: int) -> int:
        new = real_dup(fd)
        duplicated.append(new)
        return new

    def execvp(file: str, args: list[str]) -> NoReturn:
        seen["call"] = (file, list(args))
        seen["signals"] = [signal.getsignal(s) for s in entry._RESTORED_SIGNALS]
        seen["inheritable"] = [os.get_inheritable(fd) for fd in duplicated]
        os.write(2, b"to-stderr\n")
        raise Replaced

    monkeypatch.setattr(os, "dup", dup)
    monkeypatch.setattr(os, "execvp", execvp)
    try:
        with pytest.raises(Replaced):
            entry.main(["exec", "--", "make", "-j", "--", "x"])
        assert seen["call"] == ("make", ["make", "-j", "--", "x"])
        assert seen["signals"] == [signal.SIG_DFL] * len(entry._RESTORED_SIGNALS)
        assert seen["inheritable"] == [False]  # the saved stderr does not survive the exec
        captured = capfd.readouterr()
        assert "to-stderr" in captured.out  # stderr went to stdout
    finally:
        for fd in duplicated:
            os.close(fd)
        for signum, handler in before.items():  # the "replaced" process never restores them
            signal.signal(signum, handler)


def test_exec_of_a_missing_command_exits_127_with_the_reason_in_the_output(
    capfd: pytest.CaptureFixture[str],
) -> None:
    assert entry.main(["exec", "--", "lf-no-such-tool-x"]) == EXIT_NOT_FOUND
    captured = capfd.readouterr()
    assert "error: cannot run 'lf-no-such-tool-x'" in captured.out
    assert captured.err == ""


def test_exec_of_a_non_executable_exits_126(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    script = tmp_path / "script.sh"
    script.write_text("#!/bin/sh\necho hi\n")
    script.chmod(0o644)
    assert entry.main(["exec", "--", str(script)]) == EXIT_CANNOT_EXECUTE
    assert "Permission denied" in capfd.readouterr().out
