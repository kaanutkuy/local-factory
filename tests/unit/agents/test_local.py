"""Unit tests for local_factory.agents.local: the sync file helpers and LocalExecutor.

Path confinement is also covered end to end through FakeBackend in test_fake_paths.py.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
from collections.abc import Coroutine
from pathlib import Path
from typing import Any

import pytest

from local_factory.agents import _command
from local_factory.agents.local import (
    DEFAULT_MAX_OUTPUT_BYTES,
    ESCAPES_ROOT,
    MAX_TOOL_ERROR_CHARS,
    SYMLINK_LOOP,
    LocalExecutor,
    clip,
    delete_file,
    read_file,
    write_file,
)
from local_factory.agents.tools import CommandResult, ToolError, ToolExecutor

PY = sys.executable


def tree(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*"))


def run[T](coro: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coro)


def command(executor: LocalExecutor, *argv: str, timeout_s: float = 30.0) -> CommandResult:
    return run(executor.run_command(argv, timeout_s=timeout_s))


@pytest.fixture
def root(tmp_path: Path) -> Path:
    path = tmp_path / "work"
    path.mkdir()
    return path


@pytest.fixture
def outside(tmp_path: Path) -> Path:
    path = tmp_path / "outside"
    path.mkdir()
    (path / "secret.txt").write_text("keep")
    return path


# --------------------------------------------------------------------------- write_file


def test_write_then_read_round_trips_exact_bytes(root: Path) -> None:
    content = "café\r\nline two\n\ttab"
    write_file(root, "a/b/c.txt", content)
    assert (root / "a" / "b" / "c.txt").read_bytes() == content.encode("utf-8")
    assert read_file(root, "a/b/c.txt", max_bytes=100) == content.encode("utf-8")


def test_write_truncates_an_existing_file(root: Path) -> None:
    (root / "f.txt").write_text("old content that is longer")
    write_file(root, "f.txt", "new")
    assert (root / "f.txt").read_text() == "new"


def test_write_of_unencodable_content_creates_nothing(root: Path) -> None:
    with pytest.raises(ToolError, match=r"^UnicodeEncodeError: "):
        write_file(root, "sub/f.txt", "bad \udc80 surrogate")
    assert tree(root) == []


def test_write_onto_a_directory_is_refused(root: Path) -> None:
    (root / "d").mkdir()
    with pytest.raises(ToolError, match=r"^is a directory; only files can be written$"):
        write_file(root, "d", "x")


def test_write_below_a_regular_file_is_refused(root: Path) -> None:
    (root / "f").write_text("file")
    with pytest.raises(ToolError, match=r"^(FileExistsError|NotADirectoryError): "):
        write_file(root, "f/child.txt", "x")
    assert (root / "f").read_text() == "file"


def test_write_to_a_fifo_never_blocks(root: Path) -> None:
    fifo = root / "pipe"
    os.mkfifo(fifo)
    with pytest.raises(ToolError, match=r"^OSError: "):  # no reader: ENXIO instead of hanging
        write_file(root, "pipe", "x")
    reader = os.open(fifo, os.O_RDONLY | os.O_NONBLOCK)
    try:
        with pytest.raises(ToolError, match=r"^not a regular file; only files can be written$"):
            write_file(root, "pipe", "x")
    finally:
        os.close(reader)


# --------------------------------------------------------------------------- read_file


def test_read_returns_raw_bytes(root: Path) -> None:
    (root / "bin").write_bytes(b"\xff\x00\xfe")
    assert read_file(root, "bin", max_bytes=3) == b"\xff\x00\xfe"
    assert read_file(root, "bin", max_bytes=10) == b"\xff\x00\xfe"


def test_read_over_the_limit_is_refused(root: Path) -> None:
    (root / "big").write_bytes(b"x" * 11)
    with pytest.raises(ToolError, match=r"^file is larger than the limit of 10 bytes$"):
        read_file(root, "big", max_bytes=10)
    (root / "empty").write_bytes(b"")
    assert read_file(root, "empty", max_bytes=0) == b""


def test_read_negative_limit_is_a_caller_error(root: Path) -> None:
    with pytest.raises(ValueError, match="max_bytes"):
        read_file(root, "x", max_bytes=-1)


@pytest.mark.parametrize(
    ("setup", "message"),
    [
        ("missing", "^no such file$"),
        ("dir", "^is a directory; only files can be read$"),
        ("fifo", "^not a regular file; only files can be read$"),
    ],
)
def test_read_of_a_non_file_is_refused(root: Path, setup: str, message: str) -> None:
    if setup == "dir":
        (root / "x").mkdir()
    elif setup == "fifo":
        os.mkfifo(root / "x")
    with pytest.raises(ToolError, match=message):
        read_file(root, "x", max_bytes=100)


def test_read_through_symlinks_inside_the_root(root: Path) -> None:
    (root / "real").mkdir()
    (root / "real" / "f.txt").write_text("x")
    (root / "alias").symlink_to("real")
    (root / "link.txt").symlink_to("real/f.txt")
    assert read_file(root, "alias/f.txt", max_bytes=10) == b"x"
    assert read_file(root, "link.txt", max_bytes=10) == b"x"


# --------------------------------------------------------------------------- delete_file


def test_delete_removes_files_and_links_but_never_link_targets(root: Path) -> None:
    (root / "target.txt").write_text("keep")
    (root / "link.txt").symlink_to("target.txt")
    (root / "dangling").symlink_to("missing")
    (root / "f.txt").write_text("x")
    for path in ("link.txt", "dangling", "f.txt"):
        delete_file(root, path)
    assert tree(root) == ["target.txt"]


def test_delete_of_missing_or_directory_is_refused(root: Path) -> None:
    (root / "d").mkdir()
    with pytest.raises(ToolError, match=r"^no such file$"):
        delete_file(root, "missing.txt")
    with pytest.raises(ToolError, match=r"^is a directory; only files and symlinks can be"):
        delete_file(root, "d")
    assert tree(root) == ["d"]


# --------------------------------------------------------------------------- confinement


@pytest.mark.parametrize(
    ("path", "message"),
    [
        ("../outside/secret.txt", ESCAPES_ROOT),
        ("a/../../outside/secret.txt", ESCAPES_ROOT),
        ("/etc/passwd", "absolute paths are not allowed"),
        ("", "path does not name a file"),
        (".", "path does not name a file"),
        ("..", "path does not name a file"),
        ("a/..", "path does not name a file"),
        ("a\x00b", "path contains a NUL byte"),
    ],
)
def test_every_helper_refuses_paths_outside_the_root(
    root: Path, outside: Path, path: str, message: str
) -> None:
    with pytest.raises(ToolError, match=message):
        write_file(root, path, "pwned")
    with pytest.raises(ToolError, match=message):
        read_file(root, path, max_bytes=100)
    with pytest.raises(ToolError, match=message):
        delete_file(root, path)
    assert (outside / "secret.txt").read_text() == "keep"
    assert tree(root) == []


def test_symlinks_out_of_the_root_are_refused(root: Path, outside: Path) -> None:
    (root / "dir").symlink_to(outside)
    (root / "file").symlink_to(outside / "secret.txt")
    (root / "dangling").symlink_to(outside / "new.txt")
    (root / "self").symlink_to(".")
    for path in ("dir/secret.txt", "dir/new.txt", "file", "dangling", "self"):
        with pytest.raises(ToolError, match=ESCAPES_ROOT):
            write_file(root, path, "pwned")
        with pytest.raises(ToolError, match=ESCAPES_ROOT):
            read_file(root, path, max_bytes=100)
        with pytest.raises(ToolError, match=ESCAPES_ROOT):
            delete_file(root, path)
    assert tree(outside) == ["secret.txt"]
    assert (outside / "secret.txt").read_text() == "keep"


@pytest.mark.parametrize("path", ["a", "a/f.txt", "b/f.txt"])
def test_symlink_loops_are_one_message_on_every_python_version(root: Path, path: str) -> None:
    (root / "a").symlink_to("b")
    (root / "b").symlink_to("a")
    for attempt in (
        lambda: write_file(root, path, "x"),
        lambda: read_file(root, path, max_bytes=10),
        lambda: delete_file(root, path),
    ):
        with pytest.raises(ToolError) as refused:
            attempt()
        assert str(refused.value) == SYMLINK_LOOP
    assert tree(root) == ["a", "b"]


@pytest.mark.parametrize("kind", ["missing", "file", "loop"])
def test_root_must_be_an_existing_directory(tmp_path: Path, kind: str) -> None:
    root = tmp_path / "root"
    if kind == "file":
        root.write_text("")
    elif kind == "loop":
        root.symlink_to("root")
    message = f"^worker directory {root} does not exist or is not a directory$"
    with pytest.raises(ToolError, match=message):
        write_file(root, "a.txt", "x")
    with pytest.raises(ToolError, match=message):
        read_file(root, "a.txt", max_bytes=1)
    with pytest.raises(ToolError, match=message):
        delete_file(root, "a.txt")
    with pytest.raises(ToolError, match=message):
        command(LocalExecutor(root), "true")
    assert tree(tmp_path) == ([] if kind == "missing" else ["root"])


def test_relative_root_is_resolved_against_the_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "rel").mkdir()
    monkeypatch.chdir(tmp_path)
    write_file(Path("rel"), "a.txt", "x")
    assert (tmp_path / "rel" / "a.txt").read_text() == "x"


def test_error_messages_are_bounded(root: Path) -> None:
    with pytest.raises(ToolError) as refused:
        write_file(root, "p" * 100_000, "x")  # ENAMETOOLONG names the whole path
    assert len(str(refused.value)) <= MAX_TOOL_ERROR_CHARS
    long_root = root.joinpath(*["r" * 250] * 9)  # a missing root, named in the message
    with pytest.raises(ToolError) as missing:
        read_file(long_root, "x", max_bytes=1)
    assert len(str(missing.value)) == MAX_TOOL_ERROR_CHARS


@pytest.mark.parametrize(
    ("text", "limit", "expected"),
    [
        ("short", 10, "short"),
        ("exactly10!", 10, "exactly10!"),
        ("0123456789abcdef", 9, "012...def"),
        ("0123456789abcdef", 10, "012...cdef"),
        ("0123456789", 2, "..."),
    ],
)
def test_clip_keeps_head_and_tail(text: str, limit: int, expected: str) -> None:
    assert clip(text, limit) == expected


# --------------------------------------------------------------------------- LocalExecutor files


def test_executor_is_a_tool_executor_with_defaults(root: Path) -> None:
    executor = LocalExecutor(root)
    assert isinstance(executor, ToolExecutor)
    assert (executor.root, executor.max_output_bytes) == (root, DEFAULT_MAX_OUTPUT_BYTES)


def test_executor_requires_a_positive_output_bound(root: Path) -> None:
    with pytest.raises(ValueError, match="max_output_bytes"):
        LocalExecutor(root, max_output_bytes=0)


def test_executor_file_operations(root: Path) -> None:
    executor = LocalExecutor(root, max_output_bytes=8)

    async def scenario() -> str:
        await executor.write_file("d/f.txt", "héllo")
        (root / "bad").write_bytes(b"ok\xff")
        text = await executor.read_file("d/f.txt") + "|" + await executor.read_file("bad")
        await executor.write_file("d/big.txt", "123456789")
        with pytest.raises(ToolError, match="larger than the limit of 8 bytes"):
            await executor.read_file("d/big.txt")
        await executor.delete_file("d/f.txt")
        with pytest.raises(ToolError, match=ESCAPES_ROOT):
            await executor.write_file("../x", "y")
        return text

    assert run(scenario()) == "héllo|ok�"
    assert tree(root) == ["bad", "d", "d/big.txt"]


# --------------------------------------------------------------------------- LocalExecutor commands


def test_command_runs_in_the_resolved_root_with_empty_stdin(tmp_path: Path, root: Path) -> None:
    (tmp_path / "alias").symlink_to(root)
    code = "import os, sys; print(os.getcwd()); print(repr(sys.stdin.read()))"
    result = command(LocalExecutor(tmp_path / "alias"), PY, "-c", code)
    assert result == CommandResult(exit_code=0, output=f"{root.resolve()}\n''\n")


def test_command_arguments_are_not_interpreted_by_a_shell(root: Path) -> None:
    args = ("$HOME", ";", "rm -rf .", "`id`", "*")
    result = command(LocalExecutor(root), PY, "-c", "import sys; print(sys.argv[1:])", *args)
    assert result.output == f"{list(args)}\n"


def test_command_environment_is_minimal_and_its_home_is_removed(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LF_CANARY_TOKEN", "canary-0d1e")
    monkeypatch.setenv("GIT_DIR", "/somewhere")
    code = (
        "import json, os; env = dict(os.environ); env['home?'] = os.path.isdir(env['HOME']); "
        "print(json.dumps(env))"
    )
    result = command(LocalExecutor(root), PY, "-c", code)
    seen = json.loads(result.output)
    home = seen.pop("HOME")
    assert seen.pop("home?") is True
    assert seen == {"PATH": os.environ["PATH"], "LANG": "C.UTF-8", "TMPDIR": home}
    assert Path(home).name.startswith("local-factory-cmd-")
    assert not Path(home).exists()


def test_command_env_falls_back_to_the_default_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PATH")
    assert _command.command_env("/h") == {
        "PATH": os.defpath,
        "LANG": "C.UTF-8",
        "HOME": "/h",
        "TMPDIR": "/h",
    }


def test_stdout_and_stderr_are_interleaved_in_order(root: Path) -> None:
    code = (
        "import sys\n"
        "for i in range(3):\n"
        "    print(f'out{i}', flush=True); print(f'err{i}', file=sys.stderr, flush=True)\n"
        "raise SystemExit(4)"
    )
    result = command(LocalExecutor(root), PY, "-c", code)
    assert result == CommandResult(exit_code=4, output="out0\nerr0\nout1\nerr1\nout2\nerr2\n")
    assert not result.ok


def test_killed_by_a_signal_is_a_negative_exit_code(root: Path) -> None:
    result = command(LocalExecutor(root), "sh", "-c", "kill -9 $$")
    assert (result.exit_code, result.timed_out) == (-signal.SIGKILL, False)


def test_only_the_tail_of_the_output_is_kept(root: Path) -> None:
    result = command(LocalExecutor(root, max_output_bytes=10), PY, "-c", "print('a' * 100 + 'END')")
    assert result.output == "aaaaaaEND\n"


def test_the_tail_never_starts_inside_a_character(root: Path) -> None:
    executor = LocalExecutor(root, max_output_bytes=10)
    code = "import sys; sys.stdout.buffer.write('é'.encode() * 20 + b'x')"
    assert command(executor, PY, "-c", code).output == "ééééx"  # 9 of the last 10 bytes
    garbage = "import sys; sys.stdout.buffer.write(b'\\x80' * 20)"
    assert command(executor, PY, "-c", garbage).output == "�" * 7  # at most 3 dropped


def test_timeout_kills_the_whole_process_group_and_keeps_the_output(root: Path) -> None:
    script = "echo started; sleep 30 & sleep 30"  # the background sleep holds the output too
    result = command(LocalExecutor(root), "sh", "-c", script, timeout_s=1.0)
    assert result == CommandResult(exit_code=None, output="started\n", timed_out=True)
    assert not result.ok


def test_background_processes_are_killed_when_the_command_exits(root: Path) -> None:
    """Without killing the group, the background sleep would hold the output pipe open."""
    result = command(LocalExecutor(root), "sh", "-c", "sleep 30 & echo done")
    assert result == CommandResult(exit_code=0, output="done\n")


def test_a_process_that_escapes_the_group_is_reported_not_waited_for(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_command, "DRAIN_GRACE_S", 0.2)
    code = (
        "import subprocess, sys\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'],"
        " start_new_session=True)\n"
        "open('escaped.pid', 'w').write(str(child.pid))\n"
        "print('leader done', flush=True)\n"
    )
    result = command(LocalExecutor(root), PY, "-c", code)
    os.kill(int((root / "escaped.pid").read_text()), signal.SIGKILL)
    assert result.exit_code == 0
    assert result.output.startswith("leader done\n")
    assert "[local-factory] the command exited but its output stayed open" in result.output


def test_cancelling_the_caller_kills_the_command(root: Path) -> None:
    executor = LocalExecutor(root)

    async def scenario() -> BaseException | CommandResult:
        victim = asyncio.ensure_future(
            executor.run_command(["sh", "-c", "echo $$ > victim.pid; exec sleep 30"], timeout_s=60)
        )
        started = await executor.run_command(
            ["sh", "-c", "while [ ! -s victim.pid ]; do :; done"], timeout_s=30
        )
        assert started.ok
        victim.cancel()
        (outcome,) = await asyncio.gather(victim, return_exceptions=True)
        return outcome

    assert isinstance(run(scenario()), asyncio.CancelledError)
    pid = int((root / "victim.pid").read_text())
    assert not Path(f"/proc/{pid}").exists()  # killed and reaped before the cancellation ended


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (("definitely-not-a-real-binary-xyz",), r"^FileNotFoundError: "),
        (("./script.sh",), r"^PermissionError: "),
        (("echo", "a\x00b"), r"^ValueError: embedded null byte$"),
    ],
)
def test_commands_that_cannot_start_are_tool_errors(
    root: Path, argv: tuple[str, ...], message: str
) -> None:
    (root / "script.sh").write_text("#!/bin/sh\nexit 0\n")
    (root / "script.sh").chmod(0o644)
    with pytest.raises(ToolError, match=message):
        command(LocalExecutor(root), *argv)


def test_huge_unstartable_command_keeps_the_error_bounded(root: Path) -> None:
    with pytest.raises(ToolError) as refused:
        command(LocalExecutor(root), "b" * 50_000)
    assert len(str(refused.value)) <= MAX_TOOL_ERROR_CHARS


def test_invalid_command_requests(root: Path) -> None:
    executor = LocalExecutor(root)
    with pytest.raises(ToolError, match=r"^empty argv$"):
        command(executor)
    with pytest.raises(TypeError, match="not a single string"):
        run(executor.run_command("ls", timeout_s=1))
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="timeout_s"):
            command(executor, "true", timeout_s=bad)
