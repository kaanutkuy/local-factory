"""Unit tests for FakeBackend file actions and their confinement to the worker directory."""

from __future__ import annotations

from pathlib import Path

import pytest

from local_factory.agents.backend import WorkerHandle
from local_factory.agents.fake import SYMLINK_LOOP, DeleteFile, WriteFile

from .conftest import run_script


def tree(root: Path) -> list[str]:
    """Every entry under `root`, relative and sorted (symlinks are not followed)."""
    return sorted(str(p.relative_to(root)) for p in root.rglob("*"))


# --------------------------------------------------------------------------- WriteFile


def test_write_creates_nested_directories(worker: WorkerHandle) -> None:
    result = run_script(worker, WriteFile("a/b/c/deep.txt", "deep"))
    assert result.status == "ok", result.error
    assert (worker.work_dir / "a" / "b" / "c" / "deep.txt").read_text() == "deep"


def test_write_overwrites_an_existing_file(worker: WorkerHandle) -> None:
    (worker.work_dir / "f.txt").write_text("old content that is longer")
    assert run_script(worker, WriteFile("f.txt", "new")).status == "ok"
    assert (worker.work_dir / "f.txt").read_text() == "new"


def test_write_stores_exact_utf8_bytes(worker: WorkerHandle) -> None:
    content = "café\r\nline two\n\ttab"
    assert run_script(worker, WriteFile("f.txt", content)).status == "ok"
    assert (worker.work_dir / "f.txt").read_bytes() == content.encode("utf-8")


def test_write_of_unencodable_content_is_an_error(worker: WorkerHandle) -> None:
    result = run_script(worker, WriteFile("f.txt", "bad \udc80 surrogate"))
    assert result.status == "error"
    assert result.error is not None
    assert "UnicodeEncodeError" in result.error
    assert not (worker.work_dir / "f.txt").exists()


def test_write_with_in_bounds_dotdot_is_allowed(worker: WorkerHandle) -> None:
    result = run_script(worker, WriteFile("src/../top.txt", "x"))
    assert result.status == "ok", result.error
    assert tree(worker.work_dir) == ["top.txt"]


def test_write_onto_a_directory_is_an_error(worker: WorkerHandle) -> None:
    (worker.work_dir / "d").mkdir()
    result = run_script(worker, WriteFile("d", "x"))
    assert result.status == "error"
    assert (worker.work_dir / "d").is_dir()


def test_write_below_a_regular_file_is_an_error(worker: WorkerHandle) -> None:
    (worker.work_dir / "f").write_text("file")
    result = run_script(worker, WriteFile("f/child.txt", "x"))
    assert result.status == "error"
    assert (worker.work_dir / "f").read_text() == "file"


def test_write_through_a_symlinked_parent_inside_the_worker(worker: WorkerHandle) -> None:
    (worker.work_dir / "real").mkdir()
    (worker.work_dir / "alias").symlink_to("real")
    result = run_script(worker, WriteFile("alias/f.txt", "x"))
    assert result.status == "ok", result.error
    assert (worker.work_dir / "real" / "f.txt").read_text() == "x"


def test_write_through_a_symlink_to_an_inside_file_updates_the_target(
    worker: WorkerHandle,
) -> None:
    (worker.work_dir / "target.txt").write_text("old")
    (worker.work_dir / "link.txt").symlink_to("target.txt")
    assert run_script(worker, WriteFile("link.txt", "new")).status == "ok"
    assert (worker.work_dir / "target.txt").read_text() == "new"
    assert (worker.work_dir / "link.txt").is_symlink()


def test_write_through_an_inside_dangling_symlink_creates_the_target(
    worker: WorkerHandle,
) -> None:
    (worker.work_dir / "link.txt").symlink_to("sub/later.txt")
    assert run_script(worker, WriteFile("link.txt", "x")).status == "ok"
    assert (worker.work_dir / "sub" / "later.txt").read_text() == "x"


def test_write_via_a_symlink_to_the_worker_root_stays_inside(worker: WorkerHandle) -> None:
    (worker.work_dir / "self").symlink_to(".")
    assert run_script(worker, WriteFile("self/f.txt", "x")).status == "ok"
    assert (worker.work_dir / "f.txt").read_text() == "x"


# --------------------------------------------------------------------------- confinement


ESCAPE_MESSAGE = "path escapes the worker directory"


@pytest.mark.parametrize(
    ("path", "message"),
    [
        ("../escape.txt", ESCAPE_MESSAGE),
        ("a/../../escape.txt", ESCAPE_MESSAGE),
        ("a/b/../../../escape.txt", ESCAPE_MESSAGE),
        ("/tmp/escape.txt", "absolute paths are not allowed"),
        ("", "does not name a file"),
        (".", "does not name a file"),
        ("./", "does not name a file"),
        ("..", "does not name a file"),
        ("a/..", "does not name a file"),
    ],
)
def test_paths_that_do_not_name_an_inside_file_are_rejected(
    worker: WorkerHandle, outside: Path, path: str, message: str
) -> None:
    for action in (WriteFile(path, "x"), DeleteFile(path)):
        result = run_script(worker, action)
        assert result.status == "error", action
        assert result.error is not None
        assert message in result.error
    assert tree(worker.work_dir.parent) == ["outside", "work"]


def test_absolute_path_is_rejected_even_inside_the_worker(worker: WorkerHandle) -> None:
    inside = worker.work_dir / "f.txt"
    result = run_script(worker, WriteFile(str(inside), "x"))
    assert result.status == "error"
    assert not inside.exists()


def test_nul_byte_in_path_is_an_error(worker: WorkerHandle) -> None:
    for action in (WriteFile("a\x00b", "x"), DeleteFile("a\x00b")):
        result = run_script(worker, action)
        assert (
            result.error
            == f"action 1 ({type(action).__name__} 'a\\x00b'): path contains a NUL byte"
        )


def test_symlinked_parent_escaping_the_worker_is_rejected(
    worker: WorkerHandle, outside: Path
) -> None:
    (worker.work_dir / "link").symlink_to(outside)
    for action in (WriteFile("link/pwned.txt", "x"), DeleteFile("link/pwned.txt")):
        result = run_script(worker, action)
        assert result.status == "error"
        assert result.error is not None
        assert ESCAPE_MESSAGE in result.error
    assert tree(outside) == []


def test_symlinked_file_escaping_the_worker_is_rejected(
    worker: WorkerHandle, outside: Path
) -> None:
    secret = outside / "secret.txt"
    secret.write_text("keep")
    (worker.work_dir / "link.txt").symlink_to(secret)
    assert run_script(worker, WriteFile("link.txt", "pwned")).status == "error"
    assert run_script(worker, DeleteFile("link.txt")).status == "error"
    assert secret.read_text() == "keep"
    assert (worker.work_dir / "link.txt").is_symlink()


def test_dangling_symlink_escaping_the_worker_is_rejected(
    worker: WorkerHandle, outside: Path
) -> None:
    (worker.work_dir / "link.txt").symlink_to(outside / "created.txt")
    assert run_script(worker, WriteFile("link.txt", "x")).status == "error"
    assert tree(outside) == []


def test_relative_symlink_climbing_out_is_rejected(worker: WorkerHandle, outside: Path) -> None:
    (worker.work_dir / "up").symlink_to("../outside")
    assert run_script(worker, WriteFile("up/f.txt", "x")).status == "error"
    assert tree(outside) == []


def test_path_through_an_outside_symlink_back_inside_is_rejected(
    worker: WorkerHandle, outside: Path
) -> None:
    target = worker.work_dir / "target.txt"
    target.write_text("keep")
    (worker.work_dir / "out").symlink_to(outside)
    (outside / "back").symlink_to(target)
    assert run_script(worker, WriteFile("out/back", "x")).status == "error"
    assert run_script(worker, DeleteFile("out/back")).status == "error"
    assert target.read_text() == "keep"
    assert (outside / "back").is_symlink()


def test_symlink_to_the_worker_root_is_not_a_file(worker: WorkerHandle) -> None:
    (worker.work_dir / "self").symlink_to(".")
    for action in (WriteFile("self", "x"), DeleteFile("self")):
        result = run_script(worker, action)
        assert result.status == "error"
    assert (worker.work_dir / "self").is_symlink()


@pytest.mark.parametrize(
    "action",
    [WriteFile("a", "x"), WriteFile("a/f.txt", "x"), DeleteFile("a"), DeleteFile("b/f.txt")],
)
def test_symlink_loop_is_one_error_on_every_python_version(
    worker: WorkerHandle, action: WriteFile | DeleteFile
) -> None:
    """Path.resolve() raises RuntimeError for loops on 3.12 but not on 3.13+; we report one way."""
    (worker.work_dir / "a").symlink_to("b")
    (worker.work_dir / "b").symlink_to("a")
    result = run_script(worker, action)
    assert result.error == f"action 1 ({type(action).__name__} {action.path!r}): {SYMLINK_LOOP}"
    assert tree(worker.work_dir) == ["a", "b"]


# --------------------------------------------------------------------------- DeleteFile


def test_delete_removes_a_file(worker: WorkerHandle) -> None:
    (worker.work_dir / "sub").mkdir()
    (worker.work_dir / "sub" / "f.txt").write_text("x")
    assert run_script(worker, DeleteFile("sub/f.txt")).status == "ok"
    assert tree(worker.work_dir) == ["sub"]


def test_delete_of_a_missing_file_is_an_error(worker: WorkerHandle) -> None:
    result = run_script(worker, DeleteFile("nope/missing.txt"))
    assert result.error == "action 1 (DeleteFile 'nope/missing.txt'): no such file"


def test_delete_of_a_directory_is_an_error(worker: WorkerHandle) -> None:
    (worker.work_dir / "d").mkdir()
    (worker.work_dir / "d" / "keep.txt").write_text("x")
    result = run_script(worker, DeleteFile("d"))
    assert result.status == "error"
    assert result.error is not None
    assert "is a directory" in result.error
    assert (worker.work_dir / "d" / "keep.txt").exists()


def test_delete_of_an_inside_symlink_removes_the_link_not_the_target(
    worker: WorkerHandle,
) -> None:
    (worker.work_dir / "target.txt").write_text("keep")
    (worker.work_dir / "link.txt").symlink_to("target.txt")
    (worker.work_dir / "dir").mkdir()
    (worker.work_dir / "dirlink").symlink_to("dir")
    assert run_script(worker, DeleteFile("link.txt"), DeleteFile("dirlink")).status == "ok"
    assert tree(worker.work_dir) == ["dir", "target.txt"]


def test_delete_of_an_inside_dangling_symlink_removes_it(worker: WorkerHandle) -> None:
    (worker.work_dir / "link.txt").symlink_to("missing.txt")
    assert run_script(worker, DeleteFile("link.txt")).status == "ok"
    assert tree(worker.work_dir) == []


def test_delete_through_a_symlinked_parent_inside_the_worker(worker: WorkerHandle) -> None:
    (worker.work_dir / "real").mkdir()
    (worker.work_dir / "real" / "f.txt").write_text("x")
    (worker.work_dir / "alias").symlink_to("real")
    assert run_script(worker, DeleteFile("alias/f.txt")).status == "ok"
    assert tree(worker.work_dir) == ["alias", "real"]


def test_delete_outside_the_worker_is_rejected(worker: WorkerHandle, outside: Path) -> None:
    victim = outside / "victim.txt"
    victim.write_text("keep")
    assert run_script(worker, DeleteFile("../outside/victim.txt")).status == "error"
    assert victim.read_text() == "keep"
