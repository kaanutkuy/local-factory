"""BareRepo.export_tree: byte-exact, attribute-free, never writes through symlinks."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from local_factory.gitops import bare as bare_module
from local_factory.gitops.bare import BareRepo, TreeEntry

from .conftest import Git, bare_from, make_seed

pytestmark = pytest.mark.integration

ATTRIBUTES = "* text eol=crlf ident\nignored.txt export-ignore\nsubst.txt export-subst\n"


@pytest.fixture
def rich(tmp_path: Path, git: Git) -> tuple[BareRepo, str, Path]:
    outside = tmp_path / "outside"
    outside.mkdir()
    files: dict[str, str | bytes] = {
        "lf.txt": "one\ntwo\n",
        "ignored.txt": "still exported\n",
        "subst.txt": "$Format:%H$ $Id$\n",
        "bin/tool": "#!/bin/sh\necho tool\n",
        "deep/a/b/c/d.txt": "deep\n",
        "blob.bin": bytes(range(256)),
    }
    seed = make_seed(git, tmp_path / "rich", files)
    (seed / "bin" / "tool").chmod(0o755)
    (seed / "inside").symlink_to("deep/a")
    (seed / "escape").symlink_to(str(outside))
    (seed / ".gitattributes").write_text(ATTRIBUTES)  # added after the blobs were stored
    sub = git("rev-parse", "HEAD", cwd=seed)
    git("add", "-A", cwd=seed)
    git("update-index", "--add", "--cacheinfo", f"160000,{sub},vendor/sub", cwd=seed)
    crlf = git("hash-object", "-w", "--no-filters", "--stdin", cwd=seed, input=b"one\r\ntwo\r\n")
    git("update-index", "--add", "--cacheinfo", f"100644,{crlf},crlf.txt", cwd=seed)
    git("commit", "-q", "-m", "rich", cwd=seed)
    repo = bare_from(seed, tmp_path / "rich.git")
    return repo, repo.rev_parse("refs/heads/main"), outside


def test_export_is_byte_exact_and_ignores_attributes(
    rich: tuple[BareRepo, str, Path], tmp_path: Path
) -> None:
    repo, head, _ = rich
    dest = tmp_path / "out"
    repo.export_tree(head, dest)
    assert (dest / "lf.txt").read_bytes() == b"one\ntwo\n"  # no eol conversion
    assert (dest / "crlf.txt").read_bytes() == b"one\r\ntwo\r\n"
    assert (dest / "ignored.txt").read_text() == "still exported\n"  # export-ignore ignored
    assert (dest / "subst.txt").read_text() == "$Format:%H$ $Id$\n"  # no substitution
    assert (dest / "blob.bin").read_bytes() == bytes(range(256))
    assert (dest / "deep" / "a" / "b" / "c" / "d.txt").read_text() == "deep\n"
    assert not (dest / ".git").exists()
    for path in dest.rglob("*"):
        if path.is_file() and not path.is_symlink():
            assert path.read_bytes() == repo.read_file(head, str(path.relative_to(dest)))


def test_export_modes_symlinks_and_gitlinks(
    rich: tuple[BareRepo, str, Path], tmp_path: Path
) -> None:
    repo, head, outside = rich
    dest = tmp_path / "out"
    repo.export_tree(head, dest)
    assert os.access(dest / "bin" / "tool", os.X_OK)
    assert not os.access(dest / "lf.txt", os.X_OK)
    assert (dest / "inside").readlink() == Path("deep/a")
    assert (dest / "escape").readlink() == outside  # verbatim; never followed while writing
    assert list(outside.iterdir()) == []
    assert (dest / "vendor" / "sub").is_dir() and list((dest / "vendor" / "sub").iterdir()) == []


def test_export_in_small_chunks(
    rich: tuple[BareRepo, str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, head, _ = rich
    monkeypatch.setattr(bare_module, "_EXPORT_CHUNK_BYTES", 1)
    repo.export_tree(head, tmp_path / "chunked")
    repo.export_tree(head, tmp_path / "whole")
    monkeypatch.undo()
    chunked = sorted(p.relative_to(tmp_path / "chunked") for p in (tmp_path / "chunked").rglob("*"))
    whole = sorted(p.relative_to(tmp_path / "whole") for p in (tmp_path / "whole").rglob("*"))
    assert chunked == whole and len(whole) > 10


def test_export_destination_rules(rich: tuple[BareRepo, str, Path], tmp_path: Path) -> None:
    repo, head, _ = rich
    busy = tmp_path / "busy"
    busy.mkdir()
    (busy / "keep.txt").write_text("keep")
    with pytest.raises(FileExistsError, match="not empty"):
        repo.export_tree(head, busy)
    assert [p.name for p in busy.iterdir()] == ["keep.txt"]

    real = tmp_path / "real"
    real.mkdir()
    (tmp_path / "link").symlink_to(real)
    with pytest.raises(NotADirectoryError):
        repo.export_tree(head, tmp_path / "link")
    assert list(real.iterdir()) == []

    nested = tmp_path / "new" / "nested"
    repo.export_tree(head, nested)
    assert (nested / "lf.txt").exists()


def _crafted(repo: BareRepo, git: Git, entries: list[tuple[str, bytes, str]]) -> str:
    tree = git.tree(entries, cwd=repo.path)
    return git("commit-tree", tree, "-m", "crafted", cwd=repo.path)


@pytest.mark.parametrize("name", [b"..\\evil", b".GIT", b"git~1", b"caf\xe9"])
def test_export_refuses_unsafe_paths(repo: BareRepo, git: Git, tmp_path: Path, name: bytes) -> None:
    blob = git.blob(b"pwn\n", cwd=repo.path)
    commit = _crafted(repo, git, [("100644", b"ok.txt", blob), ("100644", name, blob)])
    dest = tmp_path / "out"
    with pytest.raises(ValueError, match="refusing to export"):
        repo.export_tree(commit, dest)
    assert not dest.exists()  # validated before anything is written


def test_export_refuses_unknown_modes(
    repo: BareRepo, base: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def ls_tree(self: BareRepo, rev: str) -> list[TreeEntry]:
        return [TreeEntry("100664", "0" * 40, 1, b"x")]

    monkeypatch.setattr(BareRepo, "ls_tree", ls_tree)
    with pytest.raises(ValueError, match="unknown mode 100664"):
        repo.export_tree(base, tmp_path / "out")
