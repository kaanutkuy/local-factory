"""Unit tests for `unsafe_symlinks` and the import check built on it, with a stubbed repository
(the real git is in tests/integration/gitops/test_bundles.py and the Phase 2b acceptance tests)."""

from __future__ import annotations

from collections.abc import Iterable
from typing import cast

import pytest

from local_factory.gitops import BareRepo
from local_factory.gitops.bare import TreeEntry
from local_factory.gitops.bundles import BundleRejected, _check_symlinks, unsafe_symlinks
from local_factory.models import MODE_REGULAR, MODE_SYMLINK

LONG = "x" * 5000  # longer than any real symlink target


class StubRepo:
    """Just the two reads `unsafe_symlinks` makes: `ls_tree` and `read_blobs`."""

    def __init__(self, entries: dict[bytes, tuple[str, bytes]]) -> None:
        # path -> (mode, content); the blob id is the path in hex, so every blob is distinct
        self.entries = entries

    def ls_tree(self, rev: str) -> list[TreeEntry]:
        return [
            TreeEntry(mode, path.hex(), len(content), path)
            for path, (mode, content) in self.entries.items()
        ]

    def read_blobs(self, shas: Iterable[str]) -> dict[str, bytes]:
        wanted = set(shas)
        return {
            path.hex(): content
            for path, (_, content) in self.entries.items()
            if path.hex() in wanted
        }

    def repo(self) -> BareRepo:
        return cast("BareRepo", self)


def links(**targets: str) -> StubRepo:
    return StubRepo(
        {name.encode(): (MODE_SYMLINK, target.encode()) for name, target in targets.items()}
    )


def test_safe_links_and_regular_files_are_not_listed() -> None:
    repo = StubRepo(
        {
            b"file": (MODE_REGULAR, b"x"),
            b"ok": (MODE_SYMLINK, b"file"),
            b"sub/ok": (MODE_SYMLINK, b"../file"),
        }
    )
    assert unsafe_symlinks(repo.repo(), "rev") == []


def test_every_unsafe_link_is_listed_sorted_by_path_with_its_target_and_reason() -> None:
    repo = links(zeta="/etc/passwd", alpha="../outside", middle="x\x00y")
    found = unsafe_symlinks(repo.repo(), "rev")
    assert [path for path, _ in found] == ["alpha", "middle", "zeta"]
    assert found[0] == ("alpha", "../outside: resolves outside the repository root")
    assert found[2] == ("zeta", "/etc/passwd: absolute target")
    assert found[1][1].endswith("target contains a NUL byte")


def test_an_unreadable_target_is_listed_with_a_question_mark() -> None:
    repo = StubRepo({b"long": (MODE_SYMLINK, LONG.encode())})
    assert unsafe_symlinks(repo.repo(), "rev") == [
        ("long", "?: target could not be read (too long, or not a blob)")
    ]


def test_non_utf8_names_are_shown_printably() -> None:
    repo = StubRepo({b"caf\xe9": (MODE_SYMLINK, b"/abs")})
    [(path, reason)] = unsafe_symlinks(repo.repo(), "rev")
    assert path == "caf�" and reason.startswith("/abs:")


def test_the_import_check_rejects_with_the_first_of_the_list_in_the_same_words() -> None:
    repo = links(b="/etc/shadow", a="../x")
    with pytest.raises(BundleRejected) as caught:
        _check_symlinks(repo.repo(), "head")
    assert caught.value.code == "symlink_escape"
    assert str(caught.value) == "symlink_escape: a -> ../x: resolves outside the repository root"


def test_the_import_check_passes_a_clean_tree() -> None:
    repo = StubRepo({b"real": (MODE_REGULAR, b"x"), b"a": (MODE_SYMLINK, b"real")})
    _check_symlinks(repo.repo(), "head")
