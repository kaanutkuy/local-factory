"""Unit tests for path safety and symlink resolution (local_factory.gitops._paths)."""

from __future__ import annotations

import pytest

from local_factory.gitops._paths import (
    decode_path,
    display,
    is_dotgit,
    symlink_escape_reason,
    unsafe_path_reason,
)

# --------------------------------------------------------------------------- .git detection


@pytest.mark.parametrize(
    "name",
    [
        ".git",
        ".GIT",
        ".Git",
        ".git.",  # NTFS strips trailing dots
        ".git ",  # ... and spaces
        ".git . .",
        "git~1",  # NTFS 8.3 short name
        "GIT~1",
        ".git::$INDEX_ALLOCATION",  # NTFS alternate data stream
        ".g‌it",  # HFS+ ignores zero-width non-joiner
        "﻿.git",
        ".gi‮t",
    ],
)
def test_is_dotgit_catches_aliases(name: str) -> None:
    assert is_dotgit(name)


@pytest.mark.parametrize(
    "name", [".gitignore", ".github", ".gitmodules", "git", "..git", ".git2", "x.git", "git~2"]
)
def test_is_dotgit_allows_lookalikes(name: str) -> None:
    assert not is_dotgit(name)


# --------------------------------------------------------------------------- unsafe paths


@pytest.mark.parametrize(
    "path",
    ["README.md", "src/app.py", "a/b/c.txt", ".gitignore", "docs/.github/x", "café/ü.txt"],
)
def test_safe_paths(path: str) -> None:
    assert unsafe_path_reason(path) is None


@pytest.mark.parametrize(
    ("path", "reason"),
    [
        ("/etc/passwd", "absolute"),
        ("\\windows\\x", "absolute"),
        ("..", "'..'"),
        ("a/../b", "'..'"),
        ("a\\..\\b", "'..'"),
        ("..\\evil", "'..'"),
        ("./a", "'.'"),
        ("a//b", "''"),
        ("a/", "''"),
        (".git/config", "git metadata"),
        ("sub/.GIT/hooks/x", "git metadata"),
        ("vendor/git~1/x", "git metadata"),
        (decode_path(b"caf\xe9.txt"), "UTF-8"),
    ],
)
def test_unsafe_paths(path: str, reason: str) -> None:
    found = unsafe_path_reason(path)
    assert found is not None and reason in found


def test_decode_path_round_trips_and_display_is_printable() -> None:
    raw = b"caf\xe9/\xff.txt"
    text = decode_path(raw)
    assert text.encode("utf-8", "surrogateescape") == raw
    assert display(text) == "caf�/�.txt"
    display(text).encode("utf-8")  # JSON/pydantic safe: no lone surrogates


# --------------------------------------------------------------------------- symlinks


def escapes(link: str, links: dict[str, str | None]) -> str | None:
    return symlink_escape_reason(link, links)


@pytest.mark.parametrize(
    ("link", "target"),
    [
        ("docs/readme", "../README.md"),
        ("a/b/c", "../../x"),
        ("link", "."),
        ("link", ""),
        ("link", "src/"),
        ("link", "./src/./app.py"),
        ("a/link", "b/../c/../../d"),
        ("deep/er/link", "../../deep/er/../x"),
        ("link", "missing/dir/../../file"),
    ],
)
def test_symlink_inside_repo(link: str, target: str) -> None:
    assert escapes(link, {link: target}) is None


@pytest.mark.parametrize(
    ("link", "target", "reason"),
    [
        ("evil", "/etc/passwd", "absolute"),
        ("link", "..", "outside"),
        ("link", "../x", "outside"),
        ("a/link", "../../outside", "outside"),
        ("link", "a/../../x", "outside"),
        ("link", "../seed/README.md", "outside"),  # out and back in by name still escapes
        ("link", ".git/config", "git metadata"),
        ("a/link", "../.GIT", "git metadata"),
        ("link", "sub/git~1/hooks", "git metadata"),
        ("link", "a\0/../../../x", "NUL"),
        ("link", "..\0x", "NUL"),
    ],
)
def test_symlink_escapes(link: str, target: str, reason: str) -> None:
    found = escapes(link, {link: target})
    assert found is not None and reason in found


def test_unreadable_target_escapes() -> None:
    found = escapes("link", {"link": None})
    assert found is not None and "could not be read" in found


def test_chain_through_safe_links_is_rejected() -> None:
    # Each link is harmless lexically, but "p" is the repo root, so "p/.." is its parent.
    links: dict[str, str | None] = {"p": ".", "q": "p/../x"}
    assert escapes("p", links) is None
    found = escapes("q", links)
    assert found is not None and "passes through the symlink p" in found


@pytest.mark.parametrize(
    ("links", "link", "through"),
    [
        ({"vendor": "/usr/share/lib", "mine": "vendor/x"}, "mine", "vendor"),  # escaping link
        ({"lib": "vendor/lib", "use": "lib/../file"}, "use", "lib"),  # safe link, intermediate
        ({"lib": "vendor/lib", "x/y": "../lib"}, "x/y", "lib"),  # final component
        ({"a": None, "b": "a/x"}, "b", "a"),  # unreadable link: never looked at
        ({"a/b/link": "../sibling", "a/sibling": "x"}, "a/b/link", "a/sibling"),
        ({"d/l": "e", "d/e": "f"}, "d/l", "d/e"),  # relative to the link's own directory
    ],
)
def test_traversing_any_other_symlink_is_rejected(
    links: dict[str, str | None], link: str, through: str
) -> None:
    found = escapes(link, links)
    assert found == f"passes through the symlink {through}"


@pytest.mark.parametrize("links", [{"a": "a"}, {"a": "b", "b": "a"}, {"a": "./b/../a"}])
def test_symlink_loops_are_rejected(links: dict[str, str | None]) -> None:
    found = escapes("a", links)
    assert found is not None and "passes through the symlink" in found


def test_paths_that_name_no_symlink_are_fine() -> None:
    links: dict[str, str | None] = {"lib": "vendor", "x/l": "../y", "z": "real/../file"}
    assert escapes("x/l", links) is None
    assert escapes("z", links) is None
