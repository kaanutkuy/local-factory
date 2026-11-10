"""Pure path-safety logic for worker-produced trees: unsafe paths and symlink resolution.

Paths and symlink targets are handled as ``str`` decoded with ``surrogateescape`` so that they
round-trip to the exact bytes git stored; validity as UTF-8 is a separate check.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

# Linux PATH_MAX is 4096 including the terminating NUL; a longer target cannot be created.
MAX_SYMLINK_TARGET = 4095

# Code points HFS+ ignores when comparing names (as in git's is_hfs_dotgit).
_HFS_IGNORABLE = frozenset(
    chr(code)
    for code in (0x200C, 0x200D, 0x200E, 0x200F, *range(0x202A, 0x202F), *range(0x206A, 0x2070))
) | {chr(0xFEFF)}
_SEPARATORS = re.compile(r"[/\\]")


def decode_path(raw: bytes) -> str:
    """Bytes -> str that round-trips exactly (non-UTF-8 bytes become lone surrogates)."""
    return raw.decode("utf-8", "surrogateescape")


def display(text: str) -> str:
    """A printable, JSON-safe rendering of a possibly non-UTF-8 path or target."""
    return text.encode("utf-8", "surrogateescape").decode("utf-8", "replace")


def is_dotgit(component: str) -> bool:
    """True if a path component would name git metadata on any common filesystem.

    Case-insensitive, and also catches the NTFS forms (trailing dots/spaces, ``git~1``,
    ``.git::$INDEX_ALLOCATION``) and HFS+ forms (ignorable code points), like git's own checks.
    """
    name = "".join(ch for ch in component if ch not in _HFS_IGNORABLE)
    name = name.split(":", 1)[0].rstrip(" .").lower()
    return name in {".git", "git~1"}


def unsafe_path_reason(path: str) -> str | None:
    """Why ``path`` (a tree path from git) is unsafe to materialize, or None if it is fine.

    Rejects non-UTF-8 paths, absolute paths, empty/``.``/``..`` components and any ``.git``
    component (see :func:`is_dotgit`). Backslashes count as separators, conservatively.
    """
    try:
        path.encode("utf-8")
    except UnicodeEncodeError:
        return "path is not valid UTF-8"
    if path.startswith(("/", "\\")):
        return "absolute path"
    for component in _SEPARATORS.split(path):
        if component in {"", ".", ".."}:
            return f"path component {component!r} is not allowed"
        if is_dotgit(component):
            return f"path component {component!r} names git metadata"
    return None


def symlink_escape_reason(link: str, links: Mapping[str, str | None]) -> str | None:
    """Why the symlink at tree path ``link`` is unsafe, or None if it stays inside the repository.

    ``links`` maps *every* symlink path in the tree to its target (None if unreadable). The
    target is resolved component by component from the link's directory, the way the kernel
    would, and is unsafe if it is absolute or contains a NUL byte, if any step climbs above the
    repository root (even if later components come back down: the checkout directory's name is
    unknown), if a component names git metadata, or if any component - intermediate or final,
    escaping or not - is itself a symlink in the tree. Chains are never followed, so ``a -> .``
    plus ``b -> a/..`` is caught at ``a``, and loops cannot occur. Components that are not
    symlinks in the tree are plain directories (possibly not yet existing).
    """
    target = links[link]
    if target is None:
        return "target could not be read (too long, or not a blob)"
    if target.startswith("/"):
        return "absolute target"
    if "\0" in target:
        return "target contains a NUL byte"
    resolved = link.split("/")[:-1]  # the link's parent: always a real directory in a tree
    for component in target.split("/"):
        if component in {"", "."}:
            continue
        if component == "..":
            if not resolved:
                return "resolves outside the repository root"
            resolved.pop()
            continue
        if is_dotgit(component):
            return "resolves into git metadata (.git)"
        resolved.append(component)
        through = "/".join(resolved)
        if through in links:
            return f"passes through the symlink {display(through)}"
    return None
