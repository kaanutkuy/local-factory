"""Repository path globs for policy rules.

Semantics (case-sensitive, anchored at the repository root, "/" is the only separator):

- ``**`` as a whole segment matches zero or more whole segments; runs of ``**`` act as one.
- ``*`` matches any run of characters within one segment (never ``/``); ``?`` exactly one.
- ``[abc]``, ``[a-z]``, ``[!a-z]`` / ``[^a-z]`` match one character of / not of a set. A ``]``
  right after ``[`` (or ``[!``) is a member; ``-`` first or last is literal; a reversed range
  matches nothing; a ``[`` without a closing ``]`` is literal.
- ``**`` inside a segment (``a**b``) is just ``*``. Everything else, including ``\\``, ``{`` and
  regex metacharacters, is literal. Wildcards match names starting with ``.``.
- Empty and ``.`` segments are ignored in both path and pattern, so ``/factory.toml`` and
  ``./factory.toml`` mean ``factory.toml``. A trailing ``/`` means "this directory and
  everything below it": ``migrations/`` == ``migrations/**``.
"""

from __future__ import annotations

import re
from functools import lru_cache

# A compiled pattern segment: a regex for one path segment, or None for "**".
_Segment = re.Pattern[str] | None
_IGNORED_SEGMENTS = ("", ".")


def glob_match(path: str, pattern: str) -> bool:
    """Whether the "/"-separated ``path`` matches ``pattern`` (see the module docstring)."""
    return _match(_compile(pattern), _split(path))


def _split(path: str) -> tuple[str, ...]:
    return tuple(part for part in path.split("/") if part not in _IGNORED_SEGMENTS)


@lru_cache(maxsize=1024)
def _compile(pattern: str) -> tuple[_Segment, ...]:
    parts = list(_split(pattern))
    if pattern.endswith("/"):
        parts.append("**")
    compiled: list[_Segment] = []
    for part in parts:
        if part != "**":
            compiled.append(re.compile(_translate(part)))
        elif not compiled or compiled[-1] is not None:
            compiled.append(None)
    return tuple(compiled)


def _match(pattern: tuple[_Segment, ...], segments: tuple[str, ...]) -> bool:
    # ok[j]: pattern[i:] matches segments[j:], computed for i from the end towards 0.
    n = len(segments)
    ok = [False] * n + [True]
    for part in reversed(pattern):
        cur = [False] * (n + 1)
        if part is None:
            cur[n] = ok[n]
            for j in range(n - 1, -1, -1):
                cur[j] = ok[j] or cur[j + 1]
        else:
            for j in range(n):
                cur[j] = ok[j + 1] and part.fullmatch(segments[j]) is not None
        ok = cur
    return ok[0]


def _translate(segment: str) -> str:
    """Regex source for one pattern segment (matched with ``fullmatch``)."""
    out: list[str] = []
    i = 0
    while i < len(segment):
        char = segment[i]
        if char == "*":
            if not out or out[-1] != "[^/]*":
                out.append("[^/]*")
            i += 1
        elif char == "?":
            out.append("[^/]")
            i += 1
        elif char == "[" and (bracket := _bracket(segment, i)) is not None:
            regex, i = bracket
            out.append(regex)
        else:
            out.append(re.escape(char))
            i += 1
    return "".join(out)


def _bracket(segment: str, start: int) -> tuple[str, int] | None:
    """Translate the bracket expression at ``segment[start] == "["``; None if it never closes."""
    i = start + 1
    negate = i < len(segment) and segment[i] in "!^"
    if negate:
        i += 1
    end = segment.find("]", i + 1)  # a "]" right at i is a member, not the end
    if end < 0:
        return None
    body = segment[i:end]
    items: list[str] = []
    j = 0
    while j < len(body):
        if j + 2 < len(body) and body[j + 1] == "-":
            low, high = body[j], body[j + 2]
            if low <= high:
                items.append(f"{re.escape(low)}-{re.escape(high)}")
            j += 3
        else:
            items.append(re.escape(body[j]))
            j += 1
    if not items:  # only reversed ranges: matches nothing, or (negated) any one character
        return ("[^/]" if negate else "(?!)"), end + 1
    return f"[{'^/' if negate else ''}{''.join(items)}]", end + 1
