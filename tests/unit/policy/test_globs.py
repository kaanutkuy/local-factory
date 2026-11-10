from __future__ import annotations

import pytest

from local_factory.policy.globs import glob_match


@pytest.mark.parametrize(
    ("path", "pattern", "expected"),
    [
        # "**" as a whole segment: zero or more whole segments
        ("x", "**", True),
        ("a/b/c", "**", True),
        ("", "**", True),
        ("x", "**/x", True),
        ("a/b/x", "**/x", True),
        ("a/bx", "**/x", False),
        ("a", "a/**", True),
        ("a/b/c", "a/**", True),
        ("ab", "a/**", False),
        # "**" in the middle
        ("src/test_a.py", "src/**/test_*.py", True),
        ("src/a/b/test_a.py", "src/**/test_*.py", True),
        ("src/a/b/a_test.py", "src/**/test_*.py", False),
        ("lib/a/test_a.py", "src/**/test_*.py", False),
        ("a/b", "a/**/b", True),
        ("a/x/b", "a/**/b", True),
        ("a/x/y/b", "a/**/b", True),
        ("a/b/c", "a/**/b", False),
        ("a/x/b/y/b", "a/**/b/**/b", True),
        ("a/b/b", "a/**/b/**/b", True),
        ("a/x/y/b", "a/**/b/**/b", False),
        ("a/x/b/y", "a/**/b/**/b", False),
        # consecutive "**" behave as one
        ("a/b", "a/**/**/b", True),
        ("a/x/y/b", "a/**/**/b", True),
        ("a/x/y/b", "**/**/**", True),
        ("x", "**/**/x", True),
        # "**" inside a segment is just "*"
        ("ab", "a**b", True),
        ("axyb", "a**b", True),
        ("ax/yb", "a**b", False),
        ("a/b/c.py", "**.py", False),
        ("c.py", "**.py", True),
        # "*" stays within a segment and may match nothing
        ("src/a.py", "src/*.py", True),
        ("src/.py", "src/*.py", True),
        ("src/a/b.py", "src/*.py", False),
        ("src/a.py", "*", False),
        ("src", "*", True),
        ("abc", "a*c*", True),
        ("abd", "a*c*", False),
        ("aaaaaaaaaaaaaaaaaaaaaaaaaaaaab", "*a*a*a*a*a*a*a*b", True),
        # wildcards match dot-files and dot-directories
        (".github/workflows/ci.yml", "**/*.yml", True),
        (".env", "*", True),
        (".hidden/x", "?hidden/*", True),
        # "?" is exactly one character, never "/"
        ("a1.txt", "a?.txt", True),
        ("a.txt", "a?.txt", False),
        ("a12.txt", "a?.txt", False),
        ("a/b", "a?b", False),
        # bracket expressions
        ("ab.txt", "a[bc].txt", True),
        ("ad.txt", "a[bc].txt", False),
        ("a5", "a[0-9]", True),
        ("ax", "a[0-9]", False),
        ("ax", "a[!0-9]", True),
        ("a5", "a[!0-9]", False),
        ("ax", "a[^0-9]", True),
        ("a5", "a[^0-9]", False),
        ("a]", "a[]]", True),
        ("a]", "a[]x]", True),
        ("ax", "a[]x]", True),
        ("a]", "a[!]]", False),
        ("ay", "a[!]]", True),
        ("a-", "a[x-]", True),
        ("a-", "a[-x]", True),
        ("a-", "a[!-x]", False),
        ("am", "a[z-a]", False),
        ("am", "a[!z-a]", True),
        ("ab", "a[z-ab]", True),
        ("a^", "a[x^]", True),
        ("a\\", "a[\\]", True),
        ("aB", "a[a-z]", False),
        # unclosed or empty brackets are literal
        ("a[b", "a[b", True),
        ("ab", "a[b", False),
        ("a[]", "a[]", True),
        ("a[!]", "a[!]", True),
        ("a[", "a[", True),
        ("a[!", "a[!", True),
        # "/" ends a segment, so it can never be inside a bracket
        ("a/b", "a[/]b", False),
        ("a[/]b", "a[/]b", True),
        # everything else is literal, including regex metacharacters and braces
        ("a+b.txt", "a+b.txt", True),
        ("aab.txt", "a+b.txt", False),
        ("axtxt", "a.txt", False),
        ("a{b,c}", "a{b,c}", True),
        ("ab", "a{b,c}", False),
        ("(x)|y$", "(x)|y$", True),
        ("a\\*", "a\\*", True),
        ("a\\b", "a\\*", True),
        # case-sensitive
        ("Factory.toml", "factory.toml", False),
        ("SRC/a.py", "src/*.py", False),
        # anchoring: patterns start at the repository root
        ("sub/factory.toml", "factory.toml", False),
        ("factory.toml", "/factory.toml", True),
        ("factory.toml", "./factory.toml", True),
        ("a/b", "a/./b", True),
        ("a/b", "a//b", True),
        # trailing "/" means the directory and everything below it
        ("migrations", "migrations/", True),
        ("migrations/001.py", "migrations/", True),
        ("migrations/a/b.sql", "migrations/", True),
        ("migrationsx/001.py", "migrations/", False),
        ("anything/at/all", "/", True),
        ("x", "**/", True),
        # paths are normalized the same way
        ("./factory.toml", "factory.toml", True),
        ("a//b", "a/b", True),
        ("a/b/", "a/b", True),
        # empty pattern only matches an empty path
        ("", "", True),
        ("a", "", False),
        # a path with fewer segments than the pattern needs
        ("a", "a/b", False),
        ("a/b", "a", False),
    ],
)
def test_glob_match(path: str, pattern: str, expected: bool) -> None:
    assert glob_match(path, pattern) is expected


def test_deep_paths_with_many_double_stars_stay_fast() -> None:
    path = "/".join(["d"] * 200) + "/x.py"
    assert glob_match(path, "**/d/**/d/**/d/**/*.py")
    assert not glob_match(path, "**/d/**/d/**/d/**/*.rs")
