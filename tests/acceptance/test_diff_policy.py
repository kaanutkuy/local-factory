"""Acceptance: diff policy (docs/specs/phase-1a.md § policy). Protected."""

from __future__ import annotations

from typing import Any

import pytest

from local_factory.config import ProjectConfig
from local_factory.models import (
    MODE_GITLINK,
    MODE_REGULAR,
    MODE_SYMLINK,
    AddedLine,
    ChangedFile,
    ChangeKind,
    Waiver,
)
from local_factory.policy.diff_policy import DiffPolicy, check_changes, glob_match

K = ChangeKind
POLICY = DiffPolicy(protected=("tests/acceptance/**", "factory.toml", ".github/workflows/**"))


def changed(
    path: str,
    kind: ChangeKind = K.MODIFIED,
    lines: tuple[str, ...] = (),
    **kw: Any,
) -> ChangedFile:
    mode = None if kind is K.DELETED else kw.pop("mode", MODE_REGULAR)
    return ChangedFile(
        path=path,
        kind=kind,
        mode=mode,
        added_lines=tuple(AddedLine(line_no=i, text=t) for i, t in enumerate(lines, start=1)),
        **kw,
    )


def rules(changes: list[ChangedFile], **kw: Any) -> list[tuple[str, str, int | None]]:
    return [(v.rule, v.path, v.line) for v in check_changes(changes, POLICY, **kw)]


@pytest.mark.parametrize(
    ("path", "pattern", "expected"),
    [
        ("tests/acceptance/T-0001/test_x.py", "tests/acceptance/**", True),
        ("tests/acceptance", "tests/acceptance/**", True),
        ("tests/acceptancex/a.py", "tests/acceptance/**", False),
        ("test_a.py", "**/test_*.py", True),
        ("a/b/test_a.py", "**/test_*.py", True),
        ("a/b/atest_a.py", "**/test_*.py", False),
        ("src/a.py", "src/*.py", True),
        ("src/a/b.py", "src/*.py", False),
        ("factory.toml", "factory.toml", True),
        ("sub/factory.toml", "factory.toml", False),
        ("a1.txt", "a?.txt", True),
        ("ab.txt", "a[bc].txt", True),
        ("Factory.toml", "factory.toml", False),
    ],
)
def test_glob_match(path: str, pattern: str, expected: bool) -> None:
    assert glob_match(path, pattern) is expected


def test_clean_change_passes() -> None:
    assert check_changes([changed("src/app.py", lines=("x = 1",))], POLICY) == []


def test_protected_paths_any_kind() -> None:
    assert rules(
        [
            changed("tests/acceptance/T-0001/test_tags.py"),
            changed("factory.toml", K.DELETED),
            changed("src/moved.py", K.RENAMED, old_path=".github/workflows/ci.yml"),
            changed("tests/acceptance/new_test.py", K.ADDED),
        ]
    ) == [
        ("protected_path", "factory.toml", None),
        ("protected_path", "src/moved.py", None),
        ("protected_path", "tests/acceptance/T-0001/test_tags.py", None),
        ("protected_path", "tests/acceptance/new_test.py", None),
    ]


def test_deleting_or_renaming_away_tests() -> None:
    assert rules(
        [
            changed("tests/unit/test_core.py", K.DELETED),
            changed("src/helpers.py", K.RENAMED, old_path="pkg/test_helpers.py"),
            changed("tests/unit/test_renamed.py", K.RENAMED, old_path="tests/unit/test_old.py"),
        ]
    ) == [
        ("test_deleted", "src/helpers.py", None),
        ("test_deleted", "tests/unit/test_core.py", None),
    ]


@pytest.mark.parametrize(
    "line",
    [
        "@pytest.mark.skip(reason='later')",
        "@pytest.mark.xfail",
        "    pytest.skip('flaky')",
        "@unittest.skip('x')",
        "it.skip('renders', () => {})",
        "describe.skip('suite', fn)",
        "test.skip('x', fn)",
        "describe.only('suite', fn)",
        "xit('renders', fn)",
        "xdescribe('suite', fn)",
    ],
)
def test_skip_markers(line: str) -> None:
    assert rules([changed("tests/unit/test_core.py", lines=("ok = 1", line))]) == [
        ("skip_marker", "tests/unit/test_core.py", 2)
    ]


@pytest.mark.parametrize(
    "line",
    [
        "if debug:  # pragma: no cover",
        "x = 1  # pragma: no mutate",
        "import os  # noqa: F401",
        "y = f()  # type: ignore[no-any-return]",
        "subprocess.call(cmd)  # nosec",
        "z = 2  # pylint: disable=invalid-name",
        "/* eslint-disable */",
    ],
)
def test_suppressions(line: str) -> None:
    assert rules([changed("src/app.py", lines=(line,))]) == [("suppression", "src/app.py", 1)]


def test_suppression_with_effective_waiver_is_allowed() -> None:
    change = changed("src/app.py", lines=("ok = 1", "x = f()  # type: ignore[misc]"))

    def w(scope: str, *, by: str = "verifier") -> Waiver:
        return Waiver(
            id="W-1",
            gate="typecheck",
            scope=scope,
            reason="third-party stub bug",
            requested_by="implementer",
            adjudicated_by=by,
            approved=True,
        )

    assert check_changes([change], POLICY, [w("src/app.py:2")]) == []
    assert rules([change], waivers=[w("src/app.py:1")]) == [("suppression", "src/app.py", 2)]
    assert rules([change], waivers=[w("src/app.py:2", by="implementer")]) == [
        ("suppression", "src/app.py", 2)
    ]


def test_gate_config_changes() -> None:
    assert rules(
        [
            changed(".coveragerc"),
            changed("pytest.ini", K.ADDED),
            changed("pyproject.toml", lines=("[tool.coverage.report]", "fail_under = 10")),
            changed("sub/pyproject.toml", lines=('version = "2"',)),
        ]
    ) == [
        ("gate_config", ".coveragerc", None),
        ("gate_config", "pyproject.toml", 1),
        ("gate_config", "pyproject.toml", 2),
        ("gate_config", "pytest.ini", None),
    ]


def test_structural_rules() -> None:
    assert rules(
        [
            changed("vendor/sub", K.ADDED, mode=MODE_GITLINK),
            changed("evil", K.ADDED, mode=MODE_SYMLINK, symlink_target="/etc/passwd"),
            changed("a/up", K.ADDED, mode=MODE_SYMLINK, symlink_target="../../x"),
            changed("docs/ok", K.ADDED, mode=MODE_SYMLINK, symlink_target="../README.md"),
            changed("data/huge.bin", K.ADDED, size=2_000_000, is_binary=True),
        ]
    ) == [
        ("symlink_escape", "a/up", None),
        ("file_too_large", "data/huge.bin", None),
        ("symlink_escape", "evil", None),
        ("gitlink", "vendor/sub", None),
    ]


def test_one_violation_per_rule_path_line_and_instructive_messages() -> None:
    change = changed(
        "tests/acceptance/test_x.py", lines=("@pytest.mark.skip  # noqa", "@pytest.mark.skip")
    )
    violations = check_changes([change], POLICY)
    assert [(v.rule, v.line) for v in violations] == [
        ("protected_path", None),
        ("skip_marker", 1),
        ("suppression", 1),
        ("skip_marker", 2),
    ]
    for v in violations:
        assert v.path in v.message or "tests/acceptance" in v.message
        assert len(v.message) > 30


def test_policy_from_project_config() -> None:
    config = ProjectConfig(name="p", protected=("migrations/**",))
    policy = DiffPolicy.from_project(config)
    assert set(policy.protected) == {"migrations/**", "factory.toml"}
    violations = check_changes([changed("migrations/001.py"), changed("factory.toml")], policy)
    assert [v.path for v in violations] == ["factory.toml", "migrations/001.py"]
