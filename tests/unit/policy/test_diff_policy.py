from __future__ import annotations

from typing import Any

import pytest

from local_factory.config import ProjectConfig
from local_factory.models import (
    MODE_EXECUTABLE,
    MODE_GITLINK,
    MODE_REGULAR,
    MODE_SYMLINK,
    AddedLine,
    ChangedFile,
    ChangeKind,
    Violation,
    Waiver,
)
from local_factory.policy import DiffPolicy, check_changes
from local_factory.policy.diff_policy import GATE_CONFIG_FILES

K = ChangeKind
POLICY = DiffPolicy(protected=("tests/acceptance/**", "factory.toml"))
Row = tuple[str, str, int | None]


def changed(
    path: str, kind: ChangeKind = K.MODIFIED, lines: tuple[str, ...] = (), **kw: Any
) -> ChangedFile:
    mode = None if kind is K.DELETED else kw.pop("mode", MODE_REGULAR)
    return ChangedFile(
        path=path,
        kind=kind,
        mode=mode,
        added_lines=tuple(AddedLine(line_no=i, text=t) for i, t in enumerate(lines, start=1)),
        **kw,
    )


def check(
    changes: list[ChangedFile], policy: DiffPolicy = POLICY, waivers: tuple[Waiver, ...] = ()
) -> list[Violation]:
    return check_changes(changes, policy, waivers)


def rows(changes: list[ChangedFile], **kw: Any) -> list[Row]:
    return [(v.rule, v.path, v.line) for v in check(changes, **kw)]


def line_rules(path: str, *lines: str) -> list[tuple[str, int | None]]:
    return [(v.rule, v.line) for v in check([changed(path, lines=lines)])]


def waiver(scope: str, **kw: Any) -> Waiver:
    fields: dict[str, Any] = {
        "id": "W-1",
        "gate": "lint",
        "scope": scope,
        "reason": "third-party stub",
        "requested_by": "implementer",
        "adjudicated_by": "verifier",
        "approved": True,
    }
    return Waiver(**{**fields, **kw})


# --------------------------------------------------------------------------- DiffPolicy


def test_policy_defaults() -> None:
    policy = DiffPolicy(protected=())
    assert policy.test_globs == ("tests/**", "**/test_*.py", "**/*_test.py")
    assert policy.max_file_bytes == 1_000_000


def test_policy_rejects_a_bare_string_instead_of_globs() -> None:
    a_string: Any = "factory.toml"  # the mistake under test; Any keeps it past mypy
    with pytest.raises(TypeError, match="protected"):
        DiffPolicy(protected=a_string)
    with pytest.raises(TypeError, match="test_globs"):
        DiffPolicy(protected=(), test_globs=a_string)


def test_policy_rejects_negative_size_limit() -> None:
    with pytest.raises(ValueError, match="max_file_bytes"):
        DiffPolicy(protected=(), max_file_bytes=-1)


def test_from_project_keeps_order_without_duplicates() -> None:
    config = ProjectConfig(name="p", protected=("a/**", "factory.toml", "b/**"))
    assert DiffPolicy.from_project(config).protected == ("a/**", "factory.toml", "b/**")


def test_from_project_protects_factory_toml_even_if_config_bypassed_validation() -> None:
    config = ProjectConfig.model_construct(name="p", protected=("migrations/**",))
    assert DiffPolicy.from_project(config).protected == ("migrations/**", "factory.toml")


# --------------------------------------------------------------------------- protected_path


def test_renaming_into_or_out_of_a_protected_path() -> None:
    violations = check(
        [
            changed("tests/acceptance/x.py", K.RENAMED, old_path="src/x.py"),
            changed("src/y.py", K.RENAMED, old_path="tests/acceptance/y.py"),
        ]
    )
    assert [(v.rule, v.path) for v in violations] == [
        ("protected_path", "src/y.py"),
        ("test_deleted", "src/y.py"),  # tests/acceptance/** is also a test path
        ("protected_path", "tests/acceptance/x.py"),
    ]
    out_of = violations[0].message
    assert "'tests/acceptance/y.py'" in out_of and "'tests/acceptance/**'" in out_of
    assert out_of.startswith("Renaming 'src/y.py' (renamed from 'tests/acceptance/y.py')")


def test_renamed_protected_file_within_protected_tree_is_one_violation() -> None:
    assert rows(
        [changed("tests/acceptance/b.py", K.RENAMED, old_path="tests/acceptance/a.py")]
    ) == [("protected_path", "tests/acceptance/b.py", None)]


def test_protected_type_change_and_unprotected_neighbours() -> None:
    assert rows(
        [
            changed("factory.toml", K.TYPE_CHANGED, mode=MODE_SYMLINK, symlink_target="x"),
            changed("sub/factory.toml"),
            changed("tests/acceptancex.py"),
        ]
    ) == [("protected_path", "factory.toml", None)]


def test_protection_uses_the_policy_globs() -> None:
    policy = DiffPolicy(protected=("docs/*.md",))
    assert rows([changed("docs/a.md"), changed("docs/x/a.md")], policy=policy) == [
        ("protected_path", "docs/a.md", None)
    ]


# --------------------------------------------------------------------------- test_deleted


def test_deleting_any_file_under_a_test_glob() -> None:
    assert rows(
        [
            changed("tests/fixtures/data.json", K.DELETED),
            changed("pkg/core_test.py", K.DELETED),
            changed("pkg/test_core.py", K.DELETED),
            changed("src/core.py", K.DELETED),
        ]
    ) == [
        ("test_deleted", "pkg/core_test.py", None),
        ("test_deleted", "pkg/test_core.py", None),
        ("test_deleted", "tests/fixtures/data.json", None),
    ]


def test_renames_that_keep_or_gain_test_status_are_fine() -> None:
    assert (
        check(
            [
                changed("pkg/test_b.py", K.RENAMED, old_path="tests/test_b.py"),
                changed("tests/test_c.py", K.RENAMED, old_path="src/c.py"),
                changed("src/d.py", K.RENAMED, old_path="src/e.py"),
            ]
        )
        == []
    )


def test_rename_away_from_tests_names_both_paths() -> None:
    [violation] = check([changed("src/helpers.py", K.RENAMED, old_path="tests/helpers.py")])
    assert violation.rule == "test_deleted"
    assert "'tests/helpers.py'" in violation.message and "'src/helpers.py'" in violation.message


def test_custom_test_globs() -> None:
    policy = DiffPolicy(protected=(), test_globs=("**/*.spec.ts",))
    assert rows(
        [changed("web/a.spec.ts", K.DELETED), changed("tests/test_x.py", K.DELETED)],
        policy=policy,
    ) == [("test_deleted", "web/a.spec.ts", None)]


def test_modifying_a_test_is_not_deleting_it() -> None:
    assert check([changed("tests/test_x.py", lines=("def test_x(): ...",))]) == []


# --------------------------------------------------------------------------- skip_marker


@pytest.mark.parametrize(
    "line",
    [
        "@pytest.mark.skipif(sys.platform == 'win32', reason='x')",
        "@unittest.skipIf(True, 'x')",
        "@unittest.skipUnless(False, 'x')",
        "pytestmark = pytest.mark.skip",
        "@pytest.mark.xfail(strict=False)",
        "pytest . mark . skip",
        "pytest.skip ('later')",
        "np = pytest.importorskip('numpy')",
        "importorskip ('numpy')",
        "        self.skipTest('not today')",
        "raise unittest.SkipTest('x')",
        "raise SkipTest",
        "from unittest import SkipTest",
        "  it.only('focus', fn)",
        "test.only('x', fn)",
        "fit.only ( 'x' )",
        "  xit ('x', fn)",
        "x = [xdescribe('a')]",
        "fit('focused', () => {})",
        "  fit ('focused', fn)",
        "fdescribe('suite', fn)",
        "it . skip ('x')",
        "describe.skip('suite')",
    ],
)
def test_skip_marker_variants(line: str) -> None:
    assert line_rules("tests/test_x.py", line) == [("skip_marker", 1)]


@pytest.mark.parametrize(
    "line",
    [
        "sys.exit(main())",
        "raise SystemExit(exit(1))",
        "unit.skip('x')",
        "latest.skip('x')",
        "mytest.skip('x')",
        "suite_xdescribe('x')",
        "pytest.mark.parametrize('a', [1])",
        "it.skips = 3",
        "# skip this for now",
        "only(1)",
        "model.fit(X, y)",
        "self.fit(data)",
        "outfit('red')",
        "refit(x)",
        "obj.fdescribe(x)",
        "obj.xit(x)",
        "MySkipTest = 1",
        "self.skipTests = []",
        "do_importorskip = 1",
    ],
)
def test_lookalikes_are_not_skip_markers(line: str) -> None:
    assert line_rules("tests/test_app.py", line) == []


@pytest.mark.parametrize(
    "path",
    ["tests/unit/helpers.py", "pkg/test_x.py", "pkg/x_test.py", "conftest.py", "a/conftest.py"],
)
def test_skip_markers_count_in_test_files_and_every_conftest(path: str) -> None:
    assert line_rules(path, "pytest.skip('everything')") == [("skip_marker", 1)]


@pytest.mark.parametrize(
    ("path", "line"),
    [
        ("src/app.py", "rows = qs.only('id', 'name')"),
        ("src/app.py", "pytest.skip('x')  # a helper that wraps pytest"),
        ("docs/testing.md", "Use `@pytest.mark.skip` only with a waiver."),
        ("web/app.spec.js", "describe.only('x', fn)"),
        ("src/conftest.py.txt", "pytest.skip()"),
        ("src/myconftest.py", "pytest.skip()"),
    ],
)
def test_skip_markers_are_ignored_outside_test_files(path: str, line: str) -> None:
    assert line_rules(path, line) == []


def test_skip_markers_follow_custom_test_globs() -> None:
    policy = DiffPolicy(protected=(), test_globs=("**/*.spec.js",))
    change = changed("web/app.spec.js", lines=("describe.only('x', fn)",))
    assert rows([change], policy=policy) == [("skip_marker", "web/app.spec.js", 1)]


def test_skip_marker_uses_the_new_path_of_a_rename() -> None:
    assert (
        rows([changed("src/helper.py", K.RENAMED, old_path="src/old.py", lines=("pytest.skip()",))])
        == []
    )
    assert rows(
        [changed("tests/helper.py", K.RENAMED, old_path="src/helper.py", lines=("pytest.skip()",))]
    ) == [("skip_marker", "tests/helper.py", 1)]


# --------------------------------------------------------------------------- suppression


@pytest.mark.parametrize(
    "line",
    [
        "x = 1  # NOQA",
        "x = 1  # noqa:E501",
        "# ruff: noqa",
        "x = f()  #type:ignore",
        "x = f()  # type : ignore[attr-defined]",
        "if x:  # pragma:no cover",
        "if x:  #pragma nocover",
        "if x:  # PRAGMA: NO COVER",
        "x = 1  # pragma:no mutate",
        "x = 1  # NOSEC B101",
        "x = 1  # pylint:disable=all",
        "x = 1  # pylint: disable-next=all",
        "// eslint-disable-next-line no-console",
        "/* eslint-disable-line */",
        "else:  # pragma: no branch",
        "while x:  #pragma:nobranch",
        "x = f()  # pyright: ignore[reportGeneralTypeIssues]",
        "x = f()  #pyright:ignore",
        "# mypy: ignore-errors",
        "# mypy: ignore_errors",
        "# mypy: disallow-any-generics, ignore-errors",
        "# MYPY : IGNORE-ERRORS",
        "// @ts-ignore",
        "// @ts-expect-error: wrong types upstream",
        "// @ts-nocheck",
        "// @TS-IGNORE",
        "/* istanbul ignore next */",
        "/* istanbul  ignore else */",
        "/* c8 ignore start */",
        "// c8 ignore next 3",
    ],
)
def test_suppression_variants(line: str) -> None:
    assert line_rules("src/app.py", line) == [("suppression", 1)]


@pytest.mark.parametrize(
    "line",
    [
        "x = 1  # cover all cases",
        "type_ = 'ignore'",
        "# pylint is configured elsewhere",
        "# mypy: strict",
        "# mypy errors are not ignored here  # see docs",
        "// ts-ignore is not used",
        "istanbul = ignore_list",
    ],
)
def test_lookalikes_are_not_suppressions(line: str) -> None:
    assert line_rules("src/app.py", line) == []


def test_suppressions_count_in_any_file() -> None:
    assert line_rules("README.md", "use `# noqa` sparingly") == [("suppression", 1)]


def test_suppression_waiver_must_be_effective_and_exact() -> None:
    change = changed("src/app.py", lines=("ok = 1", "x = f()  # noqa: F821"))
    flagged: list[Row] = [("suppression", "src/app.py", 2)]
    assert rows([change], waivers=(waiver("src/app.py:2"),)) == []
    assert rows([change], waivers=(waiver("src/app.py:2", gate="anything"),)) == []
    assert rows([change], waivers=(waiver("src/app.py:2", approved=False),)) == flagged
    assert rows([change], waivers=(waiver("src/app.py:2", adjudicated_by=None),)) == flagged
    assert rows([change], waivers=(waiver("src/app.py:2", adjudicated_by="implementer"),)) == (
        flagged
    )
    assert rows([change], waivers=(waiver("src/app.py:1"),)) == flagged
    assert rows([change], waivers=(waiver("src/app.py:20"),)) == flagged
    assert rows([change], waivers=(waiver("src/other.py:2"),)) == flagged
    assert rows([change], waivers=(waiver("app.py:2"),)) == flagged
    assert rows([change], waivers=(waiver("src/app.py"),)) == flagged


def test_waiver_scope_is_per_path() -> None:
    a = changed("src/a.py", lines=("x = 1  # noqa",))
    b = changed("src/b.py", lines=("x = 1  # noqa",))
    assert rows([a, b], waivers=(waiver("src/a.py:1"),)) == [("suppression", "src/b.py", 1)]


def test_waivers_never_cover_skip_markers() -> None:
    change = changed("tests/test_x.py", lines=("@pytest.mark.skip  # noqa",))
    assert rows([change], waivers=(waiver("tests/test_x.py:1"),)) == [
        ("skip_marker", "tests/test_x.py", 1)
    ]


# --------------------------------------------------------------------------- gate_config


@pytest.mark.parametrize("name", sorted(GATE_CONFIG_FILES))
def test_gate_config_files_anywhere_in_the_tree(name: str) -> None:
    assert rows([changed(name), changed(f"pkg/sub/{name}", K.ADDED)]) == sorted(
        [("gate_config", name, None), ("gate_config", f"pkg/sub/{name}", None)]
    )


def test_gate_config_files_cover_every_kind_of_change() -> None:
    assert rows(
        [
            changed(".coveragerc", K.DELETED),
            changed("old.ini", K.RENAMED, old_path="pytest.ini"),
            changed("tox.ini", K.RENAMED, old_path="notes.ini"),
            changed("setup.cfg", K.TYPE_CHANGED, mode=MODE_SYMLINK, symlink_target="x.cfg"),
        ]
    ) == [
        ("gate_config", ".coveragerc", None),
        ("gate_config", "old.ini", None),
        ("gate_config", "setup.cfg", None),
        ("gate_config", "tox.ini", None),
    ]


def test_gate_config_file_message_names_the_file() -> None:
    [violation] = check([changed("old.ini", K.RENAMED, old_path="pytest.ini")])
    assert "'pytest.ini'" in violation.message and "'old.ini'" in violation.message


def test_lookalike_config_names_are_fine() -> None:
    assert check([changed("docs/pytest.ini.md"), changed("my-tox.ini"), changed("setup.py")]) == []


@pytest.mark.parametrize(
    "line",
    [
        # tables
        "[tool.coverage.run]",
        "[tool.pytest.ini_options]",
        "[tool.mutmut]",
        "[tool.mypy]",
        "[[tool.mypy.overrides]]",
        "[tool.ruff]",
        "[tool.ruff.lint.per-file-ignores]",
        "[ tool . coverage . report ]  # comment",
        '[tool."coverage".report]',
        "[ 'tool' . 'pytest' ]",
        "[[tool.mutmut.paths]]",
        "tool.mypy.ignore_errors = true",
        "tool.ruff = { line-length = 200 }",
        # keys
        "fail_under = 0",
        "  fail_under=0",
        '"fail_under" = 0',
        "'omit' = []",
        "report.fail_under = 0",
        'tool.coverage.report.exclude_also = [".*"]',
        'tool."coverage".run.omit = ["src/*"]',
        'addopts = "-p no:cacheprovider"',
        'omit = ["src/*"]',
        'exclude_also = ["def "]',
        'exclude_lines = ["."]',
        'testpaths = ["nowhere"]',
        "norecursedirs = ['*']",
        'python_files = "none_*.py"',
        'python_classes = "Nope"',
        'python_functions = "nope_*"',
        "report = { fail_under = 0 }",
        'run = { source = ["src"], omit = ["src/*"] }',
    ],
)
def test_pyproject_gate_config_lines(line: str) -> None:
    assert line_rules("pyproject.toml", 'name = "x"', line) == [("gate_config", 2)]


@pytest.mark.parametrize(
    "line",
    [
        "[project]",
        'version = "2.0"',
        'dependencies = ["pytest>=8", "coverage"]',
        "[tool.cov]",
        "[mytool.coverage]",
        "[tool.pytest-watcher]",
        "[tool.ruffian]",
        "[tool.poetry.group.dev.dependencies]",
        'pytest = "^8.0"',
        'mypy = { version = "^1.10" }',
        "omitted = 1",
        "my_addopts = 1",
        "fail_under_note = 1",
        "# we omit tests from the wheel",
        "# fail_under = 0",
        'description = "never omit testpaths"',
        "readme.omit_badges = true",
    ],
)
def test_harmless_pyproject_lines(line: str) -> None:
    assert line_rules("pyproject.toml", line) == []


def test_pyproject_tokens_only_matter_in_pyproject() -> None:
    assert line_rules("src/conf.py", "addopts = '-x'  # see [tool.pytest]") == []


def test_nested_pyproject_is_checked_too() -> None:
    assert line_rules("pkg/pyproject.toml", "[tool.coverage.report]") == [("gate_config", 1)]


@pytest.mark.parametrize(
    "change",
    [
        changed("pyproject.toml", K.DELETED),
        changed("old.toml", K.RENAMED, old_path="pyproject.toml"),
        changed("pyproject.toml", K.RENAMED, old_path="evil.toml"),
        changed("pyproject.toml", K.TYPE_CHANGED, mode=MODE_SYMLINK, symlink_target="evil"),
        changed("pyproject.toml", K.ADDED, mode=MODE_SYMLINK, symlink_target="evil.toml"),
        changed("pyproject.toml", K.MODIFIED, is_binary=True),
        changed("sub/pyproject.toml", K.ADDED, mode=None),
    ],
)
def test_pyproject_replaced_wholesale(change: ChangedFile) -> None:
    [violation] = check([change])
    assert (violation.rule, violation.path, violation.line) == ("gate_config", change.path, None)
    assert "line-level checks" in violation.message


def test_pyproject_regular_edits_are_checked_line_by_line() -> None:
    assert (
        check(
            [
                changed("pyproject.toml", lines=('version = "2"',)),
                changed("tools/pyproject.toml", K.ADDED, mode=MODE_EXECUTABLE),
            ]
        )
        == []
    )


# --------------------------------------------------------------------------- structural rules


def test_gitlink_only_for_the_new_mode() -> None:
    assert rows(
        [
            changed("vendor/a", K.MODIFIED, mode=MODE_GITLINK, old_mode=MODE_GITLINK),
            changed("vendor/b", K.DELETED, old_mode=MODE_GITLINK),
        ]
    ) == [("gitlink", "vendor/a", None)]


@pytest.mark.parametrize(
    ("path", "target"),
    [
        ("evil", "/etc/passwd"),
        ("evil", "//etc"),
        ("up", ".."),
        ("up", "./a/../../x"),
        ("up", "a/../.."),
        ("a/up", "../../x"),
        ("a/b/up", "../../../x"),
        ("a/up", "../../a/x"),
        ("a/up", "x/../../../a"),
        ("up", "../repo/x"),
    ],
)
def test_symlink_escapes(path: str, target: str) -> None:
    change = changed(path, K.ADDED, mode=MODE_SYMLINK, symlink_target=target)
    [violation] = check([change])
    assert (violation.rule, violation.path) == ("symlink_escape", path)
    assert repr(target) in violation.message


@pytest.mark.parametrize(
    ("path", "target"),
    [
        ("lnk", "a/./b"),
        ("lnk", "./README.md"),
        ("lnk", "."),
        ("lnk", ""),
        ("lnk", "dir/"),
        ("lnk", "a//b"),
        ("a/lnk", ".."),
        ("a/b/lnk", "../../c"),
        ("a/lnk", "../a/../b"),
        ("lnk", "a/b/../../c"),
        ("lnk", "C:/Windows"),
        ("lnk", "..."),
        ("lnk", "..x/y"),
    ],
)
def test_symlinks_inside_the_repo(path: str, target: str) -> None:
    assert check([changed(path, K.ADDED, mode=MODE_SYMLINK, symlink_target=target)]) == []


def test_symlink_without_a_recorded_target_fails_closed() -> None:
    [violation] = check([changed("lnk", K.ADDED, mode=MODE_SYMLINK)])
    assert violation.rule == "symlink_escape"
    assert "not recorded" in violation.message


def link(path: str, target: str, kind: ChangeKind = K.ADDED, **kw: Any) -> ChangedFile:
    return changed(path, kind, mode=MODE_SYMLINK, symlink_target=target, **kw)


@pytest.mark.parametrize(
    ("links", "flagged"),
    [
        # lexically harmless, but `a` is the repo root, so a/.. is its parent
        ([link("a", "."), link("b", "a/..")], ["b"]),
        ([link("a", "sub"), link("b", "a/../..")], ["b"]),
        # the intermediate link stays inside, but the combination escapes: real path d/../../..
        ([link("a/b/c/lnk", "../../../d"), link("x", "a/b/c/lnk/../../..")], ["x"]),
        # a link to a link that stays inside is rejected too (fail closed)
        ([link("docs/latest", "v2"), link("current", "docs/latest")], ["current"]),
        ([link("current", "docs/latest/index.md"), link("docs/latest", "v2")], ["current"]),
        # links next to each other in a subdirectory
        ([link("d/a", "."), link("d/b", "a/../..")], ["d/b"]),
        # loops
        ([link("loop", "loop")], ["loop"]),
        ([link("loop", "./x/../loop")], ["loop"]),
        ([link("a", "b"), link("b", "a")], ["a", "b"]),
        # an escaping link and a link through it are both reported
        ([link("out", ".."), link("via", "out/x")], ["out", "via"]),
        # renamed links count by their new path
        ([link("new", "real", K.RENAMED, old_path="old"), link("x", "new/..")], ["x"]),
        ([link("new", "real", K.RENAMED, old_path="old"), link("x", "old/f")], []),
    ],
)
def test_symlink_chains(links: list[ChangedFile], flagged: list[str]) -> None:
    assert [(v.rule, v.path) for v in check(links)] == [("symlink_escape", p) for p in flagged]


def test_symlink_chain_message_names_the_link_it_passes_through() -> None:
    violations = check([link("a", "."), link("b", "a/..")])
    [violation] = violations
    assert "'a/..'" in violation.message
    assert "passes through the symlink 'a'" in violation.message
    assert "instead" in violation.message


def test_only_symlinks_in_the_change_set_count_as_links() -> None:
    assert (
        check(
            [
                changed("gone", K.DELETED),  # was perhaps a symlink; now it is not there at all
                changed("dir/file.txt", K.ADDED),
                link("x", "gone/f"),
                link("y", "dir/file.txt"),
                link("docs/readme", "../README.md"),
            ]
        )
        == []
    )


def test_symlink_target_on_a_regular_file_is_ignored() -> None:
    assert check([changed("f", K.ADDED, symlink_target="/etc/passwd")]) == []


def test_file_size_limit_is_exclusive() -> None:
    policy = DiffPolicy(protected=(), max_file_bytes=10)
    assert rows(
        [
            changed("at-limit", K.ADDED, size=10),
            changed("over", K.ADDED, size=11),
            changed("unknown", K.ADDED),
            changed("gone", K.DELETED),
        ],
        policy=policy,
    ) == [("file_too_large", "over", None)]
    [violation] = check([changed("big.bin", K.ADDED, size=2_500_000, is_binary=True)])
    assert "2,500,000 bytes" in violation.message and "1,000,000-byte" in violation.message


# --------------------------------------------------------------------------- aggregation


def test_duplicate_changes_yield_one_violation_per_rule_path_line() -> None:
    change = changed("tests/acceptance/t.py", lines=("x = 1  # noqa",))
    assert rows([change, change]) == [
        ("protected_path", "tests/acceptance/t.py", None),
        ("suppression", "tests/acceptance/t.py", 1),
    ]


def test_several_markers_on_one_line_are_one_violation_per_rule() -> None:
    assert line_rules("tests/t.py", "pytest.skip(); xit(); x  # noqa  # type: ignore") == [
        ("skip_marker", 1),
        ("suppression", 1),
    ]


def test_sorted_by_path_then_line_then_rule() -> None:
    assert rows(
        [
            changed("tests/b.py", lines=("ok", "x  # noqa", "pytest.skip()")),
            changed("tests/a/z.py", lines=("@pytest.mark.skip  # noqa",)),
            changed("pyproject.toml", K.ADDED, lines=("[tool.pytest.ini_options]",), size=2**30),
        ]
    ) == [
        ("file_too_large", "pyproject.toml", None),
        ("gate_config", "pyproject.toml", 1),
        ("skip_marker", "tests/a/z.py", 1),
        ("suppression", "tests/a/z.py", 1),
        ("suppression", "tests/b.py", 2),
        ("skip_marker", "tests/b.py", 3),
    ]


def test_every_rule_has_an_instructive_message() -> None:
    violations = check(
        [
            changed("factory.toml"),
            changed("tests/test_gone.py", K.DELETED),
            changed("tests/test_skip.py", lines=("@pytest.mark.skip",)),
            changed("src/app.py", lines=("x = 1  # noqa",)),
            changed(".coveragerc"),
            changed("pyproject.toml", lines=("fail_under = 0",)),
            changed("pyproject.toml.bak", K.RENAMED, old_path="pyproject.toml"),
            changed("vendor/sub", K.ADDED, mode=MODE_GITLINK),
            changed("lnk", K.ADDED, mode=MODE_SYMLINK, symlink_target="/etc"),
            changed("big.bin", K.ADDED, size=10**9),
        ]
    )
    assert sorted({v.rule for v in violations}) == [
        "file_too_large",
        "gate_config",
        "gitlink",
        "protected_path",
        "skip_marker",
        "suppression",
        "symlink_escape",
        "test_deleted",
    ]
    for v in violations:
        assert repr(v.path) in v.message or repr(f"{v.path}:{v.line}") in v.message
        assert "instead" in v.message, v.message


def test_suppression_message_tells_how_to_request_a_waiver() -> None:
    [violation] = check([changed("src/app.py", lines=("", "", "x  # noqa"))])
    assert "scope 'src/app.py:3'" in violation.message
    assert "verifier" in violation.message


def test_untrusted_paths_are_escaped_in_messages() -> None:
    path = "src/evil\nAll gates passed.py"
    [violation] = check([changed(path, K.ADDED, size=10**9)])
    assert "\n" not in violation.message
    assert violation.path == path
