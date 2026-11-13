"""Deterministic diff policy for imported task diffs (docs/specs/phase-1a.md § policy, PLAN §4.2).

Every rule answers one question about a ``ChangedFile`` and, when it fires, says what was blocked,
why, and what the implementer should do instead. The policy is fail-closed: when a change cannot be
shown to be safe (an unknown symlink target, a gate config replaced wholesale), it is a violation.

Line rules match the listed markers robustly: skip markers (checked in test files only) tolerate
whitespace around ``.`` and ``(`` and only match where the marker starts a name (so ``sys.exit(``
is not ``xit(``); suppressions also tolerate the spacing and letter-case variants their tools
accept (``# NOQA``, ``#type:ignore``, ``# pragma:no cover``); pyproject markers follow TOML syntax.

Symlinks are resolved component by component against the other symlinks in the change set. Only
those are known here; ``import_task_bundle`` repeats the check against the whole head tree.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from local_factory.config import DEFAULT_TEST_GLOBS, PROJECT_CONFIG_FILENAME, ProjectConfig
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
from local_factory.policy.globs import glob_match

__all__ = ["DEFAULT_TEST_GLOBS", "DiffPolicy", "check_changes", "glob_match"]

DEFAULT_MAX_FILE_BYTES = 1_000_000

PROTECTED_PATH = "protected_path"
TEST_DELETED = "test_deleted"
SKIP_MARKER = "skip_marker"
SUPPRESSION = "suppression"
GATE_CONFIG = "gate_config"
GITLINK = "gitlink"
SYMLINK_ESCAPE = "symlink_escape"
FILE_TOO_LARGE = "file_too_large"

_Marker = tuple[str, re.Pattern[str]]  # (the marker as listed in the spec, its matcher)


def _markers(flags: int, *pairs: tuple[str, str]) -> tuple[_Marker, ...]:
    return tuple((label, re.compile(regex, flags)) for label, regex in pairs)


# Identifier-like markers match only where a name starts (``sys.exit(`` is not ``xit(``). The
# Jasmine/Jest globals are called bare, so an attribute call (``model.fit(``) is not a marker.
_NAME_START = r"(?<!\w)"
_BARE_CALL = r"(?<![\w.])"
_SKIP_MARKERS = _markers(
    0,
    ("pytest.mark.skip", _NAME_START + r"pytest\s*\.\s*mark\s*\.\s*skip"),
    ("pytest.mark.xfail", _NAME_START + r"pytest\s*\.\s*mark\s*\.\s*xfail"),
    ("pytest.skip(", _NAME_START + r"pytest\s*\.\s*skip\s*\("),
    ("importorskip(", _NAME_START + r"importorskip\s*\("),
    ("unittest.skip", _NAME_START + r"unittest\s*\.\s*skip"),
    ("skipTest(", _NAME_START + r"skipTest\s*\("),
    ("SkipTest", _NAME_START + r"SkipTest"),
    ("it.skip(", _NAME_START + r"it\s*\.\s*skip\s*\("),
    ("describe.skip(", _NAME_START + r"describe\s*\.\s*skip\s*\("),
    ("test.skip(", _NAME_START + r"test\s*\.\s*skip\s*\("),
    (".only(", r"\.\s*only\s*\("),
    ("xit(", _BARE_CALL + r"xit\s*\("),
    ("xdescribe(", _BARE_CALL + r"xdescribe\s*\("),
    ("fit(", _BARE_CALL + r"fit\s*\("),
    ("fdescribe(", _BARE_CALL + r"fdescribe\s*\("),
)
_SUPPRESSIONS = _markers(
    re.IGNORECASE,
    ("pragma: no cover", r"pragma\s*:?\s*no\s*cover"),
    ("pragma: no branch", r"pragma\s*:?\s*no\s*branch"),
    ("pragma: no mutate", r"pragma\s*:?\s*no\s*mutate"),
    ("noqa", r"noqa"),
    ("type: ignore", r"type\s*:\s*ignore"),
    ("pyright: ignore", r"pyright\s*:\s*ignore"),
    # mypy inline config takes a comma-separated flag list, with "-" or "_" in flag names
    ("mypy: ignore-errors", r"mypy\s*:[^#\n]*ignore[-_]errors"),
    ("nosec", r"nosec"),
    ("pylint: disable", r"pylint\s*:\s*disable"),
    ("eslint-disable", r"eslint-disable"),
    ("@ts-ignore", r"@ts-ignore"),
    ("@ts-expect-error", r"@ts-expect-error"),
    ("@ts-nocheck", r"@ts-nocheck"),
    ("istanbul ignore", r"istanbul\s*ignore"),
    ("c8 ignore", r"c8\s*ignore"),
)
# pyproject.toml markers are anchored to TOML syntax, not free text. A table may be spaced or
# quoted ([ tool . "coverage" . report ], [[tool.mypy.overrides]]) or written as a dotted key
# (tool.mypy.strict = false). A key may sit at the start of the line or of an inline-table entry,
# behind optional quotes and a dotted prefix (report.fail_under = 0, { "omit" = [] }).
_Q = r"[\"']?"
_TOML_KEY_PART = r"(?:[A-Za-z0-9_-]+|\"(?:[^\"\\]|\\.)*\"|'[^']*')"
_GATE_TABLES = ("coverage", "pytest", "mutmut", "mypy", "ruff")
_GATE_KEYS = (
    "fail_under",
    "addopts",
    "omit",
    "exclude_also",
    "exclude_lines",
    "testpaths",
    "norecursedirs",
    "python_files",
    "python_classes",
    "python_functions",
)
_PYPROJECT_MARKERS = _markers(
    0,
    *(
        (f"[tool.{table}", rf"^\s*(?:\[\[?\s*)?{_Q}tool{_Q}\s*\.\s*{_Q}{table}{_Q}\s*[.=\]]")
        for table in _GATE_TABLES
    ),
    *(
        (key, rf"(?:^|[{{,])\s*(?:{_TOML_KEY_PART}\s*\.\s*)*{_Q}{key}{_Q}\s*=")
        for key in _GATE_KEYS
    ),
)
# Whole files that configure test selection, coverage or mutation. Matched by file name anywhere
# in the tree; includes the pytest>=9 names (.pytest.ini, pytest.toml, .pytest.toml).
GATE_CONFIG_FILES = frozenset(
    {
        ".coveragerc",
        "pytest.ini",
        ".pytest.ini",
        "pytest.toml",
        ".pytest.toml",
        "tox.ini",
        "setup.cfg",
        ".mutmut-config",
    }
)
PYPROJECT = "pyproject.toml"
CONFTEST = "conftest.py"
_REGULAR_MODES = (MODE_REGULAR, MODE_EXECUTABLE)

_VERBS = {
    ChangeKind.ADDED: "Adding",
    ChangeKind.MODIFIED: "Modifying",
    ChangeKind.DELETED: "Deleting",
    ChangeKind.RENAMED: "Renaming",
    ChangeKind.TYPE_CHANGED: "Changing the file type of",
}


@dataclass(frozen=True)
class DiffPolicy:
    protected: tuple[str, ...]
    test_globs: tuple[str, ...] = DEFAULT_TEST_GLOBS
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES

    def __post_init__(self) -> None:
        for name, globs in (("protected", self.protected), ("test_globs", self.test_globs)):
            if isinstance(globs, str):
                raise TypeError(f"DiffPolicy.{name} must be a tuple of globs, not a string")
        if self.max_file_bytes < 0:
            raise ValueError(f"max_file_bytes must be >= 0, got {self.max_file_bytes}")

    @classmethod
    def from_project(cls, config: ProjectConfig) -> DiffPolicy:
        """The project's protected and test globs; ``factory.toml`` is protected no matter what."""
        return cls(
            protected=tuple(dict.fromkeys((*config.protected, PROJECT_CONFIG_FILENAME))),
            test_globs=config.test_globs,
        )


def check_changes(
    changes: Sequence[ChangedFile], policy: DiffPolicy, waivers: Sequence[Waiver] = ()
) -> list[Violation]:
    """All violations, one per (rule, path, line), sorted by (path, line or 0, rule).

    A suppression is allowed only if an *effective* waiver (adjudicated by someone other than its
    requester) has ``scope == f"{path}:{line_no}"``.
    """
    waived_lines = frozenset(w.scope for w in waivers if w.is_effective)
    links = frozenset(_normalize(c.path) for c in changes if c.mode == MODE_SYMLINK)
    found: dict[tuple[str, str, int | None], Violation] = {}
    for change in changes:
        for violation in _check_change(change, policy, waived_lines, links):
            found.setdefault((violation.rule, violation.path, violation.line), violation)
    return sorted(found.values(), key=lambda v: (v.path, v.line or 0, v.rule))


# --------------------------------------------------------------------------- per-change rules


def _check_change(
    change: ChangedFile,
    policy: DiffPolicy,
    waived_lines: frozenset[str],
    links: frozenset[str],
) -> Iterator[Violation]:
    yield from _protected(change, policy)
    yield from _test_deleted(change, policy)
    yield from _gate_config_file(change)
    yield from _structural(change, policy, links)
    test_file = _is_test_file(change.path, policy)
    pyproject = _basename(change.path) == PYPROJECT
    for added in change.added_lines:
        yield from _line_rules(
            change, added, waived_lines, test_file=test_file, pyproject=pyproject
        )


def _protected(change: ChangedFile, policy: DiffPolicy) -> Iterator[Violation]:
    for touched in (change.path, change.old_path):
        glob = None if touched is None else _first_match(touched, policy.protected)
        if glob is not None:
            yield _violation(
                PROTECTED_PATH,
                change,
                f"{_action(change)} touches {touched!r}, which matches the protected pattern "
                f"{glob!r}. Protected files belong to the independent oracle and the factory "
                "configuration (acceptance tests, factory.toml, CI), so implementers may not add, "
                "modify, rename or delete them. Revert this change; if you believe the protected "
                "file is wrong, explain why in your task report instead.",
            )
            return


def _test_deleted(change: ChangedFile, policy: DiffPolicy) -> Iterator[Violation]:
    if change.kind is ChangeKind.DELETED and _is_test(change.path, policy):
        yield _violation(
            TEST_DELETED,
            change,
            f"Deleting the test file {change.path!r} is not allowed: removing tests weakens the "
            "oracle your work is verified against. Keep the test and make it pass (fix the code, "
            "or the test if it is genuinely wrong); if you believe it is obsolete, explain why in "
            "your task report instead.",
        )
    elif (
        change.kind is ChangeKind.RENAMED
        and change.old_path is not None
        and _is_test(change.old_path, policy)
        and not _is_test(change.path, policy)
    ):
        yield _violation(
            TEST_DELETED,
            change,
            f"Renaming the test file {change.old_path!r} to {change.path!r} moves it outside the "
            f"test paths ({', '.join(policy.test_globs)}), so it would silently stop running. Keep "
            "it at a test path; if you believe it should not be a test, explain why in your task "
            "report instead.",
        )


def _gate_config_file(change: ChangedFile) -> Iterator[Violation]:
    names = {_basename(p) for p in (change.path, change.old_path) if p is not None}
    config_files = sorted(names & GATE_CONFIG_FILES)
    if config_files:
        yield _violation(
            GATE_CONFIG,
            change,
            f"{_action(change)} changes the gate configuration file {config_files[0]!r}. Test "
            "selection, coverage and mutation settings are owned by the factory because changing "
            "them can weaken the gates that judge your work. Revert this change; if the "
            "configuration really needs to change, describe the change and why in your task "
            "report instead.",
        )
    elif PYPROJECT in names and _replaces_pyproject_wholesale(change):
        yield _violation(
            GATE_CONFIG,
            change,
            f"{_action(change)} replaces a pyproject.toml as a whole (deleted, renamed, or not a "
            "regular text file), which bypasses the line-level checks of its gate configuration "
            "([tool.pytest], [tool.coverage], [tool.mutmut], [tool.mypy], [tool.ruff]). Keep "
            "pyproject.toml a regular file at its path and edit it line by line; if its gate "
            "configuration needs to change, describe the change in your task report instead.",
        )


def _replaces_pyproject_wholesale(change: ChangedFile) -> bool:
    if change.kind in (ChangeKind.DELETED, ChangeKind.RENAMED, ChangeKind.TYPE_CHANGED):
        return True
    return change.is_binary or change.mode not in _REGULAR_MODES


def _structural(
    change: ChangedFile, policy: DiffPolicy, links: frozenset[str]
) -> Iterator[Violation]:
    if change.mode == MODE_GITLINK:
        yield _violation(
            GITLINK,
            change,
            f"{change.path!r} is a git submodule (gitlink, mode {MODE_GITLINK}). Submodules pull "
            "in code from outside the repository that the factory cannot import or verify. Remove "
            "it; vendor the code as regular files or declare it as a package dependency instead.",
        )
    if change.mode == MODE_SYMLINK:
        problem = _symlink_problem(change.path, change.symlink_target, links)
        if problem is not None:
            yield _violation(
                SYMLINK_ESCAPE,
                change,
                f"{change.path!r} is a symlink {problem}. Symlinks must resolve inside the "
                "repository without passing through other symlinks, because the factory "
                "materializes your tree on the host. Point it directly at the real file or "
                "directory with a relative target, or commit a regular file instead.",
            )
    if change.size is not None and change.size > policy.max_file_bytes:
        yield _violation(
            FILE_TOO_LARGE,
            change,
            f"{change.path!r} is {change.size:,} bytes, over the {policy.max_file_bytes:,}-byte "
            "limit for a single file. Large files (build outputs, datasets, archives, binaries) "
            "do not belong in a change. Remove it, make it smaller, or generate it during the "
            "build instead.",
        )


def _line_rules(
    change: ChangedFile,
    added: AddedLine,
    waived_lines: frozenset[str],
    *,
    test_file: bool,
    pyproject: bool,
) -> Iterator[Violation]:
    where = f"{change.path}:{added.line_no}"
    marker = _first_marker(_SKIP_MARKERS, added.text) if test_file else None
    if marker is not None:
        yield _violation(
            SKIP_MARKER,
            change,
            f"{where!r} adds the test skip/focus marker {marker!r}. Skipped, xfailed and focused "
            "tests hide failures from the gates. Remove the marker and make the test pass; if the "
            "test itself is wrong, report it in your task report instead of skipping it.",
            added.line_no,
        )
    marker = _first_marker(_SUPPRESSIONS, added.text)
    if marker is not None and where not in waived_lines:
        yield _violation(
            SUPPRESSION,
            change,
            f"{where!r} adds the gate suppression {marker!r} without an approved waiver. "
            "Suppressions silence lint, type, coverage, mutation or security checks. Fix the "
            "underlying problem instead; if it truly cannot be fixed, request a waiver with scope "
            f"{where!r} and your reason in your task report - the verifier adjudicates it.",
            added.line_no,
        )
    marker = _first_marker(_PYPROJECT_MARKERS, added.text) if pyproject else None
    if marker is not None:
        yield _violation(
            GATE_CONFIG,
            change,
            f"{where!r} changes gate configuration in pyproject.toml ({marker!r}). Test "
            "selection, coverage, mutation, lint and type-check settings are owned by the factory "
            "because changing them can weaken the gates that judge your work. Revert this line; "
            "if the configuration really needs to change, describe the change and why in your "
            "task report instead.",
            added.line_no,
        )


# --------------------------------------------------------------------------- helpers


def _violation(rule: str, change: ChangedFile, message: str, line: int | None = None) -> Violation:
    return Violation(rule=rule, path=change.path, line=line, message=message)


def _action(change: ChangedFile) -> str:
    renamed = f" (renamed from {change.old_path!r})" if change.old_path is not None else ""
    return f"{_VERBS[change.kind]} {change.path!r}{renamed}"


def _first_match(path: str, globs: Sequence[str]) -> str | None:
    return next((glob for glob in globs if glob_match(path, glob)), None)


def _is_test(path: str, policy: DiffPolicy) -> bool:
    return _first_match(path, policy.test_globs) is not None


def _is_test_file(path: str, policy: DiffPolicy) -> bool:
    """Where skip markers count: test paths, plus every ``conftest.py`` (it can skip anything)."""
    return _is_test(path, policy) or _basename(path) == CONFTEST


def _first_marker(markers: Sequence[_Marker], text: str) -> str | None:
    return next((label for label, regex in markers if regex.search(text)), None)


def _basename(path: str) -> str:
    return path.rsplit("/", 1)[-1]


def _normalize(path: str) -> str:
    return "/".join(part for part in path.split("/") if part not in ("", "."))


def _symlink_problem(path: str, target: str | None, links: frozenset[str]) -> str | None:
    """Why the symlink at ``path`` is unsafe, or None if its target stays inside the repo.

    The target is resolved component by component from the link's directory, the way the kernel
    would. Climbing above the root at any point is an escape, even if later components come back
    down. Passing through any symlink of the change set - intermediate or final, escaping or not -
    is rejected outright: a chain like ``a -> .`` plus ``b -> a/..`` looks lexically harmless but
    resolves outside the repository. ``links`` holds the normalized paths of those symlinks.
    """
    if target is None:
        return "whose target was not recorded, so it cannot be shown to stay inside the repository"
    if target.startswith("/"):
        return f"to the absolute path {target!r}"
    resolved: list[str] = []
    for part in [*path.split("/")[:-1], *target.split("/")]:
        if part in ("", "."):
            continue
        if part != "..":
            resolved.append(part)
            through = "/".join(resolved)
            if through in links:
                return f"to {target!r}, which passes through the symlink {through!r}"
        elif resolved:
            resolved.pop()
        else:
            return f"to {target!r}, which resolves outside the repository"
    return None
