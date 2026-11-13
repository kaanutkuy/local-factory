"""Unit tests for the project registry: names, records, lookups and every way adding refuses.

The refusals that need a real git repository live in tests/acceptance/test_projects.py; these
cover the registry's own logic and the failures git itself reports.
"""

from __future__ import annotations

import errno
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from local_factory.config import FactoryConfig, ProjectConfig
from local_factory.projects import (
    PROJECT_FILE,
    Project,
    ProjectError,
    ProjectPaths,
    add_project,
    list_projects,
    load_project,
    project_config,
    project_paths,
)
from tests.acceptance.conftest import Git
from tests.acceptance.intake_world import make_target

NOW = datetime(2026, 11, 13, 12, 0, tzinfo=UTC)


@pytest.fixture
def config(tmp_path: Path) -> FactoryConfig:
    return FactoryConfig(home=tmp_path / "factory")


def register(config: FactoryConfig, directory: str, *, called: str | None = None) -> Path:
    """A project record on disk, without a repository: enough for the registry's lookups.

    The record names the project ``called`` (default: its directory's name).
    """
    root = config.project_state_dir(directory)
    root.mkdir(parents=True)
    record = {
        "name": called or directory,
        "source": "/src",
        "branch": "main",
        "added_at": NOW.isoformat(),
    }
    (root / PROJECT_FILE).write_text(json.dumps(record))
    return root


def leftovers(config: FactoryConfig) -> list[str]:
    root = config.home / "projects"
    return sorted(p.name for p in root.iterdir()) if root.exists() else []


# --------------------------------------------------------------------------- the record


def test_a_project_needs_an_absolute_source_and_an_aware_time() -> None:
    fields: dict[str, object] = {
        "name": "p",
        "source": Path("/src"),
        "branch": "main",
        "added_at": NOW,
    }
    assert Project.model_validate(fields).source == Path("/src")
    for bad in (
        {"source": Path("relative")},
        {"added_at": datetime(2026, 1, 1)},
        {"name": "Not Valid"},
        {"name": "-lead"},
        {"branch": ""},
        {"extra": 1},
    ):
        with pytest.raises(ValidationError):
            Project.model_validate({**fields, **bad})


@pytest.mark.parametrize("name", ["p", "cli-target", "9lives", "a-b-c"])
def test_valid_names_are_the_project_config_s_names(name: str) -> None:
    assert ProjectConfig(name=name).name == name
    assert project_paths(FactoryConfig(home=Path("/h")), name).root == Path("/h/projects") / name


@pytest.mark.parametrize("name", ["", "-x", "A", "a b", "../x", "a/b", "a\n", "x.y", "é"])
def test_invalid_names_are_refused_like_the_project_config_does(name: str) -> None:
    with pytest.raises(ValidationError):
        ProjectConfig(name=name)
    with pytest.raises(ProjectError, match="invalid project name"):
        project_paths(FactoryConfig(home=Path("/h")), name)


def test_project_paths_are_all_inside_the_project_directory(config: FactoryConfig) -> None:
    paths = project_paths(config, "p")
    assert isinstance(paths, ProjectPaths)
    for path in (paths.repo, paths.ledger, paths.holdout, paths.transcripts, paths.work):
        assert path.parent == paths.root


# --------------------------------------------------------------------------- lookups


def test_nothing_registered(config: FactoryConfig) -> None:
    assert list_projects(config) == []
    with pytest.raises(ProjectError, match="factory project add"):
        load_project(config)
    with pytest.raises(ProjectError, match="none is registered"):
        load_project(config, "p")


def test_lookups_read_the_records_sorted_by_name(config: FactoryConfig) -> None:
    register(config, "zeta")
    register(config, "alpha")
    assert [p.name for p in list_projects(config)] == ["alpha", "zeta"]
    assert load_project(config, "zeta") == Project(
        name="zeta", source=Path("/src"), branch="main", added_at=NOW
    )
    with pytest.raises(ProjectError, match=r"registered: alpha, zeta"):
        load_project(config, "beta")
    with pytest.raises(ProjectError, match=r"alpha, zeta.*--project"):
        load_project(config)


def test_a_bad_name_is_refused_before_the_disk_is_touched(config: FactoryConfig) -> None:
    with pytest.raises(ProjectError, match="invalid project name"):
        load_project(config, "../escape")


def test_directories_that_are_not_projects_are_not_listed(config: FactoryConfig) -> None:
    register(config, "real")
    root = config.home / "projects"
    (root / ".adding-abc123").mkdir()  # a staging directory left by a crash
    (root / "no-record").mkdir()
    (root / "a-file").write_text("x")
    (root / "UPPER").mkdir()
    (root / "UPPER" / PROJECT_FILE).write_text("{}")
    assert [p.name for p in list_projects(config)] == ["real"]
    assert load_project(config).name == "real"


def test_a_corrupt_or_misfiled_record_is_an_error_naming_it(config: FactoryConfig) -> None:
    root = register(config, "p")
    (root / PROJECT_FILE).write_text("{not json")
    with pytest.raises(ProjectError, match="not a readable project record"):
        load_project(config, "p")
    with pytest.raises(ProjectError, match="not a readable project record"):
        list_projects(config)
    (root / PROJECT_FILE).write_text(json.dumps({"name": "p"}))  # valid JSON, not a record
    with pytest.raises(ProjectError, match="not a readable project record"):
        load_project(config, "p")

    other = register(config, "q", called="r")
    with pytest.raises(ProjectError, match="record of 'r', not 'q'"):
        load_project(config, "q")
    assert other.name == "q"


def test_an_unreadable_record_is_an_error_too(
    config: FactoryConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    register(config, "p")

    def unreadable(self: Path) -> bytes:
        raise PermissionError(errno.EACCES, "Permission denied")

    monkeypatch.setattr(Path, "read_bytes", unreadable)
    with pytest.raises(ProjectError, match="not a readable project record"):
        load_project(config, "p")


# --------------------------------------------------------------------------- project_config


def test_project_config_needs_a_repository(config: FactoryConfig) -> None:
    with pytest.raises(ProjectError, match="not a usable repository"):
        project_config(project_paths(config, "p"))


# --------------------------------------------------------------------------- add: refusals


def test_add_refuses_a_non_directory_and_a_non_repository(
    config: FactoryConfig, tmp_path: Path
) -> None:
    (tmp_path / "file").write_text("x")
    with pytest.raises(ProjectError, match="not a directory"):
        add_project(config, tmp_path / "file")
    with pytest.raises(ProjectError, match="not a git repository"):
        add_project(config, tmp_path)
    assert not config.home.exists()  # nothing was created for a refusal this early


@pytest.mark.parametrize("branch", ["", "a b", "a..b", "-x:y", "x*", "a:b", "x.lock", "/x", "x/"])
def test_add_refuses_an_invalid_branch_name(
    config: FactoryConfig, tmp_path: Path, branch: str
) -> None:
    (tmp_path / "src" / ".git").mkdir(parents=True)
    with pytest.raises(ProjectError, match="not a valid branch name"):
        add_project(config, tmp_path / "src", branch=branch)
    assert not config.home.exists()


def test_add_reports_what_git_says_when_it_cannot_fetch(
    config: FactoryConfig, tmp_path: Path
) -> None:
    (tmp_path / "broken" / ".git").mkdir(parents=True)  # looks like a repository, is not one
    with pytest.raises(ProjectError, match=r"cannot fetch 'main' from .*broken"):
        add_project(config, tmp_path / "broken")
    assert leftovers(config) == []


def test_a_bare_repository_is_a_repository_too(config: FactoryConfig, tmp_path: Path) -> None:
    bare = tmp_path / "bare.git"
    (bare / "objects").mkdir(parents=True)
    (bare / "HEAD").write_text("ref: refs/heads/main\n")
    with pytest.raises(ProjectError, match="cannot fetch"):  # past the "is it a repository" check
        add_project(config, bare)
    assert leftovers(config) == []


# --------------------------------------------------------------------------- add: real git


def target_with(tmp_path: Path, git: Git, **files: bytes) -> Path:
    target = make_target(tmp_path, git)
    for path, content in files.items():
        (target / path).write_bytes(content)
    git("add", "-A", cwd=target)
    git("commit", "-q", "-m", "change", cwd=target)
    return target


@pytest.fixture
def git(tmp_path: Path) -> Git:
    (tmp_path / "githome").mkdir()
    return Git(tmp_path / "githome")


@pytest.mark.integration
@pytest.mark.parametrize(
    ("toml", "needle"),
    [
        (b"\xff\xfe\x00", "not valid TOML"),
        (b'[project]\nname = "x"\n[nope]\n', "unknown section"),
        (b'[project]\nname = "Bad Name"\n', "factory.toml"),
        (b"", "factory.toml"),  # no [project] name
    ],
    ids=["not-utf8", "unknown-section", "bad-name", "empty"],
)
def test_add_refuses_an_invalid_factory_toml(
    config: FactoryConfig, tmp_path: Path, git: Git, toml: bytes, needle: str
) -> None:
    target = target_with(tmp_path, git, **{"factory.toml": toml})
    with pytest.raises(ProjectError, match=needle):
        add_project(config, target)
    assert leftovers(config) == []


@pytest.mark.integration
def test_a_directory_named_factory_toml_is_a_missing_config(
    config: FactoryConfig, tmp_path: Path, git: Git
) -> None:
    target = make_target(tmp_path, git)
    (target / "factory.toml").unlink()
    (target / "factory.toml").mkdir()
    (target / "factory.toml" / "x").write_text("x")
    git("add", "-A", cwd=target)
    git("commit", "-q", "-m", "dir", cwd=target)
    with pytest.raises(ProjectError, match=r"no factory\.toml at its root"):
        add_project(config, target)


@pytest.mark.integration
def test_a_move_that_fails_leaves_nothing_behind(
    config: FactoryConfig, tmp_path: Path, git: Git, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = make_target(tmp_path, git)

    def refuse(self: Path, destination: object) -> None:
        raise OSError(errno.EXDEV, "Invalid cross-device link")

    monkeypatch.setattr(Path, "rename", refuse)
    with pytest.raises(ProjectError, match=r"cannot create .*cli-target.*cross-device"):
        add_project(config, target)
    assert leftovers(config) == []


@pytest.mark.integration
def test_add_creates_the_state_private_and_complete(
    config: FactoryConfig, tmp_path: Path, git: Git
) -> None:
    target = make_target(tmp_path, git)
    clock_calls: list[int] = []

    def clock() -> datetime:
        clock_calls.append(1)
        return NOW

    project = add_project(config, target, clock=clock)
    assert clock_calls == [1] and project.added_at == NOW
    assert leftovers(config) == ["cli-target"]  # no staging directory remains
    record = json.loads((project_paths(config, "cli-target").root / PROJECT_FILE).read_text())
    assert record["source"] == str(target.resolve())
    assert record["added_at"].startswith("2026-11-13T12:00:00")
    assert oct(config.home.stat().st_mode & 0o777) == "0o700"
    assert oct((config.home / "projects").stat().st_mode & 0o777) == "0o700"


@pytest.mark.integration
def test_a_second_add_of_another_branch_is_a_second_project(
    config: FactoryConfig, tmp_path: Path, git: Git
) -> None:
    target = make_target(tmp_path, git)
    git("branch", "release", cwd=target)
    add_project(config, target)
    add_project(config, target, name="cli-target-release", branch="release")
    assert [p.name for p in list_projects(config)] == ["cli-target", "cli-target-release"]
    with pytest.raises(ProjectError, match="already exists"):
        add_project(config, target, branch="release")
    assert leftovers(config) == ["cli-target", "cli-target-release"]
