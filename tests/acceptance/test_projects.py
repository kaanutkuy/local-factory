"""`factory project add` and the project registry (docs/specs/phase-2b.md § projects.py).
Protected.
"""

from __future__ import annotations

import json
import stat
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest

from local_factory.config import FactoryConfig
from local_factory.gitops.bare import BareRepo
from local_factory.ledger import Ledger
from local_factory.projects import (
    PROJECT_FILE,
    Project,
    ProjectError,
    add_project,
    list_projects,
    load_project,
    project_config,
    project_paths,
)
from tests.acceptance.conftest import Git
from tests.acceptance.intake_world import MAIN, make_target

pytestmark = pytest.mark.integration

NOW = datetime(2026, 11, 13, 12, 0, tzinfo=UTC)


@pytest.fixture
def git(tmp_path: Path) -> Git:
    (tmp_path / "githome").mkdir()
    return Git(tmp_path / "githome")


@pytest.fixture
def config(tmp_path: Path) -> FactoryConfig:
    return FactoryConfig(home=tmp_path / "factory")


@pytest.fixture
def target(tmp_path: Path, git: Git) -> Path:
    return make_target(tmp_path, git)


def commit(git: Git, repo: Path, message: str = "change") -> None:
    git("add", "-A", cwd=repo)
    git("commit", "-q", "-m", message, cwd=repo)


def projects_dir(config: FactoryConfig) -> Path:
    return config.home / "projects"


def leftovers(config: FactoryConfig) -> list[str]:
    root = projects_dir(config)
    return sorted(p.name for p in root.iterdir()) if root.exists() else []


def test_add_adopts_a_repository(config: FactoryConfig, target: Path, git: Git) -> None:
    project = add_project(config, target, clock=lambda: NOW)
    assert project == Project(
        name="cli-target", source=target.resolve(), branch="main", added_at=NOW
    )

    paths = project_paths(config, "cli-target")
    assert paths.root == config.project_state_dir("cli-target")
    assert (paths.repo, paths.ledger, paths.holdout, paths.transcripts, paths.work) == (
        paths.root / "repo.git",
        paths.root / "ledger.db",
        paths.root / "holdout",
        paths.root / "transcripts",
        paths.root / "work",
    )
    for directory in (paths.root, paths.holdout):
        assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    assert paths.transcripts.is_dir() and paths.work.is_dir()
    assert BareRepo.open(paths.repo).rev_parse(MAIN) == git("rev-parse", "HEAD", cwd=target)
    with Ledger(paths.ledger) as ledger:
        assert ledger.list_tickets() == []
    record = json.loads((paths.root / PROJECT_FILE).read_text())
    assert record["name"] == "cli-target" and record["branch"] == "main"

    assert project_config(paths).name == "cli-target"
    assert "acceptance" in project_config(paths).commands
    assert list_projects(config) == [project]
    assert load_project(config) == project == load_project(config, "cli-target")


def test_add_takes_a_name_and_a_branch(config: FactoryConfig, target: Path, git: Git) -> None:
    git("checkout", "-q", "-b", "trunk", cwd=target)
    (target / "extra.txt").write_text("on trunk\n")
    commit(git, target)
    project = add_project(config, target, name="greeter", branch="trunk")
    assert (project.name, project.branch) == ("greeter", "trunk")
    repo = BareRepo.open(project_paths(config, "greeter").repo)
    assert repo.read_file(MAIN, "extra.txt") == b"on trunk\n"


def test_the_config_comes_from_the_commit_not_the_worktree(
    config: FactoryConfig, target: Path, git: Git
) -> None:
    toml = (target / "factory.toml").read_text()
    (target / "factory.toml").write_text(toml.replace("acceptance", "not_acceptance"))
    project = add_project(config, target)  # the committed file still has the command
    assert "acceptance" in project_config(project_paths(config, project.name)).commands

    other = FactoryConfig(home=config.home.parent / "other-factory")
    commit(git, target)
    (target / "factory.toml").write_text(toml)  # fixed in the worktree only
    with pytest.raises(ProjectError, match="acceptance"):
        add_project(other, target)
    assert leftovers(other) == []


@pytest.mark.parametrize(
    ("change", "needle"),
    [
        (lambda t: (t / "factory.toml").unlink(), "factory.toml"),
        (lambda t: (t / "factory.toml").write_text("[project\n"), "factory.toml"),
        (
            lambda t: (t / "factory.toml").write_text(
                (t / "factory.toml").read_text().replace("acceptance = ", "nope = ")
            ),
            "acceptance",
        ),
        (
            lambda t: (t / "factory.toml").write_text(
                (t / "factory.toml").read_text().replace("{path}", "tests")
            ),
            "acceptance",
        ),
    ],
    ids=["no-config", "bad-toml", "no-acceptance", "no-path-placeholder"],
)
def test_add_refuses_a_target_it_cannot_run(
    config: FactoryConfig, target: Path, git: Git, change: Callable[[Path], object], needle: str
) -> None:
    change(target)
    commit(git, target)
    with pytest.raises(ProjectError, match=needle):
        add_project(config, target)
    assert leftovers(config) == []
    assert list_projects(config) == []


def test_add_refuses_what_is_not_a_repository(config: FactoryConfig, tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    for source in (plain, tmp_path / "missing"):
        with pytest.raises(ProjectError):
            add_project(config, source)
    assert leftovers(config) == []


def test_add_refuses_a_missing_branch(config: FactoryConfig, target: Path) -> None:
    with pytest.raises(ProjectError, match="release"):
        add_project(config, target, branch="release")
    assert leftovers(config) == []


def test_add_refuses_a_second_project_of_the_same_name(config: FactoryConfig, target: Path) -> None:
    add_project(config, target)
    with pytest.raises(ProjectError, match="already"):
        add_project(config, target)
    with pytest.raises(ProjectError):
        add_project(config, target, name="Not A Valid Name")
    assert leftovers(config) == ["cli-target"]


def test_add_reports_every_unsafe_symlink(config: FactoryConfig, target: Path, git: Git) -> None:
    (target / "passwd").symlink_to("/etc/passwd")
    (target / "docs").mkdir()
    (target / "docs" / "up").symlink_to("../../outside")
    (target / "ok").symlink_to("greet.py")
    commit(git, target, "links")
    with pytest.raises(ProjectError) as caught:
        add_project(config, target)
    message = str(caught.value)
    assert "passwd ->" in message and "docs/up ->" in message
    assert "ok ->" not in message
    assert leftovers(config) == []


def test_load_project_needs_an_unambiguous_project(
    config: FactoryConfig, target: Path, tmp_path: Path, git: Git
) -> None:
    with pytest.raises(ProjectError, match="project add"):
        load_project(config)
    add_project(config, target)
    add_project(config, target, name="second")
    with pytest.raises(ProjectError, match="--project") as caught:
        load_project(config)
    assert "cli-target" in str(caught.value) and "second" in str(caught.value)
    assert load_project(config, "second").name == "second"
    assert [p.name for p in list_projects(config)] == ["cli-target", "second"]
    with pytest.raises(ProjectError):
        load_project(config, "third")
