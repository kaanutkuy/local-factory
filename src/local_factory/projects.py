"""The project registry: `factory project add` and where a project's factory state lives
(docs/specs/phase-2b.md § projects.py).

Adopting a repository copies *one branch of it* into the factory's own bare repository and reads
the project's configuration from the fetched commit, never from the source's worktree: what the
factory later checks is exactly what it adopted. Nothing is registered unless every check passes;
the project's directory appears under ``projects/`` complete, in one rename, or not at all.
"""

from __future__ import annotations

import re
import shutil
import tempfile
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from pydantic import AwareDatetime, Field, ValidationError, field_validator

from local_factory.config import (
    ACCEPTANCE_COMMAND,
    PROJECT_CONFIG_FILENAME,
    ConfigError,
    FactoryConfig,
    ProjectConfig,
    parse_project_config,
)
from local_factory.gitops import BareRepo, GitError, run_git
from local_factory.gitops.bundles import unsafe_symlinks
from local_factory.hashing import utc_now
from local_factory.ledger import Ledger
from local_factory.models import FrozenModel

__all__ = [
    "PROJECT_FILE",
    "Project",
    "ProjectError",
    "ProjectPaths",
    "add_project",
    "list_projects",
    "load_project",
    "project_config",
    "project_paths",
]

PROJECT_FILE = "project.json"
MAIN_REF = "refs/heads/main"  # the project's main in the factory's repository
_NAME_PATTERN = r"^[a-z0-9][a-z0-9-]*$"  # ProjectConfig.name's pattern
_NAME = re.compile(r"[a-z0-9][a-z0-9-]*")
_STAGING_PREFIX = ".adding-"
_PRIVATE = 0o700


class ProjectError(Exception):
    """A project cannot be added, found or used; the message says what to do about it."""


class Project(FrozenModel):
    """A registered project: the record in ``project.json``."""

    name: str = Field(pattern=_NAME_PATTERN)
    source: Path  # absolute: the repository it was adopted from
    branch: str = Field(min_length=1)  # the source's branch that became the project's main
    added_at: AwareDatetime

    @field_validator("source")
    @classmethod
    def _absolute(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError(f"source must be an absolute path: {value}")
        return value


@dataclass(frozen=True)
class ProjectPaths:
    """Where a project's factory state lives (``FactoryConfig.project_state_dir``)."""

    root: Path
    repo: Path  # the factory's bare repository; refs/heads/main is the project's main
    ledger: Path
    holdout: Path  # 0700: the sealed holdout records
    transcripts: Path
    work: Path  # scratch: base bundles and verification trees


def project_paths(config: FactoryConfig, name: str) -> ProjectPaths:
    _check_name(name)
    root = config.project_state_dir(name)
    return ProjectPaths(
        root=root,
        repo=root / "repo.git",
        ledger=root / "ledger.db",
        holdout=root / "holdout",
        transcripts=root / "transcripts",
        work=root / "work",
    )


# --------------------------------------------------------------------------- add


def add_project(
    config: FactoryConfig,
    source: Path,
    *,
    name: str | None = None,
    branch: str = "main",
    clock: Callable[[], datetime] = utc_now,
) -> Project:
    """Adopt ``branch`` of the git repository at ``source`` as a project.

    Refuses, with a :class:`ProjectError`, unless: ``source`` is a git repository with that
    branch; its ``factory.toml`` (read from the branch's commit) is valid; the name is free;
    ``[commands] acceptance`` can run one test (it contains ``{path}``); and the tree has no
    symlink the import check would reject (every task on such a base would fail closed with
    ``symlink_escape``). A refusal leaves nothing under ``projects/``.
    """
    source = source.resolve()
    if not source.is_dir():
        raise ProjectError(f"{source} is not a directory")
    if not _is_git_repository(source):
        raise ProjectError(f"{source} is not a git repository (no .git inside it)")
    _check_branch(branch)

    projects = config.home / "projects"
    projects.parent.mkdir(mode=_PRIVATE, parents=True, exist_ok=True)
    projects.mkdir(mode=_PRIVATE, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=_STAGING_PREFIX, dir=projects))  # created 0700
    try:
        project = _stage(config, staging, source, name=name, branch=branch, clock=clock)
        _move_into_place(staging, projects / project.name)
    finally:
        shutil.rmtree(staging, ignore_errors=True)  # gone already after a successful move
    return project


def _stage(
    config: FactoryConfig,
    staging: Path,
    source: Path,
    *,
    name: str | None,
    branch: str,
    clock: Callable[[], datetime],
) -> Project:
    """Everything a project needs, built and checked in ``staging`` (the project's future root)."""
    repo = BareRepo.init(staging / "repo.git")
    _fetch_branch(repo, source, branch)
    project_cfg = _parse_config(_read_config(repo, branch))
    chosen = project_cfg.name if name is None else name
    _check_name(chosen)
    if (config.project_state_dir(chosen)).exists():
        raise ProjectError(f"project {chosen!r} already exists; pick another name with --name")
    _check_acceptance(project_cfg)
    _check_symlinks(repo, branch)

    project = Project(name=chosen, source=source, branch=branch, added_at=clock())
    Ledger(staging / "ledger.db").close()  # creates the schema
    for directory in ("holdout", "transcripts", "work"):
        (staging / directory).mkdir(mode=_PRIVATE)
    (staging / PROJECT_FILE).write_text(project.model_dump_json(indent=2) + "\n")  # last
    return project


def _move_into_place(staging: Path, destination: Path) -> None:
    try:
        staging.rename(destination)  # atomic; fails if the destination is a non-empty directory
    except OSError as exc:
        raise ProjectError(f"cannot create {destination}: {exc.strerror}") from exc


def _is_git_repository(path: Path) -> bool:
    """A worktree (``.git`` directory or file) or a bare repository, as git's own lookup sees it."""
    return (path / ".git").exists() or ((path / "HEAD").is_file() and (path / "objects").is_dir())


def _check_branch(branch: str) -> None:
    """``branch`` must be a valid branch name: it goes into a refspec."""
    result = run_git(["check-ref-format", f"refs/heads/{branch}"], check=False)
    if result.returncode != 0:
        raise ProjectError(f"{branch!r} is not a valid branch name")


def _fetch_branch(repo: BareRepo, source: Path, branch: str) -> None:
    try:
        repo.fetch_from(source, [f"refs/heads/{branch}:{MAIN_REF}"])
    except GitError as exc:
        if "couldn't find remote ref" in exc.stderr:
            raise ProjectError(f"{source} has no branch {branch!r}; pass --branch") from exc
        raise ProjectError(f"cannot fetch {branch!r} from {source}: {exc.stderr.strip()}") from exc


def _read_config(repo: BareRepo, branch: str) -> bytes:
    try:
        return repo.read_file(MAIN_REF, PROJECT_CONFIG_FILENAME)
    except GitError as exc:
        raise ProjectError(
            f"branch {branch!r} has no {PROJECT_CONFIG_FILENAME} at its root: commit one first "
            "(a project needs its commands and protected paths)"
        ) from exc


def _parse_config(data: bytes) -> ProjectConfig:
    try:
        return parse_project_config(tomllib.loads(data.decode("utf-8")))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ProjectError(f"{PROJECT_CONFIG_FILENAME} is not valid TOML: {exc}") from exc
    except ConfigError as exc:
        raise ProjectError(str(exc)) from exc


def _check_acceptance(config: ProjectConfig) -> None:
    command = config.commands.get(ACCEPTANCE_COMMAND)
    if command is None or "{path}" not in command:
        raise ProjectError(
            f"{PROJECT_CONFIG_FILENAME} needs [commands] {ACCEPTANCE_COMMAND}, the command that "
            "runs one acceptance test, containing {path} (and optionally {name}); for example "
            f'{ACCEPTANCE_COMMAND} = "python3 -m unittest {{path}} -k {{name}}"'
        )


def _check_symlinks(repo: BareRepo, branch: str) -> None:
    unsafe = unsafe_symlinks(repo, MAIN_REF)
    if unsafe:
        listing = "\n".join(f"  {path} -> {reason}" for path, reason in unsafe)
        raise ProjectError(
            f"branch {branch!r} has symlinks that escape the repository, so every task on it "
            f"would be rejected (symlink_escape). Replace them with regular files:\n{listing}"
        )


# --------------------------------------------------------------------------- look up


def list_projects(config: FactoryConfig) -> list[Project]:
    """Every registered project, sorted by name. Directories that are not projects (a staging
    directory left by a crash, a stray file) are not listed."""
    root = config.home / "projects"
    if not root.is_dir():
        return []
    return [
        _read_project(entry, entry.name)
        for entry in sorted(root.iterdir())
        if _NAME.fullmatch(entry.name) and (entry / PROJECT_FILE).is_file()
    ]


def load_project(config: FactoryConfig, name: str | None = None) -> Project:
    """The project called ``name``, or the only registered project when ``name`` is None."""
    if name is not None:
        root = project_paths(config, name).root
        if not (root / PROJECT_FILE).is_file():
            raise ProjectError(f"unknown project {name!r}; {_known(config)}")
        return _read_project(root, name)
    projects = list_projects(config)
    if not projects:
        raise ProjectError("no project is registered; add one with `factory project add PATH`")
    if len(projects) > 1:
        names = ", ".join(project.name for project in projects)
        raise ProjectError(f"several projects are registered ({names}); pass --project NAME")
    return projects[0]


def project_config(paths: ProjectPaths) -> ProjectConfig:
    """The project's ``factory.toml``, read from ``refs/heads/main`` of its repository."""
    try:
        repo = BareRepo.open(paths.repo)
    except GitError as exc:
        raise ProjectError(
            f"{paths.repo} is not a usable repository: {exc.stderr.strip()}"
        ) from exc
    return _parse_config(_read_config(repo, "main"))


def _read_project(root: Path, name: str) -> Project:
    record = root / PROJECT_FILE
    try:
        project = Project.model_validate_json(record.read_bytes())
    except (OSError, ValidationError) as exc:
        raise ProjectError(f"{record} is not a readable project record: {exc}") from exc
    if project.name != name:
        raise ProjectError(f"{record} is the record of {project.name!r}, not {name!r}")
    return project


def _known(config: FactoryConfig) -> str:
    names = [project.name for project in list_projects(config)]
    return "registered: " + ", ".join(names) if names else "none is registered"


def _check_name(name: str) -> None:
    if _NAME.fullmatch(name) is None:
        raise ProjectError(
            f"invalid project name {name!r}: start with a lowercase letter or digit, then use "
            "lowercase letters, digits and '-'"
        )
