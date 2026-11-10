"""Isolation properties with real git: config, hooks, fsmonitor, transports, and the invariant that
host-side code never runs git inside a worker directory (PLAN.md invariant 7)."""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path
from typing import Any

import pytest

from local_factory.gitops import _git
from local_factory.gitops._git import FACTORY_EMAIL, FACTORY_NAME, GitError, run_git
from local_factory.gitops.bare import BareRepo
from local_factory.gitops.bundles import import_task_bundle
from local_factory.gitops.integrate import merge_into, squash_onto
from local_factory.gitops.worker_side import commit_all, export_task_bundle

from .conftest import DEST, Git, make_seed, write

pytestmark = pytest.mark.integration

GITOPS = Path(_git.__file__).parent
HOST_MODULES = ("bare.py", "bundles.py", "integrate.py")

# --------------------------------------------------------------------------- runner


def test_hostile_repo_config_does_not_execute(tmp_path: Path, git: Git) -> None:
    repo = make_seed(git, tmp_path / "w", {"a.txt": "a\n"})
    marker = tmp_path / "pwned"
    git("config", "core.fsmonitor", f"touch {marker}; true", cwd=repo)
    # Positive control: plain git really does run the planted fsmonitor.
    git("status", cwd=repo)
    assert marker.exists()
    marker.unlink()

    run_git(["status"], git_dir=repo / ".git", cwd=repo)
    assert not marker.exists()


def test_hooks_never_run(tmp_path: Path, git: Git) -> None:
    bare = tmp_path / "plain.git"
    git("init", "-q", "--bare", str(bare), cwd=tmp_path)
    marker = tmp_path / "hooked"
    hook = bare / "hooks" / "reference-transaction"
    hook.write_text(f"#!/bin/sh\ntouch {marker}\n")
    hook.chmod(0o755)
    tree = git("hash-object", "-t", "tree", "-w", "--stdin", cwd=bare, input="")
    sha = git("commit-tree", tree, "-m", "c", cwd=bare)
    git("update-ref", "refs/heads/control", sha, cwd=bare)
    assert marker.exists()  # positive control: plain git runs the hook
    marker.unlink()

    run_git(["update-ref", "refs/heads/other", sha], git_dir=bare)
    assert not marker.exists()


def test_inherited_environment_is_ignored(
    tmp_path: Path, git: Git, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = make_seed(git, tmp_path / "w", {"a.txt": "a\n"})
    marker = tmp_path / "pwned"
    evil_home = tmp_path / "evil-home"
    evil_home.mkdir()
    (evil_home / ".gitconfig").write_text("[user]\n\tname = evil\n[alias]\n\tst = status\n")
    monkeypatch.setenv("HOME", str(evil_home))
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(evil_home / ".gitconfig"))
    monkeypatch.setenv("GIT_CONFIG_PARAMETERS", f"'core.fsmonitor=touch {marker}; true'")
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "elsewhere"))
    monkeypatch.setenv("GIT_AUTHOR_NAME", "evil")

    run_git(["status"], git_dir=repo / ".git", cwd=repo)
    assert not marker.exists()
    ident = run_git(["var", "GIT_AUTHOR_IDENT"], git_dir=repo / ".git").stdout.decode()
    assert ident.startswith(f"{FACTORY_NAME} <{FACTORY_EMAIL}>")
    with pytest.raises(GitError):  # the alias from the "global" config does not exist
        run_git(["st"], git_dir=repo / ".git", cwd=repo)
    fsmonitor = run_git(["config", "core.fsmonitor"], git_dir=repo / ".git")
    assert fsmonitor.stdout.strip() == b"false"


@pytest.mark.parametrize(
    "url", ["https://example.invalid/x.git", "ssh://example.invalid/x", "git://example.invalid/x"]
)
def test_network_transports_are_refused(repo: BareRepo, url: str) -> None:
    with pytest.raises(GitError) as exc:
        repo.fetch_from(url, ["refs/heads/main:refs/heads/x"])
    assert "not allowed" in exc.value.stderr


def test_ext_transport_cannot_run_commands(repo: BareRepo, tmp_path: Path) -> None:
    marker = tmp_path / "pwned"
    with pytest.raises(GitError) as exc:
        repo.fetch_from(f"ext::sh -c touch% {marker}", ["refs/heads/main:refs/heads/x"])
    assert "not allowed" in exc.value.stderr
    assert not marker.exists()


def test_git_error_from_real_failure(repo: BareRepo) -> None:
    with pytest.raises(GitError) as exc:
        run_git(["rev-parse", "--verify", "refs/heads/nope"], git_dir=repo.path)
    assert exc.value.returncode == 128
    assert exc.value.args == ("rev-parse", "--verify", "refs/heads/nope")
    assert "fatal" in exc.value.stderr and "rev-parse" in str(exc.value)


def test_default_cwd_is_not_the_process_cwd(
    tmp_path: Path, git: Git, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = make_seed(git, tmp_path / "w", {"a.txt": "a\n"})
    monkeypatch.chdir(repo)  # e.g. the host process happens to sit in a worker directory
    with pytest.raises(GitError, match="not a git repository"):
        run_git(["status"])  # no discovery from the process cwd


# --------------------------------------------------------------------------- invariant 7


def _calls(module: Path) -> list[ast.Call]:
    return [node for node in ast.walk(ast.parse(module.read_text())) if isinstance(node, ast.Call)]


@pytest.mark.parametrize("module", HOST_MODULES)
def test_host_modules_never_pass_cwd(module: str) -> None:
    source = (GITOPS / module).read_text()
    tree = ast.parse(source)
    for call in _calls(GITOPS / module):
        assert "cwd" not in {kw.arg for kw in call.keywords}, f"{module}:{call.lineno}"
        func = call.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        assert name not in {"chdir", "Popen", "system", "check_output", "check_call"}
        if name == "run_git" and "git_dir" not in {kw.arg for kw in call.keywords}:
            # Only `git init <absolute path>` runs without --git-dir (in the neutral cwd).
            argv = call.args[0]
            assert isinstance(argv, ast.List), f"{module}:{call.lineno}"
            first = argv.elts[0]
            assert isinstance(first, ast.Constant) and first.value == "init", (
                f"{module}:{call.lineno}"
            )
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import | ast.ImportFrom)
        for alias in node.names
    }
    assert "subprocess" not in imported


def test_only_git_module_spawns_processes() -> None:
    for module in sorted(GITOPS.glob("*.py")):
        if module.name == "_git.py":
            continue
        assert "subprocess.run" not in module.read_text(), module.name


class Spy:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], Path]] = []
        self.real = subprocess.run

    def __call__(self, argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        self.calls.append((list(argv), Path(kwargs["cwd"])))
        return self.real(argv, **kwargs)


def test_host_side_never_touches_the_worker_directory(
    repo: BareRepo, base: str, work: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write(work, {"feature.txt": "f\n", "src/app.py": "changed\n"})
    head = commit_all(work, "feature")
    assert head is not None
    bundle = work / "out" / "task.bundle"  # the bundle even lives inside the worker directory
    bundle.parent.mkdir()
    export_task_bundle(work, bundle, base_sha=base)
    repo.update_ref("refs/heads/factory/T-0001", base)

    spy = Spy()
    monkeypatch.setattr(subprocess, "run", spy)
    result = import_task_bundle(repo, bundle, base_sha=base, dest_ref=DEST)
    repo.diff(base, result.head_sha)
    merge_into(repo, target_ref="refs/heads/factory/T-0001", source=result.head_sha, message="m")
    squash_onto(repo, base=base, head=result.head_sha, message="s")
    repo.export_tree(result.head_sha, tmp_path / "checkout")

    assert spy.calls
    worker = str(work)
    for argv, cwd in spy.calls:
        assert not cwd.is_relative_to(work), argv
        assert f"--git-dir={repo.path}" in argv, argv
        assert not any(worker in arg for arg in argv), argv
