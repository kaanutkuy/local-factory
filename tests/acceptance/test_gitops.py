"""Acceptance: git topology — bare repo, bundles, worker side, integration.

docs/specs/phase-1a.md § gitops and docs/PLAN.md §5.1. Protected.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from local_factory.gitops._git import FACTORY_EMAIL, FACTORY_NAME, GitError, run_git
from local_factory.gitops._git import git_env as factory_git_env
from local_factory.gitops.bare import BareRepo
from local_factory.gitops.bundles import (
    BundleLimits,
    BundleRejected,
    export_base_bundle,
    import_task_bundle,
)
from local_factory.gitops.integrate import merge_into, squash_onto
from local_factory.gitops.worker_side import (
    TASK_BRANCH,
    commit_all,
    export_task_bundle,
    init_from_bundle,
)
from local_factory.models import MODE_EXECUTABLE, MODE_SYMLINK, ChangeKind

from .conftest import Git

pytestmark = pytest.mark.integration

DEST = "refs/factory/tasks/T-0001/task-1"
TICKET = "refs/heads/factory/T-0001"


@pytest.fixture
def repo(tmp_path: Path, seed_repo: Path) -> BareRepo:
    bare = BareRepo.init(tmp_path / "repo.git")
    bare.fetch_from(seed_repo, ["refs/heads/main:refs/heads/main"])
    return bare


@pytest.fixture
def base(repo: BareRepo) -> str:
    return repo.rev_parse("refs/heads/main")


def make_work(repo: BareRepo, sha: str, tmp_path: Path, name: str = "work") -> Path:
    bundle = tmp_path / f"{name}-base.bundle"
    ref = export_base_bundle(repo, sha=sha, out=bundle)
    work = tmp_path / name
    init_from_bundle(bundle, ref, work)
    return work


@pytest.fixture
def work(repo: BareRepo, base: str, tmp_path: Path) -> Path:
    return make_work(repo, base, tmp_path)


def bundle_of(work: Path, base: str, tmp_path: Path, name: str = "task.bundle") -> Path:
    out = tmp_path / name
    export_task_bundle(work, out, base_sha=base)
    return out


def all_refs(repo: BareRepo, git: Git) -> list[str]:
    return git("for-each-ref", "--format=%(refname)", cwd=repo.path).splitlines()


def assert_rejected(
    repo: BareRepo,
    bundle: Path,
    base: str,
    git: Git,
    codes: set[str],
    limits: BundleLimits | None = None,
) -> None:
    kwargs = {"limits": limits} if limits is not None else {}
    with pytest.raises(BundleRejected) as exc:
        import_task_bundle(repo, bundle, base_sha=base, dest_ref=DEST, **kwargs)
    assert exc.value.code in codes, exc.value.detail
    assert not repo.ref_exists(DEST)
    assert all_refs(repo, git) == ["refs/heads/main"]


# --------------------------------------------------------------------------- git environment


def test_git_env_is_sanitized(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GIT_DIR", "/tmp/evil")
    monkeypatch.setenv("GIT_CONFIG_PARAMETERS", "'core.fsmonitor=touch /tmp/pwned'")
    env = factory_git_env()
    assert "GIT_DIR" not in env
    assert "GIT_CONFIG_PARAMETERS" not in env
    assert env["GIT_CONFIG_GLOBAL"] == os.devnull
    assert env["GIT_CONFIG_NOSYSTEM"] == "1"
    assert env["GIT_TERMINAL_PROMPT"] == "0"
    assert env["GIT_AUTHOR_NAME"] == env["GIT_COMMITTER_NAME"] == FACTORY_NAME
    assert env["GIT_AUTHOR_EMAIL"] == env["GIT_COMMITTER_EMAIL"] == FACTORY_EMAIL


def test_run_git_errors(repo: BareRepo) -> None:
    with pytest.raises(GitError):
        run_git(["rev-parse", "--verify", "refs/heads/nope"], git_dir=repo.path)
    result = run_git(["rev-parse", "--verify", "refs/heads/nope"], git_dir=repo.path, check=False)
    assert result.returncode != 0


# --------------------------------------------------------------------------- bare repo


def test_init_is_hardened(repo: BareRepo, git: Git) -> None:
    def cfg(key: str) -> str:
        return git("config", "--file", str(repo.path / "config"), key, cwd=repo.path)

    assert cfg("core.bare") == "true"
    assert cfg("core.hooksPath") == "/dev/null"
    assert cfg("core.fsmonitor") == "false"
    for key in ("fetch.fsckObjects", "receive.fsckObjects", "transfer.fsckObjects"):
        assert cfg(key) == "true"
    assert BareRepo.open(repo.path).path == repo.path


def test_open_rejects_non_bare(seed_repo: Path) -> None:
    with pytest.raises(GitError):
        BareRepo.open(seed_repo)


def test_ref_and_object_queries(repo: BareRepo, base: str, work: Path, git: Git) -> None:
    (work / "a.txt").write_text("1\n")
    first = commit_all(work, "one")
    (work / "a.txt").write_text("2\n")
    second = commit_all(work, "two")
    assert first and second
    import_task_bundle(repo, bundle_of(work, base, work.parent), base_sha=base, dest_ref=DEST)

    assert repo.rev_parse(DEST) == second
    with pytest.raises(GitError):
        repo.rev_parse("refs/heads/does-not-exist")
    assert repo.ref_exists(DEST) and not repo.ref_exists("refs/heads/nope")
    assert repo.is_ancestor(base, second) and not repo.is_ancestor(second, base)
    assert repo.list_commits(base, second) == [first, second]
    assert repo.tree_hash(second) == git("rev-parse", f"{second}^{{tree}}", cwd=repo.path)
    assert repo.read_file(second, "a.txt") == b"2\n"

    with pytest.raises(GitError):
        repo.update_ref(DEST, base, old_sha=first)  # CAS mismatch
    repo.update_ref(DEST, base, old_sha=second)
    assert repo.rev_parse(DEST) == base
    repo.delete_ref(DEST)
    assert not repo.ref_exists(DEST)


def test_export_tree_materializes_without_git_metadata(
    repo: BareRepo, base: str, tmp_path: Path
) -> None:
    dest = tmp_path / "checkout"
    dest.mkdir()
    repo.export_tree(base, dest)
    assert (dest / "README.md").read_text() == "# seed\n"
    assert (dest / "src" / "app.py").exists()
    assert not (dest / ".git").exists()


# ------------------------------------------------------------------ worker side + round trip


def test_worker_clone_is_standalone(work: Path, git: Git) -> None:
    assert TASK_BRANCH == "task"
    assert (work / ".git").is_dir()  # its own metadata, not a worktree pointer
    assert git("rev-parse", "--abbrev-ref", "HEAD", cwd=work) == "task"
    assert (work / "src" / "app.py").exists()


def test_commit_all_returns_none_without_changes(work: Path) -> None:
    assert commit_all(work, "noop") is None


def test_bundle_round_trip(repo: BareRepo, base: str, work: Path, tmp_path: Path, git: Git) -> None:
    (work / "src" / "app.py").write_text(
        "def add(a, b):\n    return a + b\n\n\ndef sub(a, b):\n    return a - b\n"
    )
    sha = commit_all(work, "add sub")
    assert sha is not None and len(sha) == 40

    result = import_task_bundle(repo, bundle_of(work, base, tmp_path), base_sha=base, dest_ref=DEST)

    assert result.head_sha == sha
    assert result.commits == (sha,)
    assert repo.rev_parse(DEST) == sha
    [change] = result.changes
    assert change.path == "src/app.py"
    assert change.kind is ChangeKind.MODIFIED
    assert [(ln.line_no, ln.text) for ln in change.added_lines] == [
        (3, ""),
        (4, ""),
        (5, "def sub(a, b):"),
        (6, "    return a - b"),
    ]
    # export's temporary ref and the quarantine ref are both gone
    assert sorted(all_refs(repo, git)) == [DEST, "refs/heads/main"]


def test_import_reads_only_the_bundle(
    repo: BareRepo, base: str, work: Path, tmp_path: Path, git: Git
) -> None:
    (work / "x.txt").write_text("x\n")
    commit_all(work, "x")
    bundle = bundle_of(work, base, tmp_path)
    # The worker turns hostile after producing its bundle...
    marker = tmp_path / "pwned"
    git("config", "core.fsmonitor", f"touch {marker}; true", cwd=work)
    hooks = work / ".git" / "hooks"
    for hook in ("post-checkout", "post-merge", "reference-transaction", "pre-auto-gc"):
        (hooks / hook).write_text(f"#!/bin/sh\ntouch {marker}\n")
        (hooks / hook).chmod(0o755)
    # ...and its directory is not even reachable from the host any more.
    work.rename(tmp_path / "gone")

    result = import_task_bundle(repo, bundle, base_sha=base, dest_ref=DEST)
    assert repo.rev_parse(DEST) == result.head_sha
    assert not marker.exists()


def test_diff_reports_kinds_modes_and_lines(
    repo: BareRepo, base: str, work: Path, tmp_path: Path, git: Git
) -> None:
    (work / "tests" / "test_app.py").unlink()
    (work / "docs").mkdir()
    git("mv", "README.md", "docs/README.md", cwd=work)
    (work / "assets").mkdir()
    (work / "assets" / "logo.bin").write_bytes(b"\x00\x01\x02")
    (work / "run.sh").write_text("#!/bin/sh\necho hi\n")
    (work / "run.sh").chmod(0o755)
    head = commit_all(work, "mixed change")
    assert head

    result = import_task_bundle(repo, bundle_of(work, base, tmp_path), base_sha=base, dest_ref=DEST)
    assert result.changes == tuple(repo.diff(base, head))
    by_path = {c.path: c for c in result.changes}
    assert list(by_path) == sorted(by_path)

    deleted = by_path["tests/test_app.py"]
    assert (deleted.kind, deleted.mode, deleted.old_mode) == (ChangeKind.DELETED, None, "100644")
    renamed = by_path["docs/README.md"]
    assert (renamed.kind, renamed.old_path) == (ChangeKind.RENAMED, "README.md")
    binary = by_path["assets/logo.bin"]
    assert (binary.kind, binary.is_binary, binary.size, binary.added_lines) == (
        ChangeKind.ADDED,
        True,
        3,
        (),
    )
    script = by_path["run.sh"]
    assert script.mode == MODE_EXECUTABLE
    assert [(ln.line_no, ln.text) for ln in script.added_lines] == [
        (1, "#!/bin/sh"),
        (2, "echo hi"),
    ]


# --------------------------------------------------------------------------- rejections


def test_rejects_unreadable_bundle(repo: BareRepo, base: str, tmp_path: Path, git: Git) -> None:
    bundle = tmp_path / "garbage.bundle"
    bundle.write_bytes(b"not a bundle\n")
    assert_rejected(repo, bundle, base, git, {"invalid_bundle"})


def test_rejects_extra_refs(
    repo: BareRepo, base: str, work: Path, tmp_path: Path, git: Git
) -> None:
    (work / "x.txt").write_text("x\n")
    commit_all(work, "x")
    git("branch", "evil", cwd=work)
    bundle = tmp_path / "extra.bundle"
    git("bundle", "create", str(bundle), "refs/heads/task", "refs/heads/evil", f"^{base}", cwd=work)
    assert_rejected(repo, bundle, base, git, {"unexpected_refs"})


def test_rejects_wrong_ref_name(
    repo: BareRepo, base: str, work: Path, tmp_path: Path, git: Git
) -> None:
    (work / "x.txt").write_text("x\n")
    commit_all(work, "x")
    git("branch", "-m", "task", "other", cwd=work)
    bundle = tmp_path / "other.bundle"
    git("bundle", "create", str(bundle), "refs/heads/other", f"^{base}", cwd=work)
    assert_rejected(repo, bundle, base, git, {"unexpected_refs"})


def test_rejects_missing_prerequisite(
    repo: BareRepo, base: str, work: Path, tmp_path: Path, git: Git
) -> None:
    (work / "x.txt").write_text("1\n")
    unknown = commit_all(work, "not in the factory repo")
    (work / "x.txt").write_text("2\n")
    commit_all(work, "on top")
    bundle = tmp_path / "prereq.bundle"
    git("bundle", "create", str(bundle), "refs/heads/task", f"^{unknown}", cwd=work)
    assert_rejected(repo, bundle, base, git, {"invalid_bundle"})


def test_rejects_unrelated_history(repo: BareRepo, base: str, tmp_path: Path, git: Git) -> None:
    other = tmp_path / "other"
    other.mkdir()
    git("init", "-q", "-b", "task", cwd=other)
    (other / "a.txt").write_text("unrelated\n")
    git("add", "-A", cwd=other)
    git("commit", "-q", "-m", "unrelated", cwd=other)
    bundle = tmp_path / "unrelated.bundle"
    git("bundle", "create", str(bundle), "refs/heads/task", cwd=other)
    assert_rejected(repo, bundle, base, git, {"base_not_ancestor"})


def test_rejects_no_changes(
    repo: BareRepo, base: str, work: Path, tmp_path: Path, git: Git
) -> None:
    bundle = tmp_path / "same.bundle"
    git("bundle", "create", str(bundle), "refs/heads/task", cwd=work)
    assert_rejected(repo, bundle, base, git, {"no_changes"})


def test_rejects_too_many_commits(
    repo: BareRepo, base: str, work: Path, tmp_path: Path, git: Git
) -> None:
    for i in range(2):
        (work / "x.txt").write_text(f"{i}\n")
        commit_all(work, f"c{i}")
    limits = BundleLimits(max_commits=1)
    assert_rejected(repo, bundle_of(work, base, tmp_path), base, git, {"too_many_commits"}, limits)


def test_rejects_gitlink(repo: BareRepo, base: str, work: Path, tmp_path: Path, git: Git) -> None:
    git("update-index", "--add", "--cacheinfo", f"160000,{base},vendor/sub", cwd=work)
    git("commit", "-q", "-m", "gitlink", cwd=work)
    assert_rejected(repo, bundle_of(work, base, tmp_path), base, git, {"gitlink"})


@pytest.mark.parametrize(
    ("link", "target"),
    [("evil", "/etc/passwd"), ("a/link", "../../outside"), ("link", "../x")],
)
def test_rejects_escaping_symlinks(
    repo: BareRepo, base: str, work: Path, tmp_path: Path, git: Git, link: str, target: str
) -> None:
    path = work / link
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to(target)
    commit_all(work, "symlink")
    assert_rejected(repo, bundle_of(work, base, tmp_path), base, git, {"symlink_escape"})


def test_accepts_symlink_inside_repo(repo: BareRepo, base: str, work: Path, tmp_path: Path) -> None:
    (work / "docs").mkdir()
    (work / "docs" / "readme").symlink_to("../README.md")
    commit_all(work, "internal symlink")
    result = import_task_bundle(repo, bundle_of(work, base, tmp_path), base_sha=base, dest_ref=DEST)
    [change] = result.changes
    assert (change.mode, change.symlink_target, change.added_lines) == (
        MODE_SYMLINK,
        "../README.md",
        (),
    )


def test_rejects_dot_git_path_component(
    repo: BareRepo, base: str, work: Path, tmp_path: Path, git: Git
) -> None:
    blob = git("hash-object", "-w", "--stdin", cwd=work, input="pwn\n")
    inner = git("mktree", cwd=work, input=f"100644 blob {blob}\tconfig\n")
    outer = git("mktree", cwd=work, input=f"040000 tree {inner}\t.GIT\n")
    commit = git("commit-tree", outer, "-p", base, "-m", "dotgit", cwd=work)
    git("update-ref", "refs/heads/task", commit, cwd=work)
    bundle = tmp_path / "dotgit.bundle"
    git("bundle", "create", str(bundle), "refs/heads/task", f"^{base}", cwd=work)
    assert_rejected(repo, bundle, base, git, {"fetch_failed", "unsafe_path"})


def test_rejects_oversized_content(
    repo: BareRepo, base: str, work: Path, tmp_path: Path, git: Git
) -> None:
    (work / "big.txt").write_text("x" * 100)
    commit_all(work, "big")
    limits = BundleLimits(max_file_bytes=10)
    assert_rejected(repo, bundle_of(work, base, tmp_path), base, git, {"file_too_large"}, limits)


def test_rejects_oversized_total(
    repo: BareRepo, base: str, work: Path, tmp_path: Path, git: Git
) -> None:
    (work / "a.txt").write_text("12345678")
    (work / "b.txt").write_text("abcdefgh")
    commit_all(work, "two files")
    limits = BundleLimits(max_file_bytes=100, max_total_bytes=10)
    assert_rejected(repo, bundle_of(work, base, tmp_path), base, git, {"too_large"}, limits)


# --------------------------------------------------------------------------- integration


def _task_head(repo: BareRepo, base: str, tmp_path: Path, name: str, files: dict[str, str]) -> str:
    work = make_work(repo, base, tmp_path, name)
    for rel, content in files.items():
        (work / rel).parent.mkdir(parents=True, exist_ok=True)
        (work / rel).write_text(content)
    commit_all(work, name)
    out = tmp_path / f"{name}.bundle"
    export_task_bundle(work, out, base_sha=base)
    return import_task_bundle(
        repo, out, base_sha=base, dest_ref=f"refs/factory/tasks/T-0001/{name}"
    ).head_sha


def test_merge_fast_forwards_then_merges(
    repo: BareRepo, base: str, tmp_path: Path, git: Git
) -> None:
    repo.update_ref(TICKET, base)
    h1 = _task_head(repo, base, tmp_path, "task-1", {"one.txt": "1\n"})
    h2 = _task_head(repo, base, tmp_path, "task-2", {"two.txt": "2\n"})

    first = merge_into(repo, target_ref=TICKET, source=h1, message="merge task-1")
    assert (first.sha, first.conflicts) == (h1, ())  # fast-forward
    assert repo.rev_parse(TICKET) == h1

    second = merge_into(repo, target_ref=TICKET, source=h2, message="merge task-2")
    assert second.sha is not None and second.conflicts == ()
    assert repo.rev_parse(TICKET) == second.sha
    parents = git("rev-list", "--parents", "-n", "1", second.sha, cwd=repo.path).split()[1:]
    assert parents == [h1, h2]
    assert repo.read_file(second.sha, "one.txt") == b"1\n"
    assert repo.read_file(second.sha, "two.txt") == b"2\n"


def test_merge_conflict_leaves_target_untouched(repo: BareRepo, base: str, tmp_path: Path) -> None:
    repo.update_ref(TICKET, base)
    h1 = _task_head(repo, base, tmp_path, "task-1", {"src/app.py": "one\n"})
    h2 = _task_head(repo, base, tmp_path, "task-2", {"src/app.py": "two\n"})
    merge_into(repo, target_ref=TICKET, source=h1, message="merge task-1")
    outcome = merge_into(repo, target_ref=TICKET, source=h2, message="merge task-2")
    assert outcome.sha is None
    assert outcome.conflicts == ("src/app.py",)
    assert repo.rev_parse(TICKET) == h1


def test_squash_onto_builds_single_parent_candidate(
    repo: BareRepo, base: str, seed_repo: Path, tmp_path: Path, git: Git
) -> None:
    head = _task_head(repo, base, tmp_path, "task-1", {"feature.txt": "f\n"})
    (seed_repo / "upstream.txt").write_text("u\n")
    git("add", "-A", cwd=seed_repo)
    git("commit", "-q", "-m", "upstream moved", cwd=seed_repo)
    repo.fetch_from(seed_repo, ["refs/heads/main:refs/heads/main"])
    new_main = repo.rev_parse("refs/heads/main")
    refs_before = all_refs(repo, git)

    outcome = squash_onto(repo, base=new_main, head=head, message="T-0001: feature")
    assert outcome.sha is not None and outcome.conflicts == ()
    parents = git("rev-list", "--parents", "-n", "1", outcome.sha, cwd=repo.path).split()[1:]
    assert parents == [new_main]
    assert repo.read_file(outcome.sha, "feature.txt") == b"f\n"
    assert repo.read_file(outcome.sha, "upstream.txt") == b"u\n"
    assert all_refs(repo, git) == refs_before
