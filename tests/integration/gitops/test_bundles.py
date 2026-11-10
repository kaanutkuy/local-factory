"""export_base_bundle / import_task_bundle edge cases with real git."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from local_factory.gitops._git import GitError
from local_factory.gitops.bare import BareRepo
from local_factory.gitops.bundles import (
    BASE_REF_PREFIX,
    BundleLimits,
    BundleRejected,
    export_base_bundle,
    import_task_bundle,
)
from local_factory.gitops.worker_side import commit_all
from local_factory.models import MODE_SYMLINK, ChangeKind

from .conftest import (
    DEST,
    SEED_FILES,
    Git,
    all_refs,
    bare_from,
    bundle_of,
    make_seed,
    make_work,
    set_task,
    write,
)

pytestmark = pytest.mark.integration


def rejected(
    repo: BareRepo,
    bundle: Path,
    base: str,
    git: Git,
    code: str,
    limits: BundleLimits | None = None,
) -> str:
    """Assert the import is rejected with ``code`` and leaves the refs exactly as they were."""
    before = all_refs(repo, git)
    kwargs = {"limits": limits} if limits is not None else {}
    with pytest.raises(BundleRejected) as exc:
        import_task_bundle(repo, bundle, base_sha=base, dest_ref=DEST, **kwargs)
    assert exc.value.code == code, exc.value.detail
    assert all_refs(repo, git) == before
    assert not repo.ref_exists(DEST)
    return exc.value.detail


def accepted(repo: BareRepo, bundle: Path, base: str, limits: BundleLimits | None = None) -> str:
    kwargs = {"limits": limits} if limits is not None else {}
    result = import_task_bundle(repo, bundle, base_sha=base, dest_ref=DEST, **kwargs)
    assert repo.rev_parse(DEST) == result.head_sha
    return result.head_sha


def git_bundle(git: Git, work: Path, out: Path, *revs: str) -> Path:
    git("bundle", "create", str(out), *revs, cwd=work)
    return out


# --------------------------------------------------------------------------- export


def test_export_base_bundle(repo: BareRepo, base: str, tmp_path: Path, git: Git) -> None:
    out = tmp_path / "base.bundle"
    ref = export_base_bundle(repo, sha=base[:10], out=out)
    assert ref.startswith(BASE_REF_PREFIX)
    assert git("bundle", "list-heads", str(out), cwd=tmp_path) == f"{base} {ref}"
    assert all_refs(repo, git) == ["refs/heads/main"]  # temporary ref removed
    assert export_base_bundle(repo, sha=base, out=tmp_path / "again.bundle") != ref


def test_export_base_bundle_relative_out(
    repo: BareRepo, base: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    export_base_bundle(repo, sha=base, out=Path("rel.bundle"))
    assert (tmp_path / "rel.bundle").is_file()


def test_export_base_bundle_cleans_up_on_failure(
    repo: BareRepo, base: str, tmp_path: Path, git: Git
) -> None:
    with pytest.raises(GitError):
        export_base_bundle(repo, sha=base, out=tmp_path / "missing-dir" / "base.bundle")
    assert all_refs(repo, git) == ["refs/heads/main"]
    with pytest.raises(GitError):
        export_base_bundle(repo, sha="0123456789" * 4, out=tmp_path / "x.bundle")
    with pytest.raises(GitError):  # a tree is not a commit
        export_base_bundle(repo, sha=repo.tree_hash(base), out=tmp_path / "y.bundle")
    assert all_refs(repo, git) == ["refs/heads/main"]


# --------------------------------------------------------------------------- the bundle file


def test_bundle_file_must_be_a_regular_file(
    repo: BareRepo, base: str, work: Path, tmp_path: Path, git: Git
) -> None:
    write(work, {"x.txt": "x\n"})
    commit_all(work, "x")
    real = bundle_of(work, base, tmp_path)
    link = tmp_path / "link.bundle"
    link.symlink_to(real)
    assert "cannot open" in rejected(repo, link, base, git, "invalid_bundle")
    fifo = tmp_path / "fifo.bundle"
    os.mkfifo(fifo)
    assert "not a regular file" in rejected(repo, fifo, base, git, "invalid_bundle")
    rejected(repo, tmp_path / "missing.bundle", base, git, "invalid_bundle")
    rejected(repo, tmp_path, base, git, "invalid_bundle")  # a directory


def test_bundle_without_refs(repo: BareRepo, base: str, tmp_path: Path, git: Git) -> None:
    bundle = tmp_path / "empty.bundle"
    bundle.write_bytes(b"# v2 git bundle\n\n")
    assert "none" in rejected(repo, bundle, base, git, "unexpected_refs")


def test_bundle_with_head_ref(
    repo: BareRepo, base: str, work: Path, tmp_path: Path, git: Git
) -> None:
    write(work, {"x.txt": "x\n"})
    commit_all(work, "x")
    bundle = git_bundle(git, work, tmp_path / "head.bundle", "HEAD", "refs/heads/task", f"^{base}")
    assert "HEAD" in rejected(repo, bundle, base, git, "unexpected_refs")


def test_truncated_bundle(repo: BareRepo, base: str, work: Path, tmp_path: Path, git: Git) -> None:
    write(work, {"x.txt": "x" * 5000})
    commit_all(work, "x")
    data = bundle_of(work, base, tmp_path).read_bytes()
    truncated = tmp_path / "truncated.bundle"
    truncated.write_bytes(data[: len(data) // 2])
    rejected(repo, truncated, base, git, "fetch_failed")


def test_type_confused_tree_entry(
    repo: BareRepo, base: str, work: Path, tmp_path: Path, git: Git
) -> None:
    inner = git.tree([("100644", b"config", git.blob(b"pwn\n", cwd=work))], cwd=work)
    tree = git.tree([*git.tree_entries(base, cwd=work), ("100644", b"weird", inner)], cwd=work)
    set_task(git, work, tree, base)
    bundle = git_bundle(git, work, tmp_path / "confused.bundle", "refs/heads/task", f"^{base}")
    assert "malformed tree entry weird" in rejected(repo, bundle, base, git, "fetch_failed")


# --------------------------------------------------------------------------- history checks


def test_commit_limit_boundary(
    repo: BareRepo, base: str, work: Path, tmp_path: Path, git: Git
) -> None:
    for i in range(3):
        write(work, {"x.txt": f"{i}\n"})
        commit_all(work, f"c{i}")
    bundle = bundle_of(work, base, tmp_path)
    detail = rejected(repo, bundle, base, git, "too_many_commits", BundleLimits(max_commits=2))
    assert "3 commits" in detail
    head = accepted(repo, bundle, base, BundleLimits(max_commits=3))
    assert len(repo.list_commits(base, head)) == 3


def test_result_lists_commits_oldest_first(
    repo: BareRepo, base: str, work: Path, tmp_path: Path
) -> None:
    shas = []
    for i in range(3):
        write(work, {f"f{i}.txt": f"{i}\n"})
        shas.append(commit_all(work, f"c{i}"))
    result = import_task_bundle(repo, bundle_of(work, base, tmp_path), base_sha=base, dest_ref=DEST)
    assert list(result.commits) == shas
    assert result.head_sha == shas[-1]
    assert list(result.changes) == repo.diff(base, result.head_sha)


def test_reimport_moves_the_task_ref(repo: BareRepo, base: str, work: Path, tmp_path: Path) -> None:
    write(work, {"x.txt": "1\n"})
    commit_all(work, "round 1")
    first = accepted(repo, bundle_of(work, base, tmp_path, "r1.bundle"), base)
    write(work, {"x.txt": "2\n"})
    commit_all(work, "round 2")
    second = accepted(repo, bundle_of(work, base, tmp_path, "r2.bundle"), base)
    assert second != first and repo.is_ancestor(first, second)


def test_rejection_keeps_previous_round(
    repo: BareRepo, base: str, work: Path, tmp_path: Path, git: Git
) -> None:
    write(work, {"x.txt": "1\n"})
    commit_all(work, "round 1")
    first = accepted(repo, bundle_of(work, base, tmp_path, "r1.bundle"), base)
    (work / "evil").symlink_to("/etc/passwd")
    commit_all(work, "round 2")
    before = all_refs(repo, git)
    with pytest.raises(BundleRejected):
        import_task_bundle(
            repo, bundle_of(work, base, tmp_path, "r2.bundle"), base_sha=base, dest_ref=DEST
        )
    assert repo.rev_parse(DEST) == first and all_refs(repo, git) == before


def test_gitlink_removal_is_fine(tmp_path: Path, git: Git) -> None:
    seed = make_seed(git, tmp_path / "s", {"a.txt": "a\n"})
    git(
        "update-index",
        "--add",
        "--cacheinfo",
        f"160000,{git('rev-parse', 'HEAD', cwd=seed)},sub",
        cwd=seed,
    )
    git("commit", "-q", "-m", "with gitlink", cwd=seed)
    repo = bare_from(seed, tmp_path / "r.git")
    base = repo.rev_parse("main")
    work = make_work(repo, base, tmp_path)
    git("rm", "-q", "--cached", "sub", cwd=work)
    git("commit", "-q", "-m", "drop gitlink", cwd=work)
    head = accepted(repo, bundle_of(work, base, tmp_path), base)
    [change] = repo.diff(base, head)
    assert (change.path, change.kind) == ("sub", ChangeKind.DELETED)


# --------------------------------------------------------------------------- symlinks


def commit_links(work: Path, links: dict[str, str]) -> None:
    for rel, target in links.items():
        path = work / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to(target)
    assert commit_all(work, "links") is not None


@pytest.mark.parametrize(
    ("links", "culprit"),
    [
        ({"a": ".", "b": "a/.."}, "b"),  # lexically harmless, physically the parent of the root
        ({"lib": "src", "use": "lib/app.py"}, "use"),  # traverses a safe link
        ({"x": "y", "y": "README.md"}, "x"),  # final component is a link
        ({"loop": "loop"}, "loop"),
        ({"cfg": ".git/config"}, "cfg"),
        ({"sub/cfg": "../.GIT/hooks"}, "sub/cfg"),
    ],
)
def test_symlink_chains_and_metadata_are_rejected(
    repo: BareRepo,
    base: str,
    work: Path,
    tmp_path: Path,
    git: Git,
    links: dict[str, str],
    culprit: str,
) -> None:
    commit_links(work, links)
    detail = rejected(repo, bundle_of(work, base, tmp_path), base, git, "symlink_escape")
    assert detail.startswith(f"{culprit} -> ")


def test_internal_symlinks_are_accepted(
    repo: BareRepo, base: str, work: Path, tmp_path: Path
) -> None:
    commit_links(work, {"docs/readme": "../README.md", "app": "src/app.py", "here": "."})
    head = accepted(repo, bundle_of(work, base, tmp_path), base)
    targets = {c.path: c.symlink_target for c in repo.diff(base, head) if c.mode == MODE_SYMLINK}
    assert targets == {"docs/readme": "../README.md", "app": "src/app.py", "here": "."}


@pytest.mark.parametrize(
    ("target", "reason"), [(b"x" * 4096, "could not be read"), (b"ok\0/../../x", "NUL")]
)
def test_unrepresentable_symlink_targets(
    repo: BareRepo, base: str, work: Path, tmp_path: Path, git: Git, target: bytes, reason: str
) -> None:
    blob = git.blob(target, cwd=work)
    git("update-index", "--add", "--cacheinfo", f"120000,{blob},link", cwd=work)
    git("commit", "-q", "-m", "odd link", cwd=work)
    assert reason in rejected(repo, bundle_of(work, base, tmp_path), base, git, "symlink_escape")


@pytest.fixture
def linked(tmp_path: Path, git: Git) -> tuple[BareRepo, str]:
    """A target repo that already contains safe symlinks."""
    seed = make_seed(git, tmp_path / "linked", {**SEED_FILES, "dir/file.txt": "f\n"})
    (seed / "lib").symlink_to("src")
    (seed / "into-dir").symlink_to("dir/file.txt")
    (seed / "docs").mkdir()
    (seed / "docs" / "readme").symlink_to("../README.md")
    git("add", "-A", cwd=seed)
    git("commit", "-q", "-m", "links", cwd=seed)
    repo = bare_from(seed, tmp_path / "linked.git")
    return repo, repo.rev_parse("main")


def test_existing_safe_symlinks_do_not_block_tasks(
    linked: tuple[BareRepo, str], tmp_path: Path
) -> None:
    repo, base = linked
    work = make_work(repo, base, tmp_path)
    write(work, {"feature.txt": "f\n"})
    commit_all(work, "unrelated change")
    accepted(repo, bundle_of(work, base, tmp_path), base)


def test_new_link_through_an_existing_link(
    linked: tuple[BareRepo, str], tmp_path: Path, git: Git
) -> None:
    repo, base = linked
    work = make_work(repo, base, tmp_path)
    (work / "via-lib").symlink_to("lib/app.py")
    commit_all(work, "new chain")
    detail = rejected(repo, bundle_of(work, base, tmp_path), base, git, "symlink_escape")
    assert detail == "via-lib -> lib/app.py: passes through the symlink lib"


def test_unchanged_link_made_unsafe_by_other_changes(
    linked: tuple[BareRepo, str], tmp_path: Path, git: Git
) -> None:
    repo, base = linked
    work = make_work(repo, base, tmp_path)
    # "into-dir -> dir/file.txt" was safe; turning "dir" into a symlink makes it traverse one.
    (work / "dir" / "file.txt").unlink()
    (work / "dir").rmdir()
    (work / "elsewhere").mkdir()
    (work / "elsewhere" / "file.txt").write_text("f\n")
    (work / "dir").symlink_to("elsewhere")
    commit_all(work, "swap dir for a link")
    detail = rejected(repo, bundle_of(work, base, tmp_path), base, git, "symlink_escape")
    assert detail == "into-dir -> dir/file.txt: passes through the symlink dir"


def test_pre_existing_unsafe_symlink_blocks_every_task(tmp_path: Path, git: Git) -> None:
    # No exemption: whether an existing link escapes depends on the rest of the tree, which the
    # worker controls (e.g. retargeting "lib -> ." under "current -> lib/../../x").
    seed = make_seed(git, tmp_path / "unsafe", {"a.txt": "a\n"})
    (seed / "system").symlink_to("/usr/share")
    git("add", "-A", cwd=seed)
    git("commit", "-q", "-m", "absolute link", cwd=seed)
    repo = bare_from(seed, tmp_path / "unsafe.git")
    base = repo.rev_parse("main")
    work = make_work(repo, base, tmp_path)
    write(work, {"feature.txt": "f\n"})
    commit_all(work, "unrelated change")
    detail = rejected(repo, bundle_of(work, base, tmp_path), base, git, "symlink_escape")
    assert detail == "system -> /usr/share: absolute target"


def test_retargeting_a_safe_link_under_a_chain_is_rejected(tmp_path: Path, git: Git) -> None:
    seed = make_seed(git, tmp_path / "chain", {"a/b/c/x": "x\n"})
    (seed / "lib").symlink_to("a/b/c")
    git("add", "-A", cwd=seed)
    git("commit", "-q", "-m", "lib", cwd=seed)
    repo = bare_from(seed, tmp_path / "chain.git")
    base = repo.rev_parse("main")
    work = make_work(repo, base, tmp_path)
    (work / "lib").unlink()
    (work / "lib").symlink_to(".")  # fine on its own...
    (work / "current").symlink_to("lib/../../x")  # ...but now this would leave the repo
    commit_all(work, "retarget")
    detail = rejected(repo, bundle_of(work, base, tmp_path), base, git, "symlink_escape")
    assert detail.startswith("current -> lib/../../x")


# --------------------------------------------------------------------------- paths


@pytest.mark.parametrize(("name", "reason"), [(b"caf\xe9.txt", "UTF-8"), (b"..\\evil", "'..'")])
def test_paths_fsck_lets_through_are_still_rejected(
    repo: BareRepo, base: str, work: Path, tmp_path: Path, git: Git, name: bytes, reason: str
) -> None:
    blob = git.blob(b"x\n", cwd=work)
    tree = git.tree([*git.tree_entries(base, cwd=work), ("100644", name, blob)], cwd=work)
    set_task(git, work, tree, base)
    detail = rejected(repo, bundle_of(work, base, tmp_path), base, git, "unsafe_path")
    assert reason in detail


# --------------------------------------------------------------------------- sizes


def test_file_size_boundary(
    repo: BareRepo, base: str, work: Path, tmp_path: Path, git: Git
) -> None:
    write(work, {"ten.txt": "x" * 10})
    commit_all(work, "ten bytes")
    bundle = bundle_of(work, base, tmp_path)
    limits = BundleLimits(max_file_bytes=9)
    assert "ten.txt: 10 bytes" in rejected(repo, bundle, base, git, "file_too_large", limits)
    accepted(repo, bundle, base, BundleLimits(max_file_bytes=10))


def test_total_size_boundary(
    repo: BareRepo, base: str, work: Path, tmp_path: Path, git: Git
) -> None:
    write(work, {"a.txt": "12345", "b.txt": "abcde", "c.txt": "12345"})  # c duplicates a
    commit_all(work, "files")
    bundle = bundle_of(work, base, tmp_path)
    limits = BundleLimits(max_file_bytes=100, max_total_bytes=9)
    assert "10 new bytes" in rejected(repo, bundle, base, git, "too_large", limits)
    accepted(repo, bundle, base, BundleLimits(max_file_bytes=100, max_total_bytes=10))


def test_intermediate_blobs_count(
    repo: BareRepo, base: str, work: Path, tmp_path: Path, git: Git
) -> None:
    write(work, {"huge.bin": b"\0" * 1000})
    commit_all(work, "add a big file")
    (work / "huge.bin").unlink()
    write(work, {"small.txt": "s\n"})
    commit_all(work, "and delete it again")
    limits = BundleLimits(max_file_bytes=100)
    detail = rejected(repo, bundle_of(work, base, tmp_path), base, git, "file_too_large", limits)
    assert "intermediate commit" in detail


def test_renaming_a_large_base_file_brings_no_new_bytes(tmp_path: Path, git: Git) -> None:
    seed = make_seed(git, tmp_path / "s", {"big.dat": "x" * 1000})
    repo = bare_from(seed, tmp_path / "r.git")
    base = repo.rev_parse("main")
    work = make_work(repo, base, tmp_path)
    (work / "big.dat").rename(work / "moved.dat")
    commit_all(work, "rename")
    accepted(repo, bundle_of(work, base, tmp_path), base, BundleLimits(max_file_bytes=10))


# --------------------------------------------------------------------------- precedence


@pytest.mark.parametrize(
    ("commits", "gitlink", "link", "unsafe", "code"),
    [
        (4, True, True, True, "too_many_commits"),
        (1, True, True, True, "gitlink"),
        (1, False, True, True, "symlink_escape"),
        (1, False, False, True, "unsafe_path"),
        (1, False, False, False, "file_too_large"),
    ],
)
def test_rejection_order(
    repo: BareRepo,
    base: str,
    tmp_path: Path,
    git: Git,
    commits: int,
    gitlink: bool,
    link: bool,
    unsafe: bool,
    code: str,
) -> None:
    """When several checks fail, the earliest in the spec's order wins (every case is also
    oversized, and each row removes the previous row's winning problem)."""
    work = make_work(repo, base, tmp_path)
    for i in range(commits):
        write(work, {"big.txt": f"{i}" * 50})
        commit_all(work, f"c{i}")
    if link:
        (work / "evil").symlink_to("/etc/passwd")
        commit_all(work, "link")
    entries = git.tree_entries("HEAD", cwd=work)
    if gitlink:
        entries.append(("160000", b"sub", base))
    if unsafe:
        entries.append(("100644", b"..\\x", git.blob(b"x\n", cwd=work)))
    if gitlink or unsafe:
        set_task(git, work, git.tree(entries, cwd=work), git("rev-parse", "HEAD", cwd=work))
    limits = BundleLimits(max_commits=3, max_file_bytes=10)
    rejected(repo, bundle_of(work, base, tmp_path), base, git, code, limits)


def test_base_not_ancestor_beats_later_checks(
    repo: BareRepo, base: str, tmp_path: Path, git: Git
) -> None:
    other = make_seed(git, tmp_path / "other", {"evil": b""})
    git("branch", "-m", "main", "task", cwd=other)
    (other / "evil").unlink()
    (other / "evil").symlink_to("/etc/passwd")
    git("add", "-A", cwd=other)
    git("commit", "-q", "-m", "unrelated", cwd=other)
    bundle = git_bundle(git, other, tmp_path / "unrelated.bundle", "refs/heads/task")
    rejected(repo, bundle, base, git, "base_not_ancestor")


# --------------------------------------------------------------------------- caller errors


def test_caller_errors_leave_no_refs(
    repo: BareRepo, base: str, work: Path, tmp_path: Path, git: Git
) -> None:
    write(work, {"x.txt": "x\n"})
    commit_all(work, "x")
    bundle = bundle_of(work, base, tmp_path)
    with pytest.raises(ValueError, match="not a full ref name"):
        import_task_bundle(repo, bundle, base_sha=base, dest_ref="task-1")
    with pytest.raises(ValueError, match="quarantine"):
        import_task_bundle(repo, bundle, base_sha=base, dest_ref="refs/factory/quarantine/x")
    with pytest.raises(GitError):
        import_task_bundle(repo, bundle, base_sha="0123456789" * 4, dest_ref=DEST)
    assert all_refs(repo, git) == ["refs/heads/main"]


def test_unexpected_failure_after_fetch_leaves_no_refs(
    repo: BareRepo, base: str, work: Path, tmp_path: Path, git: Git, monkeypatch: pytest.MonkeyPatch
) -> None:
    write(work, {"x.txt": "x\n"})
    commit_all(work, "x")

    def explode(self: BareRepo, raw: object) -> list[object]:
        assert any(r.startswith("refs/factory/quarantine/") for r in all_refs(self, git))
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(BareRepo, "changes_from_raw", explode)
    with pytest.raises(RuntimeError, match="disk on fire"):
        import_task_bundle(repo, bundle_of(work, base, tmp_path), base_sha=base, dest_ref=DEST)
    assert all_refs(repo, git) == ["refs/heads/main"]


def test_attributes_in_the_task_cannot_hide_added_lines(
    repo: BareRepo, base: str, work: Path, tmp_path: Path
) -> None:
    write(work, {".gitattributes": "* -diff binary\n", "src/app.py": "x = 1  # noqa\n"})
    commit_all(work, "hide")
    result = import_task_bundle(repo, bundle_of(work, base, tmp_path), base_sha=base, dest_ref=DEST)
    app = next(c for c in result.changes if c.path == "src/app.py")
    assert not app.is_binary
    assert [(ln.line_no, ln.text) for ln in app.added_lines] == [(1, "x = 1  # noqa")]
