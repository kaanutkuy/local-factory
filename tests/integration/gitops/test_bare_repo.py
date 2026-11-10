"""BareRepo against real git: lifecycle, refs (CAS, atomicity), history and object queries."""

from __future__ import annotations

from pathlib import Path

import pytest

from local_factory.gitops._git import NULL_SHA, GitError
from local_factory.gitops.bare import HARDENED_CONFIG, BareRepo, RefCommand
from local_factory.models import MODE_EXECUTABLE, MODE_GITLINK, MODE_REGULAR, MODE_SYMLINK

from .conftest import Git, all_refs, bare_from, make_seed

pytestmark = pytest.mark.integration


def config(git: Git, repo: BareRepo, key: str) -> str:
    return git("config", "--file", str(repo.path / "config"), key, cwd=repo.path)


# --------------------------------------------------------------------------- lifecycle


def test_init_applies_every_hardening_key(tmp_path: Path, git: Git) -> None:
    repo = BareRepo.init(tmp_path / "r.git")
    assert repo.path == tmp_path / "r.git"
    for key, value in HARDENED_CONFIG:
        assert config(git, repo, key) == value
    assert git("symbolic-ref", "HEAD", cwd=repo.path) == "refs/heads/main"


def test_init_is_idempotent_and_reapplies_hardening(repo: BareRepo, base: str, git: Git) -> None:
    git("config", "--file", str(repo.path / "config"), "fetch.fsckObjects", "false", cwd=repo.path)
    git("config", "--file", str(repo.path / "config"), "core.hooksPath", "hooks", cwd=repo.path)
    again = BareRepo.init(repo.path)
    assert again == repo
    assert config(git, repo, "fetch.fsckObjects") == "true"
    assert config(git, repo, "core.hooksPath") == "/dev/null"
    assert again.rev_parse("refs/heads/main") == base  # content preserved


def test_init_into_existing_empty_dir(tmp_path: Path) -> None:
    target = tmp_path / "empty"
    target.mkdir()
    assert BareRepo.open(BareRepo.init(target).path).path == target


def test_init_refuses_a_non_repository_dir(tmp_path: Path) -> None:
    target = tmp_path / "busy"
    target.mkdir()
    (target / "file.txt").write_text("x")
    with pytest.raises(GitError):
        BareRepo.init(target)
    assert sorted(p.name for p in target.iterdir()) == ["file.txt"]


def test_init_accepts_relative_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    repo = BareRepo.init(Path("rel.git"))
    assert repo.path == tmp_path / "rel.git" and repo.path.is_absolute()


def test_open_errors(tmp_path: Path, seed_repo: Path) -> None:
    with pytest.raises(GitError):
        BareRepo.open(tmp_path / "missing")
    with pytest.raises(GitError) as exc:
        BareRepo.open(seed_repo / ".git")  # a git dir, but not bare
    assert exc.value.returncode == 0
    assert "not a bare repository" in exc.value.stderr


# --------------------------------------------------------------------------- fetch


def test_fetch_from_creates_only_the_requested_refs(tmp_path: Path, git: Git) -> None:
    seed = make_seed(git, tmp_path / "s", {"a.txt": "a\n"})
    git("tag", "v1", cwd=seed)
    git("branch", "other", cwd=seed)
    repo = bare_from(seed, tmp_path / "r.git")
    assert all_refs(repo, git) == ["refs/heads/main"]
    assert not (repo.path / "FETCH_HEAD").exists()


def test_fetch_from_bundle_path_as_string(tmp_path: Path, git: Git, seed_repo: Path) -> None:
    bundle = tmp_path / "seed.bundle"
    git("bundle", "create", str(bundle), "main", cwd=seed_repo)
    repo = BareRepo.init(tmp_path / "r.git")
    repo.fetch_from(str(bundle), ["refs/heads/main:refs/heads/imported"])
    assert repo.rev_parse("refs/heads/imported") == git("rev-parse", "main", cwd=seed_repo)


def test_fetch_from_requires_refspecs(repo: BareRepo, seed_repo: Path) -> None:
    with pytest.raises(ValueError, match="refspec"):
        repo.fetch_from(seed_repo, [])


def test_fetch_from_option_like_source_is_not_an_option(repo: BareRepo) -> None:
    with pytest.raises(GitError) as exc:
        repo.fetch_from("--upload-pack=touch /tmp/pwned", ["refs/heads/main:refs/heads/x"])
    assert "upload-pack" in exc.value.stderr or "not a git repository" in exc.value.stderr.lower()


# --------------------------------------------------------------------------- refs


def test_rev_parse(repo: BareRepo, base: str, git: Git) -> None:
    assert repo.rev_parse("main") == base
    assert repo.rev_parse("HEAD") == base
    assert repo.rev_parse(base[:12]) == base
    tree = repo.rev_parse(f"{base}^{{tree}}")
    assert tree == git("rev-parse", f"{base}^{{tree}}", cwd=repo.path) == repo.tree_hash(base)
    for bad in ("refs/heads/nope", "--all", "", "nope~1"):
        with pytest.raises(GitError):
            repo.rev_parse(bad)


def test_ref_exists(repo: BareRepo, tmp_path: Path) -> None:
    assert repo.ref_exists("refs/heads/main")
    assert not repo.ref_exists("refs/heads/mai")  # exact names, no prefix or DWIM matching
    with pytest.raises(ValueError, match="not a full ref name"):
        repo.ref_exists("main")
    broken = BareRepo(tmp_path / "not-a-repo")
    with pytest.raises(GitError):
        broken.ref_exists("refs/heads/main")


def test_update_ref_create_only_and_cas(repo: BareRepo, base: str, git: Git) -> None:
    repo.update_ref("refs/x/a", base, old_sha=NULL_SHA)
    with pytest.raises(GitError, match="already exists"):
        repo.update_ref("refs/x/a", base, old_sha=NULL_SHA)
    other = git("commit-tree", f"{base}^{{tree}}", "-p", base, "-m", "o", cwd=repo.path)
    with pytest.raises(GitError, match="expected"):
        repo.update_ref("refs/x/a", other, old_sha=other)  # stale expectation
    assert repo.rev_parse("refs/x/a") == base
    repo.update_ref("refs/x/a", other, old_sha=base)
    repo.update_ref("refs/x/a", base)  # unconditional
    assert repo.rev_parse("refs/x/a") == base


def test_update_ref_rejects_non_commit_for_branch(repo: BareRepo, base: str) -> None:
    with pytest.raises(GitError):
        repo.update_ref("refs/heads/tree", repo.tree_hash(base))


def test_delete_ref(repo: BareRepo, base: str, git: Git) -> None:
    repo.delete_ref("refs/x/missing")  # idempotent
    repo.update_ref("refs/x/a", base)
    other = git("commit-tree", f"{base}^{{tree}}", "-m", "o", cwd=repo.path)
    with pytest.raises(GitError):
        repo.delete_ref("refs/x/a", old_sha=other)
    assert repo.ref_exists("refs/x/a")
    repo.delete_ref("refs/x/a", old_sha=base)
    assert not repo.ref_exists("refs/x/a")


def test_ref_transaction_is_atomic(repo: BareRepo, base: str, git: Git) -> None:
    before = all_refs(repo, git)
    with pytest.raises(GitError):
        repo.ref_transaction(
            [
                RefCommand("update", "refs/x/new", new=base),
                RefCommand("update", "refs/heads/main", new=base, old=NULL_SHA),  # fails
            ]
        )
    assert all_refs(repo, git) == before
    repo.ref_transaction(
        [RefCommand("update", "refs/x/a", new=base), RefCommand("update", "refs/x/b", new=base)]
    )
    assert repo.ref_exists("refs/x/a") and repo.ref_exists("refs/x/b")


# --------------------------------------------------------------------------- history


@pytest.fixture
def branchy(tmp_path: Path, git: Git) -> tuple[BareRepo, dict[str, str]]:
    """main: root - a - merge(a, side)   side: root - s"""
    seed = make_seed(git, tmp_path / "h", {"f.txt": "0\n"})
    shas = {"root": git("rev-parse", "HEAD", cwd=seed)}
    git("checkout", "-q", "-b", "side", cwd=seed)
    (seed / "s.txt").write_text("s\n")
    git("add", "-A", cwd=seed)
    git("commit", "-q", "-m", "s", cwd=seed)
    shas["s"] = git("rev-parse", "HEAD", cwd=seed)
    git("checkout", "-q", "main", cwd=seed)
    (seed / "a.txt").write_text("a\n")
    git("add", "-A", cwd=seed)
    git("commit", "-q", "-m", "a", cwd=seed)
    shas["a"] = git("rev-parse", "HEAD", cwd=seed)
    git("merge", "-q", "--no-edit", "side", cwd=seed)
    shas["merge"] = git("rev-parse", "HEAD", cwd=seed)
    return bare_from(seed, tmp_path / "h.git"), shas


def test_list_and_count_commits(branchy: tuple[BareRepo, dict[str, str]]) -> None:
    repo, shas = branchy
    commits = repo.list_commits(shas["root"], shas["merge"])
    assert sorted(commits) == sorted([shas["a"], shas["s"], shas["merge"]])
    assert commits[-1] == shas["merge"]  # parents before children
    assert repo.count_commits(shas["root"], shas["merge"]) == 3
    assert repo.list_commits(shas["merge"], shas["merge"]) == []
    assert repo.count_commits(shas["merge"], shas["root"]) == 0


def test_is_ancestor(branchy: tuple[BareRepo, dict[str, str]]) -> None:
    repo, shas = branchy
    assert repo.is_ancestor(shas["root"], shas["merge"])
    assert repo.is_ancestor(shas["s"], shas["merge"])
    assert not repo.is_ancestor(shas["a"], shas["s"])
    assert repo.is_ancestor(shas["a"], shas["a"])
    with pytest.raises(GitError):
        repo.is_ancestor("0123456789" * 4, shas["a"])


def test_new_objects(branchy: tuple[BareRepo, dict[str, str]], git: Git) -> None:
    repo, shas = branchy
    new = set(repo.new_objects(shas["root"], shas["a"]))
    blob_a = git("rev-parse", f"{shas['a']}:a.txt", cwd=repo.path)
    blob_f = git("rev-parse", f"{shas['root']}:f.txt", cwd=repo.path)
    assert {shas["a"], blob_a, repo.tree_hash(shas["a"])} <= new
    assert blob_f not in new and shas["root"] not in new


# --------------------------------------------------------------------------- objects


@pytest.fixture
def mixed(tmp_path: Path, git: Git) -> tuple[BareRepo, str]:
    seed = make_seed(git, tmp_path / "m", {"dir/run.sh": "#!/bin/sh\n", "plain.txt": "p\n"})
    (seed / "dir" / "run.sh").chmod(0o755)
    (seed / "link").symlink_to("plain.txt")
    sub = git("rev-parse", "HEAD", cwd=seed)
    git("add", "-A", cwd=seed)
    git("update-index", "--add", "--cacheinfo", f"160000,{sub},vendor/sub", cwd=seed)
    git("commit", "-q", "-m", "mixed", cwd=seed)
    repo = bare_from(seed, tmp_path / "m.git")
    return repo, repo.rev_parse("main")


def test_ls_tree(mixed: tuple[BareRepo, str]) -> None:
    repo, head = mixed
    entries = {e.path: (e.mode, e.size) for e in repo.ls_tree(head)}
    assert entries == {
        b"dir/run.sh": (MODE_EXECUTABLE, 10),
        b"link": (MODE_SYMLINK, 9),
        b"plain.txt": (MODE_REGULAR, 2),
        b"vendor/sub": (MODE_GITLINK, None),
    }


def test_read_file(mixed: tuple[BareRepo, str]) -> None:
    repo, head = mixed
    assert repo.read_file(head, "plain.txt") == b"p\n"
    assert repo.read_file(head, "link") == b"plain.txt"  # a symlink's blob is its target
    for bad in ("missing.txt", "dir"):
        with pytest.raises(GitError):
            repo.read_file(head, bad)


def test_object_info_and_read_blobs(mixed: tuple[BareRepo, str], git: Git) -> None:
    repo, head = mixed
    blob = git("rev-parse", f"{head}:plain.txt", cwd=repo.path)
    missing = "1234567890" * 4
    assert repo.object_info([blob, blob, missing, head]) == {
        blob: ("blob", 2),
        head: ("commit", repo.object_info([head])[head][1]),
    }
    assert repo.object_info([]) == {}
    assert repo.read_blobs([blob, blob]) == {blob: b"p\n"}
    assert repo.read_blobs([]) == {}
    with pytest.raises(GitError, match="expected a blob"):
        repo.read_blobs([blob, repo.tree_hash(head)])
    with pytest.raises(GitError, match="expected a blob"):
        repo.read_blobs([missing])


def test_read_blobs_handles_binary_content(tmp_path: Path, git: Git) -> None:
    payload = bytes(range(256)) * 3 + b"\n\n"
    seed = make_seed(git, tmp_path / "b", {"bin": payload, "text": "t\n"})
    repo = bare_from(seed, tmp_path / "b.git")
    shas = [git("rev-parse", f"main:{name}", cwd=repo.path) for name in ("bin", "text")]
    assert repo.read_blobs(shas) == {shas[0]: payload, shas[1]: b"t\n"}
