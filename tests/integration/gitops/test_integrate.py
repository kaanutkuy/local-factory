"""merge_into / squash_onto edge cases: no-ops, CAS races, conflicts, attribute-free merges."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from local_factory.gitops import integrate
from local_factory.gitops._git import FACTORY_EMAIL, FACTORY_NAME, GitError
from local_factory.gitops.bare import BareRepo
from local_factory.gitops.bundles import import_task_bundle
from local_factory.gitops.integrate import MergeOutcome, merge_into, squash_onto
from local_factory.gitops.worker_side import commit_all

from .conftest import Git, all_refs, bundle_of, import_files, make_seed, make_work

pytestmark = pytest.mark.integration

TICKET = "refs/heads/factory/T-0001"


def task(repo: BareRepo, base: str, tmp_path: Path, name: str, files: dict[str, str]) -> str:
    return import_files(
        repo, base, tmp_path, files, name=name, dest_ref=f"refs/factory/tasks/T-0001/{name}"
    )


def parents(git: Git, repo: BareRepo, sha: str) -> list[str]:
    return git("rev-list", "--parents", "-n", "1", sha, cwd=repo.path).split()[1:]


def test_already_merged_and_equal_sources_are_no_ops(
    repo: BareRepo, base: str, tmp_path: Path
) -> None:
    head = task(repo, base, tmp_path, "t1", {"a.txt": "a\n"})
    repo.update_ref(TICKET, head)
    assert merge_into(repo, target_ref=TICKET, source=head, message="m") == MergeOutcome(head, ())
    assert merge_into(repo, target_ref=TICKET, source=base, message="m") == MergeOutcome(head, ())
    assert repo.rev_parse(TICKET) == head


def test_merge_of_disjoint_changes(repo: BareRepo, base: str, tmp_path: Path, git: Git) -> None:
    repo.update_ref(TICKET, base)
    h1 = task(repo, base, tmp_path, "t1", {"src/one.py": "1\n"})
    h2 = task(repo, base, tmp_path, "t2", {"docs/two.md": "2\n", "src/app.py": "changed\n"})
    merge_into(repo, target_ref=TICKET, source=h1, message="one")
    outcome = merge_into(repo, target_ref=TICKET, source=h2, message="merge two")
    assert outcome.sha is not None and repo.rev_parse(TICKET) == outcome.sha
    assert parents(git, repo, outcome.sha) == [h1, h2]
    assert repo.read_file(outcome.sha, "src/one.py") == b"1\n"
    assert repo.read_file(outcome.sha, "src/app.py") == b"changed\n"
    log = git("log", "-1", "--format=%an <%ae>|%B", outcome.sha, cwd=repo.path)
    assert log == f"{FACTORY_NAME} <{FACTORY_EMAIL}>|merge two"


def test_conflicts_are_sorted_unique_and_leave_the_ref(
    repo: BareRepo, base: str, tmp_path: Path, git: Git
) -> None:
    repo.update_ref(TICKET, base)
    h1 = task(repo, base, tmp_path, "t1", {"src/app.py": "one\n", "README.md": "one\n", "n": "1"})
    h2 = task(repo, base, tmp_path, "t2", {"src/app.py": "two\n", "README.md": "two\n", "n": "2"})
    merge_into(repo, target_ref=TICKET, source=h1, message="one")
    refs = all_refs(repo, git)
    outcome = merge_into(repo, target_ref=TICKET, source=h2, message="two")
    assert outcome == MergeOutcome(None, ("README.md", "n", "src/app.py"))  # add/add included
    assert repo.rev_parse(TICKET) == h1 and all_refs(repo, git) == refs


def test_modify_delete_conflict(repo: BareRepo, base: str, tmp_path: Path, git: Git) -> None:
    repo.update_ref(TICKET, base)
    work = make_work(repo, base, tmp_path, "deleter")
    (work / "src" / "app.py").unlink()
    commit_all(work, "delete")
    deleted = import_task_bundle(
        repo, bundle_of(work, base, tmp_path, "d.bundle"), base_sha=base, dest_ref="refs/t/d"
    ).head_sha
    modified = task(repo, base, tmp_path, "t2", {"src/app.py": "modified\n"})
    merge_into(repo, target_ref=TICKET, source=deleted, message="d")
    outcome = merge_into(repo, target_ref=TICKET, source=modified, message="m")
    assert outcome == MergeOutcome(None, ("src/app.py",))


def test_in_tree_attributes_cannot_pick_merge_drivers(
    repo: BareRepo, base: str, tmp_path: Path
) -> None:
    repo.update_ref(TICKET, base)
    h1 = task(repo, base, tmp_path, "t1", {".gitattributes": "* merge=union\n", "f": "one\n"})
    h2 = task(repo, base, tmp_path, "t2", {".gitattributes": "* merge=union\n", "f": "two\n"})
    merge_into(repo, target_ref=TICKET, source=h1, message="one")
    # With the union driver this would silently "merge" to "one\ntwo\n".
    assert merge_into(repo, target_ref=TICKET, source=h2, message="two").conflicts == ("f",)


def test_fast_forward_cas_failure(
    repo: BareRepo, base: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo.update_ref(TICKET, base)
    h1 = task(repo, base, tmp_path, "t1", {"a": "a\n"})
    racer = task(repo, base, tmp_path, "t2", {"b": "b\n"})
    real = BareRepo.is_ancestor

    def racing(self: BareRepo, ancestor: str, descendant: str) -> bool:
        answer = real(self, ancestor, descendant)
        self.update_ref(TICKET, racer)  # another integrator wins the race
        return answer

    monkeypatch.setattr(BareRepo, "is_ancestor", racing)
    with pytest.raises(GitError, match="expected"):
        merge_into(repo, target_ref=TICKET, source=h1, message="ff")
    assert repo.rev_parse(TICKET) == racer


def test_merge_commit_cas_failure(
    repo: BareRepo, base: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo.update_ref(TICKET, base)
    h1 = task(repo, base, tmp_path, "t1", {"a": "a\n"})
    h2 = task(repo, base, tmp_path, "t2", {"b": "b\n"})
    racer = task(repo, base, tmp_path, "t3", {"c": "c\n"})
    merge_into(repo, target_ref=TICKET, source=h1, message="one")
    real = integrate._merge_trees

    def racing(r: BareRepo, ours: str, theirs: str) -> integrate._TreeMerge:
        r.update_ref(TICKET, racer)
        return real(r, ours, theirs)

    monkeypatch.setattr(integrate, "_merge_trees", racing)
    with pytest.raises(GitError, match="expected"):
        merge_into(repo, target_ref=TICKET, source=h2, message="two")
    assert repo.rev_parse(TICKET) == racer  # the winner's value is never overwritten


def test_merge_errors(repo: BareRepo, base: str, tmp_path: Path, git: Git) -> None:
    with pytest.raises(GitError):
        merge_into(repo, target_ref="refs/heads/missing", source=base, message="m")
    with pytest.raises(GitError):
        merge_into(repo, target_ref="refs/heads/main", source="0123456789" * 4, message="m")
    other = make_seed(git, tmp_path / "other", {"z": "z\n"})
    repo.fetch_from(other, ["refs/heads/main:refs/heads/unrelated"])
    with pytest.raises(GitError, match="unrelated"):
        merge_into(repo, target_ref="refs/heads/main", source="refs/heads/unrelated", message="m")
    assert repo.rev_parse("refs/heads/main") == base


def test_conflict_without_paths_fails_closed(
    repo: BareRepo, base: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake(self: BareRepo, args: list[str], **_: object) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(args, 1, b"0" * 40 + b"\0", b"")  # exit 1, no paths

    monkeypatch.setattr(BareRepo, "git", fake)
    with pytest.raises(GitError, match="without paths"):
        integrate._merge_trees(repo, base, base)


def test_squash_onto(repo: BareRepo, base: str, tmp_path: Path, git: Git) -> None:
    head = task(repo, base, tmp_path, "t1", {"feature.txt": "f\n"})
    refs = all_refs(repo, git)
    outcome = squash_onto(repo, base=base, head=head, message="T-0001: feature")
    assert outcome.sha is not None and outcome.sha != head
    assert parents(git, repo, outcome.sha) == [base]
    assert repo.tree_hash(outcome.sha) == repo.tree_hash(head)  # base is an ancestor: head's tree
    assert git("log", "-1", "--format=%B", outcome.sha, cwd=repo.path) == "T-0001: feature"
    assert all_refs(repo, git) == refs


def test_squash_onto_conflict_creates_nothing(
    repo: BareRepo, base: str, tmp_path: Path, git: Git
) -> None:
    upstream = task(repo, base, tmp_path, "up", {"src/app.py": "upstream\n"})
    head = task(repo, base, tmp_path, "t1", {"src/app.py": "task\n"})
    refs = all_refs(repo, git)
    assert squash_onto(repo, base=upstream, head=head, message="x") == MergeOutcome(
        None, ("src/app.py",)
    )
    assert all_refs(repo, git) == refs


def test_squash_onto_message_newline_is_preserved(
    repo: BareRepo, base: str, tmp_path: Path, git: Git
) -> None:
    head = task(repo, base, tmp_path, "t1", {"x": "x\n"})
    outcome = squash_onto(repo, base=base, head=head, message="subject\n\nbody\n")
    assert outcome.sha is not None
    raw = git("cat-file", "commit", outcome.sha, cwd=repo.path)
    assert raw.endswith("subject\n\nbody")
