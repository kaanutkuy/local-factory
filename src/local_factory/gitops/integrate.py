"""Integration on the factory's bare repository: merges without any working tree.

Merges use ``git merge-tree --write-tree`` (merge-ort, attributes disabled so in-tree
``.gitattributes`` cannot pick merge drivers) and ``git commit-tree``; ref updates are
compare-and-swap, so a concurrent update makes the merge fail with ``GitError`` instead of being
silently overwritten.
"""

from __future__ import annotations

from dataclasses import dataclass

from ._git import GitError
from .bare import NO_ATTRIBUTES, BareRepo


@dataclass(frozen=True)
class MergeOutcome:
    sha: str | None  # None iff there are conflicts
    conflicts: tuple[str, ...]

    def __post_init__(self) -> None:
        if (self.sha is None) != bool(self.conflicts):
            raise ValueError("MergeOutcome: sha must be None exactly when there are conflicts")


@dataclass(frozen=True)
class _TreeMerge:
    tree: str | None
    conflicts: tuple[str, ...]


def _merge_trees(repo: BareRepo, ours: str, theirs: str) -> _TreeMerge:
    """Three-way merge of two commits into a tree object (no refs, no working tree)."""
    args = [NO_ATTRIBUTES, "merge-tree", "--write-tree", "-z", "--name-only", "--no-messages"]
    args += ["--end-of-options", ours, theirs]
    result = repo.git(args, check=False)
    if result.returncode not in (0, 1):
        raise GitError(args, result.returncode, result.stderr.decode("utf-8", "replace"))
    tree, *paths = [field.decode("utf-8", "replace") for field in result.stdout.split(b"\0")]
    conflicts = tuple(sorted({path for path in paths if path}))
    if result.returncode == 0:
        return _TreeMerge(tree, ())
    if not conflicts:  # fail closed: never report a conflicted merge as clean
        raise GitError(args, result.returncode, "merge-tree reported conflicts without paths")
    return _TreeMerge(None, conflicts)


def _commit_tree(repo: BareRepo, tree: str, parents: tuple[str, ...], message: str) -> str:
    args = ["commit-tree", "--no-gpg-sign"]
    for parent in parents:
        args += ["-p", parent]
    args += ["-F", "-", tree]
    text = message if message.endswith("\n") else f"{message}\n"
    return repo.git(args, input=text.encode()).stdout.decode().strip()


def merge_into(repo: BareRepo, *, target_ref: str, source: str, message: str) -> MergeOutcome:
    """Merge ``source`` into ``target_ref``.

    Fast-forwards when the target is an ancestor of ``source``; returns the target unchanged
    when ``source`` is already merged; otherwise writes a merge commit with parents
    (target, source). The ref update is a compare-and-swap against the value read at the start
    (``GitError`` if it moved). On conflicts the ref is not touched.
    """
    target = repo.rev_parse(f"{target_ref}^{{commit}}")
    incoming = repo.rev_parse(f"{source}^{{commit}}")
    if repo.is_ancestor(target, incoming):
        repo.update_ref(target_ref, incoming, old_sha=target)
        return MergeOutcome(incoming, ())
    if repo.is_ancestor(incoming, target):
        return MergeOutcome(target, ())  # already merged
    merged = _merge_trees(repo, target, incoming)
    if merged.tree is None:
        return MergeOutcome(None, merged.conflicts)
    commit = _commit_tree(repo, merged.tree, (target, incoming), message)
    repo.update_ref(target_ref, commit, old_sha=target)
    return MergeOutcome(commit, ())


def squash_onto(repo: BareRepo, *, base: str, head: str, message: str) -> MergeOutcome:
    """A candidate commit whose tree is merge(base, head) and whose only parent is ``base``.

    Updates no refs. On conflicts returns the conflicting paths and creates no commit.
    """
    parent = repo.rev_parse(f"{base}^{{commit}}")
    merged = _merge_trees(repo, parent, repo.rev_parse(f"{head}^{{commit}}"))
    if merged.tree is None:
        return MergeOutcome(None, merged.conflicts)
    return MergeOutcome(_commit_tree(repo, merged.tree, (parent,), message), ())
