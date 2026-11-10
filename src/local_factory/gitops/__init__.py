"""Git topology for the factory: hardened bare repos, the bundle boundary, and integration.

Security model (docs/PLAN.md invariants 3 and 7, §5.1):

- Every git process goes through :func:`._git.run_git`: sanitized environment, no user config,
  no hooks, no fsmonitor, ``file`` transport only, never the process's cwd.
- Host-side code (:mod:`.bare`, :mod:`.bundles`, :mod:`.integrate`) only ever runs git against
  factory-owned bare repositories via ``--git-dir``. Workers are reached only through bundle
  files, which are copied and validated before any ref points at their content.
- :mod:`.worker_side` is the only code that runs git inside a worker directory, and in
  production it runs inside the worker's sandbox.
"""

from __future__ import annotations

from ._git import FACTORY_EMAIL, FACTORY_NAME, GitError, git_env, run_git
from .bare import BareRepo
from .bundles import (
    BundleLimits,
    BundleRejected,
    ImportResult,
    export_base_bundle,
    import_task_bundle,
)
from .integrate import MergeOutcome, merge_into, squash_onto
from .worker_side import (
    TASK_BRANCH,
    commit_all,
    export_task_bundle,
    has_task_commits,
    init_from_bundle,
)

__all__ = [
    "FACTORY_EMAIL",
    "FACTORY_NAME",
    "TASK_BRANCH",
    "BareRepo",
    "BundleLimits",
    "BundleRejected",
    "GitError",
    "ImportResult",
    "MergeOutcome",
    "commit_all",
    "export_base_bundle",
    "export_task_bundle",
    "git_env",
    "has_task_commits",
    "import_task_bundle",
    "init_from_bundle",
    "merge_into",
    "run_git",
    "squash_onto",
]
