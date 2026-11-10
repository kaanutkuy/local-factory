"""Policy: deterministic rules that every imported task diff must satisfy."""

from __future__ import annotations

from local_factory.policy.diff_policy import DiffPolicy, check_changes
from local_factory.policy.globs import glob_match

__all__ = ["DiffPolicy", "check_changes", "glob_match"]
