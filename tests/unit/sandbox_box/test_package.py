from __future__ import annotations

import subprocess
import sys

import pytest

import local_factory.sandbox as sandbox_pkg
from local_factory.sandbox.box import ExportOutcome, Sandbox
from local_factory.sandbox.executor import SandboxExecutor


def test_public_api_is_re_exported() -> None:
    assert sandbox_pkg.Sandbox is Sandbox
    assert sandbox_pkg.ExportOutcome is ExportOutcome
    assert sandbox_pkg.SandboxExecutor is SandboxExecutor
    assert sorted(sandbox_pkg.__all__) == ["ExportOutcome", "Sandbox", "SandboxExecutor"]
    missing = "Nope"
    with pytest.raises(AttributeError, match="no attribute 'Nope'"):
        getattr(sandbox_pkg, missing)
    assert {"ExportOutcome", "Sandbox", "SandboxExecutor"} <= set(dir(sandbox_pkg))


def test_names_resolve_lazily_and_once() -> None:
    probe = (
        "import sys, local_factory.sandbox as p; "
        "before = 'local_factory.sandbox.box' in sys.modules; "
        "first = p.Sandbox; "
        "print(before, 'Sandbox' in vars(p), p.Sandbox is first, 'SandboxExecutor' in vars(p))"
    )
    out = subprocess.run(
        [sys.executable, "-I", "-c", probe], capture_output=True, text=True, check=True
    ).stdout
    assert out.split() == ["False", "True", "True", "False"]


def test_the_box_entrypoint_loads_no_host_machinery() -> None:
    """lf-entry imports only what runs in the box: not the docker runtime, the facade or asyncio."""
    probe = (
        "import sys; import local_factory.sandbox.entry; "
        "prefixes = ('local_factory.sandbox', 'asyncio'); "
        "print(sorted(m for m in sys.modules if m.startswith(prefixes)))"
    )
    out = subprocess.run(
        [sys.executable, "-I", "-c", probe], capture_output=True, text=True, check=True
    ).stdout
    assert out.strip() == (
        "['local_factory.sandbox', 'local_factory.sandbox.entry', 'local_factory.sandbox.protocol']"
    )
