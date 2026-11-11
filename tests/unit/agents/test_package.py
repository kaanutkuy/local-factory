"""Unit tests for the local_factory.agents package: lazy re-exports and a light agents.local."""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

import local_factory
import local_factory.agents as agents

SRC = Path(local_factory.__file__).resolve().parent.parent
# The lazy loader exists at runtime only (mypy is not shown it), so reach it through the namespace.
LAZY_GETATTR = vars(agents)["__getattr__"]


@pytest.mark.parametrize("name", agents.__all__)
def test_every_export_resolves_to_its_defining_module(name: str) -> None:
    value = LAZY_GETATTR(name)
    defining = {
        "LocalExecutor": "local",
        "CommandResult": "tools",
        "ToolError": "tools",
        "ToolExecutor": "tools",
    }.get(name)
    candidates = [defining] if defining else ["backend", "fake"]
    owners = [
        module
        for module in candidates
        if getattr(importlib.import_module(f"local_factory.agents.{module}"), name, None) is value
    ]
    assert owners, name
    assert getattr(agents, name) is value


def test_from_imports_work() -> None:
    from local_factory.agents import FakeBackend, LocalExecutor, WorkerHandle
    from local_factory.agents.fake import FakeBackend as Defined

    assert FakeBackend is Defined
    assert WorkerHandle(work_dir=Path()).tools().__class__ is LocalExecutor


def test_unknown_names_are_attribute_errors() -> None:
    with pytest.raises(AttributeError, match="no attribute 'Nope'"):
        LAZY_GETATTR("Nope")
    unknown = "Nope"
    with pytest.raises(AttributeError):
        getattr(agents, unknown)


def test_dir_lists_the_exports() -> None:
    assert set(agents.__all__) <= set(dir(agents))


def test_the_box_helpers_import_without_heavy_dependencies() -> None:
    """The box imports agents.local under `python3 -I` for every file operation."""
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]);"
        "import local_factory.agents.local as local;"
        "heavy = sorted(m for m in ('pydantic', 'asyncio', 'local_factory.models',"
        " 'local_factory.agents.backend', 'local_factory.agents.fake') if m in sys.modules);"
        "print(heavy)"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", code, str(SRC)],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    assert result.stdout == "[]\n", result.stderr
