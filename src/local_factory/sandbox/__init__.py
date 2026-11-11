"""The box: hardened, network-less containers for every worker action (docs/specs/phase-1b.md).

Public API: :class:`Sandbox` (workspaces, the in-box tools, task export, gates),
:class:`ExportOutcome` and :class:`SandboxExecutor`. Lower layers: :mod:`.spec` (container specs
and their unconditional hardening), :mod:`.runtime` (the docker CLI), :mod:`.workspace` (tmpfs task
workspaces). :mod:`.entry` is the code that runs *inside* the box.

The names below are loaded lazily (PEP 562): ``lf-entry`` imports ``local_factory.sandbox.entry``
inside the box, which runs this file first, and the box needs none of the host-side docker
machinery. ``from local_factory.sandbox import X`` works as usual.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

_EXPORTS: dict[str, str] = {
    "ExportOutcome": "box",
    "Sandbox": "box",
    "SandboxExecutor": "executor",
}

__all__ = ["ExportOutcome", "Sandbox", "SandboxExecutor"]


if TYPE_CHECKING:
    from local_factory.sandbox.box import ExportOutcome, Sandbox
    from local_factory.sandbox.executor import SandboxExecutor
else:
    # Hidden from mypy (it skips this branch), so mypy knows exactly the names imported above and
    # rejects any other `from local_factory.sandbox import ...`.

    def __getattr__(name: str) -> object:
        module = _EXPORTS.get(name)
        if module is None:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
        value = getattr(importlib.import_module(f"{__name__}.{module}"), name)
        globals()[name] = value  # resolve each name once
        return value

    def __dir__() -> list[str]:
        return sorted({*globals(), *__all__})
