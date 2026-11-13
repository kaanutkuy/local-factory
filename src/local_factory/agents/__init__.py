"""Agent backends: the protocol every runtime implements, its tools, and the scripted `FakeBackend`.

The names below are loaded lazily (PEP 562): importing a submodule such as
`local_factory.agents.local`, which the box imports under `python3 -I` for its file helpers,
must not drag in pydantic and the domain models. `from local_factory.agents import X` works as
usual.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

_EXPORTS: dict[str, str] = {
    **dict.fromkeys(
        (
            "EFFORT_LEVELS",
            "AgentBackend",
            "AgentResult",
            "AgentTask",
            "Capability",
            "Effort",
            "RoleSpec",
            "WorkerHandle",
        ),
        "backend",
    ),
    **dict.fromkeys(
        (
            "Action",
            "DeleteFile",
            "EmitOutput",
            "Fail",
            "FakeBackend",
            "RaiseHalt",
            "RunCommand",
            "WriteFile",
        ),
        "fake",
    ),
    "LocalExecutor": "local",
    **dict.fromkeys(("CommandResult", "ToolError", "ToolExecutor"), "tools"),
    **dict.fromkeys(("BOX_TOOLS", "BoxTools", "ToolOutcome"), "box_tools"),
    "ClaudeAPIBackend": "claude_api",  # the only name that imports the `anthropic` SDK
}

__all__ = [
    "BOX_TOOLS",
    "EFFORT_LEVELS",
    "Action",
    "AgentBackend",
    "AgentResult",
    "AgentTask",
    "BoxTools",
    "Capability",
    "ClaudeAPIBackend",
    "CommandResult",
    "DeleteFile",
    "Effort",
    "EmitOutput",
    "Fail",
    "FakeBackend",
    "LocalExecutor",
    "RaiseHalt",
    "RoleSpec",
    "RunCommand",
    "ToolError",
    "ToolExecutor",
    "ToolOutcome",
    "WorkerHandle",
    "WriteFile",
]


if TYPE_CHECKING:
    from local_factory.agents.backend import (
        EFFORT_LEVELS,
        AgentBackend,
        AgentResult,
        AgentTask,
        Capability,
        Effort,
        RoleSpec,
        WorkerHandle,
    )
    from local_factory.agents.box_tools import BOX_TOOLS, BoxTools, ToolOutcome
    from local_factory.agents.claude_api import ClaudeAPIBackend
    from local_factory.agents.fake import (
        Action,
        DeleteFile,
        EmitOutput,
        Fail,
        FakeBackend,
        RaiseHalt,
        RunCommand,
        WriteFile,
    )
    from local_factory.agents.local import LocalExecutor
    from local_factory.agents.tools import CommandResult, ToolError, ToolExecutor
else:
    # Hidden from mypy (it skips this branch), so mypy knows exactly the names imported above and
    # rejects any other `from local_factory.agents import ...`.

    def __getattr__(name: str) -> object:
        module = _EXPORTS.get(name)
        if module is None:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
        value = getattr(importlib.import_module(f"{__name__}.{module}"), name)
        globals()[name] = value  # resolve each name once
        return value

    def __dir__() -> list[str]:
        return sorted({*globals(), *__all__})
