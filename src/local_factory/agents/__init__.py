"""Agent backends: the protocol every runtime implements, and the scripted `FakeBackend`."""

from __future__ import annotations

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

__all__ = [
    "EFFORT_LEVELS",
    "Action",
    "AgentBackend",
    "AgentResult",
    "AgentTask",
    "Capability",
    "DeleteFile",
    "Effort",
    "EmitOutput",
    "Fail",
    "FakeBackend",
    "RaiseHalt",
    "RoleSpec",
    "RunCommand",
    "WorkerHandle",
    "WriteFile",
]
