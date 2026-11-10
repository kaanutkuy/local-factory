"""The backend contract every agent runtime implements (docs/PLAN.md §11).

A backend runs one role against one task inside one worker directory and reports what happened as
an `AgentResult`. Backends never own workflow state: the orchestrator records results in the
ledger and decides what happens next. Nothing safety- or correctness-critical depends on a
backend capability.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal, Protocol, Self, get_args, runtime_checkable

from pydantic import BaseModel, Field, JsonValue, model_validator

from local_factory.models import (
    TASK_ID_PATTERN,
    TICKET_ID_PATTERN,
    FrozenModel,
    Halt,
    RunStatus,
    Tier,
    Usage,
)


class Capability(StrEnum):
    """Optional features a backend may offer. Never required for safety or correctness."""

    NATIVE_FS_TOOLS = "native_fs_tools"
    HOOKS = "hooks"
    STRUCTURED_OUTPUT = "structured_output"
    SESSIONS = "sessions"
    SUBAGENTS = "subagents"


Effort = Literal["low", "medium", "high", "xhigh", "max"]
EFFORT_LEVELS: tuple[Effort, ...] = get_args(Effort)


@dataclass(frozen=True)
class RoleSpec:
    """What an agent is asked to be: prompt, model tier, effort, tools and output schema."""

    name: str
    prompt: str
    tier: Tier
    effort: Effort = "high"
    tools: tuple[str, ...] = ()
    max_turns: int = 50
    output_schema: type[BaseModel] | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("RoleSpec.name must not be empty")
        if self.effort not in EFFORT_LEVELS:
            raise ValueError(f"RoleSpec.effort must be one of {EFFORT_LEVELS}, got {self.effort!r}")
        if self.max_turns < 1:
            raise ValueError(f"RoleSpec.max_turns must be >= 1, got {self.max_turns}")


class AgentTask(FrozenModel):
    """One unit of work handed to a backend."""

    ticket_id: str = Field(pattern=TICKET_ID_PATTERN)
    task_id: str | None = Field(default=None, pattern=TASK_ID_PATTERN)
    instructions: str
    context: dict[str, JsonValue] = Field(default_factory=dict)


@dataclass(frozen=True)
class WorkerHandle:
    """The worker directory a backend operates in (inside the sandbox in production)."""

    work_dir: Path


class AgentResult(FrozenModel):
    """What one backend run produced. Backends report failures here instead of raising."""

    status: RunStatus
    halt: Halt | None = None
    structured_output: dict[str, JsonValue] | None = None
    text: str | None = None
    usage: Usage = Usage()
    session_id: str | None = None
    transcript_path: Path | None = None
    permission_denials: tuple[dict[str, JsonValue], ...] = ()
    error: str | None = None

    @model_validator(mode="after")
    def _status_matches_details(self) -> Self:
        if (self.status == "halted") != (self.halt is not None):
            raise ValueError("a halt must be given exactly when status is 'halted'")
        if self.status == "error" and not self.error:
            raise ValueError("status 'error' requires a non-empty error message")
        return self


@runtime_checkable
class AgentBackend(Protocol):
    """An agent runtime: Claude Agent SDK, an OpenAI-compatible loop, or the scripted fake."""

    name: str
    capabilities: frozenset[Capability]

    async def run(self, role: RoleSpec, task: AgentTask, worker: WorkerHandle) -> AgentResult: ...
