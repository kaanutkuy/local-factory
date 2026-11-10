"""Unit tests for FakeBackend structured output: pass-through, schema validation, normalization."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from local_factory.agents.backend import RoleSpec, WorkerHandle
from local_factory.agents.fake import EmitOutput, FakeBackend, WriteFile
from local_factory.models import Tier

from .conftest import run, run_script


class Report(BaseModel):
    summary: str
    files_changed: int


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    summary: str


class Open(BaseModel):
    model_config = ConfigDict(extra="allow")
    summary: str


class WithDefaults(BaseModel):
    summary: str
    risks: list[str] = Field(default_factory=list)
    confidence: float = 0.5


class Aliased(BaseModel):
    files_changed: int = Field(alias="filesChanged")


class Timed(BaseModel):
    at: datetime


def role(schema: type[BaseModel] | None) -> RoleSpec:
    return RoleSpec(name="implementer", prompt="p", tier=Tier.STANDARD, output_schema=schema)


# --------------------------------------------------------------------------- without a schema


def test_without_schema_and_without_output_there_is_none(worker: WorkerHandle) -> None:
    assert run_script(worker, WriteFile("a", "x")).structured_output is None


def test_without_schema_the_last_emit_wins(worker: WorkerHandle) -> None:
    result = run_script(worker, EmitOutput({"n": 1}), EmitOutput({"n": 2}), EmitOutput({"n": 3}))
    assert result.structured_output == {"n": 3}


def test_without_schema_nested_json_passes_through(worker: WorkerHandle) -> None:
    data: dict[str, Any] = {"a": [1, 2.5, "s", None, True], "b": {"c": {"d": []}}}
    assert run_script(worker, EmitOutput(data)).structured_output == data


def test_without_schema_non_json_output_is_an_error(worker: WorkerHandle) -> None:
    data: Any = {"when": datetime(2026, 1, 1), "tags": {"a", "b"}}
    result = run_script(worker, EmitOutput(data))
    assert result.status == "error"
    assert result.error is not None
    assert "not a JSON object" in result.error


def test_without_schema_a_non_object_is_an_error(worker: WorkerHandle) -> None:
    data: Any = [1, 2, 3]
    assert run_script(worker, EmitOutput(data)).status == "error"


def test_output_is_a_copy_of_the_scripted_data(worker: WorkerHandle) -> None:
    backend = FakeBackend({"implementer": [EmitOutput({"items": [1]})]})
    first = run(backend, worker).structured_output
    assert first is not None
    items = first["items"]
    assert isinstance(items, list)
    items.append(2)
    assert run(backend, worker).structured_output == {"items": [1]}


# --------------------------------------------------------------------------- with a schema


def test_schema_validates_the_last_emit(worker: WorkerHandle) -> None:
    good: dict[str, JsonValue] = {"summary": "done", "files_changed": 1}
    bad: dict[str, JsonValue] = {"summary": "done"}
    ok = run_script(worker, EmitOutput(bad), EmitOutput(good), role=role(Report))
    assert ok.structured_output == good
    failed = run_script(worker, EmitOutput(good), EmitOutput(bad), role=role(Report))
    assert failed.status == "error"
    assert failed.structured_output is None


def test_schema_mismatch_error_names_the_schema_and_field(worker: WorkerHandle) -> None:
    result = run_script(worker, EmitOutput({"summary": "done"}), role=role(Report))
    assert result.error is not None
    assert result.error.startswith("structured output does not match Report")
    assert "files_changed" in result.error


def test_schema_without_any_emit_is_an_error(worker: WorkerHandle) -> None:
    result = run_script(worker, role=role(Report))
    assert result.error == (
        "role 'implementer' requires structured output (Report) but the script emitted none"
    )


def test_schema_ignoring_extras_drops_them(worker: WorkerHandle) -> None:
    data: dict[str, JsonValue] = {"summary": "s", "files_changed": 1, "chatter": "ignored"}
    result = run_script(worker, EmitOutput(data), role=role(Report))
    assert result.status == "ok", result.error
    assert result.structured_output == {"summary": "s", "files_changed": 1}


def test_schema_allowing_extras_keeps_them(worker: WorkerHandle) -> None:
    result = run_script(worker, EmitOutput({"summary": "s", "extra": [1]}), role=role(Open))
    assert result.structured_output == {"summary": "s", "extra": [1]}


def test_schema_forbidding_extras_rejects_them(worker: WorkerHandle) -> None:
    result = run_script(worker, EmitOutput({"summary": "s", "extra": 1}), role=role(Strict))
    assert result.status == "error"
    assert result.error is not None
    assert "extra" in result.error


def test_schema_output_is_normalized_to_the_validated_model(worker: WorkerHandle) -> None:
    result = run_script(worker, EmitOutput({"summary": "s"}), role=role(WithDefaults))
    assert result.structured_output == {"summary": "s", "risks": [], "confidence": 0.5}
    coerced = run_script(
        worker, EmitOutput({"summary": "s", "files_changed": "3"}), role=role(Report)
    )
    assert coerced.structured_output == {"summary": "s", "files_changed": 3}


def test_schema_output_uses_aliases_so_it_revalidates(worker: WorkerHandle) -> None:
    result = run_script(worker, EmitOutput({"filesChanged": 4}), role=role(Aliased))
    assert result.structured_output == {"filesChanged": 4}
    assert Aliased.model_validate(result.structured_output).files_changed == 4


def test_schema_output_is_json_compatible(worker: WorkerHandle) -> None:
    result = run_script(worker, EmitOutput({"at": "2026-01-01T05:00:00Z"}), role=role(Timed))
    assert result.structured_output == {"at": "2026-01-01T05:00:00Z"}
