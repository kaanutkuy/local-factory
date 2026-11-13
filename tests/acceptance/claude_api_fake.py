"""An in-process stand-in for the Claude Messages API. Orchestrator-owned and protected.

`FakeClaudeAPI` answers ``POST /v1/messages`` through ``httpx2.MockTransport``, so a real
``anthropic.AsyncAnthropic`` client (its request building, SSE streaming, accumulation and error
mapping) runs unmodified with no network. Each request is recorded (headers + JSON body); each
reply comes from a script: a list consumed in order, or a function of the recorded call.

Replies are plain API JSON (`reply(...)`, `api_error(...)`), or an exception instance, which the
transport raises (the SDK maps it to ``APIConnectionError``).
"""

from __future__ import annotations

import copy
import itertools
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx2
from anthropic import AsyncAnthropic, DefaultAsyncHttpxClient

BASE_URL = "http://anthropic.test"
DEFAULT_KEY = "sk-ant-test-0000"
DEFAULT_USAGE = {"input_tokens": 10, "output_tokens": 5}

_ids = itertools.count(1)


@dataclass(frozen=True)
class Reply:
    """A successful Messages API response."""

    message: dict[str, Any]


@dataclass(frozen=True)
class ApiError:
    status: int
    type: str
    message: str = "fake API error"
    headers: dict[str, str] = field(default_factory=dict)


Scripted = Reply | ApiError | BaseException


@dataclass
class Call:
    """One request the fake received."""

    index: int
    headers: dict[str, str]
    body: dict[str, Any]

    @property
    def system_text(self) -> str:
        system = self.body.get("system", "")
        if isinstance(system, str):
            return system
        return "".join(block.get("text", "") for block in system)

    @property
    def messages(self) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = self.body["messages"]
        return messages

    @property
    def tool_names(self) -> list[str]:
        return [tool["name"] for tool in self.body.get("tools", [])]

    @property
    def betas(self) -> list[str]:
        raw = self.headers.get("anthropic-beta", "")
        return [b.strip() for b in raw.split(",") if b.strip()]

    def tool_results(self) -> list[dict[str, Any]]:
        """The tool_result blocks of the last message (a user message after tool calls)."""
        content = self.messages[-1]["content"]
        if isinstance(content, str):
            return []
        return [block for block in content if block.get("type") == "tool_result"]

    def all_tool_results(self) -> list[dict[str, Any]]:
        """Every tool_result block anywhere in this request's conversation."""
        found: list[dict[str, Any]] = []
        for message in self.messages:
            content = message["content"]
            if isinstance(content, list):
                found.extend(b for b in content if b.get("type") == "tool_result")
        return found


def result_text(block: dict[str, Any]) -> str:
    """The text of a tool_result block, whether its content is a string or text blocks."""
    content = block.get("content", "")
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content)


# --------------------------------------------------------------------------- reply builders


def text(value: str) -> dict[str, Any]:
    return {"type": "text", "text": value}


def thinking(signature: str = "sig-0001", summary: str = "") -> dict[str, Any]:
    return {"type": "thinking", "thinking": summary, "signature": signature}


def compaction(summary: str = "<summary>earlier work</summary>") -> dict[str, Any]:
    return {"type": "compaction", "content": summary}


def tool_use(name: str, tool_input: dict[str, Any], *, id: str | None = None) -> dict[str, Any]:
    return {
        "type": "tool_use",
        "id": id or f"toolu_{next(_ids):06d}",
        "name": name,
        "input": tool_input,
    }


def bash(command: str, *, id: str | None = None) -> dict[str, Any]:
    return tool_use("bash", {"command": command}, id=id)


def edit(*, id: str | None = None, **tool_input: Any) -> dict[str, Any]:
    return tool_use("str_replace_based_edit_tool", tool_input, id=id)


def submit(payload: dict[str, Any], *, id: str | None = None) -> dict[str, Any]:
    return tool_use("submit_result", payload, id=id)


def reply(
    *blocks: dict[str, Any],
    stop_reason: str | None = None,
    usage: dict[str, Any] | None = None,
    model: str | None = None,
    stop_details: dict[str, Any] | None = None,
) -> Reply:
    """A response. ``stop_reason`` defaults to ``tool_use`` if a block is one, else ``end_turn``."""
    if stop_reason is None:
        stop_reason = "tool_use" if any(b["type"] == "tool_use" for b in blocks) else "end_turn"
    message: dict[str, Any] = {
        "id": f"msg_{next(_ids):06d}",
        "type": "message",
        "role": "assistant",
        "content": list(blocks),
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 0, "output_tokens": 0, **(usage or DEFAULT_USAGE)},
    }
    if model is not None:
        message["model"] = model
    if stop_details is not None:
        message["stop_details"] = stop_details
    return Reply(message)


def api_error(
    status: int, type: str, message: str = "fake API error", headers: dict[str, str] | None = None
) -> ApiError:
    return ApiError(status=status, type=type, message=message, headers=dict(headers or {}))


# --------------------------------------------------------------------------- the fake


class FakeClaudeAPI:
    """Serves scripted replies to ``POST /v1/messages`` and records every request."""

    def __init__(self, script: Sequence[Scripted] | Callable[[Call], Scripted]) -> None:
        self._script = script
        self.calls: list[Call] = []
        self.overrun = False  # a request arrived after a list script was used up

    def client(self, *, api_key: str = DEFAULT_KEY, max_retries: int = 0) -> AsyncAnthropic:
        return AsyncAnthropic(
            api_key=api_key,
            base_url=BASE_URL,
            max_retries=max_retries,
            http_client=DefaultAsyncHttpxClient(transport=httpx2.MockTransport(self.handle)),
        )

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        if request.method != "POST" or not request.url.path.endswith("/v1/messages"):
            return _error_response(ApiError(404, "not_found_error", f"fake: {request.url.path}"))
        call = Call(
            index=len(self.calls), headers=dict(request.headers), body=json.loads(request.content)
        )
        self.calls.append(call)
        scripted = self._next(call)
        if isinstance(scripted, BaseException):
            raise scripted
        if isinstance(scripted, ApiError):
            return _error_response(scripted)
        message = copy.deepcopy(scripted.message)
        message.setdefault("model", call.body.get("model", "claude-test"))
        if call.body.get("stream"):
            return httpx2.Response(
                200,
                headers={"content-type": "text/event-stream", "request-id": f"req_{call.index}"},
                content=_sse(message),
            )
        return httpx2.Response(200, json=message, headers={"request-id": f"req_{call.index}"})

    def _next(self, call: Call) -> Scripted:
        if callable(self._script):
            return self._script(call)
        if call.index < len(self._script):
            return self._script[call.index]
        self.overrun = True
        return ApiError(400, "invalid_request_error", "fake API: the script is used up")


def _error_response(error: ApiError) -> httpx2.Response:
    body = {"type": "error", "error": {"type": error.type, "message": error.message}}
    return httpx2.Response(error.status, json=body, headers=error.headers)


def _sse(message: dict[str, Any]) -> bytes:
    """The streaming events a real server sends for ``message``."""
    start = dict(message, content=[], stop_reason=None, stop_sequence=None)
    start["usage"] = {
        "input_tokens": message["usage"].get("input_tokens", 0),
        "output_tokens": 1,
    }
    events: list[dict[str, Any]] = [{"type": "message_start", "message": start}]
    for index, block in enumerate(message["content"]):
        events.extend(_block_events(index, block))
        events.append({"type": "content_block_stop", "index": index})
    delta: dict[str, Any] = {"stop_reason": message["stop_reason"], "stop_sequence": None}
    if "stop_details" in message:
        delta["stop_details"] = message["stop_details"]
    events.append({"type": "message_delta", "delta": delta, "usage": message["usage"]})
    events.append({"type": "message_stop"})
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


def _block_events(index: int, block: dict[str, Any]) -> list[dict[str, Any]]:
    kind = block["type"]

    def start(content_block: dict[str, Any]) -> dict[str, Any]:
        return {"type": "content_block_start", "index": index, "content_block": content_block}

    def delta(body: dict[str, Any]) -> dict[str, Any]:
        return {"type": "content_block_delta", "index": index, "delta": body}

    if kind == "text":
        return [
            start({"type": "text", "text": ""}),
            delta({"type": "text_delta", "text": block["text"]}),
        ]
    if kind == "thinking":
        return [
            start({"type": "thinking", "thinking": "", "signature": ""}),
            delta({"type": "thinking_delta", "thinking": block["thinking"]}),
            delta({"type": "signature_delta", "signature": block["signature"]}),
        ]
    if kind == "tool_use":
        return [
            start({"type": "tool_use", "id": block["id"], "name": block["name"], "input": {}}),
            delta({"type": "input_json_delta", "partial_json": json.dumps(block["input"])}),
        ]
    if kind == "compaction":
        return [
            start({"type": "compaction", "content": None}),
            delta({"type": "compaction_delta", "content": block["content"]}),
        ]
    return [start(block)]
