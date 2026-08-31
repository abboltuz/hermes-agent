from __future__ import annotations

import asyncio
import io
import threading
import time
from typing import Any, cast

import pytest

from agent.cursor_bridge_client import (
    _ActiveRun,
    _CallbackHandler,
    AsyncCursorBridgeClient,
    CursorBridgeClient,
)
from agent.cursor_bridge_transport import CursorBridgeError


TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "safe_tool",
            "description": "A test tool",
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "string"}},
            },
        },
    }
]


class _Process:
    def __init__(self):
        self.stopped = False

    def is_alive(self):
        return not self.stopped

    def stop(self):
        self.stopped = True


class _Transport:
    def __init__(self, client, *, stream_error=False, final_text=""):
        self.client = client
        self.stream_error = stream_error
        self.final_text = final_text
        self.calls = []
        self.delete_timeouts = []
        self.callback_result = None

    def unary(self, service, method, message, timeout=0):
        self.calls.append((service, method, message, timeout))
        if method == "CreateAgent":
            return {"agentId": "agent-1"}
        if method == "ListRuns":
            return {"items": []}
        if method == "DeleteAgent":
            self.delete_timeouts.append(timeout)
            return {}
        if method == "Shutdown":
            return {}
        raise AssertionError((service, method, message))

    def server_stream(self, service, method, message, *, deadline, read_timeout=90.0):
        self.callback_result = self.client._handle_tool_callback(
            {
                "agentId": message["agentId"],
                "toolName": "safe_tool",
                "args": {"value": "ok"},
                "toolCallId": "stable-tool-id",
            }
        )
        if self.stream_error:
            raise CursorBridgeError("cancelled after captured tool")
        if self.final_text:
            yield {
                "result": {
                    "status": "RUN_LIFECYCLE_STATUS_FINISHED",
                    "result": {"result": self.final_text},
                }
            }
        else:
            yield {"done": {}}


def _client_with_transport(*, stream_error=False, final_text=""):
    client = CursorBridgeClient(api_key="test-redacted", bridge_command="unused")
    fake = _Transport(client, stream_error=stream_error, final_text=final_text)
    cast(Any, client)._transport = fake
    cast(Any, client)._process = _Process()
    return client, fake


def test_loop_mode_exposes_only_host_custom_tools_and_stable_ids():
    from openai.types.chat.chat_completion_message_tool_call import (
        ChatCompletionMessageToolCall,
        Function,
    )

    client, fake = _client_with_transport()
    try:
        response = client.chat.completions.create(
            model="auto",
            messages=[{"role": "user", "content": "use a tool"}],
            tools=TOOL_SCHEMA,
            timeout=2,
        )
    finally:
        client.close()

    create = next(call for call in fake.calls if call[1] == "CreateAgent")
    options = create[2]["options"]
    assert options["tools"] == {"names": ["mcp"]}
    assert options["local"]["settingSources"] == []
    assert set(options["local"]["customTools"]) == {"safe_tool"}
    tool_call = response.choices[0].message.tool_calls[0]
    assert response.choices[0].finish_reason == "tool_calls"
    assert type(tool_call) is ChatCompletionMessageToolCall
    assert type(tool_call.function) is Function
    assert tool_call.id == "stable-tool-id"
    assert tool_call.function.name == "safe_tool"
    assert dict(tool_call)["id"] == "stable-tool-id"
    assert dict(tool_call.function)["name"] == "safe_tool"
    assert tool_call.model_dump()["function"]["arguments"] == '{"value": "ok"}'
    assert fake.callback_result is not None
    assert fake.callback_result["status"] == "deferred"


def test_captured_tool_survives_bridge_cancellation_error():
    client, _fake = _client_with_transport(stream_error=True)
    try:
        response = client.chat.completions.create(
            model="auto",
            messages=[{"role": "user", "content": "use a tool"}],
            tools=TOOL_SCHEMA,
            timeout=2,
        )
    finally:
        client.close()

    assert response.choices[0].finish_reason == "tool_calls"
    assert response.choices[0].message.tool_calls[0].id == "stable-tool-id"


def test_tool_free_runs_remain_text_only():
    client, fake = _client_with_transport(final_text="plain response")
    try:
        response = client.chat.completions.create(
            model="auto",
            messages=[{"role": "user", "content": "say hello"}],
            timeout=2,
        )
    finally:
        client.close()

    create = next(call for call in fake.calls if call[1] == "CreateAgent")
    options = create[2]["options"]
    assert options["tools"] == {"names": []}
    assert "customTools" not in options["local"]
    assert "settingSources" not in options["local"]
    assert response.choices[0].finish_reason == "stop"


def test_harness_and_cursor_builtin_tools_fail_closed():
    with pytest.raises(CursorBridgeError, match="harness mode"):
        CursorBridgeClient(api_key="x", tool_mode="harness")
    with pytest.raises(CursorBridgeError, match="built-in tools"):
        CursorBridgeClient(api_key="x", builtin_tools=True)


def test_non_allowlisted_callback_is_rejected():
    client = CursorBridgeClient(api_key="x")
    run = _ActiveRun("agent-1", {"safe_tool"}, time.monotonic() + 5)
    cast(Any, client)._active_runs["agent-1"] = run
    result = client._handle_tool_callback(
        {"agentId": "agent-1", "toolName": "other_tool", "args": {}}
    )
    assert result == {"error": "tool is not declared for this Hermes run"}
    assert run.captured_calls == []


def test_chunked_callback_body_enforces_size_cap():
    handler = cast(Any, object.__new__(_CallbackHandler))
    handler.headers = {"Transfer-Encoding": "chunked"}
    handler.rfile = io.BytesIO(b"100001\r\n")
    with pytest.raises(ValueError, match="size limit"):
        handler._read_body()


def test_capability_negotiation_requires_sdk_v1():
    class VersionTransport:
        def unary(self, _service, method, _message, timeout=0):
            if method == "Ping":
                return {"message": "pong"}
            return {"protocolVersion": "sdk.v2"}

    with pytest.raises(CursorBridgeError, match="required sdk.v1"):
        CursorBridgeClient._verify_capabilities(cast(Any, VersionTransport()))


def test_cleanup_timeout_is_within_request_deadline():
    client, fake = _client_with_transport(final_text="ok")
    try:
        response = client.chat.completions.create(
            model="auto",
            messages=[{"role": "user", "content": "finish"}],
            timeout=0.5,
        )
    finally:
        client.close()

    assert response.choices[0].message.content == "ok"
    assert fake.delete_timeouts
    assert 0 < fake.delete_timeouts[0] <= 0.5


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_async_cancellation_closes_sync_bridge():
    client = AsyncCursorBridgeClient(api_key="test-redacted", bridge_command="unused")
    entered = threading.Event()
    released = threading.Event()
    closed = threading.Event()

    def blocking_create(**_kwargs):
        entered.set()
        released.wait(2)
        return object()

    def abort_inflight():
        closed.set()
        released.set()

    client._sync.chat.completions.create = blocking_create
    client._sync.abort_inflight = abort_inflight
    task = asyncio.create_task(client.chat.completions.create(model="auto"))
    assert await asyncio.to_thread(entered.wait, 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed.is_set()
