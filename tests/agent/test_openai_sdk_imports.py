"""Behavioral contracts for lazy, ordered OpenAI SDK imports."""

from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path


def test_cursor_and_copilot_imports_do_not_eagerly_load_openai() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    code = """
import sys

assert not any(name == "openai" or name.startswith("openai.") for name in sys.modules)

from agent import copilot_acp_client, cursor_bridge_client

assert cursor_bridge_client.format_messages_as_prompt(
    [{"role": "user", "content": "plain text"}]
)

class Process:
    def is_alive(self):
        return True

    def stop(self):
        pass

class Transport:
    def unary(self, _service, method, _message, timeout=0):
        if method == "CreateAgent":
            return {"agentId": "agent-1"}
        if method in {"DeleteAgent", "Shutdown"}:
            return {}
        raise AssertionError(method)

    def server_stream(self, *_args, **_kwargs):
        yield {
            "result": {
                "status": "RUN_LIFECYCLE_STATUS_FINISHED",
                "result": {"result": "cursor text"},
            }
        }

cursor = cursor_bridge_client.CursorBridgeClient(
    api_key="test-redacted",
    bridge_command="unused",
)
cursor._transport = Transport()
cursor._process = Process()
try:
    cursor_response = cursor.chat.completions.create(
        model="auto",
        messages=[{"role": "user", "content": "plain text"}],
        timeout=1,
    )
finally:
    cursor.close()
assert cursor_response.choices[0].message.content == "cursor text"

copilot = copilot_acp_client.CopilotACPClient(acp_cwd="/tmp")
copilot._run_prompt = lambda *_args, **_kwargs: ("copilot text", "")
copilot_response = copilot._create_chat_completion(
    model="copilot-acp",
    messages=[{"role": "user", "content": "plain text"}],
)
assert copilot_response.choices[0].message.content == "copilot text"

loaded = sorted(
    name for name in sys.modules
    if name == "openai" or name.startswith("openai.")
)
assert loaded == [], loaded
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_openai_import_gate_serializes_root_and_type_loading(monkeypatch) -> None:
    from agent import copilot_acp_client, openai_sdk_imports, process_bootstrap

    root_entered = threading.Event()
    release_root = threading.Event()
    type_call_started = threading.Event()
    type_import_entered = threading.Event()
    errors: list[BaseException] = []
    results: list[object] = []

    class _RootModule:
        OpenAI = type("OpenAI", (), {})

    class _Function:
        def __init__(self, **values):
            self.__dict__.update(values)

    class _ToolCall:
        def __init__(self, **values):
            self.__dict__.update(values)

    class _TypeModule:
        ChatCompletionMessageToolCall = _ToolCall
        Function = _Function

    def fake_import(name: str):
        if threading.current_thread().name == "root-loader":
            root_entered.set()
            if not release_root.wait(5):
                raise TimeoutError("root import was not released")
        else:
            type_import_entered.set()
        if name == "openai":
            return _RootModule
        if name == "openai.types.chat.chat_completion_message_tool_call":
            return _TypeModule
        raise AssertionError(name)

    monkeypatch.setattr(openai_sdk_imports.importlib, "import_module", fake_import)
    monkeypatch.setattr(process_bootstrap, "_OPENAI_CLS_CACHE", None)

    def load_root() -> None:
        try:
            results.append(process_bootstrap._load_openai_cls())
        except BaseException as exc:  # pragma: no cover - failure receipt
            errors.append(exc)

    def load_types() -> None:
        type_call_started.set()
        try:
            results.append(
                copilot_acp_client._build_openai_tool_call(
                    call_id="call-1",
                    name="read_file",
                    arguments="{}",
                )
            )
        except BaseException as exc:  # pragma: no cover - failure receipt
            errors.append(exc)

    root_thread = threading.Thread(target=load_root, name="root-loader")
    type_thread = threading.Thread(target=load_types, name="type-loader")
    root_thread.start()
    assert root_entered.wait(5)
    type_thread.start()
    assert type_call_started.wait(5)

    # The second loader has started, but cannot enter importlib until the
    # root import leaves the shared gate.
    assert not type_import_entered.wait(2)
    release_root.set()
    root_thread.join(5)
    type_thread.join(5)

    assert not root_thread.is_alive()
    assert not type_thread.is_alive()
    assert errors == []
    assert type_import_entered.is_set()
    assert any(isinstance(result, _ToolCall) for result in results)


def test_copilot_tool_call_keeps_real_openai_sdk_models() -> None:
    from openai.types.chat.chat_completion_message_tool_call import (
        ChatCompletionMessageToolCall,
        Function,
    )

    from agent.copilot_acp_client import _build_openai_tool_call

    tool_call = _build_openai_tool_call(
        call_id="call-123",
        name="read_file",
        arguments='{"path":"README.md"}',
    )

    assert type(tool_call) is ChatCompletionMessageToolCall
    assert type(tool_call.function) is Function
    assert tool_call.id == "call-123"
    assert tool_call.call_id == "call-123"
    assert dict(tool_call)["id"] == "call-123"
    assert dict(tool_call.function)["name"] == "read_file"
    assert tool_call.model_dump()["function"]["arguments"] == '{"path":"README.md"}'
