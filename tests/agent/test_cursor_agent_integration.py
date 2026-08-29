from __future__ import annotations

from typing import Any, cast

from agent.cursor_bridge_client import CursorBridgeClient
from run_agent import AIAgent


class _Process:
    def is_alive(self):
        return True

    def stop(self):
        return None


class _AgentLoopTransport:
    def __init__(self, client):
        self.client = client
        self.agent_count = 0
        self.prompts = []
        self.create_options = []
        self.callback_result = None

    def unary(self, service, method, message, timeout=0):
        if method == "CreateAgent":
            self.agent_count += 1
            self.create_options.append(message["options"])
            return {"agentId": f"agent-{self.agent_count}"}
        if method == "ListRuns":
            return {"items": []}
        if method in {"DeleteAgent", "Shutdown"}:
            return {}
        raise AssertionError((service, method, message))

    def server_stream(self, service, method, message, *, deadline, read_timeout=90.0):
        self.prompts.append(message["message"]["text"])
        if len(self.prompts) == 1:
            self.callback_result = self.client._handle_tool_callback(
                {
                    "agentId": message["agentId"],
                    "toolName": "todo",
                    "args": {},
                    "toolCallId": "cursor-tool-1",
                }
            )
            yield {"done": {}}
            return
        yield {
            "result": {
                "status": "RUN_LIFECYCLE_STATUS_FINISHED",
                "result": {
                    "result": "cursor-e2e-ok",
                    "usage": {"inputTokens": 1, "outputTokens": 1},
                },
            }
        }


def test_cursor_provider_runs_real_agent_tool_loop(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    agent = AIAgent(
        api_key="test-redacted",
        base_url="sdkbridge://cursor",
        provider="cursor",
        model="auto",
        enabled_toolsets=["todo"],
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        max_iterations=4,
    )
    assert isinstance(agent.client, CursorBridgeClient)

    wire = agent._create_request_openai_client(reason="cursor_test_seed")
    assert isinstance(wire, CursorBridgeClient)
    fake = _AgentLoopTransport(wire)
    cast(Any, wire)._transport = fake
    cast(Any, wire)._process = _Process()
    agent._close_request_openai_client(wire, reason="request_complete")

    try:
        result = agent.run_conversation("Use todo once, then finish.")
    finally:
        agent.close()

    assert result is not None
    assert result["final_response"] == "cursor-e2e-ok"
    assert len(fake.prompts) == 2
    assert fake.callback_result is not None
    assert fake.callback_result["status"] == "deferred"
    assert "[tool call] todo(" in fake.prompts[1]
    assert "Tool result:\n" in fake.prompts[1]
    assert fake.create_options[0]["tools"] == {"names": []}
    assert set(fake.create_options[0]["local"]["customTools"]) == {"todo"}


def test_agent_interrupt_uses_cursor_process_abort_hook(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    agent = AIAgent(
        api_key="test-redacted",
        base_url="sdkbridge://cursor",
        provider="cursor",
        model="auto",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    wire = agent._create_request_openai_client(reason="cursor_abort_seed")
    assert isinstance(wire, CursorBridgeClient)
    process = _Process()
    cast(Any, wire)._process = process
    cache = agent._request_client_cache_ref()
    cache.update(client=wire, poisoned=False, in_use=True)
    try:
        agent._abort_request_openai_client(wire, reason="test_interrupt")
        assert wire.is_closed is True
        assert wire._process is None
        assert cache["poisoned"] is True
    finally:
        agent.close()
