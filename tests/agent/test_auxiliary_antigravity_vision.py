"""Vision resolution must retain the Antigravity bridge and its lifecycle."""

import asyncio
import threading

import pytest

from agent import antigravity_bridge_client as bridge
from agent.auxiliary_client import _to_async_client, resolve_vision_provider_client


@pytest.fixture
def local_bridge(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "model:\n  provider: antigravity\n"
        "  default: antigravity-gemini-3-pro\n"
        "auxiliary:\n  vision:\n    provider: auto\n",
        encoding="utf-8",
    )
    command = ["test-only-bridge", "--explicit-route"]
    from agent import antigravity_bridge_transport

    monkeypatch.setattr(
        antigravity_bridge_transport, "resolve_antigravity_bridge_command",
        lambda: command,
    )
    processes = []
    requests = []
    entered = threading.Event()
    released = threading.Event()

    class Process:
        def __init__(self, actual_command):
            assert actual_command == command
            self.stops = 0
            processes.append(self)

        def start(self):
            return self

        def stop(self):
            self.stops += 1
            released.set()

    class Transport:
        def __init__(self, endpoint):
            assert endpoint in processes

        def json_request(self, method, path, payload=None, **kwargs):
            if method == "GET":
                return {"data": []}
            requests.append((method, path, payload))
            entered.set()
            if payload.get("model") == "blocking-test-model":
                assert released.wait(10), "cancellation did not stop the owner"
            return {"choices": [{"message": {"content": "A test image."}}]}

    monkeypatch.setattr(bridge, "AntigravityBridgeProcess", Process)
    monkeypatch.setattr(bridge, "AntigravityHTTPTransport", Transport)
    return command, processes, requests, entered


def test_auto_vision_uses_bridge_with_multimodal_payload(local_bridge):
    command, processes, requests, _ = local_bridge

    async def exercise():
        provider, client, model = resolve_vision_provider_client(async_mode=True)
        assert provider == "antigravity"
        assert isinstance(client, bridge.AsyncAntigravityBridgeClient)
        assert client._sync._bridge_command == command
        messages = [{"role": "user", "content": [
            {"type": "text", "text": "Describe the image"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,dGVzdA=="}},
        ]}]
        try:
            response = await client.chat.completions.create(model=model, messages=messages)
            assert response.choices[0].message.content == "A test image."
            assert requests == [("POST", "/v1/chat/completions", {
                "model": model, "messages": messages,
            })]
        finally:
            await client.close()
            await client.close()
        assert len(processes) == 1
        assert processes[0].stops == 1

    asyncio.run(exercise())


def test_conversion_retains_started_owner_and_cancels_it(local_bridge):
    command, processes, _, entered = local_bridge
    sync_client = bridge.AntigravityBridgeClient(bridge_command=command)
    sync_client.list_models()

    async def exercise():
        client, model = _to_async_client(sync_client, "blocking-test-model", is_vision=True)
        assert client._sync is sync_client
        task = asyncio.create_task(client.chat.completions.create(model=model, messages=[]))
        try:
            assert await asyncio.to_thread(entered.wait, 10)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert sync_client.is_closed
            assert len(processes) == 1
            assert processes[0].stops == 1
        finally:
            task.cancel()
            await client.close()

    asyncio.run(exercise())


def test_conversion_does_not_reopen_closed_client(local_bridge):
    command, processes, _, _ = local_bridge
    sync_client = bridge.AntigravityBridgeClient(bridge_command=command)
    sync_client.close()

    async def exercise():
        client, model = _to_async_client(sync_client, "test-model")
        with pytest.raises(RuntimeError, match="closed"):
            await client.chat.completions.create(model=model, messages=[])
        await client.close()

    asyncio.run(exercise())
    assert not processes
