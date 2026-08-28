"""Tests for gateway/wake.py — background wake delivery.

Two strategies:
* push-capable adapters keep the synthetic MessageEvent / handle_message path;
* the stateless API server (supports_async_delivery=False) self-POSTs
  /v1/chat/completions with the RAW session id in X-Hermes-Session-Id, so the
  wake turn resumes the REAL session instead of a parallel invisible one
  keyed by build_session_key().
"""

import asyncio
import uuid
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import Platform
from gateway.internal_turn import INTERNAL_TURN_FIELD
from gateway.session import SessionSource
from gateway.wake import deliver_wake, adapter_supports_push


class PushAdapter:
    """Default adapter shape — no supports_async_delivery attribute."""

    def __init__(self):
        self.handled = []

    async def handle_message(self, event):
        self.handled.append(event)


class ApiServerLikeAdapter:
    supports_async_delivery = False

    def __init__(self, host="0.0.0.0", port=0, key="test-key", model="hermes"):
        self._host = host
        self._port = port
        self._api_key = key
        self._model_name = model

    async def handle_message(self, event):  # pragma: no cover — must NOT be hit
        raise AssertionError("non-push adapter must not receive handle_message wakes")


class ProfileAwareApiServerLikeAdapter(ApiServerLikeAdapter):
    def _wake_request_target(self, *, profile="", route_profile=""):
        assert profile == "writer"
        assert route_profile == "writer"
        return "/p/writer/v1/chat/completions", "writer-key"


def _source():
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="chat-1",
        chat_type="group",
    )


def test_adapter_supports_push_default_true():
    assert adapter_supports_push(PushAdapter()) is True
    assert adapter_supports_push(ApiServerLikeAdapter()) is False


async def _serve(handler):
    """Spin an in-process aiohttp server on an ephemeral loopback port."""
    from aiohttp import web

    app = web.Application()
    app.router.add_post("/v1/chat/completions", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, port


def test_deliver_wake_non_push_self_posts_raw_session_id(monkeypatch):
    """The self-post carries the RAW session id header + bearer auth and a
    single user message with stream=false — the exact entry point real
    gateway turns use."""
    from aiohttp import web

    seen = {}

    async def handler(request):
        seen["session_id"] = request.headers.get("X-Hermes-Session-Id")
        seen["auth"] = request.headers.get("Authorization")
        seen["body"] = await request.json()
        return web.json_response({"choices": [{"message": {"content": "ok"}}]})

    async def run():
        runner, port = await _serve(handler)
        try:
            adapter = ApiServerLikeAdapter(host="0.0.0.0", port=port, key="sekrit")
            await deliver_wake(
                adapter,
                text="task done — wake",
                session_id="raw-sid-42",
                display_metadata={
                    "source": "process",
                    "internal": True,
                    "kind": "process_notification",
                    "event_id": "proc-42",
                },
            )
        finally:
            await runner.cleanup()

    asyncio.run(run())
    assert seen["session_id"] == "raw-sid-42"
    assert seen["auth"] == "Bearer sekrit"
    assert seen["body"]["stream"] is False
    assert seen["body"]["messages"] == [
        {"role": "user", "content": "task done — wake"}
    ]
    assert seen["body"][INTERNAL_TURN_FIELD] == {
        "display_kind": "internal_notification",
        "display_metadata": {
            "source": "process",
            "internal": True,
            "kind": "process_notification",
            "event_id": "proc-42",
        },
        "provenance": {
            "origin_kind": "internal_system",
            "turn_kind": "notification",
            "trust_kind": "trusted_internal",
            "provenance_metadata": {
                "producer": "gateway_wake",
                "source": "process",
                "event_kind": "process_notification",
                "platform": "api_server",
                "event_id": "proc-42",
                "session_id": "raw-sid-42",
            },
        },
    }


def test_deliver_wake_retries_429_then_succeeds(monkeypatch):
    """HTTP 429 (max_concurrent_runs cap) is transient — retried with backoff."""
    from aiohttp import web

    import gateway.wake as wake_mod

    monkeypatch.setattr(wake_mod, "_RETRY_DELAYS_SECONDS", (0.01, 0.01, 0.01))
    calls = {"n": 0}

    async def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return web.json_response({"error": "busy"}, status=429)
        return web.json_response({"choices": []})

    async def run():
        runner, port = await _serve(handler)
        try:
            adapter = ApiServerLikeAdapter(port=port)
            await deliver_wake(adapter, text="x", session_id="sid")
        finally:
            await runner.cleanup()

    asyncio.run(run())
    assert calls["n"] == 2


def test_deliver_wake_named_profile_uses_qualified_route_and_scoped_key():
    from aiohttp import web

    seen = {}

    async def handler(request):
        seen["path"] = request.path
        seen["auth"] = request.headers.get("Authorization")
        return web.json_response({"choices": []})

    async def run():
        app = web.Application()
        app.router.add_post("/p/writer/v1/chat/completions", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        try:
            adapter = ProfileAwareApiServerLikeAdapter(port=port)
            await deliver_wake(
                adapter,
                text="done",
                session_id="writer-session",
                profile="writer",
                route_profile="writer",
            )
        finally:
            await runner.cleanup()

    asyncio.run(run())
    assert seen == {
        "path": "/p/writer/v1/chat/completions",
        "auth": "Bearer writer-key",
    }


def test_named_profile_never_falls_back_on_unaware_adapter():
    with pytest.raises(RuntimeError, match="refusing default-profile fallback"):
        asyncio.run(
            deliver_wake(
                ApiServerLikeAdapter(key="default-key"),
                text="done",
                session_id="writer-session",
                profile="writer",
                route_profile="writer",
            )
        )


def test_wake_retry_after_lost_response_executes_agent_once(monkeypatch):
    """A lost response after acceptance retries with one stable idempotency key."""
    from gateway.config import PlatformConfig
    from gateway.platforms.api_server import APIServerAdapter
    import gateway.wake as wake_mod

    monkeypatch.setattr(wake_mod, "_RETRY_DELAYS_SECONDS", (0.01,))
    api_adapter = APIServerAdapter(
        PlatformConfig(enabled=True, extra={"key": "wake-key"})
    )
    result = {"final_response": "ok", "messages": [], "api_calls": 1}
    usage = {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
    calls = {"http": 0, "keys": []}

    async def flaky_handler(request):
        calls["http"] += 1
        calls["keys"].append(request.headers.get("Idempotency-Key"))
        response = await api_adapter._handle_chat_completions(request)
        if calls["http"] == 1:
            request.transport.close()
        return response

    async def run():
        runner, port = await _serve(flaky_handler)
        api_adapter._host = "127.0.0.1"
        api_adapter._port = port
        try:
            with patch.object(
                api_adapter,
                "_run_agent",
                new=AsyncMock(return_value=(result, usage)),
            ) as run_agent:
                await deliver_wake(
                    api_adapter,
                    text="background task finished",
                    session_id="raw-session",
                    display_metadata={
                        "source": "process",
                        "internal": True,
                        "kind": "process_notification",
                        "event_id": f"proc-{uuid.uuid4().hex}",
                    },
                )
                assert run_agent.await_count == 1
        finally:
            await runner.cleanup()

    asyncio.run(run())
    assert calls["http"] == 2
    assert calls["keys"][0]
    assert calls["keys"][0] == calls["keys"][1]
    assert len(calls["keys"][0]) <= 128


def test_kanban_wake_self_post_reaches_api_as_agent_continuation():
    """Exercise the authenticated self-post and API envelope parser together."""
    from gateway.config import PlatformConfig
    from gateway.platforms.api_server import APIServerAdapter

    api_adapter = APIServerAdapter(
        PlatformConfig(enabled=True, extra={"key": "wake-key"})
    )
    result = {"final_response": "ok", "messages": [], "api_calls": 1}
    usage = {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}

    async def run():
        runner, port = await _serve(api_adapter._handle_chat_completions)
        api_adapter._host = "127.0.0.1"
        api_adapter._port = port
        try:
            with patch.object(
                api_adapter,
                "_run_agent",
                new=AsyncMock(return_value=(result, usage)),
            ) as run_agent:
                await deliver_wake(
                    api_adapter,
                    text="worker completed",
                    session_id="creator-session",
                    display_metadata={
                        "source": "kanban",
                        "internal": True,
                        "kind": "kanban_wake",
                        "event_id": 17,
                        "event_kind": "completed",
                        "task_id": "task-17",
                        "run_id": 23,
                    },
                )
                kwargs = run_agent.await_args.kwargs
        finally:
            await runner.cleanup()
        return kwargs

    kwargs = asyncio.run(run())
    assert kwargs["session_id"] == "creator-session"
    assert kwargs["persist_user_provenance"] == {
        "origin_kind": "agent",
        "turn_kind": "continuation",
        "trust_kind": "trusted_internal",
        "provenance_metadata": {
            "producer": "gateway_wake",
            "source": "kanban",
            "event_kind": "kanban_wake",
            "platform": "api_server",
            "event_id": 17,
            "task_id": "task-17",
            "run_id": 23,
            "session_id": "creator-session",
        },
    }
