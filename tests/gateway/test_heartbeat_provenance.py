from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.run import GatewayRunner
from gateway.session import SessionSource


@pytest.mark.asyncio
async def test_heartbeat_poller_enqueues_typed_internal_turn(monkeypatch):
    from hermes_cli import heartbeat as heartbeat_mod

    class FakeHeartbeatManager:
        fired = False

        def __init__(self, session_id: str):
            self.session_id = session_id
            self.state = SimpleNamespace(fire_count=0)

        def has_heartbeat(self) -> bool:
            return True

        def due_prompt(self):
            if self.__class__.fired:
                return None
            self.__class__.fired = True
            self.state.fire_count = 1
            return "[Heartbeat check] inspect progress"

    monkeypatch.setattr(heartbeat_mod, "HeartbeatManager", FakeHeartbeatManager)
    monkeypatch.setattr(heartbeat_mod, "POLL_SECONDS", 0)

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="chat-1",
        chat_type="dm",
    )
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._heartbeat_watch = {"quick-key": (source, "session-1")}
    runner._heartbeat_poll_task = None  # type: ignore[assignment]
    runner._background_tasks = set()
    runner._running_agents = {}

    async def warm_goals_session_db(label: str) -> None:
        await asyncio.sleep(0)

    runner._warm_goals_session_db = warm_goals_session_db  # type: ignore[method-assign]
    adapter = object()

    def adapter_for_source(source):
        return adapter

    runner._adapter_for_source = adapter_for_source  # type: ignore[method-assign]

    captured = {}
    enqueued = asyncio.Event()

    def capture_fifo(session_key, queued_event, adapter):
        captured.update(
            quick_key=session_key,
            event=queued_event,
            adapter=adapter,
        )
        enqueued.set()

    runner._enqueue_fifo = capture_fifo  # type: ignore[method-assign]
    runner._start_heartbeat_poller()
    try:
        await asyncio.wait_for(enqueued.wait(), timeout=1)
    finally:
        runner._heartbeat_poll_task.cancel()
        await asyncio.gather(runner._heartbeat_poll_task, return_exceptions=True)

    event = captured["event"]
    assert captured["quick_key"] == "quick-key"
    assert captured["adapter"] is adapter
    assert event.internal is True
    assert event.metadata == {
        "source": "heartbeat",
        "internal": True,
        "kind": "heartbeat_tick",
        "session_id": "session-1",
        "event_id": "session-1:1",
    }
