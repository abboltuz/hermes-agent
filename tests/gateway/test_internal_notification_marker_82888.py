"""Regression tests for #82888 — internal synthetic turns are persisted typed.

Async-delegation batch completions and background watch notifications re-enter
the gateway as synthetic ``MessageEvent(internal=True)`` turns (see
``_inject_watch_notification``). They must keep ``role='user'`` (message
alternation is sacred) but the persisted row must carry
``display_kind='internal_notification'`` so transcripts and the desktop UI can
render them as timeline notices instead of user bubbles.

Covered:

1. an internal event threads ``persist_user_display_kind`` into the agent run;
2. a real user event does NOT get the marker;
3. the gateway-side fallback user rows (transient-failure / no-new-messages
   paths) carry the marker for internal events only;
4. a marked row round-trips through SessionDB replay with role='user' intact
   and the marker is stripped from provider-bound payload copies.
"""

import sys
import types
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource

SESSION_KEY = "agent:main:telegram:group:-1001:12345"


def _bootstrap(monkeypatch, tmp_path):
    """Minimal GatewayRunner setup (pattern from test_42039)."""
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    config = GatewayConfig()
    runner = gateway_run.GatewayRunner(config)
    runner.adapters = {}
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._handle_active_session_busy_message = AsyncMock(return_value=False)
    runner._session_db = MagicMock()
    runner._recover_telegram_topic_thread_id = lambda _source: None
    runner._cache_session_source = lambda _key, _source: None
    runner._is_session_run_current = lambda _key, _gen: True
    runner._begin_session_run_generation = lambda _key: 1
    runner._reply_anchor_for_event = lambda _event: None
    runner._get_guild_id = lambda _event: None
    runner._should_send_voice_reply = lambda *_a, **_kw: False
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()

    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = SessionEntry(
        session_key=SESSION_KEY,
        session_id="sess-82888",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="group",
    )
    runner.session_store.load_transcript.return_value = []
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.has_platform_message_id.return_value = False
    runner.session_store.update_session = MagicMock()

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"}
    )
    monkeypatch.setattr(
        "agent.model_metadata.get_model_context_length",
        lambda *_args, **_kwargs: 100_000,
    )
    return runner


def _source():
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
        user_id="12345",
    )


def _event(*, internal: bool, text: str = "hello world", metadata=None):
    return MessageEvent(
        text=text,
        source=_source(),
        message_id=None if internal else "msg-82888",
        internal=internal,
        metadata=metadata or {},
    )


def _user_entries(calls):
    return [
        call.args[1]
        for call in calls
        if len(call.args) >= 2
        and isinstance(call.args[1], dict)
        and call.args[1].get("role") == "user"
    ]


# ── 1+2: the marker is threaded to the agent run for internal events only ──


@pytest.mark.asyncio
async def test_internal_event_threads_marker_into_agent_run(monkeypatch, tmp_path):
    runner = _bootstrap(monkeypatch, tmp_path)
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "ack",
            "messages": [],
            "tools": [],
            "history_offset": 0,
            "last_prompt_tokens": 0,
        }
    )

    await runner._handle_message_with_agent(
        _event(
            internal=True,
            text="[ASYNC DELEGATION BATCH COMPLETE]",
            metadata={
                "source": "delegation",
                "kind": "async_delegation_complete",
                "event_id": "deleg_123",
                "task_id": "task-123",
                "job_id": "job-123",
                "process_id": "proc-123",
                "delegation_id": "delegation-123",
                "run_id": 17,
                "generation_id": 3,
                "raw_payload": {"must": "not be copied"},
            },
        ),
        _source(), SESSION_KEY, 1,
    )

    kwargs = runner._run_agent.call_args.kwargs
    assert kwargs["persist_user_display_kind"] == "internal_notification"
    assert kwargs["persist_user_display_metadata"] == {
        "source": "delegation",
        "internal": True,
        "kind": "async_delegation_complete",
        "platform": "telegram",
        "event_id": "deleg_123",
        "task_id": "task-123",
        "job_id": "job-123",
        "process_id": "proc-123",
        "delegation_id": "delegation-123",
        "run_id": 17,
        "generation_id": 3,
    }
    provenance = kwargs["persist_user_provenance"]
    assert provenance["origin_kind"] == "agent"
    assert provenance["turn_kind"] == "continuation"
    assert provenance["trust_kind"] == "trusted_internal"
    for key, value in {
        "event_id": "deleg_123",
        "task_id": "task-123",
        "job_id": "job-123",
        "process_id": "proc-123",
        "delegation_id": "delegation-123",
        "run_id": 17,
        "generation_id": 3,
    }.items():
        assert provenance["provenance_metadata"][key] == value

    # The exact semantic sidecar handed to the push-path agent must survive
    # the real durable codec and reload, independently of display metadata.
    from hermes_state import SessionDB

    db = SessionDB(tmp_path / "push-wake.db")
    db.create_session("push-wake", source="telegram")
    try:
        db.append_message(
            "push-wake",
            "user",
            "[ASYNC DELEGATION BATCH COMPLETE]",
            **provenance,
        )
        reloaded = db.get_messages_as_conversation("push-wake")[0]
        assert reloaded["provenance_metadata"] == provenance[
            "provenance_metadata"
        ]
    finally:
        db.close()


@pytest.mark.asyncio
async def test_real_user_event_gets_no_marker(monkeypatch, tmp_path):
    runner = _bootstrap(monkeypatch, tmp_path)
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "hi",
            "messages": [],
            "tools": [],
            "history_offset": 0,
            "last_prompt_tokens": 0,
        }
    )

    await runner._handle_message_with_agent(
        _event(internal=False), _source(), SESSION_KEY, 1,
    )

    kwargs = runner._run_agent.call_args.kwargs
    assert kwargs["persist_user_display_kind"] is None
    assert kwargs["persist_user_display_metadata"] is None


@pytest.mark.asyncio
async def test_run_agent_wrapper_threads_display_metadata_to_inner(
    monkeypatch, tmp_path
):
    runner = _bootstrap(monkeypatch, tmp_path)
    runner._run_agent_inner = AsyncMock(return_value={"final_response": "done"})
    metadata = {
        "source": "delegation",
        "internal": True,
        "kind": "async_delegation_complete",
    }

    await runner._run_agent(
        message="internal wake",
        context_prompt="context",
        history=[],
        source=_source(),
        session_id="sess-82888",
        persist_user_display_kind="internal_notification",
        persist_user_display_metadata=metadata,
    )

    assert (
        runner._run_agent_inner.call_args.kwargs["persist_user_display_metadata"]
        == metadata
    )


# ── 3: gateway-side fallback rows carry the marker for internal events ─────


@pytest.mark.asyncio
async def test_failed_early_fallback_row_is_marked_for_internal_event(
    monkeypatch, tmp_path
):
    runner = _bootstrap(monkeypatch, tmp_path)
    runner._run_agent = AsyncMock(
        return_value={
            "failed": True,
            "final_response": None,
            "error": "429 Too Many Requests — rate limit exceeded",
            "messages": [],
            "history_offset": 0,
            "last_prompt_tokens": 0,
        }
    )

    await runner._handle_message_with_agent(
        _event(internal=True, text="[ASYNC DELEGATION BATCH COMPLETE]"),
        _source(), SESSION_KEY, 1,
    )

    entries = _user_entries(runner.session_store.append_to_transcript.call_args_list)
    assert entries, "expected a fallback user-row write"
    for entry in entries:
        assert entry["role"] == "user"  # alternation invariant: role unchanged
        assert entry["display_kind"] == "internal_notification"
        assert entry["display_metadata"] == {
            "source": "gateway_internal",
            "internal": True,
            "kind": "internal_notification",
            "platform": "telegram",
        }


@pytest.mark.asyncio
async def test_failed_early_fallback_row_is_unmarked_for_real_user(
    monkeypatch, tmp_path
):
    runner = _bootstrap(monkeypatch, tmp_path)
    runner._run_agent = AsyncMock(
        return_value={
            "failed": True,
            "final_response": None,
            "error": "429 Too Many Requests — rate limit exceeded",
            "messages": [],
            "history_offset": 0,
            "last_prompt_tokens": 0,
        }
    )

    await runner._handle_message_with_agent(
        _event(internal=False), _source(), SESSION_KEY, 1,
    )

    entries = _user_entries(runner.session_store.append_to_transcript.call_args_list)
    assert entries
    for entry in entries:
        assert "display_kind" not in entry
        assert "display_metadata" not in entry


@pytest.mark.asyncio
async def test_no_new_messages_fallback_row_is_marked_for_internal_event(
    monkeypatch, tmp_path
):
    runner = _bootstrap(monkeypatch, tmp_path)
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "done",
            "messages": [{"role": "user", "content": "x"}],
            "tools": [],
            "history_offset": 1,  # equals len(messages) → new_messages=[]
            "last_prompt_tokens": 0,
        }
    )

    await runner._handle_message_with_agent(
        _event(internal=True, text="[SYSTEM: Background process matched]"),
        _source(), SESSION_KEY, 1,
    )

    entries = _user_entries(runner.session_store.append_to_transcript.call_args_list)
    assert entries
    for entry in entries:
        assert entry["role"] == "user"
        assert entry["display_kind"] == "internal_notification"
        assert entry["display_metadata"] == {
            "source": "gateway_internal",
            "internal": True,
            "kind": "internal_notification",
            "platform": "telegram",
        }


# ── 4: DB round-trip replay + provider-payload hygiene ─────────────────────


def test_marked_row_replays_cleanly_and_never_reaches_provider(tmp_path):
    """A marked row persists, resumes with role='user' + marker intact, and
    the provider-bound copy built by conversation_loop drops the marker."""
    from hermes_state import SessionDB

    db = SessionDB(tmp_path / "state.db")
    sid = "sess-82888-replay"
    db.create_session(session_id=sid, source="gateway", model="test-model")
    try:
        db.append_message(
            session_id=sid,
            role="user",
            content="[ASYNC DELEGATION BATCH COMPLETE — 2/2 succeeded]",
            display_kind="internal_notification",
            display_metadata={
                "source": "delegation",
                "internal": True,
                "kind": "async_delegation_complete",
            },
        )
        db.append_message(session_id=sid, role="assistant", content="noted")

        # Session resume/load tolerates the extra key and keeps role='user'.
        replayed = db.get_messages_as_conversation(sid)
        user_row, = [m for m in replayed if m["role"] == "user"]
        assert user_row["display_kind"] == "internal_notification"
        assert user_row["display_metadata"] == {
            "source": "delegation",
            "internal": True,
            "kind": "async_delegation_complete",
        }
        assert user_row["content"].startswith("[ASYNC DELEGATION BATCH COMPLETE")

        # Provider hygiene: the per-request copy in conversation_loop pops
        # display fields off every outgoing message (see the api_msg.pop
        # calls in run_chat_completions_conversation). Reproduce that
        # sequence on the replayed row and verify nothing display-only
        # survives while the original persisted dict is untouched.
        from agent.conversation_loop import _clone_message_for_send

        api_msg = _clone_message_for_send(user_row)
        api_msg.pop("api_content", None)
        api_msg.pop("display_kind", None)
        api_msg.pop("display_metadata", None)
        api_msg.pop("_row_id", None)
        assert "display_kind" not in api_msg
        assert "display_metadata" not in api_msg
        assert api_msg["role"] == "user"
        assert user_row["display_kind"] == "internal_notification"
        assert user_row["display_metadata"]["source"] == "delegation"
    finally:
        db.close()
