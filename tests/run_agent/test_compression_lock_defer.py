"""Lock-contended compression no-ops must soft-DEFER, never exhaust (#49874).

On main before this fix, nothing on the automatic compression paths consumed
the #69870 lock-skip signal (``agent._compression_skipped_due_to_lock``):

* a lock-loser preflight/pre-API no-op counted as "insufficient progress",
* the oversized request went to the provider anyway, and
* the lock-contended 413/overflow retry burned ``compression_attempts`` to
  the cap and returned ``compression_exhausted`` — which the gateway answers
  with a full session auto-reset (#9893/#35809).

A temporary lock defer misclassified as exhaustion == session wipe.

These tests pin the fix: when a compression pass returns its input unchanged
AND the type-pinned lock-skip flag is set, the attempt is refunded. A hard
provider-pressure path joins and reloads the concurrent compaction when its
durable lock API is available; only an unjoinable path ends with the legacy
soft ``compression_deferred`` result distinct from ``compression_exhausted``.

Salvaged from PR #49874 (@helix4u), rebuilt on the landed #69870 signal.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.conversation_compression import (
    CONCURRENT_COMPACTION_RESUME_STATUS,
    CONCURRENT_COMPACTION_WAIT_STATUS,
    compression_skipped_due_to_lock,
    wait_for_concurrent_compression,
)
from run_agent import AIAgent
import run_agent


LOCK_HOLDER = "pid=4242:tid=1:agent=deadbeef:nonce=abcd1234"


# ---------------------------------------------------------------------------
# Helpers (mirrors tests/run_agent/test_413_compression.py)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_compression_sleep(monkeypatch):
    import time as _time

    monkeypatch.setattr(_time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(run_agent, "jittered_backoff", lambda *a, **k: 0.0)


def _make_tool_defs(*names: str) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": n,
                "description": f"{n} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for n in names
    ]


def _mock_response(content="Hello", finish_reason="stop"):
    msg = SimpleNamespace(
        content=content,
        tool_calls=None,
        reasoning_content=None,
        reasoning=None,
    )
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    resp = SimpleNamespace(choices=[choice], model="test/model")
    resp.usage = None
    return resp


def _make_413_error(message="Request entity too large"):
    err = Exception(message)
    err.status_code = 413
    return err


def _make_overflow_error():
    return Exception(
        "Error code: 400 - {'type': 'error', 'error': {'type': "
        "'invalid_request_error', 'message': 'prompt is too long: "
        "233153 tokens > 200000 maximum'}}"
    )


@pytest.fixture()
def agent():
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        a.client = MagicMock()
        a._cached_system_prompt = "You are helpful."
        a._use_prompt_caching = False
        a.tool_delay = 0
        a.compression_enabled = True
        a.save_trajectories = False
        return a


_PREFILL = [
    {"role": "user", "content": "previous question"},
    {"role": "assistant", "content": "previous answer"},
]


def _lock_skipping_compress(agent, *, holder=LOCK_HOLDER):
    """A compress double that no-ops because 'another path holds the lock'.

    Mirrors the real ``compress_context`` lock-contended abort: returns the
    INPUT list object unchanged and sets the #69870 lock-skip signal.
    """

    def _compress(messages, _system_message, **_kwargs):
        agent._compression_skipped_due_to_lock = holder
        return messages, "You are helpful."

    return _compress


def _plain_noop_compress(agent):
    """A compress double that no-ops WITHOUT lock contention (real no-progress)."""

    def _compress(messages, _system_message, **_kwargs):
        agent._compression_skipped_due_to_lock = None
        return messages, "You are helpful."

    return _compress


# ---------------------------------------------------------------------------
# Type-pinned signal read (MagicMock test-double immunity)
# ---------------------------------------------------------------------------


class TestLockSkipSignalTypePin:
    def test_true_and_holder_string_are_lock_skips(self):
        a = SimpleNamespace(_compression_skipped_due_to_lock=True)
        assert compression_skipped_due_to_lock(a) is True
        a = SimpleNamespace(_compression_skipped_due_to_lock=LOCK_HOLDER)
        assert compression_skipped_due_to_lock(a) is True

    def test_none_and_missing_are_not_lock_skips(self):
        assert compression_skipped_due_to_lock(
            SimpleNamespace(_compression_skipped_due_to_lock=None)
        ) is False
        assert compression_skipped_due_to_lock(SimpleNamespace()) is False

    def test_magicmock_agent_auto_attribute_is_not_a_lock_skip(self):
        """MagicMock agents auto-create truthy attributes; bare truthiness
        would hijack every mocked agent in sibling suites into the lock-skip
        branch (the #69870 × #69840 incident). The read must be type-pinned."""
        assert compression_skipped_due_to_lock(MagicMock()) is False

    def test_truthy_non_true_non_str_values_are_not_lock_skips(self):
        for junk in (1, 1.0, ["holder"], {"holder": True}, object(), MagicMock()):
            a = SimpleNamespace(_compression_skipped_due_to_lock=junk)
            assert compression_skipped_due_to_lock(a) is False, junk


class TestConcurrentCompressionJoin:
    def test_in_place_winner_is_reloaded_and_request_can_continue(self):
        durable_before = [
            {"role": "user", "content": "old task", "_row_id": 1},
            {"role": "assistant", "content": "old answer", "_row_id": 2},
            {"role": "user", "content": "current task", "_row_id": 3},
        ]
        compacted = [
            {"role": "user", "content": "summary", "_row_id": 7},
            {"role": "assistant", "content": "ready", "_row_id": 8},
            {"role": "user", "content": "current task", "_row_id": 9},
        ]

        class DB:
            def __init__(self):
                self.holders = [LOCK_HOLDER, LOCK_HOLDER, None]
                self.loads = [durable_before, compacted]

            def get_compression_lock_holder(self, _session_id):
                return self.holders.pop(0)

            def get_session(self, _session_id):
                return {"ended_at": None, "end_reason": None}

            def get_messages_as_conversation(self, _session_id, **_kwargs):
                return [dict(message) for message in self.loads.pop(0)]

        statuses = []
        compressor = SimpleNamespace()
        joining_agent = SimpleNamespace(
            _compression_skipped_due_to_lock=LOCK_HOLDER,
            _session_db=DB(),
            session_id="session-1",
            _emit_status=statuses.append,
            _interrupt_requested=False,
            _cached_system_prompt="policy",
            context_compressor=compressor,
            tools=[],
        )
        original = [{"role": "user", "content": "current task"}]

        result = wait_for_concurrent_compression(
            joining_agent,
            original,
            timeout_seconds=1,
            poll_interval_seconds=0.01,
        )

        assert result == compacted
        assert statuses == [
            CONCURRENT_COMPACTION_WAIT_STATUS,
            CONCURRENT_COMPACTION_RESUME_STATUS,
        ]
        assert joining_agent._compression_skipped_due_to_lock is None
        assert joining_agent._last_compression_attempt_in_place is True
        assert joining_agent._last_flushed_db_idx == len(compacted)
        assert compressor.last_prompt_tokens == -1
        assert compressor.awaiting_real_usage_after_compression is True

    @staticmethod
    def _real_db_agent(db, session_id):
        return SimpleNamespace(
            _compression_skipped_due_to_lock=LOCK_HOLDER,
            _session_db=db,
            session_id=session_id,
            _emit_status=MagicMock(),
            _interrupt_requested=False,
            _cached_system_prompt="policy",
            context_compressor=SimpleNamespace(),
            tools=[],
            _memory_manager=None,
            platform="cli",
        )

    def test_real_session_db_proves_in_place_commit(self, tmp_path):
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "join-in-place.db")
        session_id = "join-in-place"
        db.create_session(session_id=session_id, source="test")
        original = [
            {"role": "user", "content": "old task"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "current task"},
        ]
        compacted = [
            {"role": "user", "content": "summary"},
            {"role": "user", "content": "current task"},
        ]
        db.append_messages_batch(session_id, original)
        watermark = db.get_active_message_watermark(session_id)
        assert db.try_acquire_compression_lock(
            session_id, LOCK_HOLDER, ttl_seconds=60
        )
        original_holder_getter = db.get_compression_lock_holder
        polls = 0

        def finish_on_second_poll(_session_id):
            nonlocal polls
            polls += 1
            if polls == 1:
                return original_holder_getter(session_id)
            db.archive_and_compact(
                session_id,
                compacted,
                watermark=watermark,
                lock_holder=LOCK_HOLDER,
            )
            db.release_compression_lock(session_id, LOCK_HOLDER)
            return None

        db.get_compression_lock_holder = finish_on_second_poll
        joining_agent = self._real_db_agent(db, session_id)

        result = wait_for_concurrent_compression(
            joining_agent,
            original,
            timeout_seconds=1,
            poll_interval_seconds=0.01,
        )

        assert [message["content"] for message in result] == [
            "summary",
            "current task",
        ]
        assert joining_agent._last_compaction_in_place is True
        assert db.has_archived_messages(session_id) is True
        db.close()

    def test_real_session_db_does_not_adopt_aborted_owner(self, tmp_path):
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "join-aborted.db")
        session_id = "join-aborted"
        db.create_session(session_id=session_id, source="test")
        original = [{"role": "user", "content": "unchanged task"}]
        db.append_messages_batch(session_id, original)
        assert db.try_acquire_compression_lock(
            session_id, LOCK_HOLDER, ttl_seconds=60
        )
        original_holder_getter = db.get_compression_lock_holder
        polls = 0

        def abort_on_second_poll(_session_id):
            nonlocal polls
            polls += 1
            if polls == 1:
                return original_holder_getter(session_id)
            db.release_compression_lock(session_id, LOCK_HOLDER)
            return None

        db.get_compression_lock_holder = abort_on_second_poll
        joining_agent = self._real_db_agent(db, session_id)

        result = wait_for_concurrent_compression(
            joining_agent,
            original,
            timeout_seconds=1,
            poll_interval_seconds=0.01,
        )

        assert result is None
        assert not hasattr(joining_agent, "_last_compaction_in_place")
        assert db.has_archived_messages(session_id) is False
        db.close()

    def test_real_session_db_adopts_rotated_child(self, tmp_path):
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "join-rotation.db")
        parent_id = "join-parent"
        child_id = "join-child"
        db.create_session(session_id=parent_id, source="test")
        original = [{"role": "user", "content": "old task"}]
        compacted = [{"role": "user", "content": "rotated summary"}]
        db.append_messages_batch(parent_id, original)
        watermark = db.get_active_message_watermark(parent_id)
        assert db.try_acquire_compression_lock(
            parent_id, LOCK_HOLDER, ttl_seconds=60
        )
        original_holder_getter = db.get_compression_lock_holder
        polls = 0

        def rotate_on_second_poll(_session_id):
            nonlocal polls
            polls += 1
            if polls == 1:
                return original_holder_getter(parent_id)
            db.publish_compression_child(
                parent_session_id=parent_id,
                child_session_id=child_id,
                source="test",
                messages=compacted,
                compression_lock_holder=LOCK_HOLDER,
                watermark=watermark,
            )
            db.release_compression_lock(parent_id, LOCK_HOLDER)
            return None

        db.get_compression_lock_holder = rotate_on_second_poll
        joining_agent = self._real_db_agent(db, parent_id)

        result = wait_for_concurrent_compression(
            joining_agent,
            original,
            timeout_seconds=1,
            poll_interval_seconds=0.01,
        )

        assert [message["content"] for message in result] == ["rotated summary"]
        assert joining_agent.session_id == child_id
        assert joining_agent._last_compaction_in_place is False
        db.close()


# ---------------------------------------------------------------------------
# 413 handler: lock-contended no-op → soft defer, no exhaustion
# ---------------------------------------------------------------------------


class TestLockContended413Defer:
    def test_lock_contended_413_returns_compression_deferred(self, agent):
        """A 413 whose compression pass lost the lock must end the turn as a
        soft ``compression_deferred`` — never ``compression_exhausted``."""
        agent.client.chat.completions.create.side_effect = _make_413_error()

        with (
            patch.object(
                agent, "_compress_context",
                side_effect=_lock_skipping_compress(agent),
            ) as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello", conversation_history=list(_PREFILL))

        mock_compress.assert_called_once()
        assert result.get("compression_deferred") is True
        assert not result.get("compression_exhausted")
        # Soft defer: transient, retry-next-message semantics — the gateway
        # persists the user turn (failed=False) and never auto-resets.
        assert result.get("failed") is False
        assert result.get("completed") is False
        assert result.get("partial") is True


    def test_unconfirmed_lock_skip_true_also_defers(self, agent):
        """``_compression_skipped_due_to_lock = True`` (holder unconfirmed —
        ``try_acquire`` swallowed a sqlite error) is still a lock skip."""
        agent.client.chat.completions.create.side_effect = _make_413_error()

        with (
            patch.object(
                agent, "_compress_context",
                side_effect=_lock_skipping_compress(agent, holder=True),
            ),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello", conversation_history=list(_PREFILL))

        assert result.get("compression_deferred") is True
        assert not result.get("compression_exhausted")




# ---------------------------------------------------------------------------
# Pre-API gate: a lock-skipped pass must not burn the shared attempt budget
# ---------------------------------------------------------------------------


class TestPreApiLockDeferDoesNotBurnBudget:
    def test_lock_loser_turn_recovers_after_lock_release(self, agent):
        """End-to-end shape of the live bug at cap=1:

        1. Pre-API pressure gate fires; the compression pass loses the lock
           (no-op + lock-skip flag). Pre-fix this burned the single shared
           attempt.
        2. The oversized request goes to the provider → 413.
        3. The 413 handler compresses again — the lock has been released and
           the pass now succeeds — and the retry completes.

        Pre-fix, step 3 found ``compression_attempts`` already at the cap and
        returned ``compression_exhausted`` → gateway session wipe. The defer
        refund keeps the budget intact for the provider-proven retry.
        """
        agent.max_compression_attempts = 1
        # Compressor stub: pressure only on the fully-assembled request
        # (pre-API site); the turn-context preflight stands down via the
        # cheap-gate (small message count) and low turn-context estimate.
        agent.context_compressor = SimpleNamespace(
            protect_first_n=3,
            protect_last_n=20,
            threshold_tokens=100_000,
            context_length=1_000_000,
            last_prompt_tokens=0,
            should_compress=lambda t: t >= 100_000,
            should_defer_preflight_to_real_usage=lambda _t: False,
            get_active_compression_failure_cooldown=lambda: None,
        )

        agent.client.chat.completions.create.side_effect = [
            _make_413_error(),
            _mock_response(content="Recovered after lock release"),
        ]

        compress_calls = []

        def _lock_then_success(messages, _system_message, **_kwargs):
            compress_calls.append(len(messages))
            if len(compress_calls) == 1:
                # Lock loser: no-op + #69870 signal.
                agent._compression_skipped_due_to_lock = LOCK_HOLDER
                return messages, "You are helpful."
            # Lock released: real compaction (entry clears the signal).
            agent._compression_skipped_due_to_lock = None
            return (
                [{"role": "user", "content": "hello"}],
                "You are helpful.",
            )

        with (
            patch(
                "agent.turn_context.estimate_request_tokens_rough",
                return_value=10,
            ),
            patch(
                "agent.conversation_loop.estimate_request_tokens_rough",
                return_value=500_000,
            ),
            patch(
                "agent.conversation_loop.estimate_messages_tokens_rough",
                return_value=500_000,
            ),
            patch.object(agent, "_compress_context", side_effect=_lock_then_success),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello", conversation_history=list(_PREFILL))

        # Pass 1: pre-API (lock defer, refunded). Pass 2: 413 handler
        # (succeeds within the cap because the defer did not count).
        assert len(compress_calls) == 2
        assert result.get("completed") is True
        assert result["final_response"] == "Recovered after lock release"
        assert not result.get("compression_exhausted")
        assert not result.get("compression_deferred")
