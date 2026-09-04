"""Exercise the public compression boundary with real profile-scoped storage."""

from types import SimpleNamespace

import pytest

from agent import conversation_compression as compression
from hermes_state import SessionDB


def agent(db, *, generation=0, model="route-a"):
    return SimpleNamespace(
        session_id="chat", _session_db=db,
        _conversation_root_id=lambda: "chat",
        _compression_generation=generation,
        context_compressor=SimpleNamespace(model=model, context_length=8192),
    )


def test_restart_and_new_generation_cannot_repeat_same_failure(tmp_path, monkeypatch):
    calls = []
    messages = [{"role": "user", "content": "continue"}]

    def no_progress(_agent, messages, system_message, **_kwargs):
        calls.append(1)
        return messages, system_message

    monkeypatch.setattr(compression, "_compress_context_impl", no_progress)
    path = tmp_path / "state.db"
    with SessionDB(path) as db:
        db.create_session("chat", "test")
        compression.compress_context(agent(db), messages, "policy", trigger="preflight")
    with SessionDB(path) as db:
        resumed = agent(db, generation=999)
        result = compression.compress_context(resumed, messages, "policy", trigger="overflow")
        assert result[0] is messages
        assert len(calls) == 1
        compression.compress_context(resumed, messages, "policy", trigger="manual", force=True)
        assert len(calls) == 2
        compression.compress_context(agent(db, model="route-b"), messages, "policy", trigger="preflight")
        assert len(calls) == 3


def test_receipt_records_cancellation_and_preserves_exception(tmp_path, monkeypatch):
    messages = [{"role": "user", "content": "continue"}]
    calls = []

    def interrupted(*_args, **_kwargs):
        calls.append(1)
        raise KeyboardInterrupt()

    monkeypatch.setattr(compression, "_compress_context_impl", interrupted)
    path = tmp_path / "state.db"
    with SessionDB(path) as db:
        db.create_session("chat", "test")
        with pytest.raises(KeyboardInterrupt):
            compression.compress_context(agent(db), messages, "policy", trigger="preflight")
    with SessionDB(path) as db:
        compression.compress_context(agent(db), messages, "policy", trigger="preflight")
    assert len(calls) == 1


def test_admission_storage_failure_does_not_start_summarizer_or_mutate_history(tmp_path, monkeypatch):
    messages = [{"role": "user", "content": "continue"}]

    def unavailable(*_args, **_kwargs):
        raise OSError("storage unavailable")

    def unexpected(*_args, **_kwargs):
        pytest.fail("summarizer must not run without durable admission")

    with SessionDB(tmp_path / "state.db") as db:
        db.create_session("chat", "test")
        db.append_messages_batch("chat", messages)
        before = db.get_messages("chat")
        monkeypatch.setattr(db, "claim_context_compaction", unavailable)
        monkeypatch.setattr(compression, "_compress_context_impl", unexpected)
        result = compression.compress_context(agent(db), messages, "policy", trigger="preflight")
        assert result == (messages, "policy")
        assert result[0] is messages
        assert db.get_messages("chat") == before


def test_failed_receipt_does_not_turn_success_into_error(tmp_path, monkeypatch):
    messages = [{"role": "user", "content": "continue"}]
    result = ([{"role": "user", "content": "compact"}], "policy")
    monkeypatch.setattr(compression, "_compress_context_impl", lambda *_a, **_k: result)

    def unavailable(*_args, **_kwargs):
        raise OSError("storage unavailable")

    with SessionDB(tmp_path / "state.db") as db:
        db.create_session("chat", "test")
        monkeypatch.setattr(db, "finish_context_compaction", unavailable)
        assert compression.compress_context(agent(db), messages, "policy", trigger="preflight") is result
        # No receipt means no automatic repeat, even from a new process owner.
        assert db.claim_context_compaction("chat", "another", "strategy", owner="new") == "deferred_lock"


def test_unreadable_policy_does_not_start_work_or_change_context(tmp_path, monkeypatch):
    def unavailable(*_args, **_kwargs):
        raise OSError("config unavailable")

    monkeypatch.setattr(compression, "_compression_strategy_fingerprint", unavailable)
    with SessionDB(tmp_path / "state.db") as db:
        db.create_session("chat", "test")
        messages = [{"role": "user", "content": "continue"}]
        result = compression.compress_context(agent(db), messages, "policy", trigger="preflight")
        assert result[0] is messages
        assert result[1] == "policy"
